# Oracul MCP Server

A [Model Context Protocol](https://modelcontextprotocol.io) server that exposes
the Oracul's capabilities to AI agents (Claude Desktop/Code, custom agents) as
native tools. It is **mounted on the FastAPI app** (`api/`), so it ships with the
Oracul on the same host, nginx, and auto-deploy — it is not a separate service.

- **Endpoint:** `https://oracle.daatan.com/mcp` (Streamable HTTP, stateless)
- **Auth:** OAuth 2.1 — the Oracul is a Resource Server verifying AWS Cognito
  access tokens (see [Auth](#auth))
- **Code:** `api/src/forecast_api/mcp_server.py` (tools), `mcp_auth.py` (token
  verification), mounted in `main.py`

The REST API is unchanged and still uses the `x-api-key` header; `/mcp` is
OAuth-only. Non-agent/service automation should keep calling REST directly.

## Tools

Built for a **Polymarket trader** workflow — take a market, get an independent
probability, compare to the price, find an edge — plus general forecasting.

| Tool | Scope | Rate limit | What it does |
|------|-------|------------|--------------|
| `polymarket_edge(market, edge_threshold=0.05)` | `forecast` | 10/min | Resolve a market (URL/slug/Gamma id) → live YES price; forecast its question; return the **edge** (oracle − market) and a **suggested side** (BUY YES / BUY NO / NO EDGE) |
| `polymarket_market(market)` | `read` | 30/min | Live market data: question, outcomes, prices, volume, end date |
| `forecast(question, max_articles?, verbose?)` | `forecast` | 10/min | Calibrated probability for any binary question (`probability` in [0,1]). `verbose=true` returns the full `ForecastResponse` (`std`/`n_eff`/`evidence_mass`/`claims_detail`/`provenance`) instead of the trimmed default payload — retro#593 |
| `search_news(query, limit?, date_from?, date_to?)` | `read` | 60/min | News search via the provider chain |
| `fetch_article(url)` | `read` | 30/min | Fetch + extract one article (SSRF-guarded) |
| `bayes_nodes(observations?)` | `read` | none | BayesOracle Israeli-politics DAG probabilities |
| `source_leaderboard()` | `read` | none | Live source-credibility ranking |

`forecast` / `polymarket_edge` are **slow** (a live news + LLM pipeline — tens of
seconds, occasionally past the 120s proxy timeout) and cost LLM + search credits,
which is why they sit behind the `oracle-mcp/forecast` scope.

**Rate limiting** (`mcp_limiter.py`) is enforced per tool call, independently of
the REST API's `@limiter.limit(...)` (slowapi) — MCP tools call the Oracul's
service functions in-process and never pass through those decorated REST
routes, so they need their own gate (retro#431). It mirrors each tool's REST
equivalent (see table above) but keys on the caller's Cognito identity
(`sub` for human PKCE tokens, `client_id` for M2M) rather than remote address,
since MCP traffic can arrive from a shared client-side proxy. Same
in-memory-per-worker caveat as the REST limiter: not shared across gunicorn
workers.

> `polymarket_edge` is **informational, not financial advice**, and never places
> orders. It surfaces the sign of an edge only; only binary yes/no markets are
> supported. Compliance/eligibility is the trader's responsibility.

## Auth

MCP-native OAuth 2.1. The **Authorization Server** is an AWS Cognito user pool
(Google as the upstream login); the **Resource Server** is this `/mcp` mount,
which verifies the Cognito JWT (`iss`, `token_use=access`, `client_id`, `exp`,
signature via JWKS) and maps `scope` claims to tools.

- **Scopes:** `oracle-mcp/read` (the global floor, all cheap tools) and
  `oracle-mcp/forecast` (the two expensive tools).
- **Humans** authenticate with Authorization Code + PKCE via the Cognito hosted
  UI — either a native Cognito account (admin-invite only) or self-service
  Google sign-in, once Google federation is applied (see Provisioning §1). See
  the DCR façade below. **Services** (e.g. daatan's own agents, or an
  individually-provisioned trader M2M client) use the `client_credentials`
  grant.
- **Discovery:** the protected-resource metadata (RFC 9728) is served at
  `https://oracle.daatan.com/.well-known/oauth-protected-resource/mcp`; a 401
  returns `WWW-Authenticate` pointing there.

### DCR façade (human Claude-connector login)

Claude's MCP connector **requires** OAuth Dynamic Client Registration (RFC 7591):
it discovers the authorization server, fetches that server's
`/.well-known/oauth-authorization-server`, and hard-fails ("does not support
dynamic client registration") when there is no `registration_endpoint` — it will
not fall back to a static `client_id`. Cognito publishes no such endpoint.

So when `COGNITO_HOSTED_UI_DOMAIN` + `COGNITO_CLAUDE_CLIENT_ID` are set
(`config.dcr_enabled`), the Oracul origin advertises **itself** as the
authorization server (`api/src/forecast_api/mcp_dcr.py`):

- the protected-resource metadata points `authorization_servers` at the Oracul
  origin instead of the Cognito issuer;
- the origin serves `/.well-known/oauth-authorization-server` (and
  `/.well-known/openid-configuration`) that carry the pool's `jwks_uri`, inject
  **our** `registration_endpoint`, and point `authorization_endpoint` /
  `token_endpoint` at our own `/oauth2/authorize` + `/oauth2/token` proxies;
- `POST /register` ignores the request and returns the one pre-provisioned public
  PKCE client (`COGNITO_CLAUDE_CLIENT_ID`) every time — **no** Cognito client is
  created (no `CreateUserPoolClient`, so no IAM grant, no reaping, no abuse vector);
  Claude accepts a fixed `client_id` and only loops if handed a new one each call.
- `GET /oauth2/authorize` and `POST /oauth2/token` proxy Cognito's real endpoints
  but **strip the `offline_access` scope**. Claude always requests `offline_access`
  (to get a refresh token); Cognito doesn't recognise that scope name and rejects
  the whole authorize with `invalid_scope`, which bounced the login before the user
  could sign in. Cognito issues refresh tokens from the client's own config anyway,
  so stripping the scope costs nothing — Claude still gets its refresh token.

The login and token exchange run through those proxies to Cognito, so the access
token is still Cognito-issued (`iss` = the pool) and the Resource-Server verifier is
unchanged — it just needs `COGNITO_CLAUDE_CLIENT_ID` in `COGNITO_ALLOWED_CLIENT_IDS`.
`offline_access` is also not advertised in `scopes_supported`. With the two vars
unset, none of this is served and the metadata points straight at the Cognito
issuer (the M2M-only path, unchanged).

**Target claude.ai / Claude Desktop** — both use the single fixed callback
`https://claude.ai/api/mcp/auth_callback`. Claude Code's CLI uses an ephemeral
loopback port that Cognito's exact redirect-URI match can't accept; pin a fixed
port (`http://localhost:8080/callback`, registered on the client) if you need it.

### Adding to a client

Once Google federation is applied (§1), a human needs **no prior account** —
they just sign in with their own Google account on first connect (see below).
This is the self-service trader path.

Native Cognito accounts remain available as an admin-invite alternative (e.g.
for someone without/unwilling to use a Google account):

```bash
aws cognito-idp admin-create-user --region eu-central-1 \
  --user-pool-id "$POOL_ID" --username trader@example.com \
  --user-attributes Name=email,Value=trader@example.com Name=email_verified,Value=true
# Cognito emails a temporary password; the hosted UI forces a reset on first login.
```

Then in **claude.ai** (or Claude Desktop): Settings → Connectors → Add custom
connector → `https://oracle.daatan.com/mcp`, and complete the Cognito login when
prompted — **Sign in with Google** appears alongside the native username/password
form once §1 is applied. (Claude Code's `claude mcp add --transport http oracle
https://oracle.daatan.com/mcp` also works, but its ephemeral loopback port must be
pinned to a registered callback — see the DCR façade note above.)

For a service (M2M), fetch a token and call with a bearer header. Request **both**
scopes — the RS enforces a global `oracle-mcp/read` floor, so a forecast-only token
is 403'd before it reaches a tool:

```bash
TOKEN=$(curl -s -X POST "https://<pool-domain>.auth.eu-central-1.amazoncognito.com/oauth2/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "grant_type=client_credentials&client_id=$CID&client_secret=$CSECRET" \
  --data-urlencode "scope=oracle-mcp/read oracle-mcp/forecast" \
  | python -c "import sys,json;print(json.load(sys.stdin)['access_token'])")
```

## Enabling / config

The whole `/mcp` mount is **conditional on Cognito config** (`config.mcp_enabled`):
if `COGNITO_USER_POOL_ID` is unset the endpoint is simply not mounted, so a deploy
without the env boots the REST API normally. Set in the box's
`/home/ubuntu/truthmachine/.env` (see `.env.example`):

```
COGNITO_USER_POOL_ID=eu-central-1_XXXXXXXXX
# COGNITO_REGION=eu-central-1            # derived from the pool-id prefix if omitted
COGNITO_ALLOWED_CLIENT_IDS=<claude-client-id>,<m2m-client-id>
# MCP_RESOURCE_URL=https://oracle.daatan.com/mcp
# Human Claude login (DCR façade) — set BOTH or neither:
# COGNITO_HOSTED_UI_DOMAIN=https://daatan-oracle.auth.eu-central-1.amazoncognito.com
# COGNITO_CLAUDE_CLIENT_ID=<claude-client-id>
```

The Cognito user pool itself is provisioned in `terraform/cognito.tf`.

## Provisioning runbook (first-time enablement)

End-to-end steps to flip `/mcp` from inert 404 to live. All AWS work is
`eu-central-1`, account `272007598366`; the Oracul box (`i-00ac444b94c5ff9b2`) is
**SSM-only, no SSH**.

### 1. Google federation (self-service trader sign-in)

`terraform/cognito.tf` already carries the Google IdP resource and the claude
client's `supported_identity_providers` — nothing to uncomment. This step is
just the external prerequisite: create a Google OAuth 2.0 client (Cloud
Console — this part can't be done via CLI/Terraform, a human has to click
through it) whose redirect URI is
`https://daatan-oracle.auth.eu-central-1.amazoncognito.com/oauth2/idpresponse`,
then store its creds (read via a data source — never a literal in tf/state):

```bash
aws ssm put-parameter --region eu-central-1 \
  --name /retro/prod/secrets/COGNITO_GOOGLE_OAUTH --type SecureString \
  --value '{"client_id":"<GOOGLE_CLIENT_ID>","client_secret":"<GOOGLE_CLIENT_SECRET>"}'
```

The `data "aws_ssm_parameter" "google_oauth"` block fails any `plan`/`apply`
in this file until the parameter above exists — create it first.

Native Cognito accounts (admin-invite only, `admin-create-user`) still work
independently of this — Google federation is additive, not a replacement.
**Note:** `allow_admin_create_user_only` on the pool does **not** gate
federated logins, only the native sign-up form — once the Google IdP is
applied, *any* Google account holder can sign in via the claude client. That's
intentional here (self-service trader onboarding), not an oversight.

### 2. Apply Cognito (`-target`, in dependency order)

Never a blanket apply (workspace rule). From `terraform/` on the latest `main`,
pass the **real** Claude redirect URIs (the tf default is a placeholder):

```bash
cd terraform
terraform init          # real S3 backend (state key retro/)
# export TF_VAR_cognito_domain_prefix=daatan-oracle   # only if the default is taken

# Pool, resource server, domain, m2m client, and claude client already exist.
# Adding Google federation only touches two resources — apply just those, IdP first
# (the claude client's depends_on enforces that order, but -target doesn't chain
# dependencies across separate invocations, so do it explicitly):
terraform apply -target=aws_cognito_identity_provider.google  # requires the Secrets Manager secret to exist first (§1)
terraform apply -target=aws_cognito_user_pool_client.claude   # adds "Google" to supported_identity_providers

# Capture what the box + smoke tests need:
terraform output -raw cognito_user_pool_id
terraform output -raw cognito_allowed_client_ids
terraform output -raw cognito_claude_client_id
terraform output -raw cognito_hosted_ui_domain
terraform output -raw cognito_m2m_client_secret     # sensitive (M2M only)
```

### 3. Set the env on the box + restart (SSM)

`systemd` only re-reads `EnvironmentFile` on **restart**, not on the deploy's SIGHUP
reload — so this needs a real `restart` (one-time ~2–5s 502 window). SSM runs as
root; the `chown` keeps `.env` owned by `ubuntu`.

```bash
# from terraform/
POOL_ID=$(terraform output -raw cognito_user_pool_id)
CLIENT_IDS=$(terraform output -raw cognito_allowed_client_ids)
HOSTED_UI=$(terraform output -raw cognito_hosted_ui_domain)   # DCR façade (human login)
CLAUDE_CID=$(terraform output -raw cognito_claude_client_id)  # DCR façade (human login)

CMD_ID=$(aws ssm send-command \
  --region eu-central-1 \
  --instance-ids i-00ac444b94c5ff9b2 \
  --document-name AWS-RunShellScript \
  --comment "enable Oracul MCP: Cognito env + restart" \
  --parameters "{\"commands\":[
    \"sed -i '/^COGNITO_USER_POOL_ID=/d;/^COGNITO_ALLOWED_CLIENT_IDS=/d;/^MCP_RESOURCE_URL=/d;/^COGNITO_HOSTED_UI_DOMAIN=/d;/^COGNITO_CLAUDE_CLIENT_ID=/d' /home/ubuntu/truthmachine/.env\",
    \"echo COGNITO_USER_POOL_ID=$POOL_ID >> /home/ubuntu/truthmachine/.env\",
    \"echo COGNITO_ALLOWED_CLIENT_IDS=$CLIENT_IDS >> /home/ubuntu/truthmachine/.env\",
    \"echo MCP_RESOURCE_URL=https://oracle.daatan.com/mcp >> /home/ubuntu/truthmachine/.env\",
    \"echo COGNITO_HOSTED_UI_DOMAIN=$HOSTED_UI >> /home/ubuntu/truthmachine/.env\",
    \"echo COGNITO_CLAUDE_CLIENT_ID=$CLAUDE_CID >> /home/ubuntu/truthmachine/.env\",
    \"chown ubuntu:ubuntu /home/ubuntu/truthmachine/.env\",
    \"systemctl restart oracle-api\",
    \"sleep 4\",
    \"curl -s -o /dev/null -w health:%{http_code} http://127.0.0.1:8001/health\"
  ]}" \
  --query 'Command.CommandId' --output text)

aws ssm get-command-invocation --region eu-central-1 \
  --instance-id i-00ac444b94c5ff9b2 --command-id "$CMD_ID" \
  --query '{Status:Status, Out:StandardOutputContent, Err:StandardErrorContent}'
```

### 4. Verify

```bash
# /mcp was 404 (inert) — now 401 (mounted, needs a token):
curl -s -o /dev/null -w "%{http_code}\n" -X POST https://oracle.daatan.com/mcp \
  -H 'Content-Type: application/json' -H 'Accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'

# Discovery metadata — with the DCR façade on, authorization_servers is the Oracul
# origin, and the AS metadata then advertises our registration_endpoint:
curl -s https://oracle.daatan.com/.well-known/oauth-protected-resource/mcp
curl -s https://oracle.daatan.com/.well-known/oauth-authorization-server
curl -s -X POST https://oracle.daatan.com/register -H 'Content-Type: application/json' \
  -d '{"redirect_uris":["https://claude.ai/api/mcp/auth_callback"]}'   # → the static client_id

# M2M auth smoke test — with a valid token you must NOT get 401 (a JSON-RPC
# "initialize first" error is fine; it's past auth). Token request uses the M2M
# snippet in Auth (both scopes).

# Human flow: add https://oracle.daatan.com/mcp as a custom connector in
# claude.ai and sign in with Google (self-service) or a native Cognito account
# (admin-create-user, see Adding to a client).
```

**Rollback:** re-run the Step-3 SSM with just the `sed` delete line +
`systemctl restart oracle-api` — the Cognito env drops, `config.mcp_enabled` goes
false, and `/mcp` reverts to inert 404 with the REST API untouched.

## Operational notes

- **Stateless** transport (`stateless_http=True, json_response=True`) — required
  because the API runs under gunicorn with multiple workers and no sticky
  sessions.
- **No nginx change** — the existing `location /` catch-all proxies `/mcp` and the
  root `.well-known/*`. MCP calls share the per-IP 60 r/min budget.
- **Transport host allowlist** — the streamable-HTTP transport enforces DNS-rebinding
  protection. Because the app binds `127.0.0.1`, the SDK would otherwise allow only
  localhost and reject nginx-forwarded `Host: oracle.daatan.com` with `421 Invalid
  Host header`. The allowed `Host`/`Origin` values are derived from `MCP_RESOURCE_URL`
  (`config.mcp_allowed_hosts` / `mcp_allowed_origins`), so they follow the resource
  host automatically. **One hand-maintained exception**: `mcp_allowed_origins` also
  hardcodes `https://daatan.github.io` (same trust boundary as `main.py`'s CORS
  `allow_origins` list), so `oracle-mcp-test.html` can call `/mcp` directly from the
  browser with a real Bearer token. If that page's hosting path ever moves, this
  origin needs updating by hand.
- **Slow forecasts** inherit the 120s nginx/gunicorn cap; long tails can 504. A
  dedicated `location /mcp` with a longer `proxy_read_timeout` is a possible
  follow-up.
- Tools call the Oracul's internal functions **in-process** (`run_forecast`,
  `run_search`, `compute_nodes`, …) — no self-HTTP.
