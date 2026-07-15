# Oracle MCP Server

A [Model Context Protocol](https://modelcontextprotocol.io) server that exposes
the Oracle's capabilities to AI agents (Claude Desktop/Code, custom agents) as
native tools. It is **mounted on the FastAPI app** (`api/`), so it ships with the
Oracle on the same host, nginx, and auto-deploy — it is not a separate service.

- **Endpoint:** `https://oracle.daatan.com/mcp` (Streamable HTTP, stateless)
- **Auth:** OAuth 2.1 — the Oracle is a Resource Server verifying AWS Cognito
  access tokens (see [Auth](#auth))
- **Code:** `api/src/forecast_api/mcp_server.py` (tools), `mcp_auth.py` (token
  verification), mounted in `main.py`

The REST API is unchanged and still uses the `x-api-key` header; `/mcp` is
OAuth-only. Non-agent/service automation should keep calling REST directly.

## Tools

Built for a **Polymarket trader** workflow — take a market, get an independent
probability, compare to the price, find an edge — plus general forecasting.

| Tool | Scope | What it does |
|------|-------|--------------|
| `polymarket_edge(market, edge_threshold=0.05)` | `forecast` | Resolve a market (URL/slug/Gamma id) → live YES price; forecast its question; return the **edge** (oracle − market) and a **suggested side** (BUY YES / BUY NO / NO EDGE) |
| `polymarket_market(market)` | `read` | Live market data: question, outcomes, prices, volume, end date |
| `forecast(question, max_articles?)` | `forecast` | Calibrated probability for any binary question (`probability` in [0,1]) |
| `search_news(query, limit?, date_from?, date_to?)` | `read` | News search via the provider chain |
| `fetch_article(url)` | `read` | Fetch + extract one article (SSRF-guarded) |
| `bayes_nodes(observations?)` | `read` | BayesOracle Israeli-politics DAG probabilities |
| `source_leaderboard()` | `read` | Live source-credibility ranking |

`forecast` / `polymarket_edge` are **slow** (a live news + LLM pipeline — tens of
seconds, occasionally past the 120s proxy timeout) and cost LLM + search credits,
which is why they sit behind the `oracle-mcp/forecast` scope.

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
  UI. **Services** (e.g. daatan's own agents) use the `client_credentials` grant.
- **Discovery:** the protected-resource metadata (RFC 9728) is served at
  `https://oracle.daatan.com/.well-known/oauth-protected-resource/mcp`; a 401
  returns `WWW-Authenticate` pointing there.

> **Cognito has no Dynamic Client Registration.** The Claude MCP client can't
> self-register, so its `client_id` must be configured statically (or a small DCR
> shim added). Validate the connector's static-client support before rollout.

### Adding to a client

```bash
claude mcp add --transport http oracle https://oracle.daatan.com/mcp
# then complete the OAuth flow when prompted
```

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
```

The Cognito user pool itself is provisioned in `terraform/cognito.tf`.

## Provisioning runbook (first-time enablement)

End-to-end steps to flip `/mcp` from inert 404 to live. All AWS work is
`eu-central-1`, account `272007598366`; the Oracle box (`i-00ac444b94c5ff9b2`) is
**SSM-only, no SSH**.

### 1. Google OAuth client + Secrets Manager

The Cognito hosted-UI domain is deterministic from `var.cognito_domain_prefix`
(default `daatan-oracle`), so you can set Google's redirect URI up front:
`https://daatan-oracle.auth.eu-central-1.amazoncognito.com/oauth2/idpresponse`.

Create the Google OAuth 2.0 client (Google Cloud Console), then store its creds —
never as a literal in tf/state; the IdP resource reads them via a data source:

```bash
aws secretsmanager create-secret --region eu-central-1 \
  --name daatan/cognito-google-oauth \
  --secret-string '{"client_id":"<GOOGLE_CLIENT_ID>","client_secret":"<GOOGLE_CLIENT_SECRET>"}'
```

### 2. Apply Cognito (`-target`, in dependency order)

Never a blanket apply (workspace rule). From `terraform/` on the latest `main`,
pass the **real** Claude redirect URIs (the tf default is a placeholder):

```bash
cd terraform
terraform init          # real S3 backend (state key retro/)
export TF_VAR_claude_callback_urls='["https://claude.ai/api/mcp/auth_callback"]'
# export TF_VAR_cognito_domain_prefix=daatan-oracle   # only if the default is taken

terraform apply -target=aws_cognito_user_pool.oracle_mcp
terraform apply -target=aws_cognito_resource_server.oracle_mcp
terraform apply -target=aws_cognito_identity_provider.google      # reads the Step-1 secret
terraform apply -target=aws_cognito_user_pool_domain.oracle_mcp
terraform apply -target=aws_cognito_user_pool_client.claude
terraform apply -target=aws_cognito_user_pool_client.m2m

# Capture what the box + smoke tests need:
terraform output -raw cognito_user_pool_id
terraform output -raw cognito_allowed_client_ids
terraform output -raw cognito_m2m_client_secret     # sensitive
```

### 3. Set the env on the box + restart (SSM)

`systemd` only re-reads `EnvironmentFile` on **restart**, not on the deploy's SIGHUP
reload — so this needs a real `restart` (one-time ~2–5s 502 window). SSM runs as
root; the `chown` keeps `.env` owned by `ubuntu`.

```bash
# from terraform/
POOL_ID=$(terraform output -raw cognito_user_pool_id)
CLIENT_IDS=$(terraform output -raw cognito_allowed_client_ids)

CMD_ID=$(aws ssm send-command \
  --region eu-central-1 \
  --instance-ids i-00ac444b94c5ff9b2 \
  --document-name AWS-RunShellScript \
  --comment "enable Oracle MCP: Cognito env + restart" \
  --parameters "{\"commands\":[
    \"sed -i '/^COGNITO_USER_POOL_ID=/d;/^COGNITO_ALLOWED_CLIENT_IDS=/d;/^MCP_RESOURCE_URL=/d' /home/ubuntu/truthmachine/.env\",
    \"echo COGNITO_USER_POOL_ID=$POOL_ID >> /home/ubuntu/truthmachine/.env\",
    \"echo COGNITO_ALLOWED_CLIENT_IDS=$CLIENT_IDS >> /home/ubuntu/truthmachine/.env\",
    \"echo MCP_RESOURCE_URL=https://oracle.daatan.com/mcp >> /home/ubuntu/truthmachine/.env\",
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

# Discovery metadata (should list the Cognito issuer):
curl -s https://oracle.daatan.com/.well-known/oauth-protected-resource/mcp

# M2M auth smoke test — with a valid token you must NOT get 401 (a JSON-RPC
# "initialize first" error is fine; it's past auth). Full listing is cleanest via
# `claude mcp add`. Token request uses the M2M snippet in Auth (both scopes).
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
  host automatically — no separate config to keep in sync.
- **Slow forecasts** inherit the 120s nginx/gunicorn cap; long tails can 504. A
  dedicated `location /mcp` with a longer `proxy_read_timeout` is a possible
  follow-up.
- Tools call the Oracle's internal functions **in-process** (`run_forecast`,
  `run_search`, `compute_nodes`, …) — no self-HTTP.
