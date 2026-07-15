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

For a service (M2M), fetch a token and call with a bearer header:

```bash
TOKEN=$(curl -s -X POST "https://<pool-domain>.auth.eu-central-1.amazoncognito.com/oauth2/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "grant_type=client_credentials&client_id=$CID&client_secret=$CSECRET&scope=oracle-mcp/forecast" \
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

## Operational notes

- **Stateless** transport (`stateless_http=True, json_response=True`) — required
  because the API runs under gunicorn with multiple workers and no sticky
  sessions.
- **No nginx change** — the existing `location /` catch-all proxies `/mcp` and the
  root `.well-known/*`. MCP calls share the per-IP 60 r/min budget.
- **Slow forecasts** inherit the 120s nginx/gunicorn cap; long tails can 504. A
  dedicated `location /mcp` with a longer `proxy_read_timeout` is a possible
  follow-up.
- Tools call the Oracle's internal functions **in-process** (`run_forecast`,
  `run_search`, `compute_nodes`, …) — no self-HTTP.
