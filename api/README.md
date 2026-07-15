# Oracle API (`forecast_api`)

FastAPI microservice that turns a binary question into a calibrated probability
by searching current articles, extracting predictions, and weighting each source
by its historical accuracy (from `leaderboard.json`). Live at **`oracle.daatan.com`**.

This README is the developer entry point. For the full request/response contract
see the [Oracle API contract](https://github.com/Daatan/docs/blob/main/oracle-api.md);
for deploy/rollback see [`docs/ORACLE_DEPLOY.md`](../docs/ORACLE_DEPLOY.md).

## Layout (`src/forecast_api/`)

| File | Role |
|---|---|
| `main.py` | FastAPI app, routes, CORS, rate limiter wiring |
| `forecaster.py` | Core: search → fetch → gatekeep → extract → weight → aggregate → distribution |
| `searcher.py` | `/search` + `/search/health` handlers |
| `leaderboard.py` | Load/cache `leaderboard.json`; credibility weight per source |
| `bayesoracle.py` | `/bayes/nodes` — node probabilities for the BayesOracle viewers |
| `models.py` | Pydantic request/response schemas |
| `config.py` | Settings (env-driven; reuses the `tm` pipeline package) |
| `auth.py` | `x-api-key` dependency (constant-time compare via `hmac.compare_digest`) |
| `limiter.py` / `cache.py` | slowapi rate limiting; forecast + search caches |
| `mcp_server.py` / `mcp_auth.py` | MCP server at `/mcp` (agent tools) + Cognito OAuth Resource-Server auth — see [`../docs/ORACLE_MCP.md`](../docs/ORACLE_MCP.md) |
| `article_fetch.py` / `polymarket_live.py` | `/fetch-url` extraction (shared with the MCP `fetch_article` tool); live Polymarket lookup for `polymarket_edge` |

It imports `tm.gatekeeper`, `tm.extractor`, `tm.web_search`, and `tm.net_guard`
from the `pipeline/` package (a path dependency) — no code is duplicated.

**CORS.** Allowed origins are `https://daatan.github.io` and
`https://bayes.daatan.com`; localhost dev origins (`http://localhost` /
`http://127.0.0.1`, any port) are matched via `allow_origin_regex` because
Starlette's `allow_origins` list requires exact matches (wildcard ports don't
work there).

**Fetching external URLs.** Never fetch a caller- or search-supplied URL with a
raw `httpx` client. Use `tm.net_guard.safe_get`, which rejects
non-http(s) schemes and hosts that resolve to private/loopback/link-local
addresses (including the cloud metadata IP `169.254.169.254`) and re-validates
every redirect hop — a validated public host can still 30x-redirect to an
internal one. This is what backs the public `/fetch-url` proxy.

## Run locally

```bash
cd api
uv sync --extra dev
export ORACLE_API_KEY=dev-key            # required — settings validate at import
uv run uvicorn forecast_api.main:app --reload --port 8001
# probe:
curl -s localhost:8001/health
curl -s -X POST localhost:8001/forecast -H 'x-api-key: dev-key' \
  -H 'content-type: application/json' \
  -d '{"question":"Will X happen by 2026-12-31?","deadline":"2026-12-31"}'
```

Production runs the same app under gunicorn with uvicorn workers — see
`infra/oracle-api.service` for the exact command.

> A forecast hits live search + Bedrock, so it needs AWS/search credentials in
> the environment. `/health`, `/search/health`, and unit tests need none.

## Test

```bash
cd api && uv run pytest        # 154 tests; no network/secrets needed
```

`tests/conftest.py` sets a dummy `ORACLE_API_KEY` so the suite runs without a real
secret. CI runs this on every PR and **gates the deploy** (see `.github/workflows/`).

## Endpoints (summary)

`POST /forecast`, `POST /search`, `GET /search/health`, `POST /llm`,
`POST /fetch-url`, `GET /bayes/nodes`, `GET /leaderboard`, `GET /health`,
`GET /version`, `GET /pm/markets`. All require the `x-api-key` header **except**
`/health`, `/version`, and the deliberately-public `/fetch-url` (which is
rate-limited and SSRF-guarded — http(s) only, no private/loopback/link-local
hosts). Full details in the
[Oracle API contract](https://github.com/Daatan/docs/blob/main/oracle-api.md).

Error-contract notes: `/search` returns **422** (not 500) on malformed date
params, and `/pm/markets` returns **502** on a Polymarket upstream HTTP error
(**504** on an upstream connect/timeout failure) rather than an opaque 500.
