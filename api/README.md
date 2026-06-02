# Oracle API (`forecast_api`)

FastAPI microservice that turns a binary question into a calibrated probability
by searching current articles, extracting predictions, and weighting each source
by its historical accuracy (from `leaderboard.json`). Live at **`oracle.daatan.com`**.

This README is the developer entry point. For the full request/response contract
see [`docs/ORACLE_API.md`](../docs/ORACLE_API.md); for deploy/rollback see
[`docs/ORACLE_DEPLOY.md`](../docs/ORACLE_DEPLOY.md).

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
| `auth.py` | `x-api-key` dependency |
| `limiter.py` / `cache.py` | slowapi rate limiting; forecast + search caches |

It imports `tm.gatekeeper`, `tm.extractor`, and `tm.web_search` from the
`pipeline/` package (a path dependency) — no code is duplicated.

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
cd api && uv run pytest        # 50 tests; no network/secrets needed
```

`tests/conftest.py` sets a dummy `ORACLE_API_KEY` so the suite runs without a real
secret. CI runs this on every PR and **gates the deploy** (see `.github/workflows/`).

## Endpoints (summary)

`POST /forecast`, `POST /search`, `GET /search/health`, `POST /llm`,
`POST /fetch-url`, `GET /bayes/nodes`, `GET /leaderboard`, `GET /health`,
`GET /pm/markets`. All require the `x-api-key` header **except** `/health` and the
deliberately-public `/fetch-url` (which is rate-limited and SSRF-guarded —
http(s) only, no private/loopback/link-local hosts). Full details in
[`docs/ORACLE_API.md`](../docs/ORACLE_API.md).
