# Oracle API — TruthMachine Forecast Service

> **Subdomain:** `oracle.daatan.com`
> **Status:** Phases 1–5 complete and live. Wired into daatan v1.9.0.

## Overview

The Oracle API is a FastAPI microservice that lives in `retro/api/`. Given a binary question, it:

1. Searches for relevant news articles (`web_search.py` — 10-provider chain, GDELT-first)
2. Runs each article through the TruthMachine pipeline (gatekeeper → extractor)
3. Weights each source's predictions by its historical Brier score from `leaderboard.json`
4. Aggregates into a calibrated probability distribution and returns it

It is called by **daatan** (the prediction market) to give bots and users an AI-sourced probability estimate for any question.

---

## Architecture Decision

**12 options were evaluated.** The chosen architecture:

- **FastAPI microservice inside the `retro` repo** — imports pipeline code directly, no porting
- **Separate systemd service** (`oracle-api.service`) on the retro EC2 alongside the batch pipeline
- **Retro EC2 is the intelligence backend** — daatan is the marketplace frontend
- **Git submodules were rejected** — `pip install -e ../pipeline` via `[tool.uv.sources]` is cleaner for Python
- **TypeScript port was rejected** — pipeline is ~2000 lines of Python with ML deps; porting is months of work

See `ARCHITECTURE.md` for the full comparison.

---

## Security

Two independent layers:

### Layer 1 — AWS Security Group
Port 8001 is not exposed to the public internet. Only the daatan EC2 security group ID is allowed as an inbound source. Survives IP changes.

### Layer 2 — Shared Bearer Secret
`x-api-key` header on every request. Both sides read from env. Same pattern as daatan's `BOT_RUNNER_SECRET`.

```
AWS SG: daatan-ec2-sg → retro-ec2:8001
App:    x-api-key: $ORACLE_API_KEY
```

---

## Search Provider Chain

All article searches — both `/forecast` internal searches and `/search` requests — use `web_search.py`, which tries providers in order and returns the first non-empty result.

**Current chain (as of 2026-05-20):**

| # | Provider | Cost | Notes |
|---|---|---|---|
| 1 | **GDELT Doc API** | Free | Primary. No API key. Rate-limited to 1 req/10s. 3-month rolling window. Returns titles + URLs; no snippets. |
| 1b | **GDELT BigQuery** | Free | Historical only (>90 days). Entity-based matching. Requires GCP service account with BigQuery roles. |
| 2 | SerpAPI | Paid | |
| 3 | Serper.dev | Paid | |
| 3b | **Tavily** | 1 credit/call | `topic=news`; native `start_date`/`end_date` date windowing; supports `site:` operator. |
| 4 | Brave News | Paid | |
| 5 | BrightData | Paid | |
| 6 | Nimbleway | Paid | |
| 7 | ScrapingBee | Paid | |
| 8 | Newsdata.io | Paid | |
| 9 | **DataForSEO** | Paid | Last-resort paid fallback only. |
| 10 | DuckDuckGo | Free | Final fallback. Uses DDG `/news.js` (Bing-backed) — works from EC2. Post-filtered by date. |

**To change the order or add a provider:** edit `pipeline/src/tm/web_search.py`, function `search_articles()`. The numbered comments make the chain easy to reorder.

**GDELT caveats:**
- 10-second minimum interval between requests (`GDELT_MIN_INTERVAL` constant)
- `artlist` mode returns no snippet text — only title, URL, domain, and date
- Historical date filtering works via `startdatetime`/`enddatetime` (GDELT covers its full archive)
- Rapid sequential calls (faster than 1/10s) get HTTP 429 and fall through to the next provider

---

## API Reference

### `POST /forecast`

**Auth:** `x-api-key` header required.
**Rate limit:** 10 requests/minute per IP.

```json
// Request
{
  "question": "Will the Israeli coalition government collapse in 2025?",
  "max_articles": 5
}

// Response
{
  "question": "Will the Israeli coalition government collapse in 2025?",
  "mean": 0.42,
  "std": 0.18,
  "ci_low": 0.14,
  "ci_high": 0.70,
  "articles_used": 4,
  "sources": [
    {
      "source_id": "haaretz",
      "source_name": "Haaretz",
      "url": "https://haaretz.com/...",
      "stance": 0.6,
      "certainty": 0.75,
      "credibility_weight": 1.04,
      "claims": ["Coalition crisis likely after Haredi draft bill fails"]
    }
  ]
}
```

**`mean` is in stance space `[-1, 1]`:**
- `+1` = all sources certain the event will happen
- `-1` = all sources certain it won't
- `0` = neutral / mixed
- Convert to probability: `p = (mean + 1) / 2`

### `POST /search`

**Auth:** `x-api-key` header required.

Exposes the provider chain directly. Useful for bediavad (historical article discovery) and debugging which provider served a query.

```json
// Request
{
  "query": "bitcoin price rally",
  "limit": 15,
  "date_from": "2024-10-01",
  "date_to": "2025-01-01",
  "enrich_snippets": true
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `query` | string | required | Search string. Supports `site:domain.com`. |
| `limit` | int | 5 | Max results (1–30). |
| `date_from` | string | null | ISO date `YYYY-MM-DD`. Passed to providers that support native date filtering (GDELT, Tavily, Newsdata.io, DataForSEO). Other providers receive it as post-filter. |
| `date_to` | string | null | ISO date `YYYY-MM-DD`. |
| `enrich_snippets` | bool | `false` | When `true`, scrapes each article URL to fill the `snippet` field. GDELT returns no snippet text by default; this compensates. Adds 5–15s latency. Not suitable for daatan live calls. |

```json
// Response
{
  "query": "bitcoin price rally",
  "count": 5,
  "results": [
    {
      "title": "Bitcoin Surges Past $90k",
      "url": "https://coindesk.com/...",
      "snippet": "Bitcoin hit a new all-time high on Monday as institutional...",
      "source": "coindesk.com",
      "published_date": "2024-11-12"
    }
  ]
}
```

**`enrich_snippets` implementation:** after `search_articles()` returns, `enrich_snippets()` fans out URL fetches with 8 parallel workers and an 8-second total timeout. Uses `trafilatura` (primary) and BeautifulSoup (fallback) for HTML extraction. Articles whose URLs fail or timeout keep an empty snippet.

**For bediavad / historical backtest use:** set `enrich_snippets: true` and pace requests no faster than one every 12 seconds to stay within GDELT's rate limit. If GDELT returns 0 results (common for very narrow date ranges or queries older than 3 months), the chain falls through to GDELT BigQuery (if GCP key configured), then paid providers, then DDG. DDG (`d.news()`) returns results filtered by date post-fetch; it does not accept a date range natively.

### `GET /search/health`

**Auth:** `x-api-key` header required.

Returns per-provider status with credit counts where available.

```json
{
  "overall": "degraded",
  "usable_count": 2,
  "providers": {
    "gdelt":       {"configured": true, "exhausted": false, "status": "ok"},
    "dataforseo":  {"configured": true, "exhausted": false, "status": "ok", "credits": null},
    "serper":      {"configured": true, "exhausted": true,  "status": "exhausted"},
    "brave":       {"configured": true, "exhausted": true,  "status": "exhausted"},
    "ddg":         {"configured": true, "exhausted": false, "status": "ok"}
  }
}
```

### `GET /health`

No auth required. Returns:

```json
{
  "status": "ok",
  "version": "0.1.0",
  "leaderboard_sources": 27
}
```

`version` allows clients to verify API compatibility before relying on `/forecast` responses.

### `GET /leaderboard`

**Auth:** `x-api-key` header required.

Returns the live source credibility leaderboard sorted by OpenSkill conservative score. The data is read from `data/leaderboard.json` (refreshed every 5 minutes by the background loop — same file written by the scoring pipeline).

```json
{
  "sources": [
    {
      "id": "haaretz",
      "name": "Haaretz",
      "skill_mu": 26.3,
      "skill_sigma": 7.8,
      "skill_conservative": 2.9,
      "elo": 1247.0,
      "brier_score": 0.2134,
      "accuracy": 0.61,
      "predictions": 47,
      "events": 31
    }
  ],
  "count": 27
}
```

Used by daatan's `getOracleLeaderboard()` in `src/lib/services/oracle.ts` to display live source credibility.

### `GET /bayes/nodes`

**Auth:** `x-api-key` header required.

Returns the BayesOracle node probabilities for the Israeli-politics DAG. Optionally accepts observations to propagate through the DAG via log-odds perturbation.

**Query params:**

| Param | Type | Description |
|---|---|---|
| `observations` | string | Optional. Comma-separated `NODE_ID=probability` pairs, e.g. `TRUMP=0.3,PM5=0.5` |

```json
// GET /bayes/nodes?observations=TRUMP=0.3
[
  { "id": "TRUMP",   "label": "Trump wins 2024",   "layer": 0, "prior": 0.85, "p": 0.3,  "delta": -0.55, "locked": true },
  { "id": "PM5",     "label": "Netanyahu PM 2025", "layer": 2, "prior": 0.32, "p": 0.16, "delta": -0.16, "locked": false },
  { "id": "OPP_PM",  "label": "Opposition PM",     "layer": 3, "prior": 0.38, "p": 0.69, "delta": +0.31, "locked": false }
]
```

Fields: `id`, `label`, `layer` (topological depth), `prior` (base probability), `p` (current probability after propagation), `delta` (`p − prior`), `locked` (whether this node was pinned by an observation).

The DAG covers 21 nodes and 33 edges across Israeli coalition politics, judicial reform, and regional geopolitics. Implemented in `api/src/forecast_api/bayesoracle.py`.

---

## Deployment

> **Deploy flow and one-time migration:** see [`ORACLE_DEPLOY.md`](ORACLE_DEPLOY.md). The API runs from its own checkout at `/home/ubuntu/oracle-api/` (separate from the pipeline's `/home/ubuntu/truthmachine/`) so deploys can `git reset --hard origin/main` without touching the pipeline's unpushed atlas commits. Routine deploys use `infra/deploy_oracle.sh`.

### Directory structure

```
retro/
├── api/
│   ├── pyproject.toml          ← standalone package; depends on ../pipeline
│   ├── .env.example
│   └── src/forecast_api/
│       ├── main.py             ← FastAPI app + lifespan
│       ├── config.py           ← settings (extends tm.config pattern)
│       ├── auth.py             ← x-api-key dependency
│       ├── limiter.py          ← slowapi rate limiting
│       ├── leaderboard.py      ← load/cache/refresh leaderboard.json
│       ├── forecaster.py       ← core: search → extract → weight → aggregate
│       └── models.py           ← Pydantic request/response schemas
├── infra/
│   ├── oracle-api.service      ← systemd unit for the API process
│   ├── deploy_oracle.sh        ← zero-downtime deploy (fetch → reset → sync → SIGHUP)
│   └── ...
```

### First-time EC2 setup

The pipeline's checkout at `~/truthmachine` already exists. Add a second, API-only checkout at `~/oracle-api` and install the unit file from it. Full walkthrough in [`ORACLE_DEPLOY.md`](ORACLE_DEPLOY.md#one-time-migration-pipeline-ec2-i-00ac444b94c5ff9b2).

```bash
# New: dedicated API checkout
cd /home/ubuntu
git clone https://github.com/komapc/retro.git oracle-api
cd oracle-api/api
uv sync --frozen

# .env already exists at /home/ubuntu/truthmachine/.env and is shared
# (the unit file points EnvironmentFile= there). Nothing new to add beyond
# ORACLE_API_KEY, which is already set for the existing service.

# Install the new unit file (WorkingDirectory now points at oracle-api/)
sudo cp /home/ubuntu/oracle-api/infra/oracle-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable oracle-api
sudo systemctl restart oracle-api   # first-time switchover; later deploys use `reload`
```

### Smoke test

```bash
curl -s -X POST http://127.0.0.1:8001/forecast \
  -H "x-api-key: $ORACLE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"question": "Will Netanyahu remain PM through 2025?"}'
```

### Updating (zero-downtime reload)

Since the service is supervised by gunicorn, routine deploys use `reload`, not
`restart`. The gunicorn master keeps the :8001 listening socket open while it
swaps workers with fresh code, so clients see no 502s.

Use the one-shot script:

```bash
bash /home/ubuntu/oracle-api/infra/deploy_oracle.sh              # -> origin/main
bash /home/ubuntu/oracle-api/infra/deploy_oracle.sh <commit-sha> # pin to a SHA
```

Which is equivalent to:

```bash
cd /home/ubuntu/oracle-api
git fetch origin main
git reset --hard origin/main
cd api && uv sync --frozen

# Zero-downtime swap — master keeps the socket; workers recycle gracefully.
sudo systemctl reload oracle-api

# Verify /health returns the new version.
curl -s http://127.0.0.1:8001/health | jq .
```

Verify the reload window is truly seamless (run in another terminal before
reloading):

```bash
while true; do
  curl -s -o /dev/null -w "%{http_code} " http://127.0.0.1:8001/health
  sleep 0.1
done
# Expect a continuous stream of 200s across the reload.
```

### When to use `restart` instead of `reload`

- The systemd unit file itself changed (`ExecStart`, env vars, etc.) — run
  `sudo systemctl daemon-reload && sudo systemctl restart oracle-api`. This
  incurs a 2-5s 502 window.
- Gunicorn master itself crashed or needs new flags.
- Dependency-graph changes that require a fresh Python interpreter.

For routine app-code deploys (forecaster, models, config defaults), `reload`
is always sufficient and strictly preferred.

---

## Nginx routing (oracle.daatan.com)

Add to retro EC2 nginx (or to daatan's nginx if co-hosted):

```nginx
upstream oracle_api {
    server 127.0.0.1:8001;
    keepalive 4;
}

server {
    listen 443 ssl http2;
    server_name oracle.daatan.com;

    ssl_certificate /etc/letsencrypt/live/oracle.daatan.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/oracle.daatan.com/privkey.pem;

    location / {
        proxy_pass http://oracle_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 20s;
        limit_req zone=llm_limit burst=5 nodelay;
        limit_req_status 429;
    }
}
```

---

## daatan integration (live since v1.9.0)

The Oracle is wired into two `daatan` routes, with automatic fallback to the existing LLM `guessChances` path when the Oracle is unavailable, returns a placeholder, or times out:

| Route | File |
|-------|------|
| `POST /api/forecasts/[id]/context` | `daatan/src/app/api/forecasts/[id]/context/route.ts` |
| `POST /api/forecasts/express/guess` | `daatan/src/app/api/forecasts/express/guess/route.ts` |

Client: `daatan/src/lib/services/oracle.ts` — `getOracleProbability()` returns a probability in `[0, 1]` or `null` (never throws). `checkOracleHealth()` verifies the API is reachable and its version starts with `0.1`.

### IBI analysis tool

The IBI retro analysis tool (formerly `komapc.github.io/retro/ibi.html`) is hosted in daatan at **`/ibi`** (admin-only). It calls Oracle `/fetch-url`, `/search`, and `/llm` through three Daatan proxy routes (`/api/ibi/*`) so the Oracle key never reaches the browser. The static `ibi.html` remains available as a fallback but requires manual key entry.

### Secret management

The shared `x-api-key` lives in AWS Secrets Manager at `openclaw/oracle-api-key` (region `eu-central-1`). The `openclaw/` prefix is legacy naming from the decommissioned OpenClaw stack and is retained for backwards compatibility with `ec2_bootstrap.sh`. Both sides read the key from there:

- **retro EC2** (`oracle-api.service`) — `ORACLE_API_KEY` env var
- **daatan EC2** (`~/app/.env`) — `ORACLE_URL` + `ORACLE_API_KEY`, pulled via `scripts/fetch-secrets.sh` from the `daatan-env-prod` / `daatan-env-staging` bundle secret on each deploy

To rotate: update `openclaw/oracle-api-key` in Secrets Manager, then update both `daatan-env-{prod,staging}` bundles and the EC2 `.env` on the retro side, and restart both services.

---

## Roadmap

| Phase | Description |
|---|---|
| ✅ Phase 1 | API skeleton + auth + rate limiting |
| ✅ Phase 2 | Live pipeline: `web_search.py` → gatekeeper → extractor → leaderboard weighting |
| ✅ Phase 3 | Leaderboard credibility weighting (OpenSkill PlackettLuce conservative score μ − 3σ) |
| ✅ Phase 4 | `oracle.daatan.com` DNS + TLS + EC2 deploy |
| ✅ Phase 5 | daatan integration — `oracle.ts` client wired into context + express guess routes (shipped in daatan v1.9.0) |
| 🔲 Phase 6 | Async queue for >15s requests |
