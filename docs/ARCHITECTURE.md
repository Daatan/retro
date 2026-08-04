# TruthMachine — System Architecture

## Overview

TruthMachine (Factum Atlas) is a retroactive media analysis pipeline that:
1. Collects news articles published **before** known past events
2. Extracts and quantifies forward-looking predictions from each article
3. Scores each prediction against the actual outcome (Brier score)
4. Renders an interactive coverage matrix at **daatan.github.io/retro**

---

## Repository Structure

```
retro/
├── api/                         # Oracle API — FastAPI microservice (oracle.daatan.com)
│   ├── src/forecast_api/
│   │   ├── main.py              # FastAPI app + lifespan
│   │   ├── forecaster.py        # Core: search → extract → weight → aggregate
│   │   ├── leaderboard.py       # Load/cache leaderboard.json for credibility weights
│   │   ├── models.py            # Pydantic request/response schemas
│   │   ├── config.py            # Settings (extends tm.config pattern)
│   │   ├── auth.py              # x-api-key dependency (hmac.compare_digest)
│   │   └── limiter.py           # slowapi rate limiting  (SSRF guard: imported from tm.net_guard)
│   └── pyproject.toml
├── pipeline/                    # Python pipeline (uv project)
│   ├── src/tm/
│   │   ├── config.py            # Settings (models, API keys via .env)
│   │   ├── models.py            # Pydantic models (Prediction, ExtractionOutput, CellSignal, etc.)
│   │   ├── progress.py          # progress.json read/write helpers + rich terminal visualizer
│   │   ├── net_guard.py         # SSRF guard: safe_get/safe_get_async/is_safe_url, re-checks redirect hops
│   │   │
│   │   ├── # --- Ingest ---
│   │   ├── gnews_ingest.py      # GNews RSS → URL resolution → trafilatura + Wayback fallback
│   │   ├── gdelt_ingest.py      # GDELT Doc 2.0 API batch ingestor (sequential, rate-limited)
│   │   ├── ingestor.py          # Pluggable ingestor classes: DDGIngestor, GDELTIngestor
│   │   ├── site_search.py       # Direct site-search scraper (no API key, high reliability)
│   │   ├── web_search.py        # Multi-provider news search: news-indexer → GDELT → GDELT BQ → Google CSE → SerpAPI → Serper → Brave → Tavily → Newsdata.io → BrightData → Nimbleway → ScrapingBee → DataForSEO → DDG
│   │   ├── polymarket.py        # Polymarket Gamma API: fetch market history per event
│   │   ├── polymarket_harvest.py # Bulk harvest of all resolved Polymarket political markets
│   │   │
│   │   ├── # --- Extraction ---
│   │   ├── gatekeeper.py        # LLM stage 1: topic-relevance filter (is article on-topic for event?)
│   │   ├── extractor.py         # LLM stage 2: extract up to 5 structured predictions per article
│   │   ├── runner.py            # Orchestrates gatekeeper → extractor → article-aggregator per article
│   │   ├── aggregator.py        # Stage 2b (LLM, article-level): collapse high-spread predictions
│   │   │                        # within one article. Stage 3 (no LLM, cell-level): collapse all
│   │   │                        # article predictions for (event, source) → CellSignal.
│   │   ├── reaggregate.py       # Post-processing: re-run article-level aggregation on existing data
│   │   │
│   │   ├── # --- Scoring & Output ---
│   │   ├── orchestrator.py      # Batch runner: events × sources → vault → atlas
│   │   ├── scorer.py            # Brier score + calibration utilities + per-category scoring
│   │   ├── backtest.py          # LightGBM backtest + Polymarket comparison
│   │   ├── render_atlas.py      # Renders factum_atlas.html from atlas/ data
│   │   ├── generate_pages.py    # Generates per-event/source static HTML pages
│   │   ├── sync_atlas.py        # Parses event table and syncs atlas entry JSON files
│   │   │
│   │   ├── # --- One-off scripts ---
│   │   ├── init_db.py           # Initialize SQLite DB for progress tracking
│   │   ├── migrate_cell_signals.py # One-time: compute cell_signal.json from existing vault data
│   │   ├── poc_event_gen.py     # Convert harvested Polymarket events → pipeline event JSONs
│   │   ├── poc_report.py        # Generate duel.html — TruthMachine vs Polymarket comparison report
│   │   ├── create_real_samples.py  # Create real sample data for testing
│   │   └── create_sample_data.py   # Create synthetic sample data for testing
│   ├── scripts/
│   │   └── improve_keywords.py  # One-time: LLM-generate search keywords for events
│   ├── tests/                   # pytest test suite
│   ├── smoke_test.py            # 3 hardcoded articles through full pipeline
│   ├── test_run.py              # Manual test runner
│   ├── docker-compose.yml       # Local pipeline stack
│   ├── pyproject.toml
│   └── Dockerfile
├── data/                        # Gitignored except events/ and sources/
│   ├── events/                  # Event definitions (TRACKED IN GIT)
│   ├── sources/                 # Source definitions (TRACKED IN GIT)
│   ├── raw_ingest/              # Scraped articles (regeneratable, not in git)
│   ├── vault2/
│   │   ├── articles/            # Deduplicated article cache by SHA-256 hash
│   │   └── extractions/         # LLM extraction cache: {hash}_{event_id}_v1.json
│   ├── atlas/                   # Atlas link files: atlas/{event_id}/{source_id}/entry_*.json
│   └── progress.json            # Cell status: done/pending/no_predictions/failed
├── infra/
│   ├── ec2_bootstrap.sh         # One-time EC2 setup script (incl. restore_atlas before service start)
│   ├── ec2_run.sh               # Continuous pipeline loop (runs on EC2; snapshots atlas at tail of cycle)
│   ├── ec2_run_poc.sh           # PoC pipeline run script
│   ├── snapshot_atlas.sh        # Tar data/atlas + data/vault2 → S3 (per-cycle + latest.tgz)
│   ├── restore_atlas.sh         # Pull latest.tgz from S3 if data/atlas/ is empty (fresh boot only)
│   ├── deploy_oracle.sh         # Zero-downtime API deploy: fetch → reset → uv sync → reload, then a
│   │                            # consecutive-200 health gate that escalates to a full restart if the
│   │                            # reload isn't healthy (see docs/ORACLE_DEPLOY.md). Fails red on failure.
│   ├── monitor.sh               # Local monitoring script (polls EC2 via SSM)
│   ├── logs.sh                  # Tail EC2 pipeline logs via SSM
│   ├── check_keys.sh            # Verify required AWS Secrets Manager keys exist
│   ├── remote_stats.sh          # Fetch pipeline progress stats from EC2
│   ├── oracle-api.service       # systemd unit for the Oracle API (gunicorn + uvicorn workers)
│   ├── truthmachine.service     # systemd unit for the pipeline batch process
│   ├── truthmachine-poc.service # systemd unit for PoC pipeline variant
│   ├── iam/                     # IAM policy templates (GH Actions OIDC, S3 snapshots) — see infra/iam/README.md
│   └── nginx/                   # Nginx config fragments (oracle.daatan.com vhost)
├── case-studies/                # Interactive case study pages
├── .github/workflows/
│   ├── deploy-atlas.yml         # Deploy factum_atlas.html + oracle-test + duel to GitHub Pages
│   └── deploy-oracle.yml        # On push to main affecting api/, redeploy Oracle via SSM (OIDC, no static keys)
├── factum_atlas.html            # Generated atlas (committed by EC2 after each cycle)
├── oracle-test.html             # Oracle API test console (deployed to GitHub Pages)
└── duel.html                    # TruthMachine vs Polymarket comparison report (generated by poc_report.py)
```

---

## Data Model

### Event (`data/events/{id}.json`)
```json
{
  "id": "C09",
  "name": "Assad regime falls in Syria",
  "outcome": true,
  "outcome_date": "2024-12-08",
  "predictive_window_days": 14,
  "search_keywords": ["נפילת אסד סוריה", "מרד סוריה דמשק", "Assad regime collapse Syria", ...],
  "llm_referee_criteria": "The Assad government loses effective control of Damascus.",
  "category": ["Regional Geopolitics"],
  "tags": ["Assad", "Syria", "regime collapse", "rebels"]
}
```

**`category`** — multi-label list from the taxonomy:
`Israeli Politics`, `Gaza War`, `Regional Geopolitics`, `Israeli Economy`, `Israeli Society`, `AI & Tech`, `Global`.
Used to compute per-category source accuracy scores for the forecasting model.

**`tags`** — free-form keywords for fine-grained topic matching at inference time.

### Source (`data/sources/{id}.json`)
```json
{
  "id": "toi",
  "name": "Times of Israel",
  "url": "https://www.timesofisrael.com",
  "language": "en"
}
```

### Atlas Entry (`data/atlas/{event_id}/{source_id}/entry_{hash[:8]}.json`)
```json
{
  "article_hash": "...",
  "extraction_id": "...",
  "headline": "...",
  "article_url": "https://...",
  "author": "...",
  "article_date": "2024-04-13",
  "event_date": "2024-04-14",
  "extractor_model": "bedrock/amazon.nova-lite-v1:0",
  "gatekeeper_model": "bedrock/amazon.nova-micro-v1:0",
  "gatekeeper_reason": "Article directly predicts an imminent Iranian missile strike on Israel.",
  "predictions": [...]
}
```

### Vault Extraction (`data/vault2/extractions/{hash}_{event}_v1.json`)
```json
{
  "extraction": { "predictions": [...] },
  "prompt_version": "v1",
  "extractor_model": "bedrock/amazon.nova-lite-v1:0",
  "gatekeeper_model": "bedrock/amazon.nova-micro-v1:0",
  "gatekeeper_reason": "...",
  "run_date": "2026-04-14T12:00:00"
}
```

### Prediction (extracted by LLM)
Each prediction has: `quote`, `claim`, `stance` (−1 to +1, event probability), `certainty`, `settled` (bool — true when the source reports the outcome as an accomplished fact, not a prediction; the prompt explicitly excludes historical background such as a past removal/ban, see #244), and `quantitative_estimate` (optional [0,1] — an explicit modeled probability, poll number, or market price the source cites for the event itself; carries the quantitative-anchor weight premium).
Older atlas entries also carry: `sentiment`, `specificity`, `hedge_ratio`, `conditionality`, `magnitude`, `time_horizon`, `prediction_type`, `source_authority` (these fields were dropped from the extractor prompt in PR #102 to reduce latency; retained as Optional for backward compatibility).

**stance** = how strongly the prediction implies the related event WILL occur.
- `+1.0` = author is certain the event will happen
- `−1.0` = author is certain the event will NOT happen
- `0.0`  = neutral / genuinely uncertain

**certainty** = how much weight this claim should carry, independent of stance's
direction or magnitude — it answers "how much should we trust that this signal
really bears on the outcome," not "which way does it point."
- `~1.0` = decisive/explicit ("they clinched it", "the vote failed")
- `~0.2` = hedged, vague, or only loosely connected ("pressure is mounting")

Stance and certainty are meant to be independent axes (a hedged claim can point
strongly in one direction with low certainty; a flatly-stated fact can carry high
certainty with only mild stance), but the extractor prompt (`pipeline/src/tm/extractor.py`)
only ever demonstrates them via correlated examples — it never states the distinction
as a rule. See "Known limitations" below.

---

## Pipeline Flow

> For the **live** lane and for what happens before an article reaches this pipeline —
> discovery, retrieval, the gatekeeper rescue paths, and the per-stage drop taxonomy across
> news-indexer / retro / daatan — see [funnel.md](https://github.com/Daatan/docs/blob/main/funnel.md).

```
Ingest (choose one):
  gnews_ingest.py  — GNews RSS → URL resolution (Brave/SerpAPI/Serper/DDG) → trafilatura
                     If 0 articles: CDX/Wayback fallback
  gdelt_ingest.py  — GDELT Doc 2.0 API, sequential with rate-limiting
  ingestor.py      — Pluggable DDGIngestor / GDELTIngestor classes
  site_search.py   — Direct site search scraper (no API key)
  All save to: data/raw_ingest/{source}/{event}/article_NN.json
  ▼
orchestrator.py  (local_file mode)
  │  For each (event, source) cell not yet done:
  │    runner.py → gatekeeper.py (LLM: topic-relevant for event?)
  │                extractor.py  (LLM: extract up to 5 structured predictions)
  │                aggregator.aggregate_article_predictions
  │                              (LLM: collapse to one signal if stance spread > 0.4)
  │    aggregator.aggregate_predictions → cell_signal.json (no LLM, weighted mean)
  │    Save extraction to vault2/extractions/{hash}_{event}_v1.json
  │    Save atlas link to atlas/{event}/{source}/entry_{hash[:8]}.json
  │    Update progress.json → status: done | no_predictions | failed
  ▼
render_atlas.py
  │  Load all atlas/ entries + cell signals
  │  Compute competitive Brier scores (scorer.py)
  │  Render factum_atlas.html (interactive matrix)
  ▼
git push → GitHub Actions → GitHub Pages
  │
  └─ snapshot_atlas.sh: tar data/atlas + data/vault2 → S3 (latest.tgz + per-cycle snapshot)
```

---

## LLM Models

| Role | Model | Notes |
|---|---|---|
| Gatekeeper | `bedrock/amazon.nova-micro-v1:0` | Topic-relevance filter: is this article on-topic for the event? Uses a directive coarse-gate prompt that passes INDIRECT evidence (rival collapse, coalition dynamics, etc.), not just explicit predictions; regression-guarded by `pipeline/eval_gatekeeper.py`. |
| Extractor | `bedrock/amazon.nova-lite-v1:0` | Structured extraction of up to 5 predictions per article (6 fields: quote, claim, stance, certainty, settled, quantitative_estimate) |
| Article Aggregator | `bedrock/amazon.nova-lite-v1:0` | Collapses high-spread (>0.4) predictions within a single article into one editorial signal |
| Keywords | `bedrock/amazon.nova-micro-v1:0` | One-time: generate search keywords per event (via `tm.llm`) |

All defaults via AWS Bedrock. Override via env vars in `pipeline/src/tm/config.py`. The `model_api_base` and `model_api_key` settings allow routing through any LiteLLM-compatible provider (OpenRouter, etc.).

---

## Scoring

**Competitive Brier Score** — only predictions from time windows where ≥2 sources published are scored. Single-source windows are excluded.

```
p = (stance + 1.0) / 2.0      # normalize stance to [0, 1]
brier = (p - outcome)²         # outcome = 1.0 if event happened, 0.0 if not
```

Configured in `render_atlas.py`:
```python
SCORING_CONFIG = ScoringConfig(window_hours=48, min_per_window=2)
```

**Confidence-weighted Brier Score** (`scorer.py`) — predictions with higher `certainty` carry more weight:

```
weight          = 0.5 + 1.5 × certainty      # range [0.5, 2.0]
weighted_brier  = brier × weight
```

**Per-category scoring** — `scorer.py` computes `brier_score` and `weighted_brier_score` both globally and broken down by `category` (e.g. "Gaza War", "AI & Tech"). Stored in `leaderboard.json` under `by_category`. Used by the forecasting model to select trusted sources per topic.

**ELO** — zero-sum rating updated after each event: sources that predicted correctly gain points from those that predicted incorrectly. Global only (per-category ELO planned).

---

## Ingest Sources (27 defined)

### Israeli — Hebrew
| Source | Domain |
|---|---|
| Haaretz | haaretz.co.il |
| Ynet | ynet.co.il |
| Israel Hayom | israelhayom.co.il |
| Walla News | news.walla.co.il |
| N12 (Mako) | n12.co.il |
| Maariv | maariv.co.il |
| Channel 13 | 13tv.co.il |
| Channel 14 (Now 14) | now14.co.il |
| Kan 11 | kan.org.il |
| Globes | globes.co.il |
| Calcalist | calcalist.co.il |
| The Marker | themarker.com |
| Uri Kurlianchik | kurlianchik.substack.com |

### Israeli — English
| Source | Domain |
|---|---|
| Times of Israel | timesofisrael.com |
| Jerusalem Post | jpost.com |

### International
| Source | Domain |
|---|---|
| Reuters | reuters.com |
| BBC News | bbc.com |
| CNN | cnn.com |
| Al Jazeera | aljazeera.com |
| Bloomberg | bloomberg.com |
| Wall Street Journal | wsj.com |
| The Guardian | theguardian.com |
| Axios | axios.com |
| Financial Times | ft.com |
| The New York Times | nytimes.com |
| The Washington Post | washingtonpost.com |

### Reference
| Source | Domain | Notes |
|---|---|---|
| Polymarket | polymarket.com | Ground truth pricing, not scored as a media source |

---

## Deployment

### Infrastructure
- **EC2**: `t4g.small` (2 GiB RAM), Ubuntu, `eu-central-1` (Frankfurt)
- **Access**: AWS SSM Session Manager (no SSH key — instance has no key pair)
- **Instance name**: `truthmachine-pipeline` (`i-00ac444b94c5ff9b2`)
- **Public IP**: `3.120.185.111` (dynamic — reassigned on stop/start)
- **Terraform**: imported and managed in [`terraform/`](../terraform/) (`aws_instance.oracle`,
  state key `retro/` in `daatan-terraform-state`). `lifecycle.prevent_destroy = true` guards
  the box; `ignore_changes = [ami, user_data]` means the stack tracks the AWS-level resource
  only — it does not provision or enforce anything inside the OS (see the swap note below).
- **Swap**: 2 GiB swapfile at `/swapfile` (added 2026-07-16 via SSM, persisted in `/etc/fstab`).

> **Note (2026-07-16): `truthmachine.service` OOM-kill crash loop, swap added as a stopgap.**
> The box has no cgroup `MemoryMax` — the kernel OOM killer was reacting to genuine memory
> pressure on a 2 GiB instance running an LLM-orchestration + search pipeline. The journal
> showed repeated `oom-kill` restarts with wildly inconsistent run lengths before each kill
> (8 min to 7+ h), which points at specific memory-heavy inputs (e.g. large evidence
> batches) rather than a steady leak. Even a "stable" 17h run showed only ~280 Mi memory
> "available". The 2 GiB swapfile above converts a hard kill (which discards that cycle's
> in-progress work) into graceful degradation, but doesn't fix the underlying pressure. If
> OOM kills recur, the next step is resizing to `t4g.medium` — that requires a stop/start,
> and this box also serves `oracle-api.service`, so it's a real (if brief) outage window,
> not a quiet SSM tweak.
>
> **Verified (2026-07-16, ~18h after the swapfile was added):** zero `oom-kill` events
> since — the same `truthmachine.service` activation has run continuously across that
> whole window (previously it never survived more than ~7h). System swap usage was 421 Mi
> of the 2 GiB, and the service's own cgroup showed a swap peak of ~58 Mi — i.e. it is
> actually dipping into swap instead of getting killed, not just carrying an idle safety
> net. Memory is still tight (~414 Mi "available"), so this isn't headroom in any
> comfortable sense — but the mitigation is holding. Re-check after a longer window (a
> slower leak could still exhaust 2 GiB eventually, just much later than it exhausted 0);
> escalate to the `t4g.medium` resize above only if `oom-kill` reappears in the journal.

### Two checkouts on one box

The instance hosts two independent `git` worktrees with two systemd services:

| Path | Service | Git lifecycle |
|---|---|---|
| `/home/ubuntu/truthmachine/` | `truthmachine.service` (batch pipeline loop) | Commits `data/progress.json` + `factum_atlas.html`, rebases on `origin/main`, pushes. May accumulate WIP commits between rebases. |
| `/home/ubuntu/oracle-api/`   | `oracle-api.service` (FastAPI under gunicorn) | `git reset --hard origin/main` on every deploy. Never diverges. |

Both checkouts read the same `data/` directory — the API's `.env` sets
`DATA_DIR=/home/ubuntu/truthmachine/data`. **Data is shared, code is not.** This
keeps API deploys trivial: they never have to reason about the pipeline's
unpushed atlas commits. See [`docs/ORACLE_DEPLOY.md`](ORACLE_DEPLOY.md).

### Atlas durability (S3 snapshots)

`data/atlas/` and `data/vault2/` (~14 MB of expensive-to-regenerate LLM output)
are not in git. `infra/snapshot_atlas.sh` tars them to
`s3://truthmachine-atlas-snapshots-<ACCOUNT_ID>/` at the tail of every pipeline
cycle. `infra/restore_atlas.sh` runs from `ec2_bootstrap.sh` between data-dir
creation and service start, restoring `latest.tgz` if `data/atlas/` is empty.
30-day per-cycle retention + 7-day versioned `latest.tgz` give point-in-time
recovery. Full design + IAM in [`docs/ATLAS_SNAPSHOTS.md`](ATLAS_SNAPSHOTS.md).

### Required Secrets (AWS Secrets Manager, `eu-central-1`)

| Secret name | Used by |
|---|---|
| `daatan/openrouter-api-key` | LLM inference via OpenRouter (fallback) |
| `daatan/serpapi-key` | Web search — SerpAPI/Google News (optional) |
| `daatan/serperdev-key` | Web search — Serper.dev/Google News (optional) |
| `daatan/brave-api-key` | Web search — Brave News Search (optional) |
| `daatan/brightdata-api-key` | Web search — BrightData SERP API (optional) |
| `daatan/nimbleway-api-key` | Web search — Nimbleway SERP API (optional) |
| `daatan/scrapingbee-api-key` | Web search — ScrapingBee Google Search (optional) |
| `daatan/github-pat` | Push `factum_atlas.html` to repo |
| `daatan/oracle-api-key` | Shared auth key between Oracle API and daatan |

> **Note (resolved 2026-07-14):** these entries were originally created under `openclaw/*` (a decommissioned stack's namespace). PR #198 pointed the *code* at `daatan/*`; the `daatan/*` entries now exist to match, and the `openclaw/*` copies are retained but read by nothing.
>
> **A missing secret does not fail loudly here.** `_secret()` returns `None` on a miss and the provider is then treated as *not configured* and skipped — no exception, no log line, just a quieter search chain. After adding or renaming any provider secret, run `bash infra/check_keys.sh`, which asserts that every secret `web_search.py` reads actually resolves.

### Bootstrap on an existing EC2 instance

```bash
# On the instance (via SSM session):
curl -sSL https://raw.githubusercontent.com/Daatan/retro/main/infra/ec2_bootstrap.sh | bash

# Start pipeline loop
nohup bash ~/truthmachine/infra/ec2_run.sh \
  >> ~/truthmachine/pipeline_log.txt 2>&1 &
```

### Monitor from Local Machine
```bash
bash infra/monitor.sh
```

### Stop / Restart
```bash
# On EC2 (via SSM):
kill $(pgrep -f ec2_run.sh)

# Restart:
nohup bash ~/truthmachine/infra/ec2_run.sh >> ~/truthmachine/pipeline_log.txt 2>&1 &
```

---

## GitHub Actions

| Workflow | Trigger | Effect |
|---|---|---|
| `deploy-atlas.yml`  | push to `main` touching `factum_atlas.html`, `oracle-test.html`, or `duel.html` | Deploys to **https://daatan.github.io/retro/** via GitHub Pages. The EC2 pipeline commits and pushes `factum_atlas.html` after each cycle, which triggers this. |
| `deploy-oracle.yml` | push to `main` touching `api/**`, `pipeline/**`, `infra/deploy_oracle.sh`, `infra/oracle-api.service`, or the workflow itself; or manual `workflow_dispatch` | Authenticates to AWS via OIDC (no static keys), runs `aws ssm send-command` against the EC2 box to invoke `infra/deploy_oracle.sh`, polls until completion, prints the script output into the Actions log. Includes a no-op fast path when the resolved SHA is already deployed. |

OIDC auth uses an IAM role whose ARN is stored as the `AWS_DEPLOY_ROLE_ARN` repo
variable. The role's trust + permissions are scoped to GH runs from this repo on
`main` (plus `workflow_dispatch`) and to `ssm:SendCommand` on the one oracle
instance with the `AWS-RunShellScript` document only. Templates live in
`infra/iam/` — see [`infra/iam/README.md`](../infra/iam/README.md) and
[`docs/ORACLE_DEPLOY.md`](ORACLE_DEPLOY.md).

---

## Forecasting System Design

> Documented 2026-04-14 based on design conversation.

### Goal

Use the historical prediction+accuracy data to build a **media-weighted forecaster**:
given a new binary question, search the web for relevant articles, weight each source's
prediction by its historical accuracy on that topic, and output a calibrated probability
distribution.

### Data Requirements Added

1. **Event categories/tags** — every event gets one or more topic tags (multi-label).
   Used to compute per-source accuracy scores per topic.
2. **Author-level tracking** — predictions should be linked to author when available,
   enabling author-level accuracy scores within a source.

### Taxonomy (v1)

| Category | Example events |
|---|---|
| `Israeli Politics` | Coalition formation, judicial reform, elections |
| `Gaza War` | Oct 7, ground invasion, hostage deals, Rafah |
| `Regional Geopolitics` | Iran attack, Saudi normalization, ICJ ruling |
| `AI & Tech` | ChatGPT, GPT-4, DeepSeek, EU AI Act, Nvidia $1T |
| `Global` | Everything else |

Events can have multiple tags (e.g. Iran attack → `["Gaza War", "Regional Geopolitics"]`).
New categories can be added freely — the taxonomy is not fixed.

### Scoring Design

- **Confidence-weighted**: high-certainty predictions that turn out correct score more;
  high-certainty wrong predictions score worse. Formula TBD (Brier score already computed).
- **Two accuracy levels**: source-level and author-level, both per topic.
- **Binary outcomes only** for now; architecture should support continuous later.

### Inference Pipeline (future)

```
New question arrives
  │
  ├─ Match to topic category
  ├─ Search all sources for relevant articles (web_search.py)
  ├─ Extract predictions from articles (extractor.py)
  ├─ Look up each source's historical accuracy on that topic
  ├─ Weight predictions by source/author accuracy score
  └─ Aggregate → probability distribution (mean + confidence interval)
```

### ML Model Candidates (ranked by recommendation)

| # | Approach | Notes |
|---|---|---|
| 1 | **Weighted Bayesian Aggregation** | No training needed; Brier-score weights; interpretable |
| 2 | **Isotonic Regression Calibration** | Calibrate #1 against historical outcomes; corrects systematic bias |
| 3 | **Logistic Regression** | Features: source accuracy, certainty, days-before, stance, hedge_ratio |
| 4 | **Gradient Boosting (XGBoost/LightGBM)** | Captures non-linear interactions; best classical ML option |
| 5 | **Fine-tuned LLM Forecaster** | Article text → calibrated probability; highest ceiling, most expensive |

**Recommended path**: Start with #1+#2 → move to #4 once dataset is large enough → #5 long-term.

---

## Forecasting Microservice

> Phases 1–5 complete and live at `oracle.daatan.com`. Wired into daatan v1.9.0
> via `oracle.ts` (context route + express guess route, with automatic fallback
> to the existing LLM `guessChances` path when the Oracle is unavailable).
> Auto-deploys on merge to `main` via `.github/workflows/deploy-oracle.yml`. See
> the [Oracle API contract](https://github.com/Daatan/docs/blob/main/oracle-api.md) and [`docs/ORACLE_DEPLOY.md`](ORACLE_DEPLOY.md).

### Purpose

Given a binary question ("Will X happen by Y?"), return a calibrated probability distribution by searching current articles and weighting predictions by each source's historical accuracy on the relevant topic.

### Input / Output

```
POST /forecast
{
  "question": "Will Israel and Hamas reach a permanent ceasefire by June 2025?",
  "max_articles": 10      // optional
}

→ 200 OK
{
  "question": "Will Israel and Hamas reach a permanent ceasefire by June 2025?",
  "mean": -0.24,                  // credibility-weighted mean stance [-1, 1]; p = (mean+1)/2 ≈ 0.38
  "std": 0.28,
  "ci_low": -0.79,
  "ci_high": 0.31,
  "articles_used": 5,
  "sources": [
    {
      "source_id": "timesofisrael.com",
      "source_name": "Times of Israel",
      "url": "https://www.timesofisrael.com/...",
      "stance": -0.4,
      "certainty": 0.7,
      "credibility_weight": 1.12,
      "claims": ["Talks stalled over the question of a permanent end to the war."],
      "published_date": "2025-05-18",
      "recency_weight": 0.83,
      "relevance_score": 0.91
    }
  ],
  "placeholder": false,
  "insufficient_data": false,
  "reason": null,
  "articles_found": 7,
  "outcome_counts": { "ok": 5, "gate_rejected": 2 },
  "provider": "news-indexer",
  "provider_chain": ["news-indexer"],
  "distilled_query": null
}
```

When the pipeline can't compute a real estimate it returns `insufficient_data: true`
with a `reason` (e.g. `no_search_results`, `all_articles_off_topic`,
`no_usable_weight`, `no_decisive_signal`) instead of a forecast — see "Deferral / insufficient-data" below.

### Pipeline

**Stage 1 — Search & Fetch**
1. `web_search.search_articles(question, limit)` — news-indexer → GDELT → GDELT BQ → Google CSE → SerpAPI → Serper → Brave → Tavily → Newsdata.io → BrightData → Nimbleway → ScrapingBee → DataForSEO → DDG fallback chain (news-indexer is first-in-chain: the local pgvector index is queried before any paid provider)
2. Per article: trafilatura full-text fetch (falls back to title+snippet)

**Stage 2 — Gatekeeper + Extractor** (parallel per article)
1. `gatekeeper.check_is_prediction()` — LLM topic-relevance screen (graded `relevance_score`); the legacy method name predates the softening to a relevance filter. Content-free input never reaches the model: `gatekeeper.carries_proposition()` strips URLs/handles/hashtags and rejects text with no letters left (`is_prediction=false`, `relevance_score=0.0`, zero usage) — a model handed a bare URL confabulates rather than abstaining (retro#359).
2. `extractor.extract_predictions()` — LLM extraction: `stance`, `certainty`, `claim`, etc.

**Stage 3 — Weight by Source Credibility**
1. `leaderboard.get_credibility_weight(source_id)` — OpenSkill conservative score (μ − 3σ) from `leaderboard.json`. That vault is **legacy**: nothing in production has regenerated it since 2026-03-28, so it returns a neutral 1.0 for almost every live source. Setting `RESOLUTION_SHADOW_CREDIBILITY_ENABLED=true` switches this to a shrunk **Brier** score over real daatan resolutions (`resolution_leaderboard.json`) instead — off by default, see [ORACLE_VARIABLES.md](ORACLE_VARIABLES.md) §9 for why Brier and not the vault's μ − 3σ transform
2. `weight = credibility × class_weight[evidence_class] × recency × relevance_weight(relevance)` per prediction (S2 cutover; `class_weight` keyed by the extractor's `evidence_class` — `cited_probability` carries the old ×4 anchor premium, `reported_fact`/`cited_share`/`reporting`/`opinion` fill out the rest of the lookup table; unclassified claims fall back to their own `certainty`) (see [ORACLE_VARIABLES.md](ORACLE_VARIABLES.md) for the audit of these knobs). **`relevance_weight` is a band lookup, not an exponent (retro#394)**: the gatekeeper emits four band edge labels rather than a gradient — zero mass in (0.60, 0.70) across 84,254 judgments, and 51.9% of live voting rows at exactly 0.70 — so squaring it was arithmetic on a categorical value. `RELEVANCE_BAND_WEIGHTS` is initialised to `band²`, so the numbers are unchanged; the table is the place to change them once outcome data can justify it

**Stage 4 — Aggregate → Distribution** (`aggregation.aggregate_pool`, wrapping `pool_sources`)
1. Each source's stance is converted to a probability, clamped to `[0.01, 0.99]`, and the sources are pooled in **log-odds (logit) space** — a weighted *mean* of log-odds (a logarithmic opinion pool), using the Stage 3 weights. The result is converted back to `{ mean, std, ci_low, ci_high }` on the stance scale. The interval's standard error divides by Kish's **effective** sample size `(Σw)²/Σw²`, not the row count — equal weights give exactly `n`, a pool one row dominates gives `1`, so near-weightless rows cannot buy precision they do not contribute (F16, retro#365; measured median `n_eff/n` on live pools is 0.50). Because that interval measures between-source *disagreement* only, a unanimous pool would otherwise publish a zero-width band; `widen_ci_for_unresolved_dispersion` floors it at `1.96·pool_dispersion_floor/√n_eff`, so the floor decays with corroboration and never narrows a band that already has real dispersion in it.
2. Convert to probability: `p = (mean + 1) / 2`
3. **Settlement override** — when ≥ `settlement_min_sources` (2) independent sources report the outcome as an accomplished fact, the estimate is pinned to ±0.94 stance (97/3%) and the response carries `settled: true`. Since #244 a source only counts as settlement-grade when its claim is decisive (`|stance| ≥ settlement_min_claim_stance` and `certainty ≥ settlement_min_claim_certainty`, both 0.9), settled claims skip the stance/certainty realignment step (no retro-fitted odds), and when the caller supplies `claim_direction`/`claim_deadline` an early settlement may only pin in the occurrence direction (arrival → YES, survival → NO) before the deadline. Both request fields are optional and fail-open. Upstream of all of this, a *positive* settlement claim must carry an `event_date` on/before the article's own date or it is demoted at extraction time (`enforce_settlement_event_date` — the Netanyahu false pin, where the sitting coalition "settled" a claim about the *next* election; see [ORACLE_VARIABLES.md](ORACLE_VARIABLES.md)). And the `event_date` both date guards consume is itself audited first: when the article spoke in relative terms ("on Friday"), the extractor also returns the verbatim expression (`event_date_reference`) and `enforce_relative_date_resolution` redoes the calendar arithmetic in Python, overriding a disagreeing model date — LLMs resolve weekdays wrong with confidence (the Knesset incident's "Friday" came back as a Saturday).
4. `aggregate_pool()` is every step of this stage extracted into one pure function, shared by the live pipeline above **and** `POST /pool/aggregate` — a recompute endpoint that reruns this exact math over a caller-supplied set of already-extracted per-source signals (no search, no LLM), so a recompute over an accumulated evidence pool (retro docs/ORACLE_VARIABLES.md, recompute-over-pool) can never silently drift from what a fresh `/forecast` run would produce. Recency is recomputed fresh against "now" for each source's `published_date`, exactly like the live pipeline. Settlement votes are **revalidated per vote** on every call (`settlement_vote_validity` — anchor date required, claim-window coherence, unanimity instead of majority; `SETTLEMENT_REVALIDATE=false` is the kill switch), so a stale stored `settled` bit can no longer pin an estimate — see ORACLE_VARIABLES.md § aggregation-time revalidation.

> **Why Oracle probabilities stay in ~20–80% (by design, not a clamp).** There is
> no hard cap on the output. `logit_clamp` (0.01) is applied in three places, all
> of them bounds rather than caps on the estimate: *per-source*, pinning each
> source's probability to `[0.01, 0.99]` so its log-odds stay finite; on the
> pooled **CI endpoints**, which can leave the range the pooled mean cannot; and,
> since F16 (retro#365), on the thin-evidence widening term, which previously
> clamped to `[0, 1]` and so was the only path by which the Oracle could publish
> a literal 0%–100% band. The settlement pin keeps its own separate bounds
> (`0.005`/`0.995`) — deliberately, since being allowed past the pooling clamp is
> what the override is for.
> The narrow aggregate range is **emergent**: because Stage 4 pools sources as a
> weighted *mean* of log-odds, the pooled probability can never be more extreme
> than its most confident member. Since news-extracted stances are typically
> moderate (rarely near ±1), the aggregate lands around 20–80%. This is
> deliberate — a logarithmic opinion pool is robust to off-topic/stale outliers
> (the original "73% on a decided series" overconfidence bug): a lone dissenter
> can't drag a confident consensus back to 0.5, and the estimate stays bounded by
> its members. The tradeoff is the flip side — the Oracle **also cannot express
> genuine near-certainty** even when warranted. To let it reach the extremes you
> would switch from a logit-*mean* (opinion pool) to a logit-*sum* (Bayesian
> evidence accumulation), which reintroduces the overconfidence risk this design
> removes. Near-certain numbers on daatan.com therefore come from the LLM-fallback
> path, not the Oracle.

### Known limitations

> **Correlated sources, not just single outliers.** The bounded-pool design above
> protects against a lone dissenter dragging a confident consensus back toward 0.5 —
> but it does nothing against the mirror case: many sources independently repeating
> the *same* narrow signal. Five match reports all framing a team as "favorite
> entering the next round" of a multi-stage tournament each add a source-level vote,
> even though none of them individually says anything about the full remaining
> bracket — the pooled estimate can end up more confident than any single article
> actually justifies (this was the root cause of a real Oracle/market divergence on
> a 2026 World Cup forecast). `extractor.py`'s prompt carries narrow, symptom-level
> patches for specific instances of this (numeric-threshold claims, multi-stage
> bracket events), but the pool itself still has no notion of *shared* narrative
> across sources. The root-cause fix — correlated-source downweighting — is gated
> future work; see "Stage C" in [`TEMPORAL_MODEL_PLAN.md`](TEMPORAL_MODEL_PLAN.md).

### Deferral / insufficient-data

Aggregation enforces three safety floors (`api/src/forecast_api/config.py`):

- **`relevance_weight_floor` (0.05)** — if the summed relevance mass
  (Σ `relevance_score²` over surviving articles) is below this, the whole set is
  treated as off-topic → `insufficient_data=true`, `reason="all_articles_off_topic"`.
- **`decisiveness_floor` (0.5)** — minimum total certainty-weighted evidence mass
  (Σ `credibility·certainty·recency·relevance²`) below which the pool is "thin."
  By default (`defer_on_thin_evidence=False`) a thin-but-on-topic pool does **not**
  defer: it still returns a forecast, with the CI **widened** toward maximal
  uncertainty in proportion to the shortfall (`widen_ci_for_thin_evidence`,
  capped by `thin_evidence_ci_inflation`), so thin evidence reads as a
  low-confidence wide-band estimate rather than an abstention. Setting
  `defer_on_thin_evidence=True` restores the old behavior — a thin pool then
  returns `insufficient_data=true`, `reason="no_decisive_signal"` instead.
- **zero total weight** — if every surviving source weighs exactly nothing
  (blocked by credibility, zeroed by relevance, or both) the pool abstains with
  `reason="no_usable_weight"`, regardless of `defer_on_thin_evidence`. Pooling
  anyway would fall through `pool_sources`' zero-total guard, which replaces the
  weights with a flat 1.0 each — the answer would then come, unweighted, from
  exactly the rows the weighting judged worthless (lane-soundness F14, design
  rule R3). A pool with no weight at all is not thin evidence, it is no evidence.

Design rule **R3 — missing data never increases influence** — governs the two
absences that feed those floors, so neither can buy influence:

- an article with **no usable date** decays to `recency_floor` rather than
  returning a neutral 1.0 (F13);
- a claim with **no `evidence_class`** still falls back to its own certainty, but
  capped at `evidence_class_weight_unclassified_cap` (0.25, the weakest class's
  weight), so an unlabelled claim can tie the weakest labelled one and never beat
  it (F10). The same cap applies on the recompute path to a persisted row with no
  stored `evidence_weight`.

Hedged/low-certainty articles are no longer dropped pre-aggregation — `certainty`
is purely a downweighting factor in `weight = credibility · certainty · recency ·
relevance²` (Stage 3 above), so a pool of only-speculative sources naturally falls
toward the `decisiveness_floor` case rather than being filtered out first.

When `relevance_weight_floor` isn't met, or `no_search_results`/`timeout`/no
usable extractions occurred, the API returns `insufficient_data=true` plus a
`reason` **instead of** a forecast. This is the deferral contract: the caller
keeps its own base rate rather than overwriting it with a ~50% coin-flip.

### Deployment (decided 2026-04-14)

**FastAPI microservice in `retro/api/`** — deployed as a second systemd service (`oracle-api.service`) on the retro EC2 alongside the batch pipeline.

- Imports `tm.gatekeeper`, `tm.extractor`, `tm.web_search` directly — no code ported
- Reads `leaderboard.json` from the same `data/` directory (refreshed daily; `leaderboard_refresh_seconds` default 86400)
- Auth: `x-api-key` header + AWS Security Group (daatan SG → port 8001 only)
- Subdomain: `oracle.daatan.com`
- Test console: https://daatan.github.io/retro/oracle-test.html
- Full docs: [Oracle API contract](https://github.com/Daatan/docs/blob/main/oracle-api.md)

**Decisions closed:**
- Source scores stay in `leaderboard.json` on the retro EC2 (no daatan DB sync needed)
- TypeScript port rejected — pipeline is ~2000 lines of Python ML, porting is months of work

---

## Cost Estimates (at scale)

| Scale | LLM cost | Search | News licenses |
|---|---|---|---|
| Current (70 events × 12 sources) | ~$2 | $0 | $0 |
| 100 events × 20 sources, 6 months | ~$2–4 | $0 | $0 |
| 100 events × 100 sources, 10 years | ~$300–500 | $0 | $0–54K |

LLM cost is negligible. The real cost at scale is licensed news archive access.
