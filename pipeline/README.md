# TruthMachine Pipeline

Python pipeline for retroactive media prediction extraction and scoring.

## Structure

```
pipeline/
  src/tm/
    config.py              # settings via pydantic-settings (models, API keys, paths)
    models.py              # pydantic schemas (Prediction, ExtractionOutput, CellSignal, etc.)
    progress.py            # matrix state tracker + rich terminal visualizer

    # Ingest
    gnews_ingest.py        # GNews RSS → URL resolution → trafilatura + Wayback fallback
    gdelt_ingest.py        # GDELT Doc 2.0 API batch ingestor (sequential, rate-limited)
    ingestor.py            # Pluggable ingestor classes: DDGIngestor, GDELTIngestor
    site_search.py         # Direct site-search scraper (no API key, high reliability)
    web_search.py          # Multi-provider search: news-indexer → GDELT → GDELT BQ → Google CSE → SerpAPI → Serper → Brave → Tavily → Newsdata.io → BrightData → Nimbleway → ScrapingBee → DataForSEO → DDG → trusted-sites
    polymarket.py          # Polymarket Gamma API: fetch market history per event
    polymarket_harvest.py  # Bulk harvest of all resolved Polymarket political markets

    # Extraction
    gatekeeper.py          # LLM stage 1: graded topic/evidence-relevance gate (emits relevance_score; passes indirect evidence, not just explicit predictions)
    extractor.py           # LLM stage 2: extract structured predictions from article
    runner.py              # Orchestrates gatekeeper → extractor per article
    aggregator.py          # Cell-level: collapse all predictions → CellSignal
    reaggregate.py         # Post-processing: re-run aggregation on high-variance cells

    # Scoring & Output
    orchestrator.py        # Batch runner: events × sources → vault → atlas
    dedup.py               # SimHash near-duplicate detection (64-bit, Hamming ≤8) — deduplicates syndicated rewrites at ingest
    scorer.py              # Brier score + ELO + OpenSkill + calibration + per-category scoring
    backtest.py            # LightGBM backtest vs Polymarket (see BACKTEST.md)
    render_atlas.py        # Renders factum_atlas.html from atlas/ data
    generate_pages.py      # Generates per-event/source static HTML pages
    sync_atlas.py          # Parses event table and syncs atlas entry JSON files

    # One-off scripts
    init_db.py             # Initialize SQLite DB for progress tracking
    migrate_cell_signals.py  # One-time: compute cell_signal.json from existing vault data
    poc_event_gen.py         # Convert harvested Polymarket events → pipeline event JSONs
    create_real_samples.py   # Create real sample data for testing
    create_sample_data.py    # Create synthetic sample data for testing

  scripts/
    improve_keywords.py    # One-time: LLM-generate search keywords for events
    audit_event_criteria.py  # Lint llm_referee_criteria fields for weak/missing anchors
  tests/
    test_models.py
  smoke_test.py            # 3 hardcoded articles through full pipeline
  test_run.py              # Manual test runner
  BACKTEST.md              # backtest design rationale and usage
```

## Setup

```bash
cd pipeline
cp .env.example .env   # configure API keys (see below)
uv sync
```

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `MODEL_API_KEY` | Yes* | API key for the LLM provider (AWS credentials or OpenRouter key) |
| `MODEL_API_BASE` | No | LiteLLM-compatible base URL (leave empty for AWS Bedrock default) |
| `AWS_REGION` | No | AWS region for Bedrock (default: `us-east-1`) |
| `OPENROUTER_API_KEY` | No | OpenRouter key — alternative LLM provider |
| `SERPAPI_API_KEY` | No | SerpAPI — news search (priority 2) |
| `SERPER_API_KEY` | No | Serper.dev — news search (priority 3) |
| `TAVILY_API_KEY` | No | Tavily — news search with date windowing (priority 3b); 1 credit/call |
| `BRAVE_API_KEY` | No | Brave News Search (priority 4); also URL resolution fallback in gnews_ingest |
| `BRIGHTDATA_API_KEY` | No | BrightData SERP API (priority 5) |
| `NIMBLEWAY_API_KEY` | No | Nimbleway SERP API (priority 6) |
| `SCRAPINGBEE_API_KEY` | No | ScrapingBee Google Search (priority 7) |
| `NEWSDATA_API_KEY` | No | Newsdata.io archive search (priority 8) |
| `DATAFORSEO_API_KEY` | No | DataForSEO — last-resort paid fallback (priority 9) |
| `GCP_SA_KEY_JSON` | No | GCP service account JSON — enables GDELT BigQuery for historical queries (>90 days) |
| `DATA_DIR` | No | Path to data directory (default: `../data`) |
| `VAULT_DIR` | No | Path to vault directory (default: `$DATA_DIR/vault`) |

\* AWS Bedrock (the default) uses the ambient AWS credentials (`~/.aws/credentials` or instance role), not `MODEL_API_KEY`. Set `MODEL_API_KEY` only when using an explicit key-based provider.

> **Note:** `vault/` may be root-owned if previously created inside Docker. Set `VAULT_DIR` to a
> user-writable path (e.g. `data/vault2/`) to avoid permission errors when running locally.

## Running the orchestrator

The orchestrator processes articles from `data/raw_ingest/` through the full LLM pipeline and
writes scored entries to `data/atlas/`.

```bash
# local_file mode — reads from data/raw_ingest/{source}/{event}/article_*.json
DATA_DIR=/path/to/data VAULT_DIR=/path/to/data/vault2 \
  uv run python -m tm.orchestrator local_file

# api mode — fetches articles via Brave Search API
DATA_DIR=/path/to/data VAULT_DIR=/path/to/data/vault2 \
  uv run python -m tm.orchestrator api
```

### Raw ingest article format

Articles go in `data/raw_ingest/{source_id}/{event_id}/article_*.json`:

```json
{
  "headline": "Article headline",
  "text": "Full article text...",
  "published_at": "YYYY-MM-DD",
  "author": "Reporter Name",
  "url": "https://..."
}
```

The `published_at` date must fall within the prediction window (3–30 days before the event's
`outcome_date`) or the article will be filtered out by the backtest.

## Running the backtest

After the orchestrator has populated `data/atlas/`, run:

```bash
# Specific events
DATA_DIR=/path/to/data \
  uv run python -m tm.backtest --events A01 A02 A04 A05 --output data/backtest/

# All resolved events
DATA_DIR=/path/to/data \
  uv run python -m tm.backtest --all-resolved --output data/backtest/

# Force weighted average (skip LightGBM)
DATA_DIR=/path/to/data \
  uv run python -m tm.backtest --all-resolved --no-lgbm
```

LightGBM requires at least 5 events with atlas entries; below that it falls back to weighted
average. See `BACKTEST.md` for full design rationale and output interpretation.

## Run smoke test

```bash
uv run python smoke_test.py
```

Runs 3 articles (1 English, 1 Hebrew, 1 non-prediction) through the full pipeline and renders the matrix progress grid.

## Matrix visualization

The progress grid shows all events × sources as a compact terminal table:

```
         YNT HAA N12 ...
  A01     ▓   ▓   ░
  A02     ░   ▒   ░
  B01     ✗   ▓   ·

░ pending  ▒ in_progress  ▓ done  ✗ failed  · no predictions
Progress: 3/250 (1.2%) | done: 2 | no_pred: 1 | failed: 0
```

## Pipeline stages

| Stage | File | Model | Purpose |
|---|---|---|---|
| 1 | `gatekeeper.py` | `bedrock/amazon.nova-micro-v1:0` | graded topic/evidence-relevance gate (emits relevance_score; passes indirect evidence, not just explicit predictions) |
| 2 | `extractor.py` | `bedrock/amazon.nova-lite-v1:0` | extract up to 5 predictions per article (4 fields: quote, claim, stance, certainty) |
| 3 | `runner.py` | — | orchestrate stages 1+2 per article |
| 4 | `aggregator.py` | — | collapse article predictions → CellSignal |
| 5 | `orchestrator.py` | — | batch across all events × sources |
| 6 | `backtest.py` | LightGBM | compare predictions to Polymarket via Brier score |

## Scoring contract (invariants)

Two guarantees are enforced at the **scoring boundary** (`scorer.py`,
`render_atlas.py`, `backtest.py`) so they hold regardless of how an article
entered the atlas:

- **Anti-lookahead** — a prediction is scored only if its `article_date` is on or
  before the event's `outcome_date` (`utils.predates_outcome`). Post-outcome
  entries are dropped, never scored — they would leak future knowledge into
  `leaderboard.json` (and thus the live Oracle's credibility weights).
- **Loud field validation** — `stance` and `certainty` are required. A prediction
  missing either (corruption / schema regression) is **skipped with a warning**,
  never silently defaulted to neutral (`utils.split_scored_predictions`). A
  systemic regression therefore surfaces as warnings + a dropped-count summary,
  not as quietly-wrong scores.

Optional fields (`specificity`, `hedge_ratio`, …) are intentionally allowed to
default — they're `Optional` and were dropped from the extractor in PR #102.

## Known issues

- `data/vault/` created inside Docker is root-owned. Use `VAULT_DIR` env var to redirect writes
  to a user-writable path when running locally.
- `data/atlas/{event}/{source}/entry_*.json` files created inside Docker are root-owned and
  cannot be overwritten locally. New entries use a `_v2` suffix to avoid the conflict.
