# Bediavad Backtest Engine

> **File:** `pipeline/src/tm/backtest.py`
> **Purpose:** Empirically validate Bediavad's retroactive source-audit thesis — does scoring media sources on historical accuracy produce a signal that predicts outcomes out-of-sample?

> ⚠️ **Scope — read first.** The **head-to-head against Polymarket is a separate
> project: the Duel** (`pipeline/src/tm/duel_report.py` → `duel.html`,
> caches in `data/duel_oracle/`). The Duel is the headline external-validation
> metric ("is TruthMachine better than a price-discovered market?") and uses the
> live **Oracul API** with a strict T-day temporal protocol and real Polymarket
> CLOB price history. **Bediavad does not own the Polymarket comparison.** The
> legacy `fetch_polymarket_price` copy that used to be wired into `backtest.py`
> (it duplicated the Duel and returned `null` for every event) has been removed.
> Bediavad's deliverable is the **source credibility leaderboard** (Brier / ELO
> per source), not beating the market. For the Polymarket numbers, see the Duel.

---

## Why This Exists

Bediavad's claim is:

> *"From the public trail of what media already published, we can reconstruct who was right — per source, per domain — without waiting for anyone to opt in."*

The backtest engine answers a narrower, internal question: **given the Factum
Atlas data we already have, does the extracted source signal predict the event
outcome out-of-sample?** Whether that signal then beats a market is the Duel's
question, not this engine's.

This is not a unit test. It is the proof-of-concept for the retroactive-audit
methodology that produces the source leaderboard.

---

## Design Decisions

### 1. Prediction window: 3–30 days before the event

**Why 3 days minimum:** Articles published in the last 72 hours before an event are mostly reactive news, not forward-looking predictions. They add noise, not signal.

**Why 30 days maximum:** Beyond one month, the information environment is too different from the resolution date. A prediction made 6 months before an election reflects a different political reality than what actually determined the outcome. Polymarket also typically opens markets 1–3 months before resolution, so this window ensures a fair comparison.

This window is configurable via `MIN_DAYS_BEFORE_EVENT` and `MAX_DAYS_BEFORE_EVENT` constants.

### 2. LightGBM with leave-one-out cross-validation

**Why LightGBM:** Gradient-boosted trees are the best-performing model family on structured/tabular data at small-to-medium scale. They handle mixed feature types (floats, categoricals, nulls) natively and degrade gracefully with small datasets.

**Why leave-one-out (LOO):** With a small Atlas (20–100 events), standard train/test splits waste data and produce unstable estimates. LOO trains on all events except the one being tested, ensuring every prediction is truly out-of-sample. This gives the most honest accuracy estimate possible at small scale.

**Why weighted average fallback:** LOO requires at least 10 training samples. Early in the Atlas build (fewer than ~15 resolved events), LightGBM is unreliable. The weighted average uses source Brier scores directly, which is a sensible baseline that works from day one.

### 3. Brier score as the metric

Brier score = `(prediction - outcome)²`. Lower is better. A perfectly calibrated random guesser scores 0.25. A perfect predictor scores 0.0.

**Why Brier and not accuracy:** Accuracy (binary correct/incorrect) ignores calibration — a model that says 0.51 when the true probability is 0.95 is penalized equally to one that says 0.49. Brier score rewards well-calibrated probabilities.

---

## Feature Vector

Each article in the Atlas window is converted to 11 features:

| Feature | Source | Rationale |
|---|---|---|
| `stance` | LLM extraction | Primary directional signal — is the source bullish or bearish on the event? |
| `certainty` | LLM extraction | High-certainty predictions are more informative |
| `specificity` | LLM extraction | Vague predictions are discounted |
| `hedge_index` | LLM extraction | Heavy hedging reduces the effective signal |
| `conditionality` | LLM extraction | Conditional predictions ("if X then Y") are weaker signals |
| `magnitude` | LLM extraction | Big predicted outcomes are more newsworthy but not necessarily more accurate |
| `source_authority` | LLM extraction | Predictions based on named sources are more reliable than opinion |
| `sentiment` | LLM extraction | Emotional charge of the article |
| `days_before` | Computed | Recent predictions carry more weight than early ones |
| `source_brier` | `data/leaderboard.json` | The source's overall historical Brier track record |
| `prediction_count` | Computed | Articles with more predictions signal a more actively covered event |

Multiple predictions within a single article are aggregated by mean before feeding to the model.

---

## Output

### Terminal report (Rich)

```
─────────────── Bediavad Backtest Report ────────────────

┌────────────────────────────────────────────┐
│ Event │ Outcome │ Our P(YES) │ Our Brier    │
├───────┼─────────┼────────────┼──────────────┤
│ A02   │ ✅ YES  │ 0.731      │ 0.0726       │
│ B01   │ ✅ YES  │ 0.612      │ 0.1488       │
│ D02   │ ✅ YES  │ 0.788      │ 0.0452       │
└───────┴─────────┴────────────┴──────────────┘

Aggregate Brier Score
  Ours: 0.0854  (3 events)

Source Contribution (avg stance weight)
  haaretz      +0.312   8 events
  bloomberg    +0.284   6 events
  israel_hayom -0.198   7 events
```

### JSON output

Per-event: `data/backtest/{event_id}_backtest.json`
Summary: `data/backtest/summary.json`

The summary includes: run timestamp, model type used, window settings, full results array.

---

## Historical Article Search (Oracul `/search`)

Bediavad needs articles published **before** a specific date, not current news. The Oracul's `/search` endpoint supports this directly.

```bash
# Example: find articles about Bitcoin published before 2025-01-01
curl -s -X POST https://oracle.daatan.com/search \
  -H "x-api-key: $ORACLE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "bitcoin price crash",
    "limit": 15,
    "date_from": "2024-10-01",
    "date_to": "2025-01-01",
    "enrich_snippets": true
  }'
```

**Provider behaviour with date filters:**
- **GDELT (primary):** Supports `startdatetime`/`enddatetime` natively. Free, no API key. Rate-limited to 1 request per 10 seconds — pace bediavad calls accordingly (≥12s between requests to be safe).
- **GDELT returns no snippets** — set `enrich_snippets: true` to scrape article text in parallel (8 workers, 8s cap). Adds 5–15s per call.
- If GDELT hits 429 (rapid sequential calls) it falls through to paid providers which may not respect the date filter.

**Recommended bediavad loop:**
```python
import time, requests

def oracle_search_historical(query, date_from, date_to, limit=15):
    r = requests.post(
        "https://oracle.daatan.com/search",
        headers={"x-api-key": ORACLE_API_KEY},
        json={"query": query, "limit": limit,
              "date_from": date_from, "date_to": date_to,
              "enrich_snippets": True},
        timeout=60,
    )
    results = r.json().get("results", [])
    time.sleep(12)  # stay within GDELT 1-req/10s rate limit
    return results
```

---

## Free, deterministic backfill via GDELT-BigQuery (`tm.gdelt_bq_ingest`)

The Oracul `/search` loop above works but pays for SERP fallback and drifts
run-to-run. For retro backfill prefer **`tm.gdelt_bq_ingest`** — a batch ingestor
that queries GDELT's GKG table in **BigQuery** directly:

- **Free.** A per-event 27-day window scans ~10 GB; the full 70-event recompute is
  ~0.7 TB, inside BigQuery's free 1 TB/month. (Measured, not estimated.)
- **Deterministic.** Same fixed table + window → same URLs every run — what a
  scientific backtest needs. No provider drift.
- **Free-only.** Never calls a paid provider; an outlet with no GKG coverage is a
  logged "miss", so cost stays provably bounded.
- **Per-source & cheap by construction.** ONE scan per event over a
  `SourceCommonName IN (tracked outlets)` filter (the filter is *free* — it adds no
  scanned bytes), then URLs are bucketed by `source_id` locally. Never query
  per-(event × source): that re-scans the same partitions 20× (~14 TB ≈ $80).
- **Wayback-first text.** GKG stores the URL, not the body. Each URL is fetched from
  the Internet Archive snapshot nearest its publish date (recovers dead 3-year-old
  URLs *and* reads the pre-outcome version, hardening anti-lookahead); `--allow-live`
  opts into a live fallback.
- **Windowed + spread-sampled.** The ingest window matches the backtest's scored
  range `[outcome − MAX_DAYS, outcome − MIN_DAYS]` (default 30d…3d), so it never
  wastes fetches on the reactive last-3-days the backtest discards. Within each
  source it samples articles **spread across the window** rather than taking the
  newest N, so cells carry forward-looking coverage, not just the reactive tail.

Requires the `daatan/gcp-service-account-key` secret (a BigQuery Job User SA). It
writes to `data/raw_ingest/{source_id}/{event_id}/`, so extraction → Atlas is the
unchanged `orchestrator local_file` step.

```bash
# Backfill specific events (Wayback-only, ≤8 articles/source)
DATA_DIR=$PWD/data uv run python -m tm.gdelt_bq_ingest --events A19 C07 B10

# Backfill everything, allow a live fetch when no pre-outcome snapshot exists
DATA_DIR=$PWD/data uv run python -m tm.gdelt_bq_ingest --all --allow-live

# Discovery: which tracked outlets covered a topic in a window (NO-event hunting)
DATA_DIR=$PWD/data uv run python -m tm.gdelt_bq_ingest --discover "Rafah offensive" \
    --date-from 2024-01-01 --date-to 2024-05-01

# Then extract → atlas as usual:
DATA_DIR=$PWD/data uv run python -m tm.orchestrator local_file
```

> A second anti-lookahead backstop now also lives at the Atlas write itself
> (`orchestrator.create_atlas_link` refuses any article dated after the outcome),
> covering the near-duplicate-reuse and cached-extraction paths that bypass the
> search-time window filter.

---

## How to Run

```bash
cd pipeline
uv sync

# Specific events
uv run python -m tm.backtest --events A01 A02 B01 D02 --output data/backtest/

# All resolved events in the Atlas
uv run python -m tm.backtest --all-resolved --output data/backtest/

# Force weighted average (no LightGBM)
uv run python -m tm.backtest --all-resolved --no-lgbm
```

---

## Interpreting Results

| Situation | Meaning |
|---|---|
| Our Brier < 0.25 | Better-than-random calibration on this event |
| LightGBM fallback message | Fewer than 10 training samples — weighted average used |
| Source contribution near 0 | Source had no clear directional stance in the window |
| Source contribution strongly negative | Source consistently predicted the opposite of what happened |

---

## Known Limitations

1. **Near-total class imbalance.** 69 of 70 events resolved `true`. With essentially no NO outcomes, Brier is uninformative (an always-YES guesser scores ≈0.014) and discrimination (AUC/calibration) cannot be measured. This is the top validity gate — add NO-outcome events before drawing conclusions.

2. **Small dataset bias.** With fewer than 20 resolved events, LOO cross-validation is noisy. Aggregate Brier differences of <0.02 are not statistically meaningful. Run on 50+ events before drawing conclusions.

3. **Calibration is in-sample.** An isotonic (Pool-Adjacent-Violators) calibration layer now post-processes the raw probabilities and reports a calibrated Brier alongside the raw one, plus an always-YES base-rate baseline. The fit is currently in-sample (no held-out calibration fold yet), so read the calibrated number as an upper bound. On the present all-YES set it trivially maps every prediction to 1.0 — which is exactly the imbalance signal, not a real gain.

4. **Source Brier scores are sparse.** Sources not yet in `data/leaderboard.json` default to 0.25 (random baseline), so early runs underweight the source track-record feature for unranked sources.

5. **Single prediction window.** The script uses one fixed window per event. A more sophisticated version would run multiple windows (7d, 14d, 30d) and compare which window produces the strongest signal.

---

## Roadmap

- [x] Add isotonic calibration post-LightGBM (in-sample; held-out fold pending)
- [x] Add always-YES base-rate baseline column
- [ ] Add confidence intervals via bootstrap resampling
- [ ] Add multi-window analysis (7d vs 14d vs 30d)
- [ ] Add SHAP feature importance output per event
