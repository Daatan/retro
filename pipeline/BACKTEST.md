# Bediavad Backtest Engine

> **File:** `pipeline/src/tm/backtest.py`
> **Purpose:** Empirically validate the TruthMachine thesis — does scoring media sources on historical accuracy produce predictions that outperform Polymarket consensus?

---

## Why This Exists

The core commercial claim of Bediavad is:

> *"Because we know who was right in the past, we can predict the future more accurately than market consensus."*

Before building the full Oracle API, we need to test whether this is actually true. The backtest engine answers the question directly: **given the Factum Atlas data we already have, does our model beat Polymarket?**

This is not a unit test. It is the primary empirical proof-of-concept for the entire business.

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

### 3. Polymarket via Gamma API (automatic)

**Why not manual:** Manual Polymarket prices would require human judgment to match event definitions — introducing bias. The Gamma API search is imperfect but reproducible and unbiased.

**Known limitation:** Polymarket coverage of Israeli events is ~30%. The script returns `None` for unmatched events and excludes them from the Brier comparison. This is intentional — better to exclude than to fake a comparison.

The script fetches the price **at the start of the prediction window** (30 days before the event), not at resolution. This is the fair comparison point: what did the market believe at the same time our media sources were publishing?

### 4. Brier score as the metric

Brier score = `(prediction - outcome)²`. Lower is better. A perfectly calibrated random guesser scores 0.25. A perfect predictor scores 0.0.

**Why Brier and not accuracy:** Accuracy (binary correct/incorrect) ignores calibration — a model that says 0.51 when the true probability is 0.95 is penalized equally to one that says 0.49. Brier score rewards well-calibrated probabilities.

**Win threshold:** A Brier difference of 0.01 is used to declare a winner. Differences smaller than this are noise given our dataset size.

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
| `source_brier` | Historical Atlas | The source's track record in this domain |
| `prediction_count` | Computed | Articles with more predictions signal a more actively covered event |

Multiple predictions within a single article are aggregated by mean before feeding to the model.

---

## Output

### Terminal report (Rich)

```
─────────────── Bediavad Backtest Report ────────────────

┌─────────────────────────────────────────────────────────┐
│ Event │ Outcome │ Our P │ Our Brier │ Poly P │ Poly B │ Winner │
├───────┼─────────┼───────┼───────────┼────────┼────────┼────────┤
│ A02   │ ✅ YES  │ 0.731 │ 0.0726    │ 0.680  │ 0.1024 │ US     │
│ B01   │ ✅ YES  │ 0.612 │ 0.1488    │ 0.710  │ 0.0841 │ POLY   │
│ D02   │ ✅ YES  │ 0.788 │ 0.0452    │ N/A    │ N/A    │ —      │
└───────┴─────────┴───────┴───────────┴────────┴────────┴────────┘

Aggregate Brier Score
  Ours:       0.0854
  Polymarket: 0.0934
  Beat Poly:  4/6 events
  Lost:       2/6 events
  Tied:       0/6 events

✅ We outperform Polymarket overall

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

## Historical Article Search (Oracle `/search`)

Bediavad needs articles published **before** a specific date, not current news. The Oracle's `/search` endpoint supports this directly.

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
| Our Brier < Poly Brier by >0.01 | We beat Polymarket on this event |
| `N/A` in Poly column | No Polymarket market found — excluded from comparison |
| LightGBM fallback message | Fewer than 10 training samples — weighted average used |
| Source contribution near 0 | Source had no clear directional stance in the window |
| Source contribution strongly negative | Source consistently predicted the opposite of what happened |

---

## Known Limitations

1. **Polymarket matching is fuzzy.** The Gamma API text search may match the wrong market, or miss a valid one. Results with `N/A` should be manually verified before publishing.

2. **Small dataset bias.** With fewer than 20 resolved events, LOO cross-validation is noisy. Aggregate Brier differences of <0.02 are not statistically meaningful. Run on 50+ events before drawing conclusions.

3. **No calibration layer yet.** The LightGBM outputs are raw probabilities, not isotonic-calibrated. They may be overconfident. Calibration will be added in Phase 2.

4. **Source Brier scores bootstrapped.** Until enough Atlas events are resolved, `source_brier` defaults to 0.25 (random baseline). This means early runs underweight the source track record feature.

5. **Single prediction window.** The script uses one fixed window per event. A more sophisticated version would run multiple windows (7d, 14d, 30d) and compare which window produces the strongest signal.

---

## Roadmap

- [ ] Add isotonic calibration post-LightGBM
- [ ] Add Kalshi as a second comparison baseline
- [ ] Add confidence intervals via bootstrap resampling
- [ ] Add multi-window analysis (7d vs 14d vs 30d)
- [ ] Add SHAP feature importance output per event
