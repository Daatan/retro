# BayesOracle

Static calibration layer that applies the **law of total probability** to a DAG of Israeli-politics Polymarket events, producing Bayes-derived probabilities that can diverge from raw PM prices.

> ## ⚠️ Experimental — not in the production forecast
>
> The live Oracle (`POST /forecast`) does **not** use this DAG. It forecasts purely
> via credibility-weighted aggregation in `api/src/forecast_api/forecaster.py`;
> there is no BayesOracle code path in `/forecast`.
>
> **What *is* live:** the `GET /bayes/nodes` endpoint
> (`api/src/forecast_api/bayesoracle.py`) serves node probabilities to the two
> offline HTML viewers (`graph.html`, `pm_analysis/`). That's a visualization
> feed, not a forecasting integration.
>
> **What's *not* built:** wiring the DAG into `/forecast` and validating it against
> `data/duel_oracle/`. `DESIGN.md` describes an eventual hard cutover to a
> Bayes-derived forecast — that is a **design goal, not implemented**. Don't treat
> anything here as authoritative for live probabilities.
>
> _Status: Phase 1 — edge weights calibrated, viewers working._

---

## What's built

| File | Purpose |
|---|---|
| `core.py` | **Single inference engine** — load/validate a JSON graph, fitted-intercept logistic-CPT propagation, exclusive-group normalization |
| `graph_political.json` | Re-noded narrative DAG (expert priors); served by the API |
| `graph_pm.json` | Polymarket-backed DAG (auto-generated; backtestable) |
| `build_pm_graph.py` | Generate `graph_pm.json` from `edge_weights.json` + `node_history/` |
| `backtest.py` | Score the PM graph against realised prices vs persistence (Brier) |
| `tests/test_core.py` | 15 unit tests for the engine |
| `fetch_node_history.py` | Pull daily Polymarket CLOB price history for all 24 DAG nodes |
| `calibrate_edges.py` | Estimate P(B\|A), P(B\|¬A) for each edge via LLM + news search |
| `compute_edge_probs.py` | Blend LLM estimates with empirical price-correlation regression |
| `edge_weights.json` | Output: 28 calibrated edges with pY/pN/blend fields |
| `node_history/` | 24 JSON files — daily price history per node |
| `graph.html` | Interactive what-if slider (legacy embedded model — see RETHINK.md) |
| `pm_analysis/index.html` | BayesOracle vs PM divergence view, ranked by surprise |

> **Architecture note (2026-06):** `core.py` is now the one engine. The API
> (`/bayes/nodes`) loads `graph_political.json` through it; the backtest loads
> `graph_pm.json` through it. The two HTML viewers still embed their own legacy
> data + propagation — porting them to fetch the JSON specs is the remaining
> follow-up. See `RETHINK.md` for the full critique and `backtest.py` output for
> the current (modest, honest) skill vs a no-change baseline.

---

## Run order

```bash
cd /home/mark/projects/retro
source pipeline/.venv/bin/activate

# 1. Fetch or refresh Polymarket price history (skips existing files)
python bayesoracle/fetch_node_history.py

# 2. Calibrate edges — LLM + news search (resumable, skips done edges)
#    ~3 min per edge × 28 edges ≈ 85 min total on first run
python bayesoracle/calibrate_edges.py

# 3. Blend LLM estimates with empirical price correlation
python bayesoracle/compute_edge_probs.py
```

After step 3, paste the JS patch printed by `calibrate_edges.py` into the HTML files (see [Updating the visualizations](#updating-the-visualizations) below).

---

## DAG structure

**24 nodes** — Israeli-politics Polymarket markets (Iran deal, elections, PM candidates, annexation, normalisation deals, etc.). Node IDs and PM market IDs are in `calibrate_edges.py:NODES` and `fetch_node_history.py:NODES`.

**28 edges** in two tiers:

| Type | Count | Use |
|---|---|---|
| Primary | 18 | Used in the BayesOracle LToTP computation: `P(B) = pY·P(A) + pN·(1−P(A))` |
| Secondary | 10 | Shown as dashed arrows for context; not included in the Bayes calculation |

---

## Edge calibration (`calibrate_edges.py`)

For each edge A → B:

1. **Search**: 3 queries via the `tm` provider chain (GDELT → paid fallbacks), deduplicated
2. **Fetch**: up to 4 full article texts via `trafilatura`
3. **LLM call**: Claude Haiku via OpenRouter — estimates `pY = P(B|A=1)` and `pN = P(B|A=0)`
4. **Consistency check**: `pY·P(A) + pN·(1−P(A))` should be within 10pp of the PM price for B
5. **Save**: appended to `edge_weights.json` immediately (crash-safe)

**Context block** (`calibrate_edges.py:115–124`): hand-curated geopolitical summary dated May 14, 2026. Update this before re-running — stale context degrades the LLM's calibration.

**Rate limit**: `time.sleep(1.2)` between edges. Each edge makes 3 search calls; GDELT needs ≥10s between calls. If GDELT is blocked (EC2 IP 429), the chain falls through to paid providers.

---

## Empirical blend (`compute_edge_probs.py`)

Requires ≥20 aligned daily price points between A and B to run a regression. Most pairs have enough history; pairs with `n_corr < 20` fall back to LLM-only.

**Algorithm**:
- Linear regression `P_B = α + β·P_A` → `pY_corr = clip(α+β)`, `pN_corr = clip(α)`
- Weight: `w_corr = R² × 3.0` (up-weighted when correlation is strong)
- Final: `pY_blend = (1·pY_llm + w_corr·pY_corr) / (1 + w_corr)`

Edges where LLM and correlation point in opposite directions are flagged as `DIRECTION FLIP` in the output — these warrant manual review.

---

## Visualizations

These are **two separate DAGs** with different scopes and purposes. Both are static HTML files with Cytoscape.js and all data embedded as inline JS — no external files are read at runtime.

### `graph.html` — political mechanics what-if

**20 nodes** covering the causal chain from current political conditions to downstream outcomes:

```
Layer 0 (roots): ELECTIONS · TRUMP · BEYACHAD · IRAN_REB · LIKUD_UNITY
Layer 1:         BIBI_LEAD · CONVICTED · SICK · DEAD
Layer 2:         A (Likud wins most seats)
Layer 3:         MANDATE · HAREDI_J · FARRIGHT_J · COAL_61
Layer 4:         PM5 · OPP_PM
Layer 5:         PARDON · JUDICIAL · HAREDI_LAW · SAUDI · IRAN2
```

Nodes have **CI ranges** (`ci:[lo, hi]`). Propagation uses **logit-space interpolation** (not pure LToTP) — when a parent's probability shifts by Δ, children update by `Δ·(logit(pY) − logit(pN))` in logit space, then sigmoid back. This keeps downstream probabilities in (0,1) regardless of chain depth.

No Polymarket market IDs — this is a self-contained narrative model. Node probabilities are hand-set from expert judgment and current polling.

Click any node → slider appears → drag to hypothesise a new P → all downstream nodes cascade. Multiple nodes can be locked (pinned) simultaneously. "Reset All" restores the baseline values baked into the file.

Open: `file:///home/mark/projects/retro/bayesoracle/graph.html`

### `pm_analysis/index.html` — PM × BayesOracle divergence view

**24 nodes** directly mapped to Polymarket markets (each has `pmId` and a `slug` for a direct link). This is the one connected to the calibration pipeline — the `pY`/`pN` values come from `edge_weights.json` (via the JS patch).

**Propagation**: primary-parent LToTP `P(B) = pY·P(A) + pN·(1−P(A))`, then a 0.5-weighted contribution from each secondary parent stacked on top.

Each node shows `PM% / Bayes%`. The sidebar ranks all nodes by `|pm − bayes|` ("Most Surprising Markets"). Click any node to see the full conditional breakdown — primary parent's pY/pN contribution, and each secondary edge's contribution.

Also shows the **PM candidate sum** (BIBI_PM + BENNETT_PM + EIZENKOT_PM + LIEBERMAN_PM + LAPID_PM). Should be ≤1.0; large slack means a significant unlisted-candidate probability or stale PM prices.

Open: `file:///home/mark/projects/retro/bayesoracle/pm_analysis/index.html`

---

## Updating `pm_analysis/index.html` after recalibration

Both the `NODES` array (with `pY`/`pN` per node) and the `EXTRA_EDGES` array are baked into the HTML. After re-running `calibrate_edges.py` + `compute_edge_probs.py`:

1. `calibrate_edges.py` prints a JS patch at the end of its output.
2. Update the `pY`/`pN` values in the `NODES` array in `pm_analysis/index.html`.
3. Replace the `EXTRA_EDGES` array in full with the printed version.
4. Also update the `pm:` field for any nodes whose Polymarket prices have moved significantly.

`graph.html` has an independent node set — update it separately when the political situation changes (polling shifts, conviction, election results, etc.).

---

## Data formats

**`edge_weights.json`** — array of edge records:
```json
{
  "source": "CEASEFIRE_X",
  "target": "ELECTIONS",
  "type": "primary",
  "pY": 0.78,
  "pN": 0.52,
  "implied_p": 0.603,
  "pm_p": 0.59,
  "drift": 0.028,
  "articles_used": 4,
  "reasoning": "...",
  "pY_corr": 0.875,
  "pN_corr": 0.412,
  "r2_corr": 0.2779,
  "n_corr": 87,
  "pY_blend": 0.875,
  "pN_blend": 0.448,
  "w_llm": 1.0,
  "w_corr": 0.834
}
```

**`node_history/{NODE_ID}.json`**:
```json
{
  "node_id": "ELECTIONS",
  "pm_id": "1280208",
  "clob_token_yes": "...",
  "prices": [
    {"date": "2025-04-01", "probability": 0.52},
    ...
  ]
}
```

---

## What's not yet implemented

See `DESIGN.md` for the full target architecture. Items not yet built:

- **P(A|B) via LLM on event articles** — `calibrate_edges.py` uses news search; DESIGN calls for using the existing Atlas article corpus
- **p1/p2 fusion** — blending DAG-derived P(A) with TruthMachine's credibility-weighted direct forecast
- **Reactive DAG propagation** — updating child nodes when new articles arrive for a parent
- **Oracle API integration** — `/forecast` returning both `base_forecast` and `bayes_forecast`
- **Correlated parents / Gaussian copula** — multi-parent joint distributions
- **`bayes_graph.html`** drilldown per DESIGN spec (current `graph.html` is a working prototype)
