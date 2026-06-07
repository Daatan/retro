# BayesOracle — Rethink

> A critical review of the current BayesOracle (methodology, code, data, nodes) with a
> prioritized set of proposed changes. This is a **design memo, not an implementation** —
> nothing here has been applied to `graph.html`, `pm_analysis/`, the API, or the pipeline.
> Read alongside `DESIGN.md` (the ambition) and `README.md` (what's actually built).

Status anchors used below (verified from the repo, 2026-06-05):
- `node_history/*.json` runs **2025-07 → 2026-06-03** for all 24 PM nodes (fresh).
- `edge_weights.json` = 28 calibrated edges; feeds **only** `pm_analysis/`.
- `graph.html` NODES are hand-set, dated "May 2026", **no PM IDs**; the live API
  (`api/src/forecast_api/bayesoracle.py`) is a line-for-line copy of those hand-set numbers.

---

## TL;DR — what I'd change first

1. **Fix the pivotal node.** Replace `A = "Likud wins most seats"` as the hinge with
   **`RIGHT_BLOC_61 = "right/Bibi bloc reaches 61 seats"`**. Largest party ≠ forms
   government (Bennett 2021). This is the single biggest *modeling* error and it's cheap to fix.
2. **Re-parameterize (don't replace) the inference engine, and relabel it.** The log-odds
   additive form is the right home for a graph with inhibitory edges — but the weights are
   mis-set and the docs call it "law of total probability," which it is not. Either fit the
   weights to a target, or promote to an exact discrete Bayes net (tractable at this size).
3. **One source of truth for the graph.** Today the node/edge tables are copy-pasted across
   `graph.html`, the API module, and a *different* set in `pm_analysis/`. Three hand-synced
   copies, three different propagation rules. Collapse to one JSON the API serves and the
   HTML fetches.
4. **Add a scoring loop before anything else fancy.** Nothing here is validated against
   anything. `node_history/` is the unused asset: backtest the DAG as-of date *T*, score
   against realized PM at *T+k* (Brier). Without this, a rethink just swaps untested numbers.
5. **Model mutually-exclusive outcomes as categoricals**, not independent binaries
   (who's PM; PM5 vs OPP_PM). The PM-candidate binaries currently sum to 0.925 with nothing
   enforcing ≤ 1.

Everything below expands these, grouped by the buckets requested: methodology, code, data,
ideas, nodes.

---

## 1. Methodology

### 1.1 The inference engine is mislabeled and mis-parameterized (not wrong-formed)

Current rule (`graph.html:275`, mirrored in `bayesoracle.py:129`):

```
logit(P_child) = logit(prior_child) + Σ_parents [logit(pYes) − logit(pNo)] · (P_parent − prior_parent)
```

What's **right** about it: a logistic / log-odds-additive CPT is the correct home for this
graph. Edges here are mixed excitatory *and* inhibitory (`CONVICTED→A` 0.16/0.50,
`BEYACHAD→A` 0.34/0.62 are inhibitions). Noisy-OR can't represent inhibition; naive
multi-parent LToTP (`Σ` of linear per-parent terms) can leave [0,1]. The sigmoid bounds the
result and a negative weight encodes inhibition cleanly. So **don't rip the engine out.**

What's **wrong**:

- **Mis-set weights.** `logit(pYes) − logit(pNo)` are derived from independently estimated
  *marginal pairwise* conditionals, then summed as if they were *jointly-fit logistic
  weights*. With correlated parents (and these parents are highly correlated — see `ρ` table
  in the sidebar) this double-counts shared variance. The combiner is uncalibrated: no target,
  no fit.
- **Baseline ≡ prior by construction.** When every parent sits at its prior, all perturbation
  terms vanish and the node returns its hand-set prior exactly. Two consequences:
  (a) the model can never *surface* an inconsistency between a node's hand-set prior and what
  its parents' priors imply through the edges; (b) only the *difference* `logit(pYes) −
  logit(pNo)` (the slope) ever affects output — the conditional **levels are ignored**. So
  the careful absolute calibration of `pY`/`pN` in `edge_weights.json` is half-wasted in any
  log-odds model (this critique applies to `graph.html` and the API, **not** to `pm_analysis`,
  which uses linear LToTP where levels do matter).
- **Mislabel.** README/DESIGN call this "law of total probability." LToTP is
  `P(A) = pY·P(B) + pN·(1−P(B))` — linear, exact, one parent. The shipped engine is a
  first-order logistic perturbation. Calling it LToTP oversells it.

**Fix ladder (pick the rung that matches ambition):**

| Rung | What | Effort | When |
|---|---|---|---|
| Honest | Relabel as a *logistic sensitivity / what-if* tool, not a forecast. Stop claiming LToTP. | trivial | now |
| Better | **Fit** intercepts + weights so baseline reproduces a target (priors or PM), so levels matter and the combiner is calibrated. | medium | once a scoring loop exists |
| Gold | **Exact discrete Bayes net.** With fan-in ≤ 3–4, a node's CPT is 2³–2⁴ = 8–16 rows. Variable elimination over 20 nodes is instant. This is what `DESIGN.md` already specifies (full joint, copula for ρ). | larger | the hard-cutover path |

### 1.2 Two purposes are tangled — separate them

The "free prior + edges = over-determined" objection only bites under one reading:

- **As a calibrated forecast** (DESIGN's hard-cutover ambition): a node should have *no*
  independent prior; its marginal must be implied by its CPT and parents. Hand-set priors that
  disagree with edge-implied marginals are a genuine bug.
- **As an interactive what-if tool** (what `graph.html` actually is, and the README hedges it
  as "experimental, not in production"): a free prior + sensitivity edges is a defensible
  design. You're exploring elasticities, not asserting a joint.

Recommendation: **state which artifact is which.** Keep `graph.html` explicitly as the
sensitivity toy. Build the forecast on the `pm_analysis` lineage (it's the one wired to data),
and hold *it* to the consistency + scoring bar.

### 1.3 The empirical blend uses level-correlation (spurious-regression risk)

`compute_edge_probs.py` regresses `P_B = α + β·P_A` on **price levels** and weights by
`R²·3.0`. Two markets drifting over eight months on a common driver (overall "Bibi strength,"
the war) will correlate in levels without B depending on A — the classic spurious-regression
trap, and the `r2≈0.02–0.09, n≈100–170` fits in `edge_weights.json` are exactly the weak,
trend-contaminated signature you'd expect. `pm_analysis`'s `edge_data` is better (binned
conditional means, `src:'bin'`, with CIs) but still levels-based.

Fixes: regress **first differences** (daily Δ) or detrend; or compute conditional means on
*returns*. Better still, treat price-correlation as evidence about **edge existence/sign**, not
as a direct estimate of the structural conditional — correlation ≠ causal conditional when a
common parent drives both.

### 1.4 Validation is the missing spine

Per `DESIGN.md §Validation`, nothing is scored against anything. This is the highest-leverage
gap: without a loop, this rethink just trades one set of untested numbers for another. Concrete
first loop (uses only assets already in the repo):

1. Freeze the graph as-of date *T* using `node_history` prices as that day's node values.
2. Propagate.
3. Score predicted child probabilities against **realized PM price at *T+k*** (Brier / log
   loss), swept over many *T*.
4. Report DAG-propagated vs raw-PM-persistence baseline. If the DAG can't beat "PM price
   tomorrow ≈ PM price today," it isn't adding value yet.

This must come **before** Monte-Carlo CI propagation (§4.2): sampling roots through an
incoherent combiner yields a distribution of meaningless numbers. Fix the combiner, prove it
on the backtest, *then* propagate uncertainty.

---

## 2. Code quality

- **Three copies of the graph, three propagation rules.** `graph.html` (logit-additive),
  `bayesoracle.py` (identical copy), `pm_analysis/index.html` (linear primary LToTP + a
  hard-coded `0.5`-weighted secondary stack). The calibration output (`edge_weights.json`)
  reaches only the third. Editing the model means hand-syncing ≥2 files and hoping. → **Single
  JSON graph spec**; API serves it at `GET /bayes/graph`; both HTML viewers `fetch()` it. The
  README frames "all data embedded inline, no external files" as a feature — it's a maintenance
  liability.
- **No validation at load.** No DAG-acyclicity check, no `pY,pN ∈ [0,1]`, no "edge endpoints
  exist," no fan-in cap. A cycle would mis-propagate silently. Add a `validate_graph()` that
  runs in a test and at API startup.
- **Redundant indices.** `graph.html` builds `parentIdx` and `condByTgt` as identical
  structures in one loop (`:264`). Collapse.
- **Mutual exclusivity encoded as a directed edge hack.** `PM5 → OPP_PM` with `pY=0.02,
  pN=0.56` is a kludge to fake "these can't both be true." That's a structural constraint, not
  a causal edge (see §5.3).
- **Magic numbers, uncommented:** the `0.5` secondary weight, `w_corr = R²·3.0`, the various
  `clip` bounds, `0.01/0.99` floors. Name them; they're modeling choices.
- **No tests** on `bayesoracle.py`. The propagation, topo order, and lock/observation logic are
  exactly the kind of pure functions that should have unit tests — especially before any
  re-parameterization.
- **Stale-by-construction served model.** Because the API copies `graph.html`'s May numbers and
  ignores `node_history` (fresh to 2026-06-03), `GET /bayes/nodes` serves month-old hand
  guesses while current PM data sits unused in the same repo.

---

## 3. New data (mostly already in the repo, unused)

- **`node_history/` time series (24 nodes × 100–286 daily points).** Currently used only for
  the level-correlation. Far higher-value uses: (a) **backtest** (§1.4); (b) **empirical
  volatility → real CIs** (the `ci:[lo,hi]` in `graph.html` are static decorations that
  propagate nowhere); (c) **lead-lag / Granger-style screening** to *suggest* edge direction
  and catch reversed arrows; (d) detect already-**resolved** markets (price pinned at ~0/1) and
  auto-pin or drop those nodes.
- **PM liquidity / order-book depth** → trust weight per node. A thin market's price is weak
  evidence; weight it down in any fusion.
- **TM base forecast** (the actual product). `graph.html` is divorced from TruthMachine
  entirely. DESIGN's p1/p2 fusion (`α·p1 + (1−α)·p2`) is the bridge; the `data/duel_oracle/`
  harness already exists to calibrate α and score it.
- **Realized outcomes as priors.** It is 2026-06; events have resolved since the model's
  "May 2026" freeze. *Method:* pin/retire resolved nodes and push their outcome through.
  *Caveat:* the specific resolutions written into `graph.html`'s context block ("Iran war done
  Feb 28," "Khamenei killed," "Gaza ceasefire Oct 2025") are **strings in the file, not facts
  verified here** — treat them as the file's own (now stale) context and re-confirm before
  baking any into priors. Anchor staleness to the verifiable price dates, not the prose.

---

## 4. New ideas

### 4.1 Categorical nodes with simplex constraints
"Who is PM" is one mutually-exclusive categorical, not five independent binaries. Model it as a
softmax/Dirichlet over {Bibi, Bennett, Eizenkot, Lieberman, Lapid, other} that **sums to 1 by
construction**. Same for the election→government branch. Removes the 0.925-sum artifact and the
`PM5→OPP_PM` edge hack in one move.

### 4.2 Monte-Carlo uncertainty propagation (after the combiner is fixed)
Sample each root from a Beta fit to its CI, propagate *N* times, report full posterior per node
instead of a point + cosmetic CI. Gives honest `chain_depth`-aware uncertainty (DESIGN wants
this). Gated on §1.1 + §1.4.

### 4.3 Prior/edge reconciliation as a diagnostic
Given root priors + the CPTs, derive descendant marginals (belief propagation to fixed point)
and **flag** every node where the hand-set prior disagrees with the edge-implied marginal. This
turns the §1.2 "inconsistency" from a hidden bug into a visible, actionable lint over the graph.

### 4.4 Edge-discovery assist
Use `node_history` lead-lag + an LLM proposer (DESIGN already imagines this) to *suggest*
candidate edges and flag likely-reversed ones, human-in-the-loop. Pairs with §3 Granger
screening.

### 4.5 Make the visualization show the decomposition, not just a number
DESIGN's per-leaf drilldown (p1, p2, α, each parent's contribution, CI, stance-flip,
chain_depth) is the genuinely useful view for a forecaster. `pm_analysis` is closer to this than
`graph.html`; converge on it.

---

## 5. Rethinking the nodes

### 5.1 The hinge is wrong: bloc, not largest party
`A = "Likud wins most seats"` is treated as the pivot from which mandate→coalition→PM flows.
But in Israel the government is decided by **bloc arithmetic to 61**, not plurality — the
sidebar even cites "blocs 50 vs 60" yet there's no bloc node. Largest-party-but-no-majority is
exactly the Bennett-2021 case. **Introduce `RIGHT_BLOC_61` as the hinge**; keep "Likud largest"
only as a *correlated companion* node if desired. Every downstream PM/coalition edge should hang
off bloc, not plurality.

### 5.2 Collapse the near-deterministic / low-information chain
`MANDATE → COAL_61 → PM5` is a chain of ~0.9 conditionals — it adds almost no independent
information and inflates apparent chain depth. `SICK → DEAD → A` has `DEAD ≈ 0.04` with weak
edges — negligible value of information. Both should collapse. Notably, `pm_analysis` **already
has the better-factored node**: `BIBI_OUT` ("Bibi not PM") cleanly absorbs the
sick/dead/convicted/loses-election machinery. The two graphs each hold the node the other is
missing — unify them.

### 5.3 Make mutually-exclusive government outcomes a single categorical
`PM5` and `OPP_PM` are forced to near-mutual-exclusion via a directed edge with hand-tuned
`pY/pN`. Replace with one categorical "next government" node (Bibi-led / alternative / another
Likud PM / unity / repeat election) under a simplex constraint (§4.1). The `−0.95` Bibi-PM ↔
opp-governs correlation in the sidebar is *describing* this constraint — encode it structurally.

### 5.4 Add the live 2026 wedge issues as first-class nodes
The model routes everything through the election, but the near-term dynamics that actually move
Israeli coalitions are under-represented:
- **Haredi draft law as a coalition-*collapse trigger***, not just a downstream outcome — it is
  the live wedge that can force early elections (cause, not effect).
- **Budget passage / failure** (a statutory auto-dissolution trigger).
- **Hostage-deal / Gaza "day-after" governance** status.
- **Trump pressure as a coalition *lever*** (currently only an exogenous root nudging `BIBI_LEAD`
  and outcomes).

### 5.5 Decouple foreign-policy outcomes from PM5-only
`PARDON / JUDICIAL / SAUDI / IRAN2` all hang off `PM5` alone. `SAUDI` realistically depends more
on the war outcome + US posture and is partly PM-agnostic; `IRAN2` depends on `IRAN_REB` + US.
The edge set is PM5-centric; give these outcomes their real (often exogenous) parents.

### 5.6 Time-stamp every node and decay toward deadlines
DESIGN's deadline effect (`P_eff = P · σ(days_remaining/halflife)`) isn't implemented. With the
election dated Oct 2026 and many markets carrying "before 2027" horizons, nodes need an
`outcome_date` and time-aware decay so "running out of runway" shows up.

---

## Suggested sequencing (proposed — not started)

1. **Unify the graph spec** into one JSON; API serves it, both HTMLs fetch it; add
   `validate_graph()` + unit tests. (Unblocks everything; low risk.)
2. **Re-node:** bloc-61 hinge, collapse the deterministic chains, merge in `BIBI_OUT`,
   categorical government node. (Pure data/spec edit on the unified graph.)
3. **Backtest harness** over `node_history` (Brier vs PM-persistence). (The scoring spine.)
4. **Re-parameterize** the combiner (fit weights, or move to exact BN) — judged by #3.
5. Only then: MC uncertainty propagation, p1/p2 TM fusion, reactive propagation.

I'd want a go-ahead before building any of these — happy to start with #1 (mechanical,
reversible) or prototype #3 (the backtest) to put numbers behind the methodology claims.

---

## Implementation status (2026-06, branch `feat/bayesoracle-engine-rethink`)

Built and tested:

- **`core.py` — one engine** replacing the three divergent propagation rules. Fitted-intercept
  logistic CPT with **exact enumeration over parent states**, which marginalises over parent
  uncertainty (E[sigmoid]) instead of the old plug-in-the-mean perturbation — they differ by
  Jensen when a parent is strictly interior. Inhibition is representable; output stays in (0,1).
  Includes `validate_graph` (cycles, ranges, endpoints, fan-in cap) and inline exclusive-group
  simplex normalisation. 16 unit tests (`tests/test_core.py`), all passing.
  **Honest caveat:** the output still depends only on `w_i = logit(pYes)−logit(pNo)` and the
  priors — absolute conditional *levels* do **not** bite yet (that needs fitting `w_i` to
  history, §1.1, not done). Baseline = prior was already true of the old model; it is not a new
  property.
- **`graph_political.json` — re-noded** per §5: `RIGHT_BLOC_61` is the hinge (not "Likud
  largest", which is kept as a correlated companion); `SICK/DEAD/CONVICTED` collapsed into
  `BIBI_OUT`; government is a mutually-exclusive group `{BIBI_PM, OPP_GOVT, OTHER_LIKUD_PM}`
  (the `PM5→OPP_PM` edge hack is gone); `HAREDI_CRISIS` added as a collapse trigger; outcomes
  decoupled from PM-only parents.
- **`graph_pm.json` + `build_pm_graph.py`** — the PM-backed graph auto-generated from
  `edge_weights.json` (edges) and `node_history/` (priors = latest price). PM-candidate
  exclusive group enforced.
- **`backtest.py` — the scoring spine.** Fit intercepts at T0, drive roots to actual prices at
  later dates, predict children, score Brier vs persistence. **First result: DAG 0.0065 vs
  persistence 0.0067 — only ~+3% skill over "markets didn't move"** (8 of 16 children improve).
  This is the honest baseline the rest of the work has to beat, and it quantifies the §1
  critique that the current edges add little. **Scope caveats:** this scores `graph_pm.json`
  only — the re-noded `graph_political.json` (bloc-61, BIBI_OUT merge, categorical government)
  has hand-set priors and is **not** empirically validated. And the PM graph's observable roots
  are foreign-policy markets weakly linked to the domestic children (BIBI_PM sits several hops
  down), so the modest skill partly reflects that root/child disconnect. n≈10 per child, no CI —
  indicative, not a verdict.
- **API wired to the engine.** `api/.../bayesoracle.py` is now a thin adapter loading
  `graph_political.json` via `core`; its hand-synced copy of the graph and its separate
  propagation are deleted. `/bayes/nodes` docstring corrected (was mislabeled "law of total
  probability"). Full API suite (75 tests) still green.

### Did fitting the edge weights to history help?  (no — `fit_edges.py`)

We tested whether learning the weights `w` from the `node_history` price series beats the
LLM/correlation-derived weights **out-of-sample** (teacher-forced per-node CPT fit; train ≤
2026-04-06 / 91d, test 59d, 18 children). Result is a **clean null**:

| Weights (OOS level Brier) | value | vs LLM | vs persistence |
|---|---|---|---|
| persistence (freeze at split) | 0.00685 | — | — |
| **LLM weights** (intercept fit) | **0.00611** | — | **+10.8%** |
| global-α scale (α≈0.68) | 0.00649 | −6% | worse |
| per-node fit (ridge 0→10) | 0.00611 | **+0.0%** (−0.1% unregularised) | +10.8% |

Two honest takeaways:
- **Freeing the weights buys nothing OOS** — the ridge fit reverts to the LLM weights, and the
  unregularised fit slightly *overfits* (−0.1%). The useful signal (the +10.8% over persistence)
  comes from the conditional **structure + intercept**, which the existing LLM weights already
  deliver; tuning the weights doesn't add to it.
- **First-difference test fails for every model**: predicting ΔP_child, no setting beats a
  zero-change baseline (LLM 0.000723 vs zero 0.000675). So even the level-Brier "skill" is mostly
  co-trending / base-rate, not learned change dynamics — consistent with §1.3. Daily PM series
  barely move; the exploitable conditional signal is in the noise on this data/horizon.

**Recommendation: keep the LLM/correlation weights.** Don't ship a fitted graph — there's no
out-of-sample case for it. Revisit only with more independent observations (more markets, longer
horizon, or event-resolution data rather than autocorrelated daily prices).

Not yet done (proposed follow-ups):

- Monte-Carlo CI propagation, p1/p2 TM fusion, time-decay toward `outcome_date`.
- The above fitting null is data-limited, not method-limited; a resolved-events corpus could
  change it.
