# Bediavad (Retro Analysis) — Deep Review

> Reviewer pass over the בדיעבד retroactive-audit engine: data analysis, engine
> bugs, and an Oracle-driven plan. Date: 2026-06-07.
> Scope reviewed: `pipeline/src/tm/backtest.py`, `pipeline/BACKTEST.md`,
> `data/events/`, `data/atlas/`, `data/sources/`, `data/leaderboard.json`,
> `docs/ORACLE_API.md`, and cross-checked against the **Duel** project
> (`pipeline/src/tm/duel_report.py`, `data/duel_oracle/`, `duel.html`).

---

## Scope correction (read first)

Bediavad and the **Duel** are two different projects and this review keeps them
apart:

| | **Bediavad** (Retro Analysis) | **Duel** |
|---|---|---|
| Question | "Who was right, per source, in hindsight?" | "Is TruthMachine better than Polymarket?" |
| Engine | `backtest.py` (LightGBM/weighted-avg over Atlas) | `duel_report.py` → `duel.html` |
| Comparison target | none — it scores sources | **Polymarket** (CLOB price history) |
| Deliverable | source credibility **leaderboard** (Brier/ELO) | head-to-head Brier, horizon sweep |
| Status | engine works; only 4 events run; bugs below | **mature**: 13 events, TM 8/12 at T=7 |

**The Polymarket head-to-head belongs to the Duel, not Bediavad.** The Duel
already works: 12-event common sweep, TM beats PM at every horizon
(T=7: Brier 0.121 vs 0.355, 8/12 wins; T=30: 0.108 vs 0.407, 9/12), with the two
lookahead-leak classes patched in PR #76. That is the headline external-validation
metric — and it has run, contrary to what my first-pass review implied.

The confusion came from `backtest.py` carrying a **second, redundant, broken**
Polymarket comparison (`fetch_polymarket_price`) that duplicates the Duel and
returns `null` for every event. `BACKTEST.md` framed that as Bediavad's "core
commercial claim." Both have now been corrected: see the scope banner added to
`pipeline/BACKTEST.md`. **Recommendation: delete `fetch_polymarket_price` and the
Polymarket columns from `backtest.py`** — Bediavad should score sources; the Duel
owns the market comparison.

---

## TL;DR

- **Bediavad's actual job** is the retroactive **source leaderboard** (Brier/ELO).
  Judge it on that, not on beating Polymarket (that's the Duel, which already
  wins).
- **#1 issue, affects both projects: the outcome data has no variance.** 69 of 70
  events resolved `true` (and all 13 Duel events are YES too). With no negatives
  you cannot measure *discrimination* (no AUC, no honest calibration), and every
  Brier looks great because "always YES" scores ≈0.014. The Duel's win is real but
  is a *calibration* win on hard YES events (PM was badly wrong on A19/C07/C09) —
  not yet a discrimination result.
- **The leaderboard is not yet trustworthy** because it's computed off the same
  all-YES data on tiny per-source samples (5–9 predictions).
- **The flagship LightGBM path has never run** (needs ≥5 events; only 4 were used)
  — every recorded number came from the weighted-average fallback.
- Three concrete Bediavad bugs make the source-scoring mechanism partly inert
  (dead `source_brier`, `domain` always `"general"`, model mislabel).

---

## 1. Data analysis (the corpus)

| Metric | Value |
|---|---|
| Events defined | 70 (A19, B13, C9, D6, E10, F5, G8) |
| **Outcomes = `true`** | **69 / 70** |
| Atlas entries (`entry_*.json`) | 899, across 20 sources |
| (event × source) cell coverage | 301 / 1400 = **21.5%** (78.5% empty) |
| Entries inside the 3–30d window | 689 (76.6%) |
| Events with ≥1 in-window entry (**usable**) | **67 / 70** |
| Events ever actually backtested | **4** (A01, A02, A04, A05) |
| Duel events scored (separate project) | 13 (12-event common sweep), all YES |

### 1.1 Finding #1 — near-total class imbalance (top issue, both projects)

69/70 positives means the data is essentially all-YES. Consequences:

- **Brier is uninformative** — "always YES" beats a real model.
- **No discrimination can be shown** — needs both classes; one negative ⇒ no AUC,
  no precision/recall, no honest calibration curve.
- **Applies to the Duel too.** All 13 Duel events resolved YES. TM's win over PM
  is a genuine *calibration* edge on events PM mispriced — but neither side has
  been tested on events that resolved NO. This is the ceiling on both claims.
- **The leaderboard inherits the flaw.** `data/leaderboard.json` Brier/ELO are
  computed off all-YES data with 5–9 predictions per source — currently noise.

**No amount of article-filling fixes validity until the event set contains a
realistic share of NO outcomes.**

### 1.2 The flagship LightGBM path has never run

`run_backtest` takes the LightGBM leave-one-out branch only when
`len(all_event_data) >= 5` (`backtest.py:479`). Only 4 events were ever run, so
**every recorded number came from the `weighted_average` fallback** — though
`summary.json` labels the model `"lightgbm_loo"`. With 67 usable events it *can*
run now, but per Finding #1 the output stays trivial until negatives exist.

---

## 2. Bediavad engine bugs (these hurt its real job: source scoring)

Not blocking, but they make the "who was right" mechanism partly inert. List, not
yet fixed.

1. **`source_brier` feature is dead.** `entry_to_features` reads source accuracy
   from `data/sources/{id}.json`'s `brier_scores` key — but those files have **no
   such key** (flat metadata only), so `source_brier` is **always 0.25**. The
   actual scores live in `data/leaderboard.json` and are never wired in. The
   central "we know who was right" signal isn't feeding the model.
   `backtest.py:77-86, 213, 225`.
2. **`domain` always `"general"`.** Events carry a `category` *list*; the code
   reads `event.get("domain", "general")` (`backtest.py:433`). Per-domain source
   scoring can never trigger even after bug #1 is fixed.
3. **Model label mismatch.** `summary.json` always records `"lightgbm_loo"`
   regardless of the branch taken (`backtest.py:523`).
4. **Legacy Polymarket code (scope bug).** `fetch_polymarket_price` and the
   Polymarket report columns belong to the Duel; remove them from `backtest.py`
   (see scope correction above).
5. **`BACKTEST.md` sample report.** The "Brier 0.0854, 4/6 beat Poly" block is an
   illustrative mock no real run reproduces; now redundant once Polymarket leaves
   this engine. Worth deleting with the legacy code.

---

## 3. Using the Oracle to fill the gap (ordered by leverage)

### 3.1 (Required for validity, both projects) Source NO-outcome events

Until the event set has a realistic share of negatives, nothing else matters —
for Bediavad's leaderboard *or* the Duel. Target ~25–35 events that **resolved
NO**: predicted escalations that fizzled, ceasefires/deals announced-then-
collapsed, downgrades/operations that never came. The Oracle `/search` endpoint
(GDELT-backed, native `date_from`/`date_to`) surfaces the *pre-event* coverage
that confidently expected the thing that didn't happen — the high-signal rows
both projects need to demonstrate discrimination. Aim for ~40/60 NO/YES.

> Note: the Duel cannot grow the NO set freely — its memo records the Polymarket
> CLOB ceiling is exhausted at 12 events (pre-2023 markets aren't in CLOB).
> Bediavad has no such ceiling: it only needs articles, not a matching market, so
> **Bediavad can balance its event set via Oracle search even where the Duel
> can't.** This is an argument for scoring sources on a broader, balanced event
> set than the Duel's 12.

### 3.2 (High) Backfill the 78.5% empty Atlas cells

301/1400 cells populated; the median usable event has only ~7 in-window entries.
Loop `(event × source × keyword)` through Oracle `/search` with
`date_from = outcome_date−30d`, `date_to = outcome_date−3d`,
`enrich_snippets: true`, pacing ≥12s/request for GDELT's rate limit (loop already
sketched in `BACKTEST.md`). Directly improves source-score reliability.

### 3.3 (Duel-owned, mostly done) Polymarket price series

This is a **Duel** task, not Bediavad's, and is largely complete: `data/
duel_oracle/` holds the per-event/per-horizon caches and the Duel already runs.
Remaining work there is bounded by CLOB availability, not by Oracle search.
Listed only to close the loop from the first-pass review, which wrongly put this
under Bediavad.

---

## 4. New ideas / roadmap

- **Report discrimination, not just Brier.** Once negatives exist, lead with AUC
  and a reliability curve in both Bediavad and the Duel. Brier alone hid the
  imbalance.
- **Always-YES base-rate baseline column.** Print it next to every Brier so the
  imbalance can't silently flatter results.
- **Wire `leaderboard.json` → `source_brier`** and add per-`category` source
  scores (fixes bugs #1/#2), so the credibility mechanism actually influences
  predictions.
- **Lookahead-leak guard in Bediavad too.** The Duel already patched two leak
  classes (PR #76: synthetic-date fallback; evergreen/encyclopedic sources). Add
  the same `article_date < outcome_date − 3d` assertion + non-news-domain block to
  the Atlas ingestion path so Bediavad's leaderboard can't be contaminated the
  same way.
- **Isotonic/Platt calibration** post-balance; **bootstrap CIs** over events for
  small-N honesty.

---

## 5. Recommended order of operations

1. Add NO-outcome events via Oracle search (3.1) → restore class balance.
   *Validity gate for both projects.*
2. Fix Bediavad bugs: wire `source_brier`, fix `domain`, remove legacy Polymarket
   code (§2).
3. Run the LightGBM LOO path for the first time on the balanced event set; report
   AUC + calibration + base-rate baseline.
4. Backfill empty Atlas cells (3.2) to thicken source signal.
5. Re-run the **Duel** on any new NO events that have a CLOB-available market;
   refresh `duel.html`.

Until step 1 lands, frame both the leaderboard and the Duel result as
"calibration on an all-YES set — discrimination not yet tested," which is the
honest current state.
