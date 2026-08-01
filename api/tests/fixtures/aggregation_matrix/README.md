# The aggregation trace matrix (R8, retro#370)

57 pinned cases covering the estimator's mechanics end to end, replayed through
the real `run_forecast` by [`api/tests/test_aggregation_matrix.py`](../../test_aggregation_matrix.py).

Why it exists: the lane-soundness audit's rule **R8** — no aggregator or
architecture change ships on "backtest agreement" alone. Agreement with the live
system measures *consistency*, not correctness, and at today's resolution volume
(N < 20) a Brier holdout is noise. Pinned fixtures are the only honest gate that
can be run today.

| File | Group | Cases | What it holds |
|---|---|---|---|
| `group_a_intra_article.json` | A | 20 | The article boundary: claims → one (stance, certainty, evidence_weight, settled, fact_signal) tuple |
| `group_b_truth_and_fabrication.json` | B | 17 | Adversarial evidence: echoes, fabricated anchors, interested parties, hoax settlements, blocked sources |
| `group_c_pool_dynamics.json` | C | 14 | The pool boundary: relevance floor, weight multipliers, logit clamp, CI and widening, settlement revalidation |
| `group_d_class_boundaries.json` | D | 6 | The evidence-class table: constants, ratios, the unclassified fallback, mixed-class collapse |

## Using it in a Phase-1 PR

Every Phase-1 PR (F16, R3, F9, F20, F1, …) must:

1. Run the suite and get **zero unexplained movement**.
2. Name in its description the case IDs it *intends* to move, and why.
3. Regenerate the intended movers, and put the resulting diff — the before/after
   numbers, not a summary — in the PR.

```bash
cd api
uv run pytest tests/test_aggregation_matrix.py -q          # gate
AGG_MATRIX_UPDATE=1 uv run pytest tests/test_aggregation_matrix.py -q   # regenerate
git diff api/tests/fixtures/aggregation_matrix/            # the movement report
```

A failure names the case, its title, the exact field-level deltas, and whether
the case is tagged `known_bad` — i.e. whether movement is plausibly the intended
fix or plainly a regression.

## Anatomy of a case

```jsonc
{
  "id": "A4",
  "name": "reported fact and the author's own conclusion collapse into one number",
  "traces": ["F1", "R5"],          // audit findings / mechanics this case is about
  "notes": "…",                     // why the case exists; read when it moves
  "articles": [                     // one entry per article in the pool
    {
      "source": "alpha",            // becomes alpha.example.test; also the credibility key
      "published_date": "2026-08-01",
      "relevance": 1.0,             // the (stubbed) gatekeeper's graded relevance
      "credibility": 1.0,           // the (stubbed) leaderboard weight
      "claims": [                   // keys pass straight to PredictionExtraction:
        {"stance": 0.2, "certainty": 0.8, "evidence_class": "reported_fact"}
      ]                             // settled, event_date, quantitative_estimate,
    }                               // specificity, fact_signal, is_occurrence, verified…
  ],
  "request": {"claim_deadline": "2026-07-01"},   // optional ForecastRequest extras
  "invariants": ["abs(sources[0]['stance'] - 0.575) < 1e-9"],
  "known_bad": {"finding": "F1", "expected_behavior": "…"},
  "expect": { /* generated — never hand-edit */ }
}
```

**`expect` is generated; `invariants` are not.** The snapshot records what the
estimator *does*; the invariant records what the case is *about*
("this dissenter must not flip the pool"). A regenerated snapshot still has to
satisfy the invariants, so a PR cannot quietly regenerate its way out of a
behavioural claim. Invariants are Python expressions evaluated against
`mean`, `std`, `ci_low`, `ci_high`, `settled`, `n`, `insufficient_data`,
`reason`, `sources` (plus `abs`, `len`, `any`, `all`, `sum`, `prob`).

**`known_bad` marks a case that pins behaviour the audit already judged wrong.**
It changes nothing at runtime — the snapshot is still enforced — but it tells the
next reader that movement here is probably the fix landing, and it tells the
Phase-1 author which cases their PR is supposed to touch. 27 of the 57 carry one:
F1 ×6, R3 ×5, F12 ×4, F2 ×3, F4 ×3, F16 ×3, F9, F20, F23.

## What is stubbed, and what is not

Stubbed: the gatekeeper call, the extractor call, `get_credibility_weight`, and
the clock (frozen at 2026-08-01 in both `forecaster` and `aggregation`).

Not stubbed — i.e. exercised as production code: `enforce_relative_date_resolution`,
`enforce_deadline_arithmetic`, `enforce_settlement_event_date`, settlement grading
and the settled-replace, `claim_weighted_stance`, `resolve_stance_certainty`,
`evidence_class_weight`, `derive_settlement_event_date`, `recency_weight`,
syndication dedup, the relevance and decisiveness gates, `pool_sources`,
`widen_ci_for_thin_evidence`, `aggregate_pool` with settlement revalidation, and
the response assembly (including its 3-decimal rounding of per-source fields).

A fixture that pinned a test-local copy of this composition would certify the
copy. That is the one thing R8 cannot afford, so the harness drives the real
`run_forecast` and stubs only what talks to a model, a database, or a clock.

Config is deliberately **not** overridden: cases run against
`forecast_api.config.settings` as prod has it. A config change (a class-weight
refit, a floor move) is therefore a fixture movement that has to be declared,
which is the intended behaviour.

## Second consumer: F22's drift canary

The same case bodies are the extraction-drift canary described in the audit's
F22: run the *live* extractor against these articles on a schedule and alert on
snapshot deltas. That job does not exist yet; when it lands it should read these
files rather than fork them.
