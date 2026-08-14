# Outlier scan — Stage A (retro#526)

Offline measurement of stored Oracle estimates against **the evidence each one
was actually computed from**. It answers "does this published number follow from
its own pool, under today's rules?" — not "was it right"; nothing here reads an
outcome.

Stage A **measures and chooses nothing.** It emits raw continuous distributions;
a human picks the cuts in retro#526 afterwards, and only then is Stage B (the
recompute arms) designed. A threshold constant appearing in `outlier_scan.py`
forfeits that discipline, which is the point of splitting the work in two:
before this scan ran, nobody had seen what these numbers look like, so nobody
was in a position to choose a cut. **"142 forecasts scored, zero outliers" is a
complete and acceptable result.**

```
api/src/forecast_api/outlier_scan.py     # pure module, no network, unit-tested
api/scripts/scan_outlier_estimates.py    # driver: `sql` | `score`
api/tests/test_outlier_scan.py
api/tests/fixtures/outlier_scan_prod_snapshot.json
```

`pyproject.toml` sets `testpaths = ["tests"]`, so nothing under `scripts/` is
collected — this is a script, not a CI test, structurally and not by convention.

## Running it

```bash
cd api

# 1. print the ready-to-run prod dump command
uv run python scripts/scan_outlier_estimates.py sql

# 2. run that on the daatan PROD box (i-04ea44d4243d35624) — the DB is there,
#    NOT on the Oracle box — and bring the file back (see "The dump" below)

# 3. score it, gated on the deployed commit
uv run python scripts/scan_outlier_estimates.py score stage_a.jsonl \
    --out scan_a.json --deployed-commit "$(curl -s https://oracle.daatan.com/version | jq -r .git_sha)"
```

Useful flags: `--limit N` (smoke run), `--no-loo` (skip leave-one-out — it costs
one aggregate per pool row), `--top N` (rows printed per signal).

**Exit codes.** `0` scored something · `1` nothing was scorable · `2` the
deployed-commit gate failed. A clean run that measured nothing must never read
as a pass (retro#395), so an empty scan is an error, not an empty table.

## Why it runs in-process

`run_pool_aggregate` is importable and fully offline — no search, no LLM — at
roughly 0.03 s per call. Over HTTP the same work would sit behind
`/pool/aggregate`'s 60/min rate limit and take hours, which is why earlier
harnesses (`backtest_fact_signal_gate.py`) were shaped around throttling. Two
env vars make the import work headlessly, and the driver sets both before
importing `forecast_api`, because settings are read at import time:

- `ORACLE_API_KEY=dummy` — a required field on `ApiSettings`; nothing calls out.
- `SETTLEMENT_VERIFIER_ENABLED=false` — skips the one Bedrock call in an
  otherwise offline recompute (the settlement match gate).

**The verifier default is not free, and `--settlement-verifier` exists because
of it.** The gate has been ENFORCING in prod since 2026-08-03; with it off,
`_apply_settlement_match_gate` returns the aggregate untouched, so a pin prod
vetoed at publication is re-applied by the recompute — which surfaces as an S1
gap and a reproduction disagreement on a forecast where the estimator did the
right thing. The default run is offline and fully deterministic; pass the flag
to score the same corpus the way prod sees it (one Bedrock call per pinned pool,
fail-open) and compare. Both runs print which mode they were in.

In-process is also what makes **leave-one-out attribution** affordable, and LOO
is the only honest way to say which row drove a pool: it runs through
`dedupe_syndicated`, cluster downweighting, `cap_source_mass`, settlement
demotion and the logit pooling. A client-side weight share does not, so it will
happily report "this row dominates" about a row the estimator had already
collapsed as a syndicated duplicate.

This is only legitimate while **local code equals deployed code** — otherwise
"today's rules" describes a system nobody is running. `--deployed-commit` makes
that a gate that aborts, not a warning.

## The universe: the frozen roster

Every estimate is scored against `context_snapshots.oracle_snapshot.sources[]`
— the rows the published number actually averaged (`snapshotSources =
pool.usableArticles` in daatan's `pooled-estimate.ts`), each carrying every
field `PoolSourceInput` needs. So pool growth since publication cannot
contaminate the measurement at all. This is what makes "a recompute is not a
replay" a solvable problem rather than a caveat.

The dump takes the **latest non-clock snapshot per prediction** that carries a
non-empty roster. Clock rows are excluded — they are arithmetic glide with no
evidence behind them — and `insufficient_data` rows too, since an abstention
published no number to be an outlier about.

Field mapping is identical to daatan's `recomputeFromPool` body, and
`tests/fixtures/outlier_scan_prod_snapshot.json` holds both halves for one real
prod forecast so the equivalence is asserted rather than assumed (V7). Two
mapping rules are easy to get wrong and are pinned by tests:

- **`source_id` is never sent.** `recomputeFromPool` omits it, and the
  snapshot's `sourceId` is the pool-row cuid, not the leaderboard outlet id
  `cap_source_mass` groups on. Sending it would put every row in its own bucket
  — inert at the shipped `max_source_share = 1.0`, wrong the moment it moves.
- **`publishedAt` is a full ISO timestamp in prod**, not `YYYY-MM-DD`.
  `recency_weight` truncates to 10 characters, so it passes through untouched; a
  well-meaning reformat here would re-date the whole corpus.

## The two clocks

Both are used, and the difference between them is itself a result.

| clock | what it is for |
|---|---|
| wall clock | the primary artifact — recency recomputed against now, exactly as a live recompute would |
| `frozen_clock(snapshot.createdAt)` | the as-published reproduction check (V6) |

Without the frozen arm, every stored number would appear to disagree with its
recompute purely because its articles have aged since — and a real
field-mapping defect would be invisible inside that decay.

`frozen_clock` patches **two** namespaces, and both are load-bearing:
`forecaster.datetime` stamps `ref_date` for every row's recency, while
`aggregation.datetime` is read independently by `settlement_vote_validity`
(called from `aggregate_pool` with no `today`) and by the deadline-glide check.
Patching one gives a recompute whose recency is as-published but whose
settlement validity is judged today — a silent hybrid that is neither clock. The
unit tests pin each patch to an outcome it changes.

## Signals

All raw continuous values. Per signal the driver prints `n | mean | p10 | p50 |
p90 | max`, split by resolved/unresolved and by the `2026-08-04` clean-corpus
boundary, plus the top rows per signal with claim text.

| id | signal |
|---|---|
| **S1** | pinned-extreme gap: `abs(p_pinned − p_no_pin)`, where `p_no_pin` re-aggregates the same roster with every `settled` flag cleared — literally the pooled mean the pin discarded |
| S1b | pin support: winning-direction summed weight (the `settlement_quality_floor` quantity), valid votes after `settlement_vote_validity`, votes demoted, suppression reason |
| **S2** | band width — stored `ai_ci_high − ai_ci_low`, the snapshot's own, and the recomputed one, all in percentage points, reported against `n_eff` |
| **S3** | centre gap: `abs(published_p − median_p(rows))`, weighted and unweighted |
| **S4** | mass/multiplicity: `n_eff`, `n_eff / articles_used`, `max(w)/sum(w)`, `evidence_mass`, rows with `w < 0.01`, `articles_used == 1` |
| S5 | corpus-hygiene **covariates**, not outlier signals: snapshot age, post-`2026-08-04`, `claims_detail` coverage, `carried_forward` share, `origin` |

S5 makes the system model's "clean corpus starts 2026-08-04" a column to
stratify on rather than a footnote re-derived by hand. A signal that only fires
on the pre-cutover half is a corpus artefact; one that fires on both is about
the estimator.

Two deliberate reporting choices:

- **A signal's `n` counts rows where it is defined**, not rows scanned. S1 only
  exists on pinned pools; averaging it over unpinned ones would report a pin gap
  made mostly of zeros nobody measured.
- **A pool that abstains without its pin reports `s1_no_pin_reason`, not a gap
  of zero.** That is a strictly stronger version of the same finding, and a zero
  would bury it among the well-supported pins.

`predictions.confidence` is **not** a reproduction target — `saveClockSnapshot →
recordEstimate` overwrites it daily for `origin='clock'`. Its distance from the
snapshot is reported separately as a clock-glide measure.

## Health metrics on every run

Reported, not asserted — both are Stage A results in their own right.

- **as-published reproduction agreement** — share of snapshots whose frozen-clock
  recompute reproduces the stored percent. The post-2026-08-04 corpus should be
  near 100% and older rows should not be; the shape of that curve is a finding.
- **V8, max-weight == max-LOO row** — share of pools where the heaviest row is
  also the most influential. `row_weights` re-derives the *pre-pooling* product
  from `run_pool_aggregate`'s own per-source loop, so a low rate on large pools
  is expected (clustering and syndication dedup are exactly what LOO sees and a
  weight share doesn't), but a collapse on small pools means the re-derivation
  has drifted from the estimator and the S4 mass signals describe a pool nobody
  computed.

## The dump

~16 MB of JSONL over the current corpus. **SSM truncates output at ~24 KB and
errors hide mid-output**, so never pull it through `get-command-invocation`:
write it to a file on the box, `gzip`, and either `base64 | split` it or push it
through S3 and delete the object afterwards. Base64 the SQL going *in*, too —
it avoids quoting hell through `send-command`.

The dump emits **one JSON object per line**, not one `json_agg` array, so a
truncated transfer fails to parse at the truncation point instead of silently
yielding a shorter array. The driver accepts either form.

Column case differs per table and bites every time: `context_snapshots` quotes
both `"predictionId"` and `"createdAt"`; `predictions` quotes `"claimText"`,
`"outcomeType"`, `"resolvedAt"` and `"createdAt"` alongside mapped snake_case
`ai_ci_low` / `claim_direction` / `claim_deadline`; `evidence_pool_articles` is
snake_case except `"predictionId"`. `published_date` and
`settlement_event_date` are TEXT — nothing formats a date.

## Scope

Stage A changes no estimator behaviour, adds no threshold, floor, gate or
weight, and **republishes nothing** — no write to `predictions.confidence`,
`ai_ci_low/high` or `context_snapshots`. Correcting a live published number
stays a separate reviewed decision. No `PoolAggregateResponse` field was added:
running in-process gives access to `PoolAggregateResult` without widening the
API on an issue scoped "not changing estimator behaviour".

Note for interpretation: **daatan#1448 is not in this corpus.** Every stored
forecast was computed under the *old* usable predicate, which ignored `status`.
Stage A does not assume the new one; before/after is a Stage B arm.
