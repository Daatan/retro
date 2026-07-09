# Oracle temporal-model plan ("formula") — v2.1

v2 incorporates the 45-agent adversarial review of v1 (48 findings raised, 47 confirmed,
1 refuted, 4 gaps; full record in `temporal-plan/review_result.json`). Every change below is traceable
to a confirmed finding. The three biggest corrections: **the recompute driver is the MVP
and is now Stage 0**, **the clock lives outside the logit pool**, and **the fitting
program is cut to what the data can actually identify**.

v2.1 folds in five confirmed findings from an external (Gemini) review of v2: glide
domain guards past T_eff (§3.3), the movement clamp scoped to fitted components only
(§3.3), resolve-alert moved to the literal deadline (§4 Stage 0.5), commitment
auto-lock restricted to non-LLM-derived pins (§4 Stage 0.6), and the silence covariate's
forward-only availability (§4 Stage B).

## 1. Problem (unchanged from v1)

The Oracle (retro, `api/src/forecast_api/`) estimates forecast probabilities by having
news articles vote on P directly: per-article stance extraction (LLM), then logit-pooling
weighted by credibility × certainty × recency × relevance². Diagnosed production failures:

- **Frozen estimates**: Knicks forecast sat at 82% for two weeks after the title was
  decided; 77 of 102 open forecasts had no update in 7+ days. Updates are 100%
  push-driven (news-indexer match → daatan POST → Oracle). Nothing recomputes on a quiet
  day.
- **No deadline awareness**: "X by Dec 31" holds its stale number regardless of time
  remaining.
- **Sign blindness**: "X will NOT happen by Dec 31" should drift UP quietly; it freezes.
- **Impossibility blindness**: statutory-notice claims (NATO withdrawal) are ~0 long
  before the deadline, but no article will ever say so.
- **Wrong-timeframe contamination** (partially fixed by extractor prompt rule, shipped).

Already shipped (retro #230, daatan #1001/#1002): settlement detection — ≥2 independent
sources reporting the outcome as accomplished fact pins the estimate to ±0.94 stance
(97/3) with `settled: true`; Telegram alert on crossing 80 from below.

Also shipped since (2026-07): settlement hardening (retro #244) — settlement-grade
gates (`|stance| ≥ 0.9`, `certainty ≥ 0.9`), settled claims skip stance/certainty
realignment, and a direction guard on early settlements driven by optional
`claim_direction`/`claim_deadline` request fields (§3.4; fail-open until callers pass
them). On the daatan side, the `recordEstimate` funnel + reader accessor
(daatan #1053/#1055) made the daily glide visible: the probability chart now includes
`kind='clock'` snapshots (hollow dots) and the gauge reads the funnel's cache instead
of the latest evidence snapshot. Background of both: the 2026-07-08 false-settlement
incident, documented with the full variable audit in
[ORACLE_VARIABLES.md](ORACLE_VARIABLES.md).

User-accepted requirements: resolved ⇒ ~0/~100; graceful degradation is fine ("won't
always work"); private/never-newsworthy forecasts are out of scope.

## 2. Review verdict driving this revision

1. **The clock never ticks** (5 blockers). No stage of v1 built a production recompute
   mechanism, yet every temporal feature requires one. The daily requote driver is the
   actual MVP; the hazard math is the escalation.
2. **Clock-in-the-pool is unsound** (blocker + majors). A logit pool is a convex
   combination bounded by its members (`aggregation.py pool_sources`, the exact reason
   the settlement override exists); article weights are floored (`recency_floor=0.02`),
   so no finite-weight clock member reaches ~0/100. It also double-counts news (articles
   vote directly AND through M(t)) and double-applies recency decay (pool 7d half-life ×
   M's τ kernel). And on the zero-article path, `run_forecast` short-circuits to
   `_empty_response(insufficient_data)` before pooling runs at all — the impossibility
   pin would be unreachable exactly when it's needed.
3. **Metadata has no home** (blocker). retro is stateless; the Prisma `Prediction` model
   has none of the fields; p_det's source table lives in news-indexer's Postgres; no
   reclassification channel exists.
4. **The math had three defects**: scheduled spikes mis-specified (δ-mass P_k gives
   survival e^(−P_k), not 1−P_k — a certain scheduled event priced at 63%); M(t) does
   not "relax with time constant τ" — it plateaus at the consensus stance for
   ~τ·ln(Σw/m₀), quietly re-creating the frozen-estimate bug for well-covered events;
   and the spike/modulation composition was ambiguous.
5. **The fitting program was over-claimed**: γ/m₀/τ are not identifiable from Polymarket
   prices (no article stream in that data); the "silence learns λ" evidence is confounded
   three ways (martingale NO-conditioning, live news flow in the exemplar, longshot-bias
   unwind); τ was fitted from effectively one deduplicated episode cluster; fitting to
   prices instead of outcomes inherits documented market miscalibration; the pilot is one
   topic, one volume stratum, ~10–15 independent event families.
6. **The kill criterion was undecidable**: "beat current behavior" is a strawman any
   decay function beats, and per-horizon buckets on daatan's resolved set are single-digit
   n.
7. **Product landmines**: daily requotes through existing writers spam the timeline and
   fire clock-driven "consider resolving" alerts (worst for survival claims, weeks before
   they're resolvable); `settled` is not persisted (only inside oracleSnapshot JSON, no
   UI) and pinned-but-ACTIVE forecasts are a live free-points window (rsChange =
   (0.25 − brier) × 100, no time discount, commitments open); retro's forecast cache
   assumes time-invariant answers; the shadow run had no isolated storage.
8. **Gaps no lens caught**: no backfill for ~100 existing forecasts; user-authored claim
   text is an adversarial (prompt-injection) surface that can force pins; no failure
   isolation for shared fitted parameters; no rollback story for clock-written history.
9. **Steelman won on sequencing**: a zero-parameter constant-hazard glide anchored to the
   last evidence-based estimate reproduces the fixed-λ reference curve exactly
   (0.545/0.408/0.231 vs 0.53/0.40/0.22 at the pilot's quartiles) with no fitting at
   all. The one refuted finding ("cut all archetypes but diffuse-deadline") confirms the
   scheduled/diffuse distinction must survive even in v1 — as a *glide suppressor*, not
   as spike pricing: gliding a "who wins the Sep election" claim toward 0 during normal
   pre-event quiet would be a new failure mode worse than freezing.

## 3. Corrected model

### 3.1 Composition: the temporal engine sits OUTSIDE the pool

The pool keeps doing what it does: articles → stances → pooled **P_evidence** (with CI).
The temporal engine is a post-pool transform with explicit precedence, evaluated even on
the zero-article path (from metadata alone, before the insufficient-data short-circuit):

    settlement pin (97/3, shipped)
      > impossibility pin (~0/100 exact: now ≥ claim_deadline − τ_lead, or statutory
        window closed)
      > deadline glide (below)
      > P_evidence as-is (non-temporal claims, unparseable deadlines — flagged)

Articles enter once (through the pool). The clock enters once (through the transform).
No pseudo-source, no weight schedule, no double-counted recency.

### 3.2 Corrected formula (v2 full model — NOT built in v1)

    S(t, T) = exp(−∫_t^{T_eff} λ_bg(u) · e^{γ·M(u)} du) · Π_{t_k ∈ (t, T_eff]} (1 − P_k)

    P_arrival = 1 − S;  P_survival = S;  T_eff = claim_deadline − τ_lead

- Scheduled occasions contribute **multiplicative Bernoulli survival (1 − P_k)**
  (equivalently hazard mass −ln(1−P_k)), exact at P_k = 1 ⇒ P_arrival = 1. Spikes are
  outside the e^{γM} modulation — unambiguous.
- **M(t) decays directly**: M(t) = M̂ · e^{−(t−t_last)/τ}, where M̂ is the mass-shrunk
  kernel-weighted stance snapshot at the last evidence time (shrunk toward 0 by
  m₀/(m₀+Σw), with Σw capped by a saturating transform so 30 syndicated copies of one
  wire story can't buy months of persistence). This restores the advertised "relaxes to
  base rate with time constant τ" semantics; the v1 ratio form plateaued instead.
- Naming this precisely: the self-excitation lives on the **article stream** (a
  genuinely repeated point process with an exponential memory kernel — Hawkes-style,
  standard MLE/EM applies there); the resolution event itself is a *terminal* event
  whose intensity λ_bg·e^{γM} is a modulated-Cox form, entering the likelihood as a
  survival term. Test a two-timescale/power-law kernel for geopolitics
  (conflict-domain evidence says single-τ exponential underfits slow-relaxing cases).
- Threshold claims (first-passage on an observed state variable) and state-variable
  spike pricing: deferred until the archetype census (§4.1) shows they earn a queue slot.

### 3.3 v1 pricing rule: the anchored constant-hazard glide (zero fitted parameters)

    P(t) = 1 − (1 − P_last)^c,   c = clamp((T_eff − t) / (T_eff − t_last), 0, 1)
                                                                  [arrival direction]

with P_last, t_last from the last evidence-anchored (article- or settlement-driven)
snapshot; survival claims apply it to the complement. **Domain guards** (Gemini review):
the exponent is clamped to [0, 1] — for t ≥ T_eff it holds at the boundary value
instead of going negative (unclamped, P_last=0.8 one interval past T_eff gives
1 − 0.2^(−1) = −400%); if T_eff ≤ t_last (anchor written at or after the horizon) the
glide is skipped entirely and the pin/precedence logic decides. This matters because
the past-T_eff domain is reachable: §3.5's divergence branch suppresses the hard pin
while leaving the glide active.

Properties: reaches ~0/~100 at T_eff by construction; re-anchors on every push so it
never fights fresh evidence; its implied λ = −ln(1−P_last)/(T_eff−t_last) is exactly
the fixed-λ reference from the pilot. Known, accepted cost: ~25–30pt mid-life
overpricing on quiet long NO-tranches vs the full model — that residual is precisely
the evidence that will justify (or kill) the v2 escalation.

**Movement clamp — scoped to fitted components only** (Gemini review): the
deterministic glide is exempt. Its travel is bounded by construction (it can never
move farther than the remaining distance to the boundary) and its only inputs are
metadata already gated by the §3.5 agreement rule; a flat per-day cap would contradict
the boundary requirement outright (P_last=0.8 with 10 days left needs ~8pts/day —
capped at 3, the glide arrives at the deadline stuck near 50 and the pin snaps it,
recreating the jump the clamp was meant to prevent). The |ΔP| ≤ 3pts/day clamp applies
to **fitted** clock components (v2: λ-learning, M(t) modulation), where poisoned
parameters are the actual threat model.

**Glide applies only when**: direction ∈ {arrival, survival}, claim_deadline parseable,
archetype = diffuse-deadline (scheduled claims hold a base-rate band, NO clock decay),
and the forecast has ≥1 historical forecast_match (boolean p_det gate — one SQL EXISTS;
the per-forecast p_det estimator is deferred with the Gamma machinery it belongs to).
Everything else keeps current behavior, flagged.

### 3.4 Per-claim metadata — owned by daatan, v1-scoped

Prisma migration on `Prediction`: `claimDeadline DateTime?` (explicit UTC instant,
end-of-day convention fixed at classification time), `claimDirection` (arrival |
survival | none), `tauLeadDays Int?`, `claimArchetype` (label only in v1: diffuse |
scheduled | threshold | none — stored for the census, only diffuse prices differently),
`classifierVersion`.

- Classifier: one LLM call at forecast creation **plus a one-shot backfill over all
  existing open forecasts** (~100 calls) with human spot-audit of every direction label
  and every claim_deadline that disagrees with resolveByDatetime. Shadow-run is blocked
  on backfill completion.
- retro stays stateless: daatan passes the metadata as **additive optional
  ForecastRequest fields**; ForecastResponse gains additive optional model-state fields
  (as_of, implied λ, T_eff, pin reason). Existing consumers (`matcher.py` reads via
  `.get()`, `oracle.ts` types a subset) are untouched.
- Reclassification channel (v2): extractor emits an optional `schedule_signal`; daatan
  applies it to the metadata on push.

### 3.5 Deadline agreement rule (data-quality + adversarial defense)

The impossibility pin is gated on a single LLM parse of **user-authored** text — both an
error surface and a prompt-injection surface (a crafted claim could force a ~100 pin and
lend platform credibility to a user's forecast). Therefore:

- Hard-pin only when claimDeadline and resolveByDatetime **agree within tolerance, or
  both have passed**. On divergence: apply only the bounded glide — with the horizon set
  to the **later** of the two dates, so a misparsed too-early deadline cannot crash the
  estimate to ~0 without review — and send a one-click "deadline disagreement" review
  alert.
- Classifier-only pins (zero article evidence) are marked **provisional** in the API
  response and excluded from the 80%-alert trigger.
- Classifier prompt hardened against instructions embedded in claim text; adversarial
  claim-text suite in Stage 0 acceptance tests.
- Classifier output persisted on the snapshot so every pin is auditable to its inputs.

## 4. Staged implementation (reordered — review's central correction)

### Stage 0 — the recompute driver + glide (production, flagged; ships first, ~days)

The actual MVP. Deliverables:

1. **Daily requote cron**: `src/app/api/cron/requote/route.ts` + GitHub Actions schedule
   (pattern: external-market-sync.yml). Iterates ACTIVE forecasts with temporal
   metadata; prices P(t) **locally in daatan from the last snapshot + metadata** (pure
   arithmetic, no Oracle call, no search, no LLM — cost ≈ zero). The retro round-trip
   and its 1h forecast cache are simply not on this path (cache staleness across a
   boundary is thereby moot; the push path still hits retro as today).
2. **Prisma migration** (§3.4 fields) + creation-hook classifier + one-shot backfill +
   spot-audit.
3. **Impossibility pin** (arithmetic + §3.5 agreement rule).
4. **Quiet snapshot semantics**: clock requotes write a snapshot with
   `kind='clock'` — excluded from `getContextTimeline` (or collapsed to one visible row
   per week), never overwrites the sources roster, carries provenance
   {clock contribution, parameter/classifier version, flag state}. Persist only on
   material change (≥1pt) — no 36k-rows/year timeline flood.
5. **Cause-aware alerts**: `notifyIfCrossedHighConfidence` fires only for
   article/settlement-driven moves. Clock-driven crossings never alert; instead a
   **single-shot "deadline passed quietly — resolve NO"** alert fires at the **literal
   claim_deadline** — not at T_eff. T_eff is a pricing horizon; τ_lead is an LLM parse,
   and prompting a resolver to act τ_lead days early on the strength of a parse would
   mis-resolve any event still landing in the final window (Gemini review). For
   τ_lead > 0 claims, T_eff instead sends a lower-key provisional note ("impossible per
   lead-time analysis — verify the reasoning; early resolution optional"). This also
   fixes the adversarial review's sharpest product point: with crossing-based alerting,
   the actionable moment (deadline) would otherwise be silent because the estimate is
   already above 80.
6. **Persist `settled` + close the free-points window**: `Prediction.settled` column,
   "outcome reported — awaiting resolution" banner on the forecast page, and lock new
   commitments once boundary-pinned — with the lock trigger split by evidence class
   (Gemini review; an LLM parse must not freeze a market on its own):
   **auto-lock** on settlement pins (≥2 independent sources) and on literally-passed
   calendar deadlines (pure arithmetic, no LLM in the loop); τ_lead-derived early
   impossibility pins lock only after **one-click admin confirmation**. The free-points
   hole is live today post-#230, independent of everything else — a +100 commit after a
   97% pin yields ~+25 RS risk-free.
7. **Failure isolation & rollback** (per gaps): per-day clamp (see §3.3); flag is
   per-archetype-class, not global; parameter/classifier version stamped on every
   influenced snapshot; daily fleet metric (distribution of clock-driven deltas per
   class) with alert threshold; rollback runbook — flag-off procedure, query to identify
   clock-influenced snapshots by `kind`, annotate-don't-delete policy, exclusion filter
   so future fitting never trains on clock-contaminated history. Staging first, then
   prod.
8. **Acceptance tests**: a Knicks-class forecast moves on a quiet day; stale 0.8-stance
   articles + quietly passed deadline ⇒ ≤0.03; certain scheduled event ⇒ no glide;
   adversarial claim-text suite; no Telegram alert from a pure clock crossing.

Stage 0 unfreezes most of the 77 stale forecasts within a day of deploy, exercises the
shipped settlement pin, and generates the residual data every escalation decision needs.

### Stage A — design doc in retro/docs (this document, landed via PR)

### Stage B — fitting (offline; re-scoped to what the data identifies)

- **Fit from outcomes, not prices**: λ₀ (a single global prior with wide uncertainty —
  per-topic λ₀ is deferred until daatan's own resolved history can supply it) via
  censored survival MLE on arrival/censoring times across the full scraped set.
  Prices are diagnostics only, after de-biasing through a per-horizon calibration curve
  (price → realized frequency) fit on the resolved set.
- **Silence-strength**: fit on the **unconditional** sample (YES markets' pre-arrival
  segments included; NO-conditioning is survivorship bias — a martingale conditioned on
  NO decays fast under any model). Quiet segments identified by an article-flow
  covariate, not assumed — but note the covariate is **forward-only**: the news-indexer
  has no archive covering market lifespans predating its own operation (Gemini review).
  For historical trajectories use an external volume proxy (e.g. GDELT daily counts per
  market keyword) or restrict the silence fit to markets overlapping the indexer's
  window; where neither is clean, keep the stated upper-bound treatment. Validate by
  quiet-streak-length buckets vs realized frequency.
- **τ episode mining — count before building**: first run the cheap independent-family
  census on the expanded scrape (the pilot has ~10–15, below threshold — building the
  detector first would be guaranteed-wasted work). Only if the census clears ≥20
  independent families: explicit bump-relax detector; deduplicate episodes across
  tranches of the same underlying event; mid-life episodes only (or fit jointly with the
  deadline decay — post-bump decline near deadline is clock, not relaxation); report
  effective N and CI. Below threshold, or CI wider than 2× ⇒ τ ships as a hand-set
  prior with sensitivity analysis, honestly labeled, and no detector is built.
- **γ, m₀**: not identifiable from PM at all (no article stream in that data). Fit on
  daatan's own resolved forecasts (context_snapshots stance/weight series + outcomes)
  with a temporal train/test split; parameters **frozen before the shadow window**. If n
  is too small — likely — ship conservative priors and let the shadow run log the data
  for the next iteration's fit.
- **Scrape hygiene**: expand beyond tag=politics and down the volume distribution;
  grouped CV holding out whole event families; label from `yes_won` never last price;
  terminal point appended at closedTime; truncation detector (gap > 2d or
  |p_final − outcome| > 0.2 ⇒ re-fetch or exclude); deadline parsed from question text,
  reconciled against Gamma endDate (which is unreliable); throttled, identified client;
  **raw scrape persisted as a frozen private S3 snapshot** (fetch timestamps + script
  version), fits pinned to the snapshot so every constant is reproducible.

### Stage C — v2 engine components, each gated on a named Stage-0 residual

| component | ships only if the glide's logged residuals show |
|---|---|
| Gamma-prior λ-learning (silence) | systematic mid-life overpricing on resolved-NO deadline forecasts matching the pilot's fixed-λ gap |
| M(t)/τ Hawkes relaxation | repeated post-news bump-then-freeze errors |
| scheduled-spike pricing (Π(1−P_k)) | archetype census: scheduled share meaningful AND fallback Brier measurably bad |
| threshold/first-passage | same, for threshold claims |
| extremization (Satopää α on the pool) + recalibration map + correlated-source downweighting | pool residuals underconfident toward 50 on resolved set |
| per-forecast p_det estimator | boolean gate demonstrably mis-gating (pre-gate candidate counts, not ForecastMatch rows, as the flow measure) |

### Stage D — shadow-run and decision (redesigned)

- **Storage isolation**: `shadow_estimates` table in daatan (predictionId, ts, p, ci,
  modelVersion) written by the same requote cron in shadow mode. No Prediction writes,
  no snapshots, no notifications. Additionally log the shadow price at every real push
  event so settlement-reaction latency is measured at event granularity.
- **Baselines**: B0 = frozen (today), **B1 = the shipped Stage-0 glide**. The fitted
  model must beat **B1**, not B0 — beating the frozen model is a strawman any decay
  function wins.
- **Scoring protocol (precommitted)**: both models evaluated on the same fixed daily
  grid; baseline = last-value-carried-forward; time-averaged Brier/log score against
  eventual outcomes (not resolution-only); reported pooled with paired per-forecast
  deltas + bootstrap CI, and split by quiet-vs-active day and onset-vs-continuation
  (hazard models flatter themselves on continuation; onset misses must be visible).
  Horizon buckets demoted to diagnostic views (single-digit n per bucket cannot gate).
- **Calibration KPIs**: reliability diagrams with bootstrap bands, Brier decomposition
  (calibration vs resolution — report sharpness), and **empirical CI coverage** vs
  nominal (the CI is a first-class shipped output and was never validated).
- **Replay backtest first**: rerun both models over stored context_snapshots history for
  already-resolved forecasts before the live shadow window, so the verdict doesn't wait
  months.
- **Time-to-repricing**: reclassified as a pipeline SLO (it measures indexer latency,
  not the model — both models share the push stream).
- **Stopping rule**: 8 weeks or 30 resolved temporal forecasts, whichever first; tie or
  insufficient data ⇒ **keep the glide**. Boundary pins and the glide itself are
  requirements verified by tests, not hypotheses — they never wait on this gate.

## 5. Open questions (carried, updated)

- Market/API data as first-class Oracle measurements (Brent spot, FIDE, polls) — still
  open; changes the Oracle's identity.
- UI explainability of clock drift: Stage 0 ships the minimal version (snapshot kind,
  gauge annotation "includes time-remaining adjustment — deadline in N days"); a richer
  drift visualization is open.
- Survival-curve grid P(by T): dropped from v1 scope — no consumer exists (no sibling
  linkage in the schema; the similar-forecasts redesign treats deadline siblings as
  things to separate). Revisit if a ladder UI ever exists.

## 6. Relevant code

- retro: `api/src/forecast_api/forecaster.py` (pipeline, settlement override,
  `_empty_response` short-circuits at 608–642/799–857), `aggregation.py` (pool_sources
  convexity, recency_floor, thin-evidence CI), `config.py` (logit_clamp,
  recency_half_life_days, cache_ttl_seconds), `models.py` (ForecastRequest — additive
  fields land here), `pipeline/src/tm/extractor.py`.
- daatan: `src/lib/services/context.ts` (3 snapshot writers, notifyIfCrossedHighConfidence
  at :28–43, getContextTimeline), `src/app/api/news-indexer/context/route.ts`,
  `src/lib/services/oracle.ts`, `src/lib/services/commitment.ts` (:43 ACTIVE gate — the
  free-points fix lands here), `src/lib/services/prediction-resolution.ts` (:132 rsChange),
  `prisma/schema.prisma` (Prediction — migration target), `src/app/api/cron/` (requote
  route lands here), `.github/workflows/external-market-sync.yml` (cron pattern).
- news-indexer: `src/news_indexer/worker/matcher.py` (push/cooldown; pre-gate candidate
  logging for p_det v2), `rematch.py`.
- Pilot data: `temporal-plan/pm_markets.json` (110 resolved political markets), `temporal-plan/pm_hist.json`
  (27 trajectories; 13 = one correlated Iran cluster — treat as ~10–15 independent
  families). Gamma: gamma-api.polymarket.com/events?tag_slug=politics&closed=true;
  CLOB: clob.polymarket.com/prices-history?market=<tokenId>&interval=max&fidelity=1440.
  Move to a frozen S3 snapshot in Stage B.
- Review record: `temporal-plan/review_result.json` (47 confirmed / 1 refuted / 4 gaps, with
  per-finding fixes and verifier notes).

## 7. Literature anchors (adopted)

- **Hawkes processes**: the relaxation formula is a self-exciting point process with
  exponential kernel — reuse standard fitting; test power-law/two-timescale kernels for
  geopolitics.
- **Satopää et al. 2014**: extremize the log-odds pool (α≈2 baseline, fit on own
  resolved set); news articles are maximally information-overlapped voters.
- **Atanasov et al. (GJP)**: decay + performance weighting + recalibration work *as a
  combination*; biggest edge at the start of long questions. Wire the source_accuracy
  Ledger in as the performance weight (v2).
- **Halawi et al. 2024**: same retrieval→extraction→aggregation shape, near
  crowd-aggregate; gains came from the IR layer (query expansion, relevance ranking);
  well-calibrated only with sufficient retrieval + abstention — empirical backing for
  the degradation ladder; LLM stances hedge near certainty, so boundary precision must
  come from pins, not the pool.
- **ViEWS (conflict forecasting)**: hazard models are strong on continuation, weak on
  onset — hence the onset/continuation KPI split and settlement reaction as the
  compensating mechanism.
- **Favorite-longshot bias**: most robust prediction-market finding; grows with horizon;
  prices are not ground truth — hence outcome-anchored fitting only.
