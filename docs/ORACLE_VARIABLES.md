# Oracle math — variable audit and simplification plan

Audited 2026-07-08/09 against retro `1fb87ce3b` and daatan `d742ce90`. Written
as a read-only audit; §8 tracks what has shipped since. Companion to `TEMPORAL_MODEL_PLAN.md`
(the glide this document references is that plan's Stage 0, live in daatan since
2026-07-05). Production evidence cited below is from the 2026-07-08
investigation (Oracle host log + prod requote runs + public page JSON).

## 1. Why this document

The estimate pipeline has accumulated ~40 variables across four layers and two
repos. Several encode the same quantity more than once, three multiplicative
weight knobs answer one question, and the same "current AI estimate" lives in
five places that are allowed to disagree. Two of the 2026-07-08 production
incidents (invisible glide, the F-35 settled-banner-plus-52% contradiction) are
consequences of that redundancy, not of any single bug.

## 2. Inventory

### 2.1 Per-claim (extractor LLM — `pipeline/src/tm/extractor.py`, `models.py`)

| variable | scale | role | notes |
|---|---|---|---|
| `stance` | −1..1 | direction & strength of "event will happen" | pydantic-bounded, no clamping (out-of-range ⇒ article dropped) |
| `certainty` | 0..1 | linguistic confidence (0 = hedged, 1 = absolute) | weight factor AND within-article claim weight |
| `quantitative_estimate` | 0..1, optional | cited model/poll/market probability | overrides stance+certainty via `resolve_stance_certainty`; triggers the 4× premium |
| `settled` | bool | outcome reported as accomplished fact | feeds the ±0.94 settlement pin |
| `specificity` | 0..1, optional | **dead** — live extractor never emits it; defaults 1.0 | remove |
| `claim`, `quote` | text | provenance | — |

### 2.2 Per-article (gatekeeper LLM — `pipeline/src/tm/gatekeeper.py`)

| variable | scale | role | notes |
|---|---|---|---|
| `is_prediction` | bool | hard in/out gate | binary form of the same judgment as `relevance_score` |
| `relevance_score` | 0..1, default **1.0** | graded "how much would a forecaster update" | fail-open default; squared downstream |
| `prediction_count_estimate` | int | debug only | — |

### 2.3 Per-source derived (`api/src/forecast_api/forecaster.py:715-794`)

| variable | formula | notes |
|---|---|---|
| `avg_stance` | certainty-weighted mean over claims; settled claims *replace* the set | the source's vote |
| `avg_certainty` | plain mean over **all** claims (incl. non-settled color quotes) | inconsistent with `avg_stance`'s settled-replacement |
| `rweight` | `0.5^(age/7d)`, floor 0.02; missing date ⇒ 1.0 | fail-open |
| `credibility` | leaderboard lookup | **1.0 for every source observed in prod** — layer currently inert |
| `quantitative_multiplier` | 4.0 if any claim carries an estimate, else 1.0 | stacks with the certainty-0.9 floor |
| **`weight`** | `credibility × avg_certainty × rweight × relevance² × quant_mult` | pool weight |

### 2.4 Pool level (`aggregation.py`, `forecaster.py:799-932`)

| variable | role |
|---|---|
| `mean, std, ci_low, ci_high` | weighted-mean logit pool + dispersion, stance scale |
| `evidence_mass = Σ weight` | thin-evidence CI widening (floor 0.5, inflation 0.45) |
| `relevance_mass = Σ relevance²` | off-topic abstention (floor 0.05) |
| `settled_directions → settled` | settlement pin ±0.94 when ≥2 sources agree |
| `insufficient_data, reason, placeholder, articles_used/found` | abstention encoding |

Config constants (12): `recency_half_life_days=7`, `recency_floor=0.02`,
`logit_clamp=0.01`, `relevance_weight_floor=0.05`,
`quantitative_anchor_weight=4`, `syndication_title_similarity=0.8`,
`decisiveness_floor=0.5`, `thin_evidence_ci_inflation=0.45`,
`defer_on_thin_evidence=False`, `settlement_min_sources=2`,
`settlement_stance=0.94`, `min_certainty=0.9`.

### 2.5 Persisted in daatan (`prisma/schema.prisma`)

| variable | scale | notes |
|---|---|---|
| `ContextSnapshot.externalProbability` | 0–100 | the estimate at that moment |
| `oracleSnapshot.mean/std/ciLow/ciHigh` | 0–100 post-v1.31.2; `mean/std` raw −1..1 in older rows | second encoding of the same estimate; two scales across history, no version marker |
| `oracleSnapshot.sources[]` | raw | stance/certainty/credibility/settled only — relevance, recency, quantitative_estimate **not persisted** |
| `ContextSnapshot.kind`, `meta` | — | `'evidence'`/`'clock'` + glide provenance `{pLast,tLast,tEff,c,direction,cause}` |
| `Prediction.confidence` | 0–100 | schema-documented as *bot metadata*, in practice the denormalized current AI estimate (glide writes it) |
| `Prediction.aiCiLow/aiCiHigh` | 0–100 | denormalized CI; glide push-forwards both bounds through its power map (`temporal-clock.ts:274-275`) |
| `Prediction.settled/settledAt` | bool | **one-way latch** — set on any Oracle `settled=true`, never cleared (`context.ts:159,257,355`) |
| `Prediction.sentiment/consensusLine/sourceSummary` | — | bot-creation-time duplicates of stance/estimate/summary |
| `ContextSnapshot.summary`, `externalReasoning` | text | two parallel free-text explanations |
| `Commitment.aiProbabilityAtCommit` | 0–100 | frozen history — legitimate, keep |

Temporal metadata: `claimDeadline`, `claimDirection`, `claimArchetype`,
`tauLeadDays`, `classifierVersion` (+ pre-existing `resolveByDatetime`); glide
constants `PIN_LOW/HIGH = 3/97`, `MATERIAL_CHANGE_PTS = 1`, divergence
tolerance 72h.

## 3. Logical duplicates

1. **"The current AI estimate" exists in five places**: latest evidence
   `ContextSnapshot.externalProbability`; `oracleSnapshot.mean` (same number,
   second encoding, two historical scales); `Prediction.confidence`;
   `Prediction.aiCiLow/High`; the UI-computed `aiEstimate`. They are allowed to
   disagree — and do: after a glide, `confidence` moves while the latest
   evidence snapshot doesn't, which is exactly why the glide is invisible on
   the detail page while visible on feed cards (§4.2), and why `elections.ts`
   could misread `oracleSnapshot.mean`.
2. **Three knobs encode "trust this source's number"**: `certainty` as a weight
   factor, the `min_certainty=0.9` floor, and the ×4 `quantitative_multiplier`.
   All answer one question — *what class of evidence is this claim?* — and they
   multiply, so a quantitative source is boosted twice for one property
   (weight ≈ 3.6 vs ≈ 0.4 typical ⇒ 60–70 % of pool mass in a 5-source pool).
3. **Two relevance judgments from one LLM call**: `is_prediction` (binary) and
   `relevance_score` (graded). The binary one is ≈ `relevance < 0.2` with extra
   failure modes.
4. **Four "how much evidence" measures**: `evidence_mass`, `relevance_mass`,
   `articles_used`, and post-widening CI width. Two floors + an inflation
   constant + an escape-hatch flag govern what is one monotone ladder
   (abstain → wide CI → trusted CI).
5. **Two boundary-pin conventions defined independently**: retro
   `settlement_stance=0.94` (⇒ 97/3) and daatan `PIN_LOW/HIGH=3/97`. They
   coincide today by arithmetic luck.
6. **Bot-era leftovers duplicating Oracle fields**: `sentiment` ≈ sign(stance);
   `confidence` ≈ the estimate (dual use is the schema's worst naming trap);
   `consensusLine`/`sourceSummary` vs `summary`/`externalReasoning`.

Deliberate non-duplicates, for the record: `claimDeadline` vs
`resolveByDatetime` (claim semantics vs platform deadline — the divergence rule
needs both); `aiProbabilityAtCommit` (frozen history).

## 4. Three questions re-examined

### 4.1 Do we need `relevance²`, or is relevance already implicit in stance?

**Relevance cannot live inside stance.** The pool is a *weighted mean* of
logits: a stance-0 vote is not an abstention, it actively drags the pool toward
50 %. "This article doesn't bear on the question" must therefore be expressed
as weight ≈ 0 — a value can't do it. And empirically the extractor does not
self-attenuate: it is conditioned on the question and maps whatever it finds
onto it (observed in prod: a 2023 live-blog got stance −0.46 / certainty 0.84
on a 2026 question; an encyclopedia background page got stance +0.5). Certainty
doesn't absorb it either — assertiveness is orthogonal to aboutness.

What *is* redundant nearby:

- `is_prediction` vs `relevance_score` — one judgment, two outputs (cluster 3).
  Collapse to the graded score with an explicit gate threshold.
- The **square** is a tuning choice, not information. The gatekeeper prompt's
  rubric (0.7–1.0 / 0.3–0.6 / 0.0–0.2) already builds convexity into the scale;
  squaring on top of that is a second, undocumented convexity. Either fold the
  desired shape into the prompt rubric and use `relevance¹`, or keep the
  exponent as a **named config constant** (`relevance_exponent`) so it is
  visibly a calibration knob. No information is lost either way.
- `relevance²` currently does triple duty (weight factor, `relevance_mass`
  floor, inside `evidence_mass`) — §5 S3 collapses the floors.

Verdict: **keep relevance as a weight; delete the binary twin; make the
exponent explicit.** The real relevance problem is not the formula but the
judgment: it is graded on topical match while the weight treats it as
evidential bearing — the wrong-timeframe leak (F-35's "removed from the
program in 2019") passes topical relevance with full marks.

### 4.2 What exactly do we do with the range (CI)?

Computed in retro (`pool_sources`: weighted SEM in probability space, 1.96×,
then thin-evidence widening; settlement pins substitute a fixed narrow band),
persisted twice (snapshot JSON + denormalized `Prediction.aiCiLow/High`),
displayed in three places, and **consumed by no decision anywhere** — not the
glide anchor, not alerts, not commitment locks, not scoring.

Display inventory and why it "sometimes" appears:

| surface | source | shown when | file |
|---|---|---|---|
| detail-page gauge band | `aiEstimate.ciLow/High ?? Prediction.aiCiLow/High` | both defined AND `ciHigh > ciLow` AND not abstained | `Speedometer.tsx:228`, `ForecastDetailClient.tsx:479-481` |
| detail-page "± N %" line | latest evidence snapshot's `oracleSnapshot` | `oracleSnapshot` present AND `ciHigh > ciLow` | `ContextTimeline.tsx:404-406` |
| feed card "AI: X±N%" | `Prediction.confidence` + `aiCiLow/High` | both non-null AND `ciHigh > ciLow`, else bare "X%" | `ForecastCard.tsx:406-418` |

Concrete hide conditions (all observed or reachable in prod):

1. **Zero-width CI**: a single-article pool has `std=0`; if its (4×-inflated)
   evidence mass clears the 0.5 widening floor, `ci_low == ci_high` and every
   surface hides the band — observed 2026-07-08 (`mean=0.900 std=0.000
   ci=[0.900,0.900]`, mass 0.587, one article). The band disappears exactly
   when uncertainty is highest.
2. **LLM-fallback snapshots** have no `oracleSnapshot` ⇒ timeline "±" hidden;
   the gauge then falls back to `Prediction.aiCiLow/High`, which may be a
   *stale* band from an earlier Oracle write around a fresh fallback needle.
3. **Abstention** (`insufficientData`) hides needle and band.
4. **Legacy rows**: forecasts whose latest write predates the denormalized
   columns have null `aiCiLow/High` ⇒ card shows bare "X%".
5. **Card vs detail disagree by design**: the card reads glide-updated
   `Prediction.*`, the detail gauge reads the latest *evidence* snapshot — the
   two surfaces can show different numbers and different bands for the same
   forecast on the same day.

Verdict: the CI is an elaborately computed decoration. Either make it
load-bearing — enforce a minimum width for n < 3 sources (fixes hide-condition
1 at the source), render it on every surface from the same accessor, and gate
high-confidence alerts / commitment locks on width — or stop computing
`std`/SEM and ship a three-level confidence label derived from evidence mass.
The half-maintained middle is what produces "sometimes I see it, sometimes
not".

### 4.3 Are the indexer-push / initial-estimate / analyze paths symmetric?

**No — and there are five paths, not three.** The estimate on
`Prediction.confidence`/`aiCiLow`/`aiCiHigh` can be written by five code paths
with four different ideas of what "evidence" is (all refs daatan `d742ce90`):

| | evidence fed to the math | limit | snapshot writer | persists `externalProbability` | notifies |
|---|---|---|---|---|---|
| **A** creation | *none* — express-generate is a plain LLM read of searched articles (`expressPrediction.ts:335`), express-guess calls the Oracle **question-only** (`express/guess/route.ts:32`) | — | **none** (writes `Prediction.confidence` directly, no snapshot — `forecast.ts:229`) | — | none |
| **B** indexer push | caller-provided matched `articles[]` (`news-indexer/context/route.ts:88`) | **uncapped** daatan-side (`.min(1)`, no `.max()` — `:35`); 15 retro-side | `saveNewsIndexerMatch` (`context.ts:218`) | yes | crossing + Telegram news-match |
| **C** analyze | daatan `oracleSearch` → 15 `articles[]` (`[id]/context/route.ts:107,177`) | 15 | `saveContextUpdate` (`context.ts:129`) | yes | crossing |
| **D** backfill | daatan `oracleSearch` → 15 `articles[]` (`oracle-backfill.ts:26`) | 15 | `saveOracleSnapshotOnly` (`context.ts:337`) | **no — left null** | crossing |
| **E** clock | none (pure arithmetic from anchor) | — | `saveClockSnapshot` (`context.ts:378`) | yes (`kind='clock'`) | none |

All of B/C/D land as identical `kind='evidence'` rows distinguished only by a
marker string in `externalReasoning` — downstream, the timeline and gauge
cannot tell them apart. The concrete asymmetries that produce flip-flops:

1. **No writer compares evidence richness before overwriting.** A 1-article
   news push clobbers a 15-article analyze estimate and vice versa; no
   `articles_used`/mass comparison exists in any writer (`context.ts:154,254,352`).
2. **A vs B/C answer different questions.** Creation numbers come from an LLM
   opinion or a question-only Oracle run (retro's own search chain); B/C feed
   explicit article sets. The first analyze after creation typically moves the
   number even with zero news — different evidence universe, not new evidence.
3. **B is uncapped, C/D are capped at 15** — different evidence masses for the
   same claim, overwriting each other.
4. **D writes `Prediction.confidence` but leaves the snapshot's
   `externalProbability` null**, so the chart and the glide anchor
   (`getLatestEvidenceEstimate` filters non-null, `context.ts:296`) ignore the
   backfill point while the gauge shows it — the clock then glides from an
   *older* anchor than the displayed number.
5. **Only C nulls the estimate on `insufficientData`** (`context.ts:152`); B
   skips the write instead, D marks-attempted without clearing — so estimates
   flip between "—" and a value depending on which path ran last.
6. **In C/D, `confidence` is written conditionally but the CI unconditionally**
   — a run can blank the band while leaving the needle from a previous run
   (value and interval from different runs on one gauge).
7. **No cross-path cooldown.** C's 1 h cooldown keys on `contextUpdatedAt`,
   which B and D deliberately don't update; and the crossing alert re-fires on
   every cross-below-then-above, so B→C→B ping-pong re-alerts.

Verdict: the *math* is shared (same retro pool for B/C/D), but the *evidence
supply and persistence* are not symmetric in any respect that matters. The
fix is not to tune the paths individually but to funnel them: one
`recordEstimate(forecastId, oracleResponse, origin)` writer that (a) always
persists the same fields including `externalProbability` and `articles_used`,
(b) stamps `origin` as a structured field instead of a reasoning string,
(c) refuses to overwrite a strictly richer recent estimate with a strictly
poorer one (or at minimum persists enough to make that policy possible), and
(d) owns the settled latch and all notifications. Path A should either call
the same funnel or be labeled a user draft, not an AI estimate.

## 5. Simplification proposals

Ordered by leverage; each independent.

- **S1 — one canonical estimate stream, one writer, one reader.** Every
  estimate change (evidence or clock, any origin path) goes through a single
  `recordEstimate` funnel (§4.3 verdict) that writes a `ContextSnapshot` with a
  structured `origin`; `Prediction.confidence/aiCiLow/aiCiHigh` become a cache
  of the latest row maintained only by that funnel; every reader (gauge, chart,
  card, elections, API) goes through one accessor. Drop
  `oracleSnapshot.mean/std/ciLow/ciHigh` (keep `sources[]` + raw response for
  audit). Kills cluster 1 and §4.3 asymmetries 1/4/5/6 — including the
  invisible-glide and card-vs-detail classes of bug.
- **S2 — replace (stance, certainty, quantitative_estimate) with
  (p, evidence_class).** One probability per claim (stance ≡ 2p−1 becomes
  derived) plus an enum: `reported_fact`, `cited_probability`,
  `cited_share` (polls/seat counts — explicitly *not* a probability),
  `reporting`, `opinion`. Source weight becomes
  `class_weight[evidence_class] × recency × relevance^k` — one lookup table
  replaces certainty-as-weight, the 0.9 floor, and the ×4 multiplier (cluster
  2), and makes "confident op-ed ≠ confident fact" expressible. *Shipped in
  two rounds (see §8): the weight-formula half above (shadow-classify → class
  weight cutover); the stance→p schema rename ("one probability per claim")
  did not — `PredictionExtraction.stance` is unchanged, evidence_class was
  added alongside it rather than replacing it.*
- **S3 — one evidence ladder.** Single monotone rule on `evidence_mass`:
  `< m_abstain` ⇒ insufficient_data; `< m_trust` ⇒ widen CI proportionally;
  else trust. Off-topic folds in naturally (relevance is already inside the
  mass); the separate `relevance_mass` floor only exists to protect the
  unweighted-mean fallback, which disappears once abstention below `m_abstain`
  replaces it (cluster 4).
- **S4 — delete dead weight.** `specificity` (never emitted), `std` (derive
  from CI), `is_prediction` (≡ `relevance < 0.2`), and either seed
  `credibility` or remove it from the formula until the leaderboard is real.
- **S5 — one shared pin constant.** Define the boundary once (97/3) with
  `pinReason ∈ {settlement, impossibility, glide_terminal}` on the response and
  snapshot (cluster 5).
- **S6 — retire or namespace bot metadata.** Rename `Prediction.confidence` →
  `aiProbability`; drop `sentiment`/`consensusLine` or mark creation-time-only;
  collapse the free-text explanation fields (cluster 6).

Net effect: ~40 → ~25 variables; three multiplicative trust knobs → one enum;
five estimate homes → one; the 2026-07-08 incident classes become structurally
impossible rather than patched.

## 6. Evidence-pool design — making the paths symmetric (accepted direction)

§4.3 showed the paths are asymmetric in evidence supply and persistence.
"Symmetric" hides two different symmetries; one is free, one has real
trade-offs. Both are adopted here as the target design.

**Goals** (product intent behind the symmetry requirement):

1. One forecast = one trustworthy estimate — never a number that depends on
   *which mechanism* ran last.
2. Every estimate built the same way, from the same evidence universe,
   whichever path triggered it.
3. Movement is explainable: a change is either new evidence or the clock,
   visible on the page; one article never rewrites the whole basis.

**Part 1 — persistence symmetry (the funnel; no downsides).** All paths write
through one `recordEstimate(forecastId, oracleResponse, origin)`: identical
fields every time (including `externalProbability` and `articles_used`),
structured `origin` instead of a marker string, one cooldown and notification
policy, single owner of the settled latch, one reader accessor (= S1).

**Part 2 — evidence symmetry (the pool; the real change).** daatan maintains a
canonical **evidence pool per forecast**. Every path only *adds* articles to
the pool; every estimate is computed over the whole pool. Retro stays
stateless — every call becomes the already-existing caller-articles form, and
retro's own search chain is demoted to a discovery step that feeds the pool.
The clock stays outside the pool as a post-transform (TEMPORAL_MODEL_PLAN
§3.1).

| aspect | current | target |
|---|---|---|
| evidence basis | whatever the triggering path had (push = matched articles only, uncapped; analyze/backfill = fresh 15-article search; creation = LLM opinion or question-only run) | the forecast's full pool, always |
| news push | *replaces* the estimate with one computed from the pushed 1–8 articles | adds article(s) → recompute; a new article moves the number by its weight share only |
| analyze | independent search that overwrites the push estimate (and vice versa) | search *discovers new* pool members → same universe, no flip-flop |
| creation number | unlabeled LLM guess written into `Prediction.confidence`, no snapshot | labeled draft; the funnel produces the first pooled estimate asynchronously |
| overwrite policy | none — a 1-article run clobbers a 15-article run | moot: estimates are cumulative, not competing (funnel persists `articles_used` regardless) |
| glide visibility | chart filters clock rows; gauge shadowed by last evidence snapshot | automatic once every reader uses the funnel's accessor |

**Known problems and their mitigations** (each is a precondition or a design
constant, not a reason to drop the direction):

| problem | mitigation |
|---|---|
| memory makes errors sticky — a false settled article keeps voting (today it washes out on the next run; the F-35 estimate "recovered" precisely because runs are memoryless) | settlement hardening (code-enforced settled-claim rules, clearable latch) **and** per-article admin exclusion ship *before* the pool |
| stale-mass accumulation: with a weighted mean and recency floor 0.02, fifty stale articles ≈ weight 1.0, enough to outvote two fresh ones | drop the recency floor *inside* the pool (it exists to protect single-old-article calls, which the pool makes moot) + cap pool at top-N by current weight or hard age cutoff |
| re-extracting the whole pool per update is too slow/expensive (~2–4 s, 1–2 k tokens per article) | cache extraction per (article, question): daatan persists extracted signals, or retro accepts pre-extracted signals / caches per-article; also kills the 25 s-timeout volatility |
| identity change: the Oracle stops being "answer a question by searching" and becomes "score this evidence set" | conscious decision — same shape later needed for polls/market prices as first-class evidence (TEMPORAL_MODEL_PLAN §5 open question) |
| duplicates across days (same wire story pushed repeatedly) | move syndication dedupe from per-call to per-pool |
| what symmetry does **not** fix: single-source dominance is a weights problem (cluster 2) | S2 (`evidence_class`) proceeds independently |

**Sequencing:** funnel (S1) → settlement hardening + exclusion → pool with
pruning + extraction cache. S2 in parallel at any point.

## 7. Production evidence (2026-07-08)

- False settlement pins: F-35-to-Turkey pinned to 3 % at 09:26 by 2-of-6
  sources whose *background sentence* ("Turkey was removed from the F-35
  program in 2019") was extracted as an accomplished-fact settlement; daatan
  latched `settled=true` (one-way), page then showed "Outcome reported" beside
  a recovered 52 % estimate. McConnell forecast pinned to 97 % for ~4 h by
  2-of-15 sources against a pool of ≈ 47 %.
- Glide live but invisible: prod requote 2026-07-08 examined 72, glided 8
  (deltas 1–4 pts), unchanged 41, skippedNoAnchor 23 — none of it rendered on
  the detail page (chart filters `kind='clock'`; gauge shadowed by the latest
  evidence snapshot).
- Volatility: same-question estimates swung 30–50 probability points across
  same-day re-runs on 1–4 surviving articles (search nondeterminism + 25 s
  per-article timeouts + memoryless runs).

## 8. Implementation status (2026-07-09)

Shipped, in the accepted sequencing order (§6):

- **S1 / funnel (persistence symmetry)** — daatan #1053: every estimate path
  (creation, analyze, news-indexer push, backfill, requote clock) writes
  through one `recordEstimate` with an `ORIGIN_POLICY` table; snapshots gain
  structured `origin` + `articles_used`; needle+band updates are atomic;
  backfill now sets `externalProbability`. Reader side — daatan #1055: one
  `getProbabilityHistory` accessor (includes clock rows), chart renders glide
  as hollow dots, gauge reads the funnel cache instead of the latest evidence
  snapshot. DB semantics documented in daatan `docs/DATABASE.md` (#1054).
- **Settlement hardening** — retro #244: settlement-grade gates
  (`settlement_min_claim_stance`/`_certainty` = 0.9), settled claims skip
  stance/certainty realignment, direction guard via optional
  `claim_direction`/`claim_deadline` (fail-open), extractor prompt rule
  against historical-background "settlements" (the F-35 failure mode),
  demotions/suppressions logged (`settlement_demoted`/`settlement_suppressed`).
- **Prod data fixes (2026-07-08)** — F-35 `settled` latch cleared (audit found
  no other bad latches); all 409 pre-v1.31.2 `oracleSnapshot` rows normalized
  to percent, removing the two-scale historical caveat from live data.
- **elections.ts scale fix** — daatan #1057 + elections #18: `meanToProbability`
  rewritten as a percent pass-through (round + clamp [0, 100]) instead of
  converting `oracleSnapshot.mean` as if it were stance-scale; moved into
  `forecast-view.ts` next to `stanceToConfidence` (which correctly stays
  stance-scale — it reads per-source `sources[].stance`, not the aggregate).
- **Settlement realignment gate** — retro #246: the #244 realignment-skip now
  keys off `settlement_grade(...)`, not the raw `settled` flag. A below-grade
  `settled` claim citing a `quantitative_estimate` previously kept its raw,
  unrealigned stance while still earning the ×4 anchor weight premium from
  that estimate; it now goes through `resolve_stance_certainty` like ordinary
  evidence.
- **daatan claim-field wiring** — daatan #1061: analyze, news-indexer push,
  the admin Oracle-sources backfill, and bot voting now forward the
  prediction's stored `claimDirection`/`claimDeadline` on every Oracle call,
  arming the #244 direction guard (previously fail-open everywhere — nothing
  sent the fields). `ClaimDirection.NONE`/null are omitted, never sent as the
  literal string `"none"` (this API's `claim_direction` is a strict
  `Literal["arrival", "survival"]` — anything else 422s).
- **Clearable settled latch** — daatan #1062: `clearSettledLatch` +
  `DELETE /api/admin/forecasts/[id]/settled`, admin-gated, surfaced as a
  "Clear settlement" button on the forecast page when `settled=true`. Was
  the §6 precondition alongside settlement hardening; the F-35 incident's fix
  is no longer a one-off manual DB `UPDATE`.
- **Evidence pool — foundation layer only** — daatan #1063: new
  `EvidencePoolArticle` table, one row per `(predictionId, urlHash)`, the row
  itself doubling as the extraction cache. analyze/news-indexer/backfill
  shadow-write their per-source signal here in addition to their existing
  writes. **Nothing reads this table to compute an estimate yet** — the
  recompute-over-pool cutover (per path, plus retro's search demoted to
  discovery-only, dedup moved per-pool, recency floor dropped inside the
  pool) is still open, per the table below. `excluded` column reserved
  (unused) for the still-open per-article admin exclusion feature.
- **S2 — shadow-classification** — retro #248: extractor prompt +
  `PredictionExtraction.evidence_class` (`reported_fact` / `cited_probability`
  / `cited_share` / `reporting` / `opinion`, optional, fail-open to null on
  ambiguity). Classified and logged (`event=evidence_class_shadow`, later
  renamed `event=evidence_class_weighted` below) for observability on real
  traffic before the cutover; not yet on the `/forecast` response.
- **S2 — weight-formula cutover** — retro (this PR): `class_weight` lookup
  table added to `ApiSettings.evidence_class_weight`, replacing
  certainty-as-weight, `resolve_stance_certainty`'s 0.9 floor's effect on the
  cross-article `weight` term, and the standalone ×4
  `quantitative_anchor_multiplier` (deleted). Cross-article
  `weight = credibility × class_weight[evidence_class] × recency ×
  relevance²`; `class_weight["cited_probability"] = 4.0` keeps the old ×4
  anchor premium verbatim (same France World Cup regression protection,
  reproduced against the new formula in `TestFranceWorldCupRegression`).
  Unclassified evidence (extractor omitted `evidence_class`) falls back to the
  claim's own `certainty` rather than a lookup, so partial classification
  coverage doesn't regress weighting quality — see `evidence_class_weight()`
  in `aggregation.py`. Full table: `cited_probability=4.0`,
  `reported_fact=1.0`, `cited_share=1.5`, `reporting=0.6`, `opinion=0.25`
  (new calibration beyond the ported ×4 anchor value; expect retuning once
  more real-traffic `event=evidence_class_weighted` volume accumulates).
  `evidence_class` still isn't on the `/forecast` response — internal
  weighting input only. **This is what actually fixes single-source
  dominance** — the France World Cup and Opta-anchor classes of regression
  are now protected end-to-end, not just at the pure-function level.
- **Per-article admin exclusion** — daatan #1068: admin-only "Evidence pool"
  panel on the forecast page, `PATCH .../evidence-pool/[articleId]` toggles
  `EvidencePoolArticle.excluded`. Not yet enforced by any computation — the
  last §6 precondition is now buildable-on-top-of, not itself the cutover.
- **`evidence_weight` exposed on `/forecast`'s `SourceSignal`** — retro #251:
  the per-source `avg_evidence_weight` forecaster.py already computes
  internally (S2's resolved `class_weight`/certainty-fallback value) is now a
  response field, so a caller can persist it per pooled article. The
  `evidence_class` taxonomy itself and the `class_weight` lookup table stay
  internal to retro — only the resolved number crosses the API boundary.
  First concrete step of the recompute-over-pool cutover below: without this,
  a future recompute would silently fall back to certainty for every article,
  since evidence_class was never persisted anywhere outside retro's own
  request lifetime.
- **`evidenceWeight`/`relevanceScore` persisted into daatan's pool** — daatan
  #1071 + #1073: `EvidencePoolArticle` gained both columns, threaded through
  the full `OracleSource` → `enrichOracleSources` → `addArticlesToPool`
  funnel. `relevance_score` was a second, independently-discovered gap of the
  same class — never captured anywhere in daatan's pipeline at all, not just
  missing from the pool; without it a recompute would have treated every
  article as fully on-topic. The pool now carries everything
  `aggregate_pool()` below needs.
- **`POST /pool/aggregate` — the recompute endpoint** — retro (this PR):
  every step of `run_forecast()` *after* its per-article extraction loop
  (relevance off-topic safety net, logit pooling, thin-evidence CI widening,
  settlement override) extracted into one pure function,
  `aggregate_pool()` (`aggregation.py`), now shared by the live pipeline
  **and** the new endpoint — a recompute over an accumulated evidence pool
  can never silently drift from what a fresh run of the same evidence would
  produce. The new endpoint takes a list of already-extracted per-source
  signals (stance/certainty/credibility/relevance/evidence_weight/
  published_date/settled — exactly what a `SourceSignal` or a persisted
  `EvidencePoolArticle` row already has) and reruns this math — no search,
  no LLM calls. Recency is recomputed fresh against "now" for each source's
  `published_date`, not a stored value, so an article decays further by the
  time of a *later* recompute even if nothing else about it changed — the
  whole point of recomputing over a pool. `forecaster.run_forecast()` itself
  now calls `aggregate_pool()` too (full existing test suite passed
  unchanged post-refactor — 200/200 — proving the extraction preserved
  behavior exactly). 8 new tests for `aggregate_pool()`, 8 more for the
  endpoint.

Open, in suggested order:

- Evidence pool — the recompute-over-pool cutover, remaining steps: daatan
  shadow-compares `/pool/aggregate`'s result against each `analyze` run's
  live estimate (log only, not user-visible); cut `analyze` over to trust
  the recompute (highest-traffic, most self-correcting path); extend to
  `news-indexer` + `backfill` once stable in prod; move dedup + drop the
  recency floor inside the pool; decide `creation`-path scope (structurally
  harder than the others — creation uses `expressPrediction.ts`, not
  retro's Oracle at all, so it can't reuse this funnel without its own
  design pass).
- S3–S6 and the remaining small known defect from §7: `credibilityWeight`
  still ≈1.0 for all real sources — **not a code bug** (investigated
  2026-07-09): `get_credibility_weight()` is correctly implemented, but
  `data/leaderboard.json` on the Oracle host has been frozen since
  2026-03-28 (no cron/systemd timer/workflow anywhere regenerates it), and
  even that stale snapshot only ever scored 5 Israeli outlets against a
  2022-dated historical backtesting harness (`data/events/*.json`)
  architecturally disconnected from the live production pipeline's 19+
  actively-scraped international sources. Fixing this for real means
  building a resolution-outcome feedback loop (daatan `Prediction` resolves
  → which sources contributed → retro scoring) — its own design pass, on par
  with the evidence pool / S2 above, not a quick fix.

## 9. Credibility feedback loop (scoped 2026-07-10)

Design: extend the existing OpenSkill/ELO/Brier scorer (`tm/scorer.py`) to
score sources against real daatan resolutions instead of only the frozen,
hand-curated vault (`data/events/*.json`) — rather than build a second,
parallel credibility mechanism. Ships in shadow mode: the resolution-informed
score is computed and logged alongside the live `credibility_weight`, but
does not affect live Oracle forecasting math until it's been watched for a
while, mirroring the pool-recompute shadow-compare rollout in §6/§8.
Excludes `opinion`-class articles from the signal — an op-ed disagreeing
with how a claim resolved isn't the same kind of credibility failure as a
reported fact being wrong. Relies on OpenSkill's own `μ − 3σ` conservative
estimate to protect established sources from one bad resolution, rather than
a second bespoke decay layer.

retro has no code path that calls out to daatan or reads its DB — the whole
system is one-directional (daatan always initiates). So daatan pushes;
retro never pulls.

Small-tasks breakdown, in order:

- **Ingestion endpoint (storage only)** — retro (this PR): `POST
  /leaderboard/ingest` accepts one resolved forecast's per-source stances
  (`ResolutionSourceInput`: source, stance, evidence_class,
  credibility_weight, evidence_weight) plus the outcome, and appends them to
  `data/resolution_feedback.jsonl`. Idempotent on `prediction_id` (an
  in-memory guard, reloaded from disk at first use, same lazy-cache shape as
  `leaderboard.py`'s `_cache`) — a fire-and-forget retry from daatan can
  never double-count a source's history. Does **not** yet compute a score or
  touch `leaderboard.json` / `get_credibility_weight()` at all — that's
  deliberately a separate step so ingestion can ship and start accumulating
  real data before the scoring design (incremental per-resolution OpenSkill
  update vs. `Scorer.run()`-style full replay from the accumulated JSONL —
  still an open question, see below) is settled.
- **daatan: resolution hook** — when a `Prediction` resolves, gather its
  `EvidencePoolArticle` rows, exclude `opinion`-class, and fire-and-forget
  POST to the endpoint above (same `.catch()`-wrapped pattern as
  `addArticlesToPool` / `shadowCompareRecompute`).
- **retro: shadow scoring** — an incremental OpenSkill update per ingested
  resolution, written to a new field separate from the vault-curated
  `skill_conservative` `get_credibility_weight()` reads today. Open question
  going in: `Scorer.run()` today always replays *all* vault history from a
  fresh `PlackettLuce()` model rather than mutating a persisted `Rating`
  between runs — genuine incremental per-match updates (what was scoped) is
  an architectural departure from that pattern, not just a new data source
  for the same one. Needs a concrete decision once this step starts, not
  before.
- **Observe** — watch the shadow score against live `credibility_weight` on
  real resolutions for a while.
- **Cutover decision** — wire the resolution-informed score into
  `get_credibility_weight()` (replace or blend with the vault-curated
  skill), gated so it can be reverted.
- **Decide the manual vault's fate** — keep `data/events/*.json` /
  `Scorer.run()` as a supplementary override for sources with too few
  resolutions to trust yet, or retire it once resolution-driven scoring is
  proven.
