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
| `quantitative_estimate` | 0..1, optional | cited model/market probability of the event itself (never a vote share/seat count — those are `cited_share`) | overrides stance+certainty via `resolve_stance_certainty` ONLY when `evidence_class=cited_probability` (retro#362); that class carries the 4× premium — and, since retro#369, only if the claim's `quote` names a source on `cited_probability_source_allowlist` (`tm/config.py`); an unattributed figure is demoted by `enforce_anchor_provenance`, which also costs it the rewrite. **Shadow until `anchor_provenance_enforced`.** |
| `settled` | bool | outcome reported as accomplished fact | feeds the ±0.94 settlement pin; a POSITIVE settlement is demoted unless dated — see `enforce_settlement_event_date` below |
| `event_date` | ISO date, optional | when the article says the event itself occurs/occurred | compared against `claim_deadline` by `enforce_deadline_arithmetic`; REQUIRED for a positive `settled`; for a NEGATIVE `settled` it carries the FORECLOSING event's date when the article dates it (the rival's win, the elimination — optional: time-expiry impossibilities stay undated) — see below |
| `event_date_reference` | text, optional | the article's verbatim relative expression behind `event_date` ("on Friday", "yesterday") | code redoes the calendar walk from it and overrides a disagreeing `event_date` (`enforce_relative_date_resolution`) — see below |
| `specificity` | 0..1, optional | **dead** — live extractor never emits it; defaults 1.0 | remove |
| `claim`, `quote` | text | provenance | — |

#### Deadline claims are decided by arithmetic, not by the LLM

A "by DATE" claim is settled by comparing two dates, and the extractor is unreliable at
exactly that — while failing *confidently*. Measured against the live prod model
(Haiku 4.5) on the forecast *"The Israeli parliament will be dissolved by July 15, 2026"*:

- Given the Guardian's *"the Knesset will dissolve **on Friday**"* (article dated Mon 2026-07-13),
  it returned **stance +1.0 / certainty 0.95 on 5 of 5 runs** — once rendering the claim as
  *"dissolved on Friday, July 15"*, snapping the weekday onto the deadline. That Friday was
  **July 17**: the claim was false, and the extractor was maximally confident it was true.
- Given Middle East Eye's article, which spells out *"July 17"* in plain text, the same model
  returned **−1.0 on 4 of 4 runs**. The only difference between a right and a wrong answer was
  whether the article had done the arithmetic for it.

So `claim_deadline` is now passed into the extractor prompt, the model reports `event_date`
(with relative references like "on Friday" resolved against the article's date), and
`enforce_deadline_arithmetic` in `extractor.py` does the comparison in Python:

    arrival  ("X happens BY D"):          event_date ≤ D → supports (+) ; after D → contradicts (−)
    survival ("X does NOT happen by D"):  mirrored

Only *confident* signals (|stance| ≥ 0.9 or `settled`) are corrected — a hedged "might slip
past the deadline" is a real judgement, not an arithmetic error. Magnitude and certainty are
preserved; only the sign moves. Every correction logs `event=deadline_arithmetic_override`.
Fail-open throughout: no deadline, no `event_date`, an unparseable date, or an unclassified
claim leaves the prediction exactly as the model returned it.

**The model still cannot resolve weekdays — so code does.** With the prompt above it
resolves "Friday" *and states the date*, which flips the Guardian case to −1.0 on 4 of 4 runs
(the incident is fixed). But it resolved it to **2026-07-18** — a Saturday, off by one; the sign
survived only because both 07-17 and 07-18 fall after the deadline, and a ±1-day error against a
date sitting *on* the deadline would still invert the answer. The prompt therefore also asks for
the verbatim expression (`event_date_reference`), and `enforce_relative_date_resolution` in
`extractor.py` redoes the calendar walk in Python, overriding a disagreeing `event_date` before
either date-consuming guard runs. Vocabulary is deliberately small — today/tonight/yesterday/
tomorrow and a weekday with an optional on/this/coming/last modifier; "next Friday" is skipped
on purpose (speakers disagree on its meaning), as is anything non-English. Every correction logs
`event=relative_date_override`; a reference never *creates* a missing `event_date` (settlement
gating stays anchored to a date the model itself asserted). Fail-open throughout.

#### A positive settlement must be dated — `enforce_settlement_event_date`

The 2026-07-15 Netanyahu false pin: *"Netanyahu will win the 2026 Israeli general election
and be appointed PM"* was pinned to 97%/settled by exactly two settlement-grade votes — a
jns.org opinion piece (*"Netanyahu secured a 64-seat Likud-led coalition, confirming his
electoral victory"*) and a Guardian claim (*"Netanyahu will serve out his full term"*) — both
describing the **sitting** government formed after the *previous* election, while the election
the claim asks about was scheduled for 2026-10-27 and hadn't happened. Both cleared the
stance/certainty gates (that guard targets *hedged* settlements, not *misattributed* ones),
and `settlement_min_sources=2` was satisfied because the two errors were correlated — raising
the count doesn't help against a narrative shared across outlets.

The tell: neither article dated the "outcome", because the outcome described wasn't this
question's. The prompt's "historical background is not settlement" rule is advisory, so —
mirroring `enforce_deadline_arithmetic` — `enforce_settlement_event_date` in `extractor.py`
enforces it deterministically. A `settled=true` claim with **positive** stance (the event
occurred) is demoted to ordinary evidence (keeps stance/certainty, loses the settlement bit,
logs `event=settlement_demoted`) when:

- it has no parseable `event_date` — an occurrence you cannot date is not this question's
  outcome; or
- its `event_date` falls **after the article's own date** — the article "reports" an outcome
  that hadn't happened when it was written (a scheduled event, not an accomplished fact).

**Negative** settlements ("became permanently impossible") date the **foreclosing** event
instead — the rival's win, the elimination, the death that made the outcome impossible — when
the article dates it; an impossibility that comes only from time expiring stays undated, so
the missing-date demotion applies to positive settlements only (a **future-dated** foreclosure
is demoted like any future-dated settlement: it's a schedule, not a fact). Because a
foreclosure date is *not* the claim-event's occurrence date, `enforce_deadline_arithmetic`
exempts settled negatives **on arrival claims** — running its comparison there would flip a
correct impossibility verdict dated within the deadline into a false YES (the 2026-07-16
France-elimination trap). On survival claims a settled negative means the underlying event
*occurred*, its date is the occurrence itself, and the arithmetic stays valid. Premature
negative pins are guarded by `settlement_direction_allowed` (§ settlement override).

The extractor prompt also carries a **single-winner contests** section (2026-07-16
stance-inversion incident: "Spain beat France" / "Argentina stun England" were extracted as
**+1 settled** for "France/England will win"): in a one-winner contest, a rival achieving the
outcome settles the subject's claim **negatively** — stance −1.0, settled, dated by the
foreclosing result — never +1, however triumphant the article.

Deliberately fails **closed** on a positive settlement's missing date — the asymmetry is the
point: a wrong demotion costs a slower pin (the stance still votes), a wrong settlement sticks
a market at 97% on history. The prompt (SETTLED section) states the same contract, so a
compliant extraction is never demoted; the guard exists for the non-compliant ones.

The anchor date survives extraction: each source's `SourceSignal.settlement_event_date`
carries the `event_date` of the highest-certainty settlement-grade claim whose sign matches
the article's collapsed stance (`derive_settlement_event_date`, forecaster.py). Callers
persist it next to `settled` and send it back on `/pool/aggregate`
(`PoolSourceInput.settlement_event_date`).

#### Claim/stance sign conflicts are logged, not corrected — `flag_claim_stance_sign_conflicts`

retro#298 found rows where the extracted `claim` text and `stance` disagree with each other in
the same row — e.g. a claim stating a withdrawal *"is mandatory"* scored stance **-0.136**. The
general case ("does this stance follow from this claim") needs a second LLM call or a verifier
stage — out of scope here. `flag_claim_stance_sign_conflicts` (`extractor.py`) is the issue's own
"cheap partial": a deterministic marker check (`is mandatory`/`must`/`is required` vs.
`will not`/`refuses`/`rejects`, etc.) that logs `event=claim_stance_sign_conflict` when a claim's
explicit marker and its stance sign disagree. Runs once, right after extraction, before any of
the guards above can touch `stance` — **observability only, never corrects a prediction**. It is
narrow by design: literal marker clashes only, so it misses subtler mismatches (a demand read as
adversarial when it is actually a climb-down, retro#298's own row 6451).

#### Aggregation-time revalidation — `settlement_vote_validity`

Extraction-time guards only protect fresh extractions; a recompute replays stored `settled`
bits written before the guards existed or re-poisoned since (the 2026-07-16 audit: 11 of 19
pins wrong; re-pushes re-flipped cleaned flags within hours). With `settlement_revalidate`
(default **on**; env kill switch `SETTLEMENT_REVALIDATE=false` + service restart), every
settlement vote re-proves its anchor inside `aggregate_pool()` on every call — live
`/forecast` and `/pool/aggregate` alike:

- **Occurrence-direction vote** (arrival:+, survival:−, unclassified:+): must carry a
  parseable `settlement_event_date`; not after `claim_deadline`; not before
  `claim_created_at` when `claim_archetype='scheduled'` (a dated fact about an *earlier
  instance* of the recurring event — the 2021/2022-article class, both signs); not after its
  own article's date.
- **Non-occurrence-direction vote** (arrival:−, survival:+): valid once the deadline passed
  (undated is fine — the absence is the evidence), or with a dated in-window **foreclosing**
  event (France's elimination legitimately pins NO before the final — the case the old
  pin-level `settlement_direction_allowed` wrongly suppressed; the per-vote rules replace
  that guard on this path). An undated non-occurrence vote before/without a known deadline
  is demoted (`undated_foreclosure`) — deliberately fail-closed on the anchor, unlike the
  old fail-open pin guard (the F-35/Netanyahu background-history class). A **dated** anchor
  more than `settlement_post_deadline_grace_days` (default 14) past a closed window is
  demoted too (`post_window_occurrence`): an out-of-window occurrence of a repeatable event
  says nothing about the window — the 2026-07-19 pool audit's "US bombs Iran in 2025" rows
  were settled NO at 0.93+ by July-2026 strike articles while ground truth was YES. Within
  the grace it is the flipped late-arrival class (Knesset dissolving July 17 vs a July 15
  deadline) and stands. A third, narrower case sits between `undated_foreclosure` and
  `post_window_occurrence`: an **undated** non-occurrence vote where the window IS already
  closed, but the article itself was *published* more than `settlement_post_deadline_grace_days`
  after the deadline, is demoted too (`stale_undated_foreclosure`, retro#295/#293) — an undated
  "nothing happened" read from an article that late is more likely a misread of a LATER,
  different-timeframe recurrence of the same event class than genuine retrospective silence on
  the closed window (the same 2026-07-19 audit's "US bombs Iran in 2025" class: mid-2026
  articles about active 2026 strikes extracted as an undated NO for the already-closed 2025
  window). The extractor prompt already forbids cross-timeframe extraction (retro#295); this
  check is the aggregation-time backstop for rows that slip through it — keyed on
  `published_date` rather than `event_date`, since there is no event date to anchor on. An
  undated non-occurrence vote from an article published within grace of a closed window is
  unaffected — that stays the ordinary, honest "window closed quietly" case.

Demoted votes keep their stance (ordinary evidence; `event=settlement_vote_demoted` with a
reason per row). Valid votes in **both** directions suppress the pin entirely
(`settlement_suppressed`, `settlement_conflict` — one extraction is provably wrong, and
facts are not decided by outvoting; the England 4-vs-1 stance-inversion pool is the
canonical case). The pin then requires `settlement_min_sources` **unanimous** valid votes.

Clearing `settlement_min_sources` is a **count**, not a quality check — a pool of uniformly
weak sources (low credibility, thin relevance, recency-decayed) that each barely clear
settlement grade could still out-count its way to a pin. `settlement_quality_floor`
(retro#279, default **0 = disabled**) additionally requires the winning direction's votes to
carry at least this much *combined weight* (`credibility × evidence_weight × recency ×
relevance²` — the same per-source `weight` term the pool itself uses, summed over the
winning direction's valid votes only) before the pin is honored; below it the pin is
suppressed too (`suppression_reason="settlement_quality_floor"`) and the pooled mean stands,
same as any other suppressed pin. Left at 0 because there is no audited incident to calibrate
it against yet — 0.5 (reusing `decisiveness_floor`'s scale) broke multiple legitimate-pin
tests once wired through real per-source weights, since credibility/recency/relevance
multiplied together lands lower than a single flat floor assumes; tune from real pool data
before enabling in prod.

`PoolAggregateResponse` exposes `settlement_suppressed`/`settlement_suppression_reason`/
`settlement_votes_demoted` for callers. Regression fixtures from the audit:
`api/tests/test_settlement_revalidation.py`; quality-floor fixtures:
`TestQualityFloor` in the same file.

### 2.2 Per-article (gatekeeper LLM — `pipeline/src/tm/gatekeeper.py`)

| variable | scale | role | notes |
|---|---|---|---|
| `is_prediction` | bool | hard in/out gate | binary form of the same judgment as `relevance_score` |
| `relevance_score` | 0..1, default **1.0** | graded "how much would a forecaster update" | fail-open default; squared downstream |
| `prediction_count_estimate` | int | debug only | — |

**Content-free input is rejected before the model is called** (retro#359, 2026-08-02).
`carries_proposition()` strips URLs, `t.me/` paths, `@handles` and `#hashtags`, then
requires a run of ≥2 Unicode letters to survive; if none does, `check_is_prediction`
returns `is_prediction=false`, `relevance_score=0.0`, zero token usage, no LLM call.
This is a **floor, not a filter** — the bar is "is there anything to judge", not "is it
substantive"; a 4-character Hebrew newsflash passes and is judged normally. It is the
one gatekeeper rule enforced in code rather than taught by the prompt, because the
failure it prevents is the model *inventing* content: measured 2026-07-31, a t.me post
whose entire text was `https://www.c14.co.il/article/1641278` was endorsed for 76 of 127
open forecasts at relevance 0.7–1.0 with fabricated justifications. Five articles of that
shape cost 644 judgments and 247 downstream Oracle runs. The guard sits inside
`check_is_prediction`, so it covers `/forecast` and `POST /relevance` alike — the latter
matters because news-indexer persists that verdict and it can be reused in place of a
fresh judgment (`reuse_supplied_relevance`), so a confabulated 1.0 does not stay
contained. Bare domains (`ynet.co.il`) are deliberately **not** stripped: the pattern that
catches them also eats ordinary abbreviations, and over-rejection here silently loses a
curated journalist's scoop, which is the failure news-indexer's rescue path exists to undo.

### 2.3 Per-source derived (`api/src/forecast_api/forecaster.py:715-794`)

| variable | formula | notes |
|---|---|---|
| `avg_stance` | certainty-weighted mean over claims; settled claims *replace* the set | the source's vote |
| `avg_certainty` | plain mean over **all** claims (incl. non-settled color quotes) | inconsistent with `avg_stance`'s settled-replacement. Per-claim, `certainty` is now capped in code at `interested_party_certainty_cap` (`tm/config.py`, **0.5**) whenever the extractor marked the claim `verified=false` — `enforce_interested_party_certainty`, the weight-side half of the interested-party rule (retro#378), in the `enforce_*` chain right after the stance half (retro#368). Unlike that cap, this number is the **prompt's own literal**, which nothing had enforced: 30.3% of live `verified=false` rows exceeded it (56/185, max 0.733, ten at exactly 0.70 with avg \|stance\| 0.76) — and that is a floor on the per-claim rate, since the stored value is the article-level reduction while `verified` is the dominant claim's. **Where it actually bites is narrower than "certainty is a weight" suggests**: `evidence_class_weight()` ignores certainty for any *classified* claim and the unclassified branch is already capped at 0.25 (F10/R3), so the pool weight does **not** move. What moves is (a) the within-article fusion — `claim_weighted_stance` weights claims by certainty, so an over-confident unverified claim now pulls its article's `avg_stance` and `fact_signal` less (matrix A20) — and (b) the number itself, which is persisted, carried on the pool wire, and is the weight fallback in `run_pool_aggregate` for legacy rows with no `evidence_weight`. It also matters ahead of R1, where per-claim certainty becomes the claim weight directly. |
| `rweight` | `0.5^(age/7d)`, floor 0.02; missing date ⇒ 1.0 | fail-open |
| `credibility` | leaderboard lookup | **1.0 for every source observed in prod** — layer currently inert |
| `quantitative_multiplier` | 4.0 if any claim carries an estimate, else 1.0 | stacks with the certainty-0.9 floor |
| **`weight`** | `credibility × avg_certainty × rweight × relevance² × quant_mult` | pool weight |
| `fact_signal` (shadow) | claim-weighted **mean** of per-claim `fact_signal` over the **same** scored claims as `avg_stance`; `None` if none carried one | Phase 2 fact-lane counterpart of `avg_stance`, un-fused from author assertion; **read by nothing in aggregation** — surfaced on `SourceSignal`/`sources[]` only for daatan persistence + the offline fact-lane gate harness (`pipeline/scripts/backtest_fact_signal_gate.py`: stance-vs-fact_signal paired Brier through the real `/pool/aggregate`; any estimator cutover is gated on it turning convincingly positive, same evidence standard as the credibility flag). Extraction-side, the FACT_SIGNAL prompt carries a **decider-statement exception** (2026-07-29, A/B-gated): an on-record statement by the actor/authority whose own act would resolve the claim — announcement or denial alike — enters the fact lane as a capped precursor instead of being nulled as opinion; assertions *about* the decider's intent by opponents or analysts stay claimed-and-unverified. A companion **negative-precursor ladder** (2026-07-29, WS5b, A/B-gated) generalizes the negative side beyond decider statements: any contrary reported fact — an obstacle emerging, a preparation reversed, an opposing development, a measured indicator moving against the event — enters the fact lane as a graded negative precursor instead of null, with the extreme negative reserved for established impossibility; this closes the measured null asymmetry (negative-stance rows nulled 33.0% vs 22.6% for positive, fact-era pool) while the mobilization regression keeps deflating (see `test_extractor_prompt.py::test_negative_precursor_ladder_present` for the A/B record). Wording is numeral-free by design (magnitude policy belongs in estimator config); see `test_extractor_prompt.py::test_decider_statements_exception_present` for the A/B evidence and re-run bar. Its facets `event_actors`/`event_target`/`is_occurrence`/`verified` ride from the **dominant** (max \|fact_signal\|) claim so they stay internally coherent. Magnitude for a **precursor** is enforced in code, not by the prompt (retro#367): `enforce_precursor_cap` clamps per-claim \|`fact_signal`\| to `fact_signal_precursor_cap` (`tm/config.py`, **0.3**) whenever the extractor set `is_occurrence=false`, in the `enforce_*` chain immediately before this fusion — so both the mean and the dominant-claim selection see the capped value. |
| `author_lean`, `author_lean_certainty` (shadow) | passed through from `ExtractionOutput` (retro #308/#309) — the byline author's OWN forecast | author-accuracy scoring lane; **not read by aggregation** |
| `claims_detail` | no reduction — the article's claims themselves, projected onto `ClaimDetail` (`build_claims_detail()`) | **F1/F15, retro#364.** Every other row in this table is a reduction; this is the layer they reduce *from*, and until it existed the inputs were discarded at the wire, so no reduction here was checkable and no history was re-scorable. Recorded POST-resolution — after the `enforce_*` chain and `resolve_stance_certainty()` — i.e. the values the fusion actually consumed, which is what keeps `avg_stance`/`avg_certainty`/`evidence_weight`/`fact_signal` derivable from it (`test_claims_detail.py` pins each derivation). Per claim: `claim`, `quote`, `stance`, `certainty`, `specificity`, `prediction_type`, `evidence_class`, `quantitative_estimate`, `settled`, `event_date`, `fact_signal` + its four facets. Two collapses become visible only here: `evidence_class` is per-claim (the article carries only the most common one) and the fact facets are per-claim (the article carries only the **dominant** claim's), which is why an over-cap interested-party claim diluted by in-contract siblings is invisible above this layer (retro#378). Unlike `claims`, nothing is filtered — a claim with an empty summary still voted, so it is still kept. **Read by nothing in aggregation**; persistence surface only (daatan#1235), same shadow-field rollout as `author_lean` and `fact_signal`. |

### 2.4 Pool level (`aggregation.py`, `forecaster.py:799-932`)

| variable | role |
|---|---|
| `mean, std, ci_low, ci_high` | weighted-mean logit pool + dispersion, stance scale; SEM divides by Kish `n_eff = (Σw)²/Σw²`, and the width is floored at `1.96·pool_dispersion_floor/√n_eff` so a unanimous pool cannot publish a point (F16) |
| `evidence_mass = Σ weight` | thin-evidence CI widening (floor 0.5, inflation 0.45) |
| `relevance_mass = Σ relevance²` | off-topic abstention (floor 0.05) |
| `settled_directions → settled` | settlement pin ±0.94 when ≥2 valid votes agree — **revalidated per vote** (`settlement_vote_validity`, default on): an occurrence-direction vote needs a parseable `settlement_event_date` within `[claim_created_at (scheduled), claim_deadline]` and ≤ its article's date; a non-occurrence vote needs a closed window (dated anchors at most `settlement_post_deadline_grace_days` past it, or an undated vote from an article published within that grace — else `stale_undated_foreclosure`) or a dated in-window foreclosure. Valid votes in BOTH directions ⇒ pin suppressed (`settlement_conflict`) — unanimity, not majority. Count alone isn't enough either: `settlement_quality_floor` (default 0 = off) additionally requires the winning direction's combined per-source weight to clear a bar, else the pin is suppressed (`settlement_quality_floor`). Kill switch `SETTLEMENT_REVALIDATE=false` restores flag-trusting majority vote + `settlement_direction_allowed`. |
| `insufficient_data, reason, placeholder, articles_used/found` | abstention encoding |

Config constants (12): `recency_half_life_days=7`, `recency_floor=0.02`,
`logit_clamp=0.01`, `relevance_weight_floor=0.05`,
`syndication_title_similarity=0.8`,
`decisiveness_floor=0.5`, `thin_evidence_ci_inflation=0.45`,
`defer_on_thin_evidence=False`, `pool_dispersion_floor=0.05`,
`settlement_min_sources=2`,
`settlement_stance=0.94`, `min_certainty=0.9`.
`logit_clamp` is not only the per-source log-odds guard: it also bounds the
pooled CI endpoints and (since F16) the thin-evidence widening term.
`pool_dispersion_floor` is the minimum published interval width, as a
between-source standard deviation in probability space — **a policy number, not
a measurement**, in the same class as `interested_party_stance_cap`; see the
derivation of its ceiling and its 5.2% binding rate in `config.py`.
(`quantitative_anchor_weight` was deleted in the S2 cutover — its 4× premium
lives on as `evidence_class_weight["cited_probability"]`.)

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
- **Adjacent-event prompt hardening (2026-07-12)** — extractor prompt section
  "THE EVENT ITSELF vs. ADJACENT EVENTS" (renamed/consolidated into "MATCH THE
  EVENT — do not credit a near-miss as the event" by #323 — see the
  2026-07-26/27 addendum below): a definitively reported fact about an
  adjacent event (a member leaving when the claim is about the organization, a
  similar-but-different action, a different arena) must not settle the claim or
  carry ±1.0 (the Illouz/Likud incident: "MK leaves Likud" scored stance +1.0
  settled=true for "a party withdraws from the race", pushing the estimate to
  93–99%). A/B sampling on the incident article (temp 0, n=10): nova-lite fails
  10/10 without the section, 8/10 with it — the section is hardening, not a fix;
  nova-lite at temp 0 is NOT deterministic and is sensitive to prompt whitespace.
  The reliable lever is a stronger extractor model (nova-pro 2/5; gemini-2.5-flash
  and flash-lite 0/3; gemini-2.5-pro 0/2 — all on the unmodified prompt). DONE
  2026-07-12: prod extractor switched to Claude Haiku 4.5 (0/10 failures, stance
  +0.37 deterministic, 3.7s, $1/$5 per M) via systemd drop-in on the Oracle host +
  Bedrock IAM grant (see infra/iam/README.md §4); verified live — the incident
  article now scores stance +0.31 / settled=false (~66%) instead of +1.0/settled
  (99%). Batch pipeline (`truthmachine.service`) deliberately stays on nova-lite.
- **Extractor guard consolidation (2026-07-26/27)** — three more prompt guards
  in the same "don't let a near-miss read as the event" family, all in
  `extractor.py`:
  - retro#299 "Unverified claims by an interested party — cap certainty": a
    claim of fact made by a party TO the underlying dispute about its OWN
    actions, casualties, or results (a belligerent's own damage count, a
    company's own success claim) carries certainty no higher than 0.5,
    however declaratively worded, unless the article also reports
    independent confirmation (a different party, a neutral observer,
    satellite imagery). Stance sign and magnitude are unaffected — only
    certainty is capped, since wartime/dispute self-reporting is routinely
    inflated or unverifiable.
  - retro#304 "The capability/intent cap applies PER CLAIM, not to the
    article's overall urgency": the existing capability/intent cap
    (|stance| ≤ 0.3 for a demonstrated capability, threat, or expectation
    that is not yet an occurrence) now explicitly applies to each claim
    independently — five capability/intent signals reported in one urgent,
    saturated article are still five separately-capped signals; their
    number or density does not itself aggregate into occurrence.
  - retro#300/#317, merged by #323 into "MATCH THE EVENT — do not credit a
    near-miss as the event" (the section this doc quoted above as "THE EVENT
    ITSELF vs. ADJACENT EVENTS"): the WHO/WHAT/SCOPE decomposition now nests
    three subsections instead of three separate headings — the original
    subject/action/arena adjacency test, #317's new named-actor rule (a fact
    about a different party in the same broader conflict is NOT evidence for
    a claim naming specific parties — "Iran strikes Jordan" is not "Iran
    strikes Israel", even the same night of the same crisis; capped at
    |stance| ≤ 0.2, certainty ≤ 0.3, never settled), and a short "a date does
    not excuse a near-miss" note (a clean, verifiable date on an adjacent
    fact does not promote it to a settlement — decide the match first, check
    the date second, never the other way around).
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
- **`author_lean`/`author_lean_certainty` exposed on `/forecast`'s
  `SourceSignal`** — retro (author-scoring redesign, follows #308/#309): the
  byline author's OWN directional forecast, which the extractor already emits
  on `ExtractionOutput` (per article×question), is now carried through
  `_process_article` → the per-source `SourceSignal` so daatan can persist it
  per pooled article and score author accuracy later. **Deliberately NOT a
  variable in this doc's sense** — it never enters `aggregate_pool()` or any
  weight; it is the *author-scoring lane*, kept separate from the estimate on
  purpose (the whole point of the un-fusing work). Shadow end-to-end: null on
  cached/old responses, populated only on fresh extractions. `author_lean` is
  the direction the author expects the event to **resolve**, deliberately
  independent of whether they *approve* of it — a 2026-07-24 prompt refinement
  after a wild-data analysis found critical op-eds that concede an event is
  happening (e.g. a column against an inevitable US–Saudi nuclear deal) leaking
  a negative lean; disapproval/alarm is sentiment, not a directional forecast.
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
- **The pool wire carries identity and claims** — retro#364 (F1+F15, Phase 1
  Tier S): the contract above was **eight anonymous scalars**. A pool row
  could not say which article it was, which outlet published it, what kind of
  evidence it carried, or which claims produced its numbers — so claim-level
  weighting (R1), the factored taxonomy (R5), per-claim credibility
  attribution (F3), the R6 shadow pool and **all** retroactive backtesting
  were blocked by the wire rather than by their own difficulty. We cannot
  re-score history we never kept. `PoolSourceInput` is now widened
  additively with `url` / `source_id` / `outlet` / `evidence_class` /
  `fact_signal` + its four facets / `claims_detail`, and `SourceSignal`
  emits `claims_detail` on every `/forecast` (see §2.3). **`run_pool_aggregate()`
  reads none of them** — the estimator keeps exactly the eight-scalar
  whitelist it had, deliberately, so that spending this data stays the job of
  the issues that own each mechanic (R1; #355 clustering; #372's cluster-aware
  settlement), each with its own R8 movement report. Per R8 this shipped as
  additive persistence only: the aggregation trace matrix moved **zero** cases,
  and `test_claims_detail.py` pins the estimate bit-identical (whole response
  object, ordinary and settlement paths) whether or not a caller sends the new
  fields. Storage half: daatan#1235.

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
  with the evidence pool / S2 above, not a quick fix. **That loop is now
  built (§9), and retro #337 wires it into `get_credibility_weight()` behind
  `RESOLUTION_SHADOW_CREDIBILITY_ENABLED` — still off by default, so this
  paragraph describes live behaviour until the flag is flipped.**

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

- **Ingestion endpoint (storage only)** — retro #254 (merged): `POST
  /leaderboard/ingest` accepts one resolved forecast's per-source stances
  (`ResolutionSourceInput`: source, stance, evidence_class,
  credibility_weight, evidence_weight) plus the outcome, and appends them to
  `data/resolution_feedback.jsonl`. Idempotent on `prediction_id` (an
  in-memory guard, reloaded from disk at first use, same lazy-cache shape as
  `leaderboard.py`'s `_cache`) — a fire-and-forget retry from daatan can
  never double-count a source's history.
- **`evidence_class` exposed on `/forecast`'s `SourceSignal`** — retro #255
  (merged): found while scoping the resolution hook below — daatan's
  `EvidencePoolArticle` only ever persisted the *resolved* `evidence_weight`
  number (#251/#1071), never the `evidence_class` label itself (deliberately
  kept internal by #251). The opinion-exclusion rule above needs the actual
  label — `evidence_weight` alone can't tell an opinion-class article apart
  from a low-certainty unclassified one, since both can land at a similar
  numeric weight. Same class of gap as the `relevance_score` discovery during
  the pool-recompute cutover (§8 step 2b) — a consumer need emerged that the
  original "stays internal" design didn't anticipate. Exposed as the
  article's most common non-null `evidence_class` among its claims (a claim-
  level field collapsed to article level, same shape as `avg_evidence_weight`
  itself). 4 new tests.
- **daatan: resolution hook** — daatan #1083 (merged, v1.45.0, same PR
  persisted `evidenceClass` on `EvidencePoolArticle`, mirroring #1071's
  funnel for `evidenceWeight`): when a **BINARY** `Prediction` resolves with
  a definite outcome, `pushCredibilityFeedback()` fire-and-forget POSTs the
  forecast's non-excluded, non-`opinion`-class pool articles to the endpoint
  above (same `.catch()`-wrapped pattern as `addArticlesToPool` /
  `shadowCompareRecompute`). Skips VOID/UNRESOLVABLE (no outcome to score
  against) and MULTIPLE_CHOICE (stance has no clean meaning for an option's
  correctness).
- **retro: shadow scoring** — retro (this PR): **replay-from-scratch per
  ingest, not true incremental updates** — resolved via `AskUserQuestion`
  once this step actually started, because scoping it surfaced a fact the
  original interview didn't have: `Scorer.run()` has never mutated a
  persisted `Rating` between runs — every run rebuilds all ratings from a
  fresh `PlackettLuce()` model and replays the *entire* vault from scratch.
  True incremental updates (the interview's literal Q4 answer) would have
  been a new architectural pattern in this codebase; replay reuses
  `Scorer.run()`'s exact proven logic, costs nothing extra at realistic
  resolution volumes, and is self-healing (a future scoring-logic fix
  applies to all history on the next call, not just new data) — while still
  satisfying Q4's actual goal of no new scheduled job, since the replay runs
  synchronously inside the `/leaderboard/ingest` request handler itself.
  `resolution_scorer.py`'s `rescore_from_disk()` reuses `tm.scorer`'s
  `brier_score`/`stance_to_prob` helpers and the same winners-beat-losers
  `PlackettLuce.rate()` call `Scorer._update_skill()` makes, replaying
  `data/resolution_feedback.jsonl` into a **separate**
  `data/resolution_leaderboard.json` — `get_credibility_weight()` never
  reads it. Re-enforces the opinion-class exclusion itself (not just
  trusting the caller already filtered it) so a future `/leaderboard/ingest`
  caller can't silently bypass the rule. A single-source resolution (no
  ranking counterparty) still contributes its plain Brier accuracy, just no
  OpenSkill rating movement — mirrors `Scorer.run()`'s own two-tier
  behavior (global stats always accumulate; the skill/ELO update is
  separately gated on ≥2 competing sources), a gap caught in review before
  merging. New `GET /leaderboard/resolution-shadow` exposes the shadow board
  for the observe step below. 9 new tests.
- **Author-scoring lane (author-scoring redesign, Phase 1 step 3;
  2026-07-25)** — the same ingest now also carries an optional
  `author_signals` array (byline `author`, `outlet_name`, `author_lean`
  [−1..1], `author_lean_certainty`, `evidence_class`), pushed by daatan from
  the pool's `author_lean` shadow columns. `rescore_authors_from_disk()`
  replays it per **(byline author, outlet)** into
  `data/resolution_author_leaderboard.json`, exposed at
  `GET /leaderboard/author-shadow`. Two deliberate departures from the
  stance lane: **opinion-class rows are included** (author_lean is the
  author's own lean — opinion is the signal), and within one resolution an
  author's rows are **averaged first** (one Brier per author per outcome —
  the shape validated on gate datapoint #1). Shadow only: nothing reads it
  into `/forecast` weighting.
- **Known residual in `author_lean` (retro #326 — documented, deliberately
  not fixed)** — the Behrendt/tagesschau class: commentary that *concedes the
  event is happening* but imports its downstream-consequence worry as a
  negative lean still leaks (extracted claims all arrival-affirming, lean
  still negative). Two prompt attempts did not flip it (one made the broader
  case worse), so it stands deferred on the same evidence bar as everything
  else in this lane: revisit only if the resolution-time author Brier
  (retro #315 — still outcome-starved) shows the subclass matters.
- **Byline identity merge (retro #329; wired in after datapoint #1)** —
  `rescore_authors_from_disk()` groups raw bylines through
  `_load_identity_map()`, which fetches news-indexer's curated
  `Person`/`PersonAlias` map (`GET /authors/admin/people`, the same
  `NEWS_INDEXER_URL`/`NEWS_INDEXER_API_KEY` retro already uses for search)
  and flattens it into `{alias: canonical_name}`. Fetched fresh once per
  rescore call — no caching, the curated set is small (a dozen-ish people)
  and replay-from-scratch already re-fetches everything else. **Fails open**
  to `{}` on any error (missing config, timeout, non-200, malformed
  response), falling back to the raw whitespace-normalized byline — same
  convention as every other news-indexer-backed dependency in this repo.
  Normalization also strips Hebrew niqqud/gershayim marks so diacritic
  variants of the same byline (the exact case news-indexer #161 hand-curated
  for Ynet/Ynetnews) collapse to one key even before an alias exists.
  Because scoring replays from scratch, curating a new alias in news-indexer
  retroactively merges that author's whole history on the *next* rescore —
  no backfill needed.
- **Per-resolution dedupe in the stance lane (retro #337)** — `rescore_from_disk()`
  now averages a source's rows within one resolution before scoring, as the
  author lane always did. The row-level version put a source with mixed-sign
  rows into **both** `winners` and `losers`, so it competed against itself in
  `rate()` and the loser write-back clobbered its winner update — 28 such
  source-resolutions across the first 6 real ingests (one resolution alone:
  113 rows over 30 distinct sources, 12 of them on both sides). It also made
  `predictions` count *articles*, overstating independent evidence: 14 sources
  had ≥8 rows while **none** had ≥8 distinct resolutions. Any shadow-board
  ordering eyeballed before this fix was unreliable. `articles` is now reported
  alongside `predictions`, mirroring the author board.
- **Cutover — the credibility signal is Brier, not `skill_conservative`
  (retro #337)** — `get_credibility_weight()` reads the resolution-shadow board
  when `RESOLUTION_SHADOW_CREDIBILITY_ENABLED=true` (default **off**).
  Measured before choosing: reusing the vault's `1.0 + skill_conservative/25`
  transform would have been a **no-op**. σ barely moves in these large
  multi-team OpenSkill matches, so `μ−3σ` stays pinned near 0 — weights spanned
  **0.986…1.016 (1.03×) on the real board** and 0.951…1.037 (1.09×) simulated at
  100 resolutions, i.e. a source right 90% of the time would outweigh one right
  20% of the time by ~9%, noise against `class_weight` (16×) and recency (50×).
  The *ranking* is fine (corr +0.98 with true accuracy); the absolute scale is
  not. Brier separates properly (corr −0.98, ~2.5× spread at 100 resolutions):

      b_shrunk = (brier_mean·n + prior_n·0.25) / (n + prior_n)
      weight   = clamp(1.0 + (0.25 − b_shrunk)·slope, w_min, w_max)

  Brier 0.25 — uninformed, or consistently hedged — maps to exactly 1.0, so an
  unknown source is neutral by construction. Shrinkage toward that prior
  (`resolution_shadow_brier_prior_n`, default 10) replaces a minimum per-source
  count: it degrades smoothly instead of at a cliff, so a newcomer with two
  lucky calls lands near 1.05 rather than at the upper clamp. One global gate,
  `resolution_shadow_min_global_predictions` (default **15** as of retro#341;
  originally 50 per retro#337's uncommitted simulation, corr ~0.97 there —
  but #341 found 50 unreachable in any useful timeframe on the real claim mix
  and that nobody had checked anything between n=6 and n=50.
  `pipeline/scripts/simulate_shadow_gate_correlation.py` fills that gap
  against the real weight formula: corr reaches 0.91 by n=15, the lowest n
  clearing a 0.90 bound), holds every source at 1.0 until the dataset as a
  whole is worth trusting. The OpenSkill fields stay on the board for
  display/ranking only.
  **Replace, not blend:** under the flag the vault is never consulted, and a
  source without resolution history falls back to neutral 1.0 — not to a frozen
  2022 backtest of 5 outlets, which would reintroduce exactly the stale-score
  ambiguity the cutover removes. Flag off ⇒ byte-identical to the pre-#337 path.
  **Watch `evidence_mass`:** it is `Σ weight`, so it feeds `decisiveness_floor`
  and thin-evidence CI widening. Weights centre on 1.0 by construction, but on
  the 6 real resolutions the mean lands at 0.950 — poorly-scoring pools will
  widen their CI slightly sooner.
- **Flipping it is manual, and gated on evidence** — run
  `pipeline/scripts/backtest_shadow_credibility.py` against a copy of the box's
  `resolution_feedback.jsonl`: it pools each resolution twice (shadow weights vs
  flat 1.0), leave-one-out, and compares Brier. On today's 6 resolutions it
  reports shadow **0.0018 worse** — the honest answer at that sample size, and
  the reason the gate is 15 rather than 6. When it turns convincingly positive,
  set the env var and restart `oracle-api.service`; revert is the same in
  reverse, no deploy either way (same story as `SETTLEMENT_REVALIDATE`).
- **The manual vault is legacy** — `data/leaderboard.json`, `data/events/*.json`
  and `tm.scorer.Scorer.run()` are retained *only* as the flag-off fallback and
  as test-fixture coverage. Nothing in production has called `Scorer.run()`
  since the vault froze on 2026-03-28 (no cron, timer or workflow does — it is
  reachable only via `python -m tm.scorer` and the pipeline tests), so naming it
  legacy is intent, not a functional change. Full retirement — deleting the
  module, the event fixtures and their tests — stays a separate decision;
  "unused but documented" is a stable end state too.
- **The author lane is deliberately NOT part of this cutover.**
  `author_lean` / `resolution_author_leaderboard.json` feed no live number.
  They score a *person's own directional record*, opinion-class **included**,
  because the columnist's opinion is the signal; the source lane scores
  *whether an outlet's reporting was right*, opinion-class **excluded**. The two
  boards deliberately invert that rule, so folding author scores into
  `get_credibility_weight()` would destroy what each one measures. Author-level
  weighting, if ever wanted, needs its own design pass.

## 10. Recency weighting is only as good as the upstream date (2026-07-13)

A single mis-dated source can dominate a pool exactly like a mis-scored one.
The Netanyahu "next government" forecast on elections.daatan.com sat at
83–91 % against a curated panel's 45 % even after the process-vs-outcome
prompt fix (§7-adjacent gatekeeper/extractor work, retro #262) shipped. Root
cause was one source: a Dec 29, 2022 JNS opinion column — confirmed via its
own `article:published_time` / JSON-LD `datePublished` metadata — got
re-crawled by news-indexer and stamped `published_at ≈ 2026-07-11`, then fed
to the Oracle as fresh, near-certain settlement evidence
(`stance=1.0, certainty=0.95, settled=true`) for an election three and a
half months in the future. At this repo's 7-day `recency_half_life_days`
(`api/src/forecast_api/config.py`), the wrong date alone gave the source
~50× its correct weight (`recency_weight≈1.0` vs. the ~0.02 floor it should
have gotten) — enough on its own to anchor the pooled mean high regardless
of gatekeeper/extractor prompt logic. `evidence_class_weight`'s
`opinion: 0.25` discount (§5) never got a chance to apply either: the
extractor classified the claim as `reported_fact`, not `opinion` — a
correctly-labeled op-ed would still have been weighted down, but recency was
the dominant term either way.

**This was a data-quality bug at ingestion, not a poolable-signal bug** —
fixed upstream in `news-indexer` (PR #122): `trafilatura.bare_extraction()`
defaults `with_metadata=False`, which skips `extract_metadata()` entirely, so
`result.date` was always `None` and every article — not just this one —
silently fell back to crawl time instead of its real publish date. An
extractor-prompt attempt at catching this class of error at the LLM level
("a 'next X' claim needs its precipitating milestone confirmed, not just the
outcome asserted") was drafted and validated against it first — 0/5 effect
on the live production model (Haiku 4.5), because the article body (byline
stripped) reads identically to a genuine fresh report; the one differentiator
(true publish date) is exactly the field the ingestion bug corrupts. Not
pursued further. If a pooled estimate looks anomalously confident again,
check the suspect source's `published_at` against its own page metadata
before assuming the extractor or gatekeeper prompt is at fault.

## 2026-08-01 — quantitative rewrite guarded by evidence class (retro#362, lane-soundness F5)

`resolve_stance_certainty` used to fire on ANY non-null `quantitative_estimate`
— no class guard — while the extractor prompt's qe section itself listed "poll
number, seat projection" and "the poll puts Candidate Y at 45%" as qe-worthy
figures and the `cited_share` class definition simultaneously called the same
figure "explicitly NOT a probability". Measured on prod (2026-08-01): 117
COMPLETE pool rows carried qe, 47 of them `cited_share` vs 14
`cited_probability` — Knesset seat shares rewritten to `stance = 2×share−1` at
certainty 0.9, bit-exact on single-claim rows. Both sides fixed: the rewrite
now applies only to `evidence_class == "cited_probability"` (unclassified
claims are also left alone — missing data must not increase influence), and
the prompt's qe section extracts probabilities of the event only, routing
shares/seat counts to `cited_share` + Numeric-thresholds stance comparison.
End-state remains a typed quantity `{value, kind}` (lane-soundness plan R4);
this guard is the interim protection. Persisted pool rows extracted before
this fix keep their rewritten stances (daatan does not store qe, so they are
not recomputable) — the 47 cited_share rows predate it and age out by recency.

## 2026-08-01 — the aggregation trace matrix is committed (retro#370, lane-soundness R8)

Every mechanic described above now has a pinned fixture. `api/tests/fixtures/aggregation_matrix/`
holds 57 cases in four groups — A intra-article composition (20), B fabrication
and the defences (17), C pool dynamics (14), D evidence-class boundaries (6) —
replayed through the real `run_forecast` by `api/tests/test_aggregation_matrix.py`
with only the gatekeeper/extractor calls, the leaderboard lookup and the clock
stubbed. Everything else in the chain (the date enforcers, settlement grading and
settled-replace, `claim_weighted_stance`, `resolve_stance_certainty`,
`evidence_class_weight`, `pool_sources`, thin-evidence widening, `aggregate_pool`
with settlement revalidation) executes as production code, so the numbers pin the
estimator rather than a test-local copy of its arithmetic.

Why fixtures and not a backtest: R8. Agreement with the live system measures
consistency, not correctness — it certifies any change that preserves an existing
bug — and at N < 20 resolutions a Brier holdout is noise. So no aggregator or
architecture change ships on agreement alone; the bar is zero *unexplained*
fixture movement plus a PR that names the case IDs it intends to move.

Each case carries hand-written directional invariants alongside the generated
snapshot (the snapshot says what the estimator does, the invariant says what the
case is about), so a PR cannot regenerate its way out of a behavioural claim. 27
cases are tagged `known_bad` with the finding that indicts them (F1 ×6, R3 ×5,
F12 ×4, F2/F4/F16 ×3 each, F9, F20, F23) — those are the snapshots the Phase-1
PRs are *supposed* to move. Regenerate with
`AGG_MATRIX_UPDATE=1 uv run pytest tests/test_aggregation_matrix.py`; the git diff
is the movement report. Full conventions: the README in that fixture directory.

Config is not overridden by the harness — cases run against `settings` as prod has
it, so a class-weight refit (D3) or a floor change surfaces as declared fixture
movement. The same case bodies are intended to become F22's extraction-drift
canary, run against the live extractor on a schedule.

## 2026-08-01 — missing data no longer increases influence (retro#366, lane-soundness R3: F13+F10+F14)

Three places resolved an absence to the *most favourable* value available. Each
was found separately; they are one anti-pattern, so they ship as one PR.

- **F13 — missing article date.** `recency_weight` returned a neutral **1.0**
  when the date was missing or unparseable: the single best multiplier the term
  can produce. An undated article therefore outweighed an honest, dated
  three-week-old report by 50×, and the less we knew about a source the more it
  was worth. It now decays straight to `recency_floor` (0.02) — treated as
  maximally stale rather than maximally fresh. A missing *reference* date or
  `half_life_days <= 0` is a different thing (recency switched off) and still
  returns 1.0 for every article alike.
- **F10 — missing `evidence_class`, pool side.** The unclassified fallback
  resolved to the claim's own `certainty`, uncapped. Certainty [0, 1] and the
  class table [0.25, 4.0] are incommensurable scales sharing one slot, so a
  confident unlabelled claim (0.95) out-weighed an identically confident claim
  the classifier *did* label `reporting` (0.6). The fallback is now
  `min(certainty, evidence_class_weight_unclassified_cap)` with the cap at 0.25,
  the weakest class's weight: an unlabelled claim can tie the weakest labelled
  one and never beat it, while a hedged unlabelled claim still resolves below
  that on its own certainty. The same cap applies in `run_pool_aggregate` to a
  legacy row with no stored `evidence_weight` — the same missing-data shape.
  (F10's *fusion-side* half — class weights weighting claims within an article —
  stays deferred: the destination architecture deletes the fusion step.)
- **F14 — zero-weight pool.** `pool_sources` has a zero-total guard that
  replaces the weights with a flat 1.0 each, so a pool in which every source was
  blocked by credibility and/or zeroed by relevance produced an *unweighted*
  answer from exactly the rows the weighting judged worthless. `aggregate_pool`
  now abstains first with `reason="no_usable_weight"`, regardless of
  `defer_on_thin_evidence` — no weight at all is not thin evidence, it is no
  evidence. `all_articles_off_topic` still takes precedence when both hold.

Measured before choosing constants (prod, 2026-08-01, 5729 COMPLETE evidence-pool
rows): **4 rows (0.1%)** carry no `published_date`, touching 1 of 115 pools;
**33 rows (0.6%)** are unclassified and *all* of them sit above the 0.25 cap
(certainty 0.38–0.74, mean 0.58); **22 rows (0.4%)** have no stored
`evidence_weight`, all above the cap; **0 of 115 pools** have zero total weight,
so F14 is purely defensive on today's traffic. The blast radius is small in every
direction — these are guards against the shapes that are rare precisely because
they are pathological.

R8 protocol: four matrix cases moved, all declared — **C6** (F13, the undated
article now ties the stale dated one at the floor instead of beating it 50-to-1),
**D2** and **A14** (F10, the capped fallback), **B16** (F14, now abstains). Their
`known_bad` tags are removed; the fixtures now pin the fixed behaviour and the
invariants state the R3 rule directly. No other case moved.

## 2026-08-01 — the precursor cap is enforced in code (retro#367, lane-soundness F9)

A fact that only *precedes* the event — a mobilisation, a capability, an
escalation — has been capped at \|`fact_signal`\| ≤ **0.3** by the extractor
prompt since the fact lane shipped, in the OCCURRENCE-vs-PRECURSOR rule, "no
matter how sustained, repeated, or intensifying it is". Nothing enforced it.

Prod audit before choosing anything (2026-08-01, `evidence_pool_articles`): of
the **1101** rows carrying `is_occurrence=false`, **269 — 24.4%** — are above the
cap; 187 in 0.3–0.5, 69 in 0.5–0.7, **13 above 0.7**, worst \|0.90\|, mean
magnitude among the breaches 0.464. And that 24.4% is a *floor* on the per-claim
rate, not the rate: the stored number is the claim-weighted mean over an
article's claims while `is_occurrence` comes from the single dominant one, so a
lone over-cap claim diluted by in-contract siblings never appears in the count.
Four further over-cap emissions were seen directly during the WS5/WS5b A/B runs
(retro#354).

`enforce_precursor_cap` (`extractor.py`, called from the `enforce_*` chain in
`forecaster.py`) is the enforcement, and it is deliberately narrow:

- **Only magnitude moves.** Sign, `stance`, `certainty`, `settled` and the facets
  are untouched — which direction a precursor points is a genuine judgement; how
  far it may push the estimate is policy.
- **The number lives in config** (`fact_signal_precursor_cap`, `tm/config.py`),
  not in the prompt and not in the code — the same numbers-out-of-prompts
  direction as the `evidence_class` weight table and retro#354's D1. The prompt
  keeps the literal for now; a test pins the two together so they cannot drift,
  and a later prompt cleanup can drop the numeral and keep the qualitative rule.
- **Fail-open in every direction.** A null `fact_signal`, or an `is_occurrence`
  that is null (the extractor declined to judge) or true, is left exactly as the
  model returned it. The clamp never invents a judgement the model didn't make.
- **It runs before fusion**, which fixes a second-order effect too: the dominant
  claim — whose facets are stored for the whole article — can no longer be
  captured by an over-cap precursor outranking a genuine occurrence claim.

R8 protocol: one matrix case moved, declared — **A20** (`fact_signal` 0.625 →
0.25, `mean` untouched, exactly as the case's own notes predicted). Its
`known_bad` tag is **narrowed, not removed**: F9's magnitude half is fixed and
the first invariant now pins it, but the case's second complaint survives — the
clamped precursor (0.3) still outranks the occurrence claim (0.1), so the article
is still filed with `is_occurrence=false` / `verified=false` despite containing a
verified occurrence. That is the fusion collapse itself (F1) and only per-claim
persistence fixes it, so the tag now traces retro#364. No other case moved.

Not in scope, and deliberately: the prompt numeral (see above), the
`fact_signal → aggregate_pool` cutover (still gated on the offline harness), and
the observation that in the audited over-cap rows `fact_signal` tracks `stance`
almost exactly (0.90/0.80, −0.82/−0.82, 0.75/0.75) — the prompt's "never let
fact_signal pull stance, or stance pull fact_signal" is not holding on this
population. That is evidence for the retro#354 D-family argument, recorded on the
issue rather than acted on here.

## 2026-08-01 — cited_probability gains a provenance check, in shadow (retro#369, lane-soundness F4)

`cited_probability` carries the largest weight in the table (**4.0**) and is the
only class that authorizes the stance rewrite — and nothing checked where the
number came from. One sentence of "a market prices this at 80%" in any article we
crawl buys the strongest evidence class in the system. The prompt's own canonical
example ("a poll-aggregator model gives Likud a 22% chance") names nobody, which
is precisely the shape.

Prod audit over all **16** live `cited_probability` rows (mean `evidence_weight`
**2.34**, the highest of any class): about ten genuinely name a checkable source —
**Opta ×5, Kalshi ×4, Polymarket** — and about six are not cited probabilities at
all: Goldman Sachs' "$110 Brent by year-end", Citigroup's "$82,000 price target",
JPMorgan's "$114 per barrel", and two carrying no figure whatsoever ("Fed rate
hikes are increasingly likely"). One check does both jobs: a claim naming no
verifiable source is not an anchor, whether the number was invented or there was
never a probability there to begin with.

`enforce_anchor_provenance` (`extractor.py`) scans the claim's verbatim `quote`
for a name on `cited_probability_source_allowlist` (`tm/config.py` — the two
integrated markets, named forecasting models, named pollsters; word-boundary,
case-insensitive) and demotes the class when it finds none. Because the demotion
is a class relabel, it also stops `resolve_stance_certainty` rewriting stance from
the figure. It **fails closed** — the same asymmetry `enforce_settlement_event_date`
applies to an undated positive settlement: an unverifiable premium is the exposure
itself, so absence of provenance costs the premium. The claim keeps its stance and
certainty and still votes as ordinary evidence.

**Shipped in shadow.** `anchor_provenance_enforced` defaults **off**: the check
runs and logs `event=anchor_provenance_unattributed` on every claim it would
demote, but changes nothing — so prod behaviour and every R8 snapshot are
untouched. The demotion target (`unattributed_probability_class`, typed as the
five-class Literal so a typo fails at startup) is a **placeholder pending the
policy decision**: `reporting` (0.6) says "we cannot check who produced this
figure, so it is ordinary coverage"; `cited_share` (1.5) would keep a premium on
an uncheckable number.

R8 protocol, both ways. Committed default (shadow): **no case moves**, 59 pass.
Dry run with enforcement on and the target at `reporting`, measured not predicted:

| Case | mean today | mean enforced |
|---|---|---|
| **B5** — one cited probability outguns three honest reports | +0.2059 | **−0.444** |
| **B6** — the same number from a credibility-0.3 outlet still outguns a trusted one | +0.2089 | **−0.5948** |
| **B9** — a debunking sentence's number becomes an anchor | +0.4175 | **−0.0832** |

Two corrections to the acceptance surface predicted on the issue. **A15 does not
move, and should not**: its anchor is meant to be legitimate (its `known_bad`
says the claim "should reach the pool as its own atom at weight 4.0") — its
complaint is F1's averaging, which an allowlist does not touch. And **D1 moved
for a fixture reason, not a table reason**: the prediction was that D1 would move
only if the demotion were implemented as a weight-table change; in fact four
untagged cases (**A12, A13, D1, D4**) moved because the matrix runner's synthetic
`quote` ("Fixture quote 0.") names nobody, so anchors those cases intend as
legitimate were demoted. Each of the four says so in its own notes — A12's "a
named model baseline must not be flipped", A13's "defying the models that gave
him 22%", D4's "both articles cite the same model at 20%". They now carry a
`quote` naming a real source, which is a no-op under shadow and leaves the flip
clean: enforcement moves **exactly B5, B6, B9** and nothing else.

Interim by construction: R5's provenance axis makes "who stands behind this" a
field rather than a text scan, and this function should be **deleted** then, not
migrated — do not grow the allowlist into a general-purpose source registry.
