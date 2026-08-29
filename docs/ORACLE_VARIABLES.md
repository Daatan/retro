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
| `claim_strength` | 0..1 | linguistic confidence (0 = hedged, 1 = absolute) — the **source's** commitment to the claim | weight factor AND within-article claim weight. **Renamed from `certainty`** in Oracle 1.5 Phase 1 (retro#680): same elicited number, same elicitation text, a name that no longer also reads as the *reader's* confidence in its own interpretation — that second quantity is a separate field (`reader_confidence`, retro#681). retro#664's Kenya case is the evidence the two were being conflated: an unhedged span scored 0.30 because the reader was unsure, not because the source hedged. `certainty` stays populated as an alias for one schema cycle — on `ClaimDetail` (from `provenance.schema_version` 1.1) and in the pipeline's serialized extraction — so no stored row or consumer moves; the article-level reduction `avg_certainty` keeps its own name |
| `reader_confidence` | `{level: high\|medium\|low, trap: null\|negation\|numeric_comparison\|entity_or_event_mismatch\|tone_vs_content\|inference_needed\|conflicting_signals}`, optional | the **reader's** confidence in its own reading of the span — the other half of the `certainty` split | **EXPERIMENTAL, shadow** (Oracle 1.5 Phase 1, retro#681): populated, persisted, read by nothing. Deliberately not a scalar — verbalised LLM confidence clusters at 0.8–0.9 whatever the input, so a float would record the model's register rather than its difficulty; the trap enum is what a model can actually answer. Each trap name matches a detector that already exists (retro#657 negation, the PR#671 numeric cases, `ab_cases/stance_tone_conflation.json`, the dyad facets), so the self-flag is scoreable against an independent judge from day one. Independent of `claim_strength` by construction — a flat categorical span the reader misread is high strength, low level. Rolls up to `SourceSignal.reader_confidence_level` (the WORST level over the article's claims, not a mean) and `.reader_confidence_traps` (the distinct traps, collected). Phase 4 is where `low` rows are down-weighted and kept out of the credibility bill, and where settlement gains its second bar (`level = high`) |
| `report_kind` | `level` \| `change`, optional | does the quote report the standing **situation** or a **movement** in it | **EXPERIMENTAL, shadow** (Oracle 1.5 Phase 1, retro#686 — unparked from retro#673 §2): populated, persisted, read by nothing. "The rate is 8.75%" and "the rate was cut to 8.75%" carry the same `stance` toward *"above 8%?"* and are not the same evidence — a level restates a state a prior article may already have supplied, a change is new movement, and only the second is news the pool has not already counted. The prompt's disciplining test is semantic, not grammatical (*what would the sentence still tell you a month later*): verb tense reads "held the rate at 8.75%" as a change, which is backwards. Two members, not a scale — retro#673's own caveat is that each new graded field is a fresh site for the retro#394 pathology where a scalar collapses onto its band labels, and one bit cannot collapse. Omitted, not guessed, when the quote is neither (a pure expectation about the future). Per-claim with **no article-level rollup**, unlike `reader_confidence`: Phase 4 E3b reads it per claim — a level report measures the state directly and should reset the recency integrator rather than decay out of it — so an article-level reduction would answer a question nothing asks |
| `quantitative_estimate` | 0..1, optional | cited model/market probability of the event itself (never a vote share/seat count — those are `cited_share`) | overrides stance+certainty via `resolve_stance_certainty` ONLY when `evidence_class=cited_probability` (retro#362); that class carries the 4× premium — and, since retro#369, only if the claim's `quote` names a source on `cited_probability_source_allowlist` (`tm/config.py`); an unattributed figure is demoted by `enforce_anchor_provenance`, which also costs it the rewrite. **Shadow until `anchor_provenance_enforced`.** |
| `settled` | bool | outcome reported as accomplished fact | feeds the ±0.94 settlement pin; a POSITIVE settlement is demoted unless dated — see `enforce_settlement_event_date` below; and a settlement whose `fact_signal` opposes its own stance is neutralised — see `enforce_settlement_fact_signal_agreement` |
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
stance-inversion incident: "Spain beat France" / "Argentina stun England" were elicited as
**+1 settled** for "France/England will win"): in a one-winner contest, a rival achieving the
outcome settles the subject's claim **negatively** — stance −1.0, settled, dated by the
foreclosing result — never +1, however triumphant the article.

Deliberately fails **closed** on a positive settlement's missing date — the asymmetry is the
point: a wrong demotion costs a slower pin (the stance still votes), a wrong settlement sticks
a market at 97% on history. The prompt (SETTLED section) states the same contract, so a
compliant elicitation is never demoted; the guard exists for the non-compliant ones.

The anchor date survives elicitation: each source's `SourceSignal.settlement_event_date`
carries the `event_date` of the highest-certainty settlement-grade claim whose sign matches
the article's collapsed stance (`derive_settlement_event_date`, forecaster.py). Callers
persist it next to `settled` and send it back on `/pool/aggregate`
(`PoolSourceInput.settlement_event_date`).

#### A settlement may not contradict its own fact lane — `enforce_settlement_fact_signal_agreement`

The sign-error class (retro#545): a strong `settled` stance pointing the opposite way from what
the article actually reports. Live flagship — 41 pool rows on the ACTIVE *"Andy Burnham will
REMAIN Prime Minister until 2028"* forecast, every one `stance=-1.00 settled` off articles
reporting that he **took office**, i.e. evidence *for* the claim read as its foreclosure. A
settle-pinned row clamps the published probability to floor/ceiling rather than nudging a
weighted mean, so each of those rows is a wrong number on a public page, and the sources
agreeing with each other is no defence (the same correlated-error hole `settlement_min_sources`
has against a shared narrative).

No second LLM call is needed because the model already tells us it disagrees with itself:
`fact_signal` is the fact-lane counterpart of `stance` on the *same* axis (+1 the facts
establish the event happened, −1 they establish it cannot). A `settled` claim asserts an
accomplished fact rather than a reading of one, so the two must share a sign. When they don't,
one of them is mis-signed and nothing deterministic can say which — so the claim is
**neutralised, not inverted** — `settled` stripped, `stance` zeroed, and
`event=settlement_fact_signal_conflict` logged — the `enforce_winner_entity_consistency` precedent.
`certainty`, `fact_signal`, `evidence_class` and the facets survive, so the row keeps its
weight and stays auditable in `claims_detail`.

Prod audit 2026-08-19 (head rows, complete): 230 settled rows carry a `fact_signal`; **46
oppose their own stance** at |`fact_signal`| ≥ **0.5** (`_SETTLEMENT_FACT_SIGNAL_ANCHOR`),
across exactly **3 ACTIVE forecasts** — 41 Burnham, 3 *"no Arab ministers"*, 2 *"Netanyahu will
be PM on 31 Dec"*. All 46 already sit at |stance| ≥ 0.7, so no separate strong-stance gate is
needed: the caught population is strong by construction. Coverage is effectively complete on
current traffic (96% of settled rows written in the last 30 days carry a `fact_signal`) even
though the shadow field exists on only 58% of settled rows historically — never backfilled, so
this guard is **forward-only** like the `facet`-keyed caps, and the stored 46 are a remediation
question, not a code one.

Fail-open on the same asymmetry as its siblings: not `settled`, no `fact_signal` (legitimately
omitted on opinion/advocacy rows — the null is not a zero), a `fact_signal` below the anchor, or
a zero stance all pass through untouched. Runs **last** in the `enforce_*` chain, so
`enforce_settlement_event_date` has already demoted undated settlements and `enforce_precursor_cap`
has already clamped precursor fact_signals below the anchor — both correctly keeping their rows
out of this net.

#### A named-actor claim landing on a different actor's fact — `audit_named_entity_dyad_mismatch` (log-only)

The wrong-entity class (retro#545, slice ii): a strong-stance claim about ONE specific named
actor, scored against an article whose reported fact is about someone else entirely. Issue
examples: a **Yoaz Hendel** claim scored against an Almog Cohen article (−0.851 @ 0.875), and
separately against an Oren Smadja article in the same pool. Unlike the versus/sports shape
`enforce_winner_entity_consistency` already covers (retro#401), there's no rival to compare
against — just a named actor `event_actors`/`event_target` never mention.

`_extract_named_entities` pulls entity-shaped substrings out of the question with the same
"1–4 capitalized words, minus stopwords" heuristic `enforce_winner_entity_consistency` already
uses (still not real NER). Only the FIRST extracted entity — the question's primary/subject
actor — is checked, not "any" of them: a question routinely also names a location or
organisation ("Yoaz Hendel ... in the 26th Knesset"), and generic nouns like that trivially
co-occur in most same-topic articles, which would mask exactly the mismatch this is meant to
catch. For a claim at `|stance| ≥ 0.7 & certainty ≥ 0.7`
(`_ENTITY_DYAD_AUDIT_STANCE_GATE`/`_ENTITY_DYAD_AUDIT_CERTAINTY_GATE`) with both `event_actors`
and `event_target` populated, if the subject entity appears in neither dyad field,
`event=entity_dyad_mismatch` is logged.

**Log-only — this never mutates `stance`/`certainty`/`settled`.** Coverage and precision on
this shape are both unmeasured: a prod check (2026-08-22) found `event_actors`/`event_target`
populated on only 38% of the `|stance| ≥ 0.7 & certainty ≥ 0.7` band (345/905), and — unlike
the sign-error guard above, a symmetric comparison of two already-reliable fields — a
regex-based single-entity extractor over free-text claims is a new, unvalidated detector. Same
rollout shape as the Gate-0 evidence-window shadow (`evidence_window_outside`): ship as a pure
audit log, review real trigger/precision rate, then decide whether to promote to an enforcing
guard in a follow-up slice.

Fail-open throughout: no entity parses out of the question, either dyad field missing, below
either gate, or the subject entity mentioned on either side of the dyad — all no-op.

**2026-08-24 precision review and fix.** A 244-event sample of live `entity_dyad_mismatch`
shadow logs found ~0% real precision on the wrong-entity class this guard targets. Three
false-positive shapes, all "the exact-phrase `_mentions_entity` check requires more than it
should": **institutional-alias** (`"Donald Trump"` vs `event_actors="Trump administration"` —
the surname is present, the full phrase isn't), **adjectival-form** (`"Israel"` vs
`"Israeli government"` — the word-boundary regex fails on the trailing `-i`), and
**topic-vs-responder** (`"Ebola"` vs `event_actors="WHO"`/`event_target="Africa CDC"` — not an
alias relationship at all; `_extract_named_entities` grabbed a topic noun as the "subject"
instead of an actor noun).

Fixed the first two: the comparison now uses `_mentions_entity_stem` (audit-only, next to
`_mentions_entity`), which additionally matches when the field text contains a word starting
with the entity's last (most distinctive) word — closing both gaps with one mechanism, no
curated alias table. A small closed exclusion list (`_GENERIC_ENTITY_ANCHOR_WORDS`: "party",
"administration", "government", ...) keeps generic multi-word org names from loose-matching on
their common trailing noun, and a 4-character floor keeps short acronyms (US/UK/EU/UN) from
matching unrelated words. `enforce_winner_entity_consistency` keeps the stricter exact-phrase
`_mentions_entity` — this loosening is audit-only.

**Third shape (topic-vs-responder) is explicitly not fixed here** — tracked as retro#644.
Fixing it means changing `_extract_named_entities`'s span-selection, which is shared verbatim
with the *enforcing* `enforce_winner_entity_consistency`; that needs its own dedicated review,
not a bundled fix in a shadow-precision pass. A pinned test documents this as current, known
behavior rather than silently-uncovered.

Also added: one `event=entity_dyad_mismatch_shadow` summary line per call (per article),
logged unconditionally with `eligible=`/`fired=`/`n=` counts, matching the
`evidence_window_shadow` convention below — so a future precision review can compute a real
trigger rate instead of only a raw hit count.

**Still log-only.** This fix does not change the shadow-only status — a fresh precision review
against the stem-matched logs is needed before any promotion to enforcement is considered.

#### Any strong-stance claim's sign-error, not just settled ones — `audit_fact_signal_sign_mismatch` (log-only)

retro#602's follow-up on the sign-error class above: `enforce_settlement_fact_signal_agreement`
only ever looks at `settled = True` claims. This audits the same `sign(stance) != sign(fact_signal)`
disagreement across **every** claim at `|stance| ≥ 0.7 & certainty ≥ 0.7`
(`_FACT_SIGNAL_SIGN_STANCE_GATE`/`_FACT_SIGNAL_SIGN_CERTAINTY_GATE`) with `|fact_signal| ≥ 0.3`
(`_FACT_SIGNAL_SIGN_MAGNITUDE_GATE`, looser than the settlement guard's 0.5 anchor since this
isn't clamping anything).

A 2026-08-23 sweep of `evidence_pool_articles` (COMPLETE, that stance/certainty band, `fact_signal`
populated) found 44/531 (8.3%) rows disagree in sign. Grouped by prediction, 82% of those 44
collapse onto the already-tracked Burnham cluster this guard's settled-only sibling exists for; a
20-row hand-check found 18/20 (90%) genuine sign-inversions, 1/20 borderline, 1/20 a fact_signal
too close to zero to call a real polarity flip — the reason for the 0.3 floor. That ~90% per-row
precision is well above `audit_named_entity_dyad_mismatch`'s ~0% on the same "promote?" question
(not promoted), which is why this one *is* promoted — but only to `event=fact_signal_sign_mismatch`
as a warning, not to enforcement: it exists to catch **new** instances of the defect class before
any decision to neutralise rows the way the settled-only guard already does.

**Log-only — never mutates `stance`/`fact_signal`/`settled`.** Runs immediately after
`enforce_settlement_fact_signal_agreement` so an already-neutralised settled row (stance zeroed)
can't also fire here. Fail-open like every sibling: missing or sub-0.3-magnitude `fact_signal`,
below either gate, or a matching sign — all no-op.

#### author_lean disagreeing with the article's own stance — `audit_author_lean_sign_mismatch` (log-only)

retro#326: the 2026-07-24 PR#314 fix corrected the primary sentiment-vs-forecast leak (an author
who condemns an event while treating it as happening was getting a negative `author_lean` instead
of positive), validated against exactly 3 real article bodies at the time. A 2026-08-25 prod sweep
of author_lean rows added *since* that fix deployed found the leak is broader than tracked: ~26-30
of 1467 rows (~2%) still disagree in sign with the article's own claim-weighted `avg_stance` —
spanning many outlets/bylines, not just the one known "downstream-consequence" residual. Spot-check
example: a hnaftali.com piece explicitly declaring Israel will NOT withdraw from Lebanon
(`avg_stance=+1.0`) still scored `author_lean=-0.9`, its outrage about a separate US-Iran deal
leaking into the author's own directional score.

This audits `sign(author_lean) != sign(avg_stance)` at `|avg_stance| ≥ 0.7 & avg_certainty ≥ 0.7`
(`_AUTHOR_LEAN_SIGN_STANCE_GATE`/`_AUTHOR_LEAN_SIGN_CERTAINTY_GATE`) with `|author_lean| ≥ 0.3`
(`_AUTHOR_LEAN_SIGN_MAGNITUDE_GATE`) — the same gate shape and thresholds as
`audit_fact_signal_sign_mismatch` above, since that guard's 90%-precision sizing is the only
precedent in this codebase for where to set this kind of bar; several real leaks in the 2026-08-25
sweep sat at `avg_stance` 0.3-0.66, below this gate, so the 0.7 bar trades recall for the same
precision discipline rather than a fresh sizing pass — a broader hand-check to loosen it (the way
retro#602 did with a 20-row sample) is a natural follow-up, not a claim that those are non-leaks.

Runs only on the **live** `/forecast` path (`forecaster.py`, right after `avg_stance` is computed)
— that is the path that actually populates daatan's `evidence_pool_articles`, unlike the batch/
atlas pipeline in `runner.py` which has no per-article stance aggregate to compare against.
Genuine author/fact disagreement (the author's own claims also read negative, so `avg_stance` is
negative too) is left alone by construction — only a *disagreement* between the two signs fires.

**Log-only — never mutates `author_lean`/`author_lean_certainty`/`stance`.** Fail-open like every
sibling: a null or sub-0.3-magnitude `author_lean` (no position taken — most reporting), or below
either gate, is a no-op.

#### A fabricated quote — the extracted text is the event, not the article — `audit_quote_provenance_mismatch` (log-only)

retro#545's 2026-08-25 cross-model extractor survey (700 matched-quote comparisons, 78
model-pairs × 50 real prod articles) surfaced a class distinct from the sign-error and
wrong-entity shapes above: in 2 of 10 flagged articles, the extracted `quote` field was
verbatim the event's own `event_name`/`event_description`, not text pulled from the article.
Two clean tells confirmed it wasn't coincidence — the article had nothing to do with its
assigned event (a Beitar Jerusalem soccer-ban article scored against "Global oil price drops
below $70/barrel"), and the "quote" was the event description restated, not a sentence from
the piece. No guard checked quote provenance at all before this: `extract_predictions`'s
prompt tells the model to quote verbatim, but nothing verified compliance.

Deliberately narrow: compares `quote` against `event_name`/`event_description` only — after
casefolding, whitespace-collapsing and punctuation-stripping
(`_normalize_for_provenance_compare`) — not a quote-vs-`article_text` substring check. The two
known real examples were exact restatements of the event, which this catches precisely; a full
article-body check would need translation- and extraction-artifact-aware normalization to avoid
drowning in noise, deferred until this narrower signal's precision is measured. A short-quote
floor (`_QUOTE_PROVENANCE_MIN_LEN`, 20 chars) skips anything short enough that an on-topic
quote could coincidentally overlap the event text without being fabricated.

**Log-only — never mutates `stance`/`claim`/anything else.** `event=quote_provenance_mismatch`
fires per match; `event=quote_provenance_mismatch_shadow` logs `eligible=`/`fired=`/`n=` once
per call regardless of outcome, same convention as the guards above, so a future precision
review has a real denominator. Precision on this shape is unmeasured (only 2 known examples) —
same rollout shape as the other audit-only guards: ship the shadow log, review real
trigger/precision rate, decide on promotion in a follow-up slice.

#### Claim/stance sign conflicts are logged, not corrected — `flag_claim_stance_sign_conflicts`

retro#298 found rows where the extracted `claim` text and `stance` disagree with each other in
the same row — e.g. a claim stating a withdrawal *"is mandatory"* scored stance **-0.136**. The
general case ("does this stance follow from this claim") needs a second LLM call or a verifier
stage — out of scope here. `flag_claim_stance_sign_conflicts` (`extractor.py`) is the issue's own
"cheap partial": a deterministic marker check (`is mandatory`/`must`/`is required` vs.
`will not`/`refuses`/`rejects`, etc.) that logs `event=claim_stance_sign_conflict` when a claim's
explicit marker and its stance sign disagree. Runs once, right after elicitation, before any of
the guards above can touch `stance` — **observability only, never corrects a prediction**. It is
narrow by design: literal marker clashes only, so it misses subtler mismatches (a demand read as
adversarial when it is actually a climb-down, retro#298's own row 6451).

#### Aggregation-time revalidation — `settlement_vote_validity`

Elicitation-time guards only protect fresh elicitations; a recompute replays stored `settled`
bits written before the guards existed or re-poisoned since (the 2026-07-16 audit: 11 of 19
pins wrong; re-pushes re-flipped cleaned flags within hours). With `settlement_revalidate`
(default **on**; env kill switch `SETTLEMENT_REVALIDATE=false` + service restart), every
settlement vote re-proves its anchor inside `aggregate_pool()` on every call — live
`/forecast` and `/pool/aggregate` alike:

- **Occurrence-direction vote** (arrival:+, survival:−, unclassified:+): must carry a
  parseable `settlement_event_date`; not after `claim_deadline`; not before
  `claim_created_at` — **every archetype** since 2026-08-16, previously `'scheduled'` only
  (a dated fact from before the claim existed, both signs: the 2021/2022-article class, and
  the 2026-08-16 audit's survival/unclassified leaks — the Putin claim pinned by 2024-election
  articles, a 2022 maritime-deal story re-pinning the 2026 Lebanon claim; strict `<` at date
  granularity, so a creation-day event still settles, and fail-open when `claim_created_at`
  is absent); not after its own article's date.
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
  articles about active 2026 strikes elicited as an undated NO for the already-closed 2025
  window). The extractor prompt already forbids cross-timeframe elicitation (retro#295); this
  check is the aggregation-time backstop for rows that slip through it — keyed on
  `published_date` rather than `event_date`, since there is no event date to anchor on. An
  undated non-occurrence vote from an article published within grace of a closed window is
  unaffected — that stays the ordinary, honest "window closed quietly" case.

Demoted votes keep their stance (ordinary evidence; `event=settlement_vote_demoted` with a
reason per row — plus, since retro#554, the audit fields that tie a demotion back to its
forecast: `question=` (question hash), the claim-window bounds actually compared against
(`created=`/`deadline=`), and `event_date_state=absent|unparseable|parsed`, which separates an
article that genuinely carried no date from one whose date string failed ISO parsing). Valid
votes in **both** directions suppress the pin entirely
(`settlement_suppressed`, `settlement_conflict` — one elicitation is provably wrong, and
facts are not decided by outvoting; the England 4-vs-1 stance-inversion pool is the
canonical case). The pin then requires `settlement_min_sources` **unanimous** valid votes.

**Evidence window** (retro#545 slice iii, Gate-0 decision 2026-08-19, **enforced 2026-08-26**):
separately from the settlement rules above — which demote *votes* — the *estimation* evidence
window for non-`scheduled` archetypes is `[claim_created_at − evidence_window_lookback_days,
claim_deadline]` (previously effectively `(−∞, deadline]`). Rows dated outside the window are
**excluded from the pooled estimate** (weight zeroed — still counted in `n`, still logged as
`event=evidence_window_outside`, reason `before_window`/`after_deadline`; a row's date is its
`settlement_event_date` when parseable, else `published_date`, and undated rows are skipped, fail
open). Shadow-only from 2026-08-19 through 2026-08-26 per the decision ("enforce only after the
shadow numbers are reviewed"); the review was a scoped election-pool sweep on 2026-08-26 (127 of
2,057 recent evidence rows across 17 forecasts fell outside the window, driven mostly by a
2026-08-24 full-pool re-extraction that re-validated topicality but not the evidence date — see
retro#545 comment). The settlement lane's own, separately-shipped temporal check
(`event_before_claim_window`, the 2026-08-16 decision) is unaffected: this zeroing happens before
`settlement_decision` and only removes a row's contribution to `settlement_quality_floor`'s mass
accounting, never a vote's validity. The lookback (default **30**, `EVIDENCE_WINDOW_LOOKBACK_DAYS`,
negative disables) is the tunable knob: it keeps the precursor/trend coverage that makes a young
forecast estimable while catching the adjacent-event class (an earlier, similar incident counted
as evidence for the forecasted one — the Baltic-drone case, where the incident the claim was
anchored on settled it at 97%).

Clearing `settlement_min_sources` is a **count**, not a quality check — a pool of uniformly
weak sources (low credibility, thin relevance, recency-decayed) that each barely clear
settlement grade could still out-count its way to a pin. `settlement_quality_floor`
(retro#279/#372, **0.20 — enabled 2026-08-02**) additionally requires the winning direction's
votes to carry at least this much *combined weight* (`credibility × evidence_weight × recency ×
relevance²` — the same per-source `weight` term the pool itself uses, summed over the
winning direction's valid votes only) before the pin is honored; below it the pin is
suppressed too (`suppression_reason="settlement_quality_floor"`) and the pooled mean stands,
same as any other suppressed pin.

**Where 0.20 comes from.** Every pin production has ever published (the latest pinning snapshot
per prediction, 29 of them; 2 excluded as unmeasurable because their stored rows predate
`evidence_weight`/`relevance_score` persistence), with the winning-direction weight
reconstructed from the stored snapshot and recency recomputed from `published_at` against the
snapshot's own timestamp:

| min | p25 | median | p75 | max |
|---|---|---|---|---|
| 0.022 | 0.28 | 0.60 | 1.65 | 12.2 |

0.20 suppresses **3 of the 27 measurable pins**, and the cut lands in a real gap (next pin up:
0.22). Each suppressed pin is indefensible on its face: one settled on articles published in
**2021**, one on a single settled vote in a one-row pool, one at 0.171. What it does **not**
buy, stated so the number isn't oversold: it suppresses none of the three pins the **Oracle**
got wrong (0.432, 2.292, 12.245) — mass was not what was wrong with them; sign, subject and
timeframe were (retro#360, #388) — and it suppresses retro#388's live pin at the snapshot where it fired
(0.134) but not permanently, since that pool's settled mass had grown to 0.297 by the latest
snapshot. A fixed floor delays a wrong pin that keeps accumulating corroboration.

The earlier note here said 0.5 "broke multiple legitimate-pin tests". That turned out to be a
fixture artifact, not a property of the floor: those tests hard-coded article dates in the
fixed past, so they decayed further toward the recency floor every day the suite aged and were
by then asserting that month-old coverage may pin. Their dates are now relative to the run
(`_FRESH` in `test_settlement*.py`, `test_pool_aggregate.py`), which is what they always meant.

`PoolAggregateResponse` exposes `settlement_suppressed`/`settlement_suppression_reason`/
`settlement_votes_demoted` for callers. Regression fixtures from the audit:
`api/tests/test_settlement_revalidation.py`; quality-floor fixtures:
`TestQualityFloor` in the same file.

#### The residual the floor does not close — `verified=null` pins (retro#449)

The floor scores **mass**, not provenance, so a pin carried entirely by claims the extractor
never independently verified clears it on weight alone. F20's `enforce_interested_party_stance_cap`
is the only thing that touches the `verified` flag, and it keys on `verified is False` —
`None` is untouched. Fixture case **B21** is the shape: two moderate-credibility reports,
`verified` absent, independently phrased (so they form two clusters rather than one echo),
summing to 0.656 against a 0.20 floor. It pins at 0.94 and is tagged `known_bad`.

**Stage A instrumentation** (`event=settlement_vote_weight`, retro#449/PR#515, live since
2026-08-10) logs weight/credibility/verified for every settlement vote on both the live and
recompute paths. Its first measurement — **987 votes across 176 forecasts, 2026-08-11/12** —
established four things worth not re-deriving:

| Measured | Value |
|---|---|
| Settlement votes with `verified=false` | **0 of 987** — F20's clamp keeps them below settlement grade entirely |
| Settlement votes with `verified=null` | 523 (53%) · mean weight 0.0435 · median 0.0269 · max 0.2153 |
| Forecasts pinned on `verified=null` evidence **only** | **0 of 176** (the 6 null-only vote sets all sum ≤ 0.0193, ~10× under the floor) |
| Live pins removed by downweighting null votes ×0.5 / ×0.75 | **0 of 78** — nothing sits near the floor |
| Live pins removed by excluding null votes outright | **22 of 78 (28%)**, nearly all legitimate |

Note the 53% null rate is a *current-traffic* figure; the frequently-quoted 87% is over the
historical pool, which is dominated by rows predating 2026-07-09, when the flag started being
written (never backfilled).

**No threshold ships, by decision (2026-08-12).** The measurement closes the question in both
directions: a graded discount is inert, and the only mechanism that bites suppresses legitimate
pins at 28%. Calibrating a number here would be fitting to an empty cell. The exposure is
instead **watched**: `event=unverified_only_pin` (WARNING, both paths, emitted *after* the match
gate so a pin that never shipped cannot raise it) fires on the first prod instance. The real fix
belongs to Phase 2 / R7 — settlement decided once at claim level — rather than to another
`verified`-keyed patch on an article-level rollup.

**The settlement match gate** (retro#388/#360, `api/src/forecast_api/settlement_verifier.py`,
applied in `_apply_settlement_match_gate`). Everything above is arithmetic and temporal: it
counts votes, re-proves anchors, weighs mass. None of it can ask the one question that both
documented pin failures turned on — *are these facts this claim's own outcome?* The gate is one
LLM call, made only when a pin is about to fire, that decomposes the question into who acts,
what action, and within what scope, and answers NO when the facts settle it the other way, when
a different party acts or the action lands on a different target, when the action was announced
or agreed but not carried out, or when the fact belongs to a different instance or timeframe of
a recurring event. Where a claim's summary and its quoted sentence disagree it believes the
**quote** — the summary is a paraphrase by the same elicitation step that may have misread the
sentence. The pin's **direction** is part of what is asked, because facts that decide a question
*against* the answer about to be published are not proof of it however clearly they decide it.

Why semantic rather than a field comparison: the check #388 originally proposed — demote a
settlement vote whose `event_actors`/`event_target` miss the claim's — would have fired on
**neither** incident. Both Patriot rows carry `event_actors="United States"`,
`event_target="Ukraine"`, the claim's own entities; the England–Argentina rows name both teams
correctly. The question has four slots (who, what action, what **aspect**, what scope) and the
extractor emits two. The deterministic end state — per-claim aspect and role as extractor shadow
fields, enforced in code the way `enforce_precursor_cap` enforces F9 — is unchanged and still the
target; this is the net that can be measured against known ground truth today.

Two flags: `settlement_verifier_enabled` runs it and logs the verdict,
`settlement_verifier_enforce` lets that verdict act. It shipped enabled-but-shadow on
2026-08-03 so the enforcement decision could be made on measurement rather than on the design
argument, and **`enforce` was turned on the same day** on this evidence
(`scripts/replay_settlement_verifier.py`, every pin production has ever published — 33 at the
time, 0 errored):

| | pins | gate keeps | gate vetoes |
|---|---|---|---|
| known outcome | 5 | 2 | 3 |
| still active | 27 | 16 | 11 |

On the five with ground truth it is **5 for 5**, scoring against whether the *Oracle's pin* was
right rather than whether the prediction resolved true — a NO pin on a claim that resolved wrong
is a pin the Oracle got right. It vetoes all three the Oracle got wrong (France winning the World
Cup and England winning their semi-final, both settled by reports of the **loss**; "Will USA bomb
Iran in 2025?", settled by bombing in **2026**) and keeps both it got right (Messi, the Knesset
dissolution).

All 11 vetoes on active pins were reviewed individually and each is defensible; they fall into
four groups. **Wrong instance or timeframe** — a pin about the *next* Israeli government settled
on a 2021 coalition agreement, one about the *next* general election settled on the 2022 one.
**Wrong actor or scope** — nine European countries and Ukraine announcing an initiative read as
*the European Union* announcing it; an earthquake that struck Egypt, and a second report of the
same tremor whose Hebrew quote says only "the Dead Sea region", read as an earthquake inside
Israel's internationally recognized borders (the quote rule doing exactly its job: that row's
summary asserts the border claim its own quote does not); a settlement in the West Bank read as
one in Lebanon. **Announced, not carried out** — the F-35 sale to Turkey, settled on "plans,
reviews, openness and hints"; the formal end of the US–Iran conflict, settled on a memorandum
while strikes continued. **Present tenure read as a future outcome** — three pins (Netanyahu on
31 Dec 2026, Burnham until 2028, Putin through May 2027) settled on facts establishing only that
the incumbent holds office *now*. The last group is where the gate is strictest, and it is also
where being wrong costs least.

That asymmetry is what makes enforcement the safer setting, not the riskier one. **A veto is a
demotion, not a deletion**: enforcement re-runs the *same* `aggregate_pool` with the vetoed rows'
`settled` flags cleared, so those rows keep voting as ordinary evidence and the published number
is still one the pooling code produced (a recompute over the stored pool reproduces it). A false
**pin** publishes a confidently wrong 97% over a pool reading 44% — the documented harm. A false
**veto** publishes the pooled estimate instead: less confident, not wrong. Every failure path is
fail-**open** — unreachable model, timeout, unparseable reply → `errored`, never a veto — so an
LLM outage cannot silently change published numbers. Because of that, a totally broken replay
(bad model id, no credentials) prints a well-formed table of zeros that reads like "the gate
vetoes nothing"; the script now exits non-zero and says so rather than letting the absence of a
result be quoted as one.

**Decided once, remembered (retro#532).** The verdict is an LLM judgment, and it is not
idempotent even at `temperature=0`: of the 13 questions the gate had seen more than once by
2026-08-14, **6 returned both verdicts on an unchanged vote-set** (the US–Saudi sequence: four
NOs at an unchanged 2-vote set, then a YES ninety minutes later — the YES is what published
97%). Combined with daatan's one-way `settled` latch, re-rolling per recompute was a ratchet:
a question the gate vetoes 31 times out of 32 still pins permanently on its one lucky roll.
Since retro#532 the gate therefore asks once per *input*, not once per recompute. A first
decision samples the model `settlement_verifier_votes` times (default 3, keep it odd) and takes
the majority; the result is stored (`settlement_verdict_store.py` — diskcache under `data_dir`,
cross-worker, survives deploys) keyed on the **built prompt** (question, direction, votes,
prefix text — so editing the prompt invalidates naturally) plus model, sample count and the
settlement-config fingerprint. Both directions are sticky: a YES and a NO are equally durable
until the vote-set, config, model or prompt changes — determinism is the property being bought,
and a veto that could be out-rolled by recomputing until it flips would be no latch at all.
Never stored: **errored** verdicts (fail-open must stay transient — caching one would turn a
timeout into a permanent pin-keeper) and **undecided rolls** (a sample errored or the decided
samples tied; the roll stays fail-open for that recompute and is re-taken in full next time).
Kill switch `SETTLEMENT_VERDICT_CACHE_ENABLED=false` restores the legacy roll-every-time
behaviour without a deploy; deleting the store directory on the box is the manual invalidation
lever. The `event=settlement_verifier` log line carries `cached=`, `samples=` and `agree=` so a
hit, a fresh majority roll and a degraded roll are all distinguishable in the log.

**Coverage caveat.** The gate needs the claim text and per-claim `claims_detail`. On `/forecast`
it has both (`question` is required; `claims_detail` comes from the in-process elicitation), and
that is the path that publishes pins. On `/pool/aggregate` both fields are optional and daatan
sends **neither**, so the gate skips explicitly (`outcome=skipped reason=no_question`) rather
than guessing from the rows — every live firing since deploy has been such a skip. That path is
shadow-compare and publishes nothing today, so enforcement is unaffected; it must be closed
before the recompute-over-pool cutover in §6/§8, or enforcement will be silently half-covered
exactly when the recompute becomes a writer. Fixtures: `api/tests/test_settlement_verifier.py`
(prompt payload, parsing, shadow, enforcement, the three skip paths). Note the suite sets
`SETTLEMENT_VERIFIER_ENABLED=false` in `conftest.py` so no test run depends on Bedrock.

### 2.2 Per-article (gatekeeper LLM — `pipeline/src/tm/gatekeeper.py`)

| variable | scale | role | notes |
|---|---|---|---|
| `is_prediction` | bool | hard in/out gate | binary form of the same judgment as `relevance_score` |
| `relevance_score` | 0..1, default **1.0** when `is_prediction=true`, **0.0** when `is_prediction=false` | graded "how much would a forecaster update" | default is is_prediction-dependent (retro#524) — an unscored rejection can never read as "relevant" to a caller that checks only this field; an explicitly graded score (e.g. a 0.1 near-miss on a rejection) always passes through unchanged; squared downstream |
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
contained. **Reuse is allowlisted by caller** (retro#536, 2026-08-15):
`relevance_reuse_allowed_clients` (default `"default"`, the primary `ORACLE_API_KEY` that
the daatan backend uses) names the API-key clients whose supplied verdict may be reused.
`reuse_supplied_relevance` only ever said *reuse is on*, never *whose verdict* — so any
holder of any valid key could skip claim-aware judging for its own requests by setting
`relevance`/`is_prediction` on the request body. A non-allowlisted caller's verdict is
dropped at the API boundary (`event=supplied_verdict_dropped`) and the article is judged
normally: fail-safe, never a 4xx. Bare domains (`ynet.co.il`) are deliberately **not** stripped: the pattern that
catches them also eats ordinary abbreviations, and over-rejection here silently loses a
curated journalist's scoop, which is the failure news-indexer's rescue path exists to undo.

### 2.3 Per-source derived (`reduce_article()` in `api/src/forecast_api/forecaster.py`)

Every row below is a reduction over the article's `claims_detail` — one pure
function, so the same reduction can be replayed over a persisted pool row
(F1 item 3, retro#364). Before that the scalars were computed over the
in-memory extraction list and the claims were projected onto the wire
separately: one source, two computations, free to drift.

| variable | formula | notes |
|---|---|---|
| `avg_stance` | certainty-weighted mean over claims; settled claims *replace* the set | the source's vote. Per-claim, `stance` is clamped in code by two location-side guardrails before fusion, sign always preserved: `interested_party_stance_cap` (`tm/config.py`, **0.3**) on `verified=false` claims — an interested party's unverified assertion (retro#368, F20) — and `decider_intent_stance_cap` (**0.3**, retro#518) on `is_occurrence=false` claims with facet `announcement`/`denial` — a decider's own stated future intent, which the fact lane already capped at ±0.3 (`enforce_precursor_cap`) while the stance lane voted it at full magnitude (prod audit 2026-08-15: 71 of 119 such rows above 0.3, to \|0.85\|). The second cap is **forward-only**: `facet` ships 2026-08-10, never backfilled (1.9% of stored rows), so legacy rows age out. Deliberately a separate constant from the fact-lane `decider_statement_*_cap` knobs reserved for retro#486's refit. |
| `avg_certainty` | plain mean over **all** claims (incl. non-settled color quotes) | inconsistent with `avg_stance`'s settled-replacement. Per-claim, `certainty` is now capped in code at `interested_party_certainty_cap` (`tm/config.py`, **0.5**) whenever the extractor marked the claim `verified=false` — `enforce_interested_party_certainty`, the weight-side half of the interested-party rule (retro#378), in the `enforce_*` chain right after the stance half (retro#368). Unlike that cap, this number is the **prompt's own literal**, which nothing had enforced: 30.3% of live `verified=false` rows exceeded it (56/185, max 0.733, ten at exactly 0.70 with avg \|stance\| 0.76) — and that is a floor on the per-claim rate, since the stored value is the article-level reduction while `verified` is the dominant claim's. **Where it actually bites is narrower than "certainty is a weight" suggests**: `evidence_class_weight()` ignores certainty for any *classified* claim and the unclassified branch is already capped at 0.25 (F10/R3), so the pool weight does **not** move. What moves is (a) the within-article fusion — `claim_weighted_stance` weights claims by certainty, so an over-confident unverified claim now pulls its article's `avg_stance` and `fact_signal` less (matrix A20) — and (b) the number itself, which is persisted, carried on the pool wire, and is the weight fallback in `run_pool_aggregate` for legacy rows with no `evidence_weight`. It also matters ahead of R1, where per-claim certainty becomes the claim weight directly. |
| `rweight` | `0.5^(age/7d)`, floor 0.02; missing date ⇒ 1.0 | fail-open |
| `credibility` | leaderboard lookup | **1.0 for every source observed in prod** — layer currently inert |
| `quantitative_multiplier` | 4.0 if any claim carries an estimate, else 1.0 | stacks with the certainty-0.9 floor |
| **`weight`** | `credibility × avg_certainty × rweight × relevance_weight(relevance) × quant_mult` | pool weight. **`relevance_weight` is a lookup table, not an exponent (retro#394).** The gatekeeper does not emit a graded score — across 84,254 judgments it emits the edge labels of the prompt's own bands, with **exactly zero mass in (0.60, 0.70)**, so `relevance²` was arithmetic on a categorical value. In the live daatan pool 51.9% of voting rows sit at exactly 0.70 and 25.2% at 0.80, making this a **three-position switch, not a continuous dial** — and the 1.31× ratio between them was whatever squaring produced, not a choice. `RELEVANCE_BAND_WEIGHTS` (`aggregation.py`) is initialised to exactly `band²`, so nothing has moved yet; it exists so the numbers can be *chosen* once there is outcome data to choose them with (as of 2026-08-04 only 6 resolved BINARY forecasts have a usable pool — see retro#393). Off-band values still fall back to squaring. |
| `fact_signal` (shadow) | claim-weighted **mean** of per-claim `fact_signal` over the **same** scored claims as `avg_stance`; `None` if none carried one | Phase 2 fact-lane counterpart of `avg_stance`, un-fused from author assertion; **read by nothing in aggregation** — surfaced on `SourceSignal`/`sources[]` only for daatan persistence + the offline fact-lane gate harness (`pipeline/scripts/backtest_fact_signal_gate.py`: stance-vs-fact_signal paired Brier through the real `/pool/aggregate`). **Status honestly stated (retro#533, decided 2026-08-15): this is a diagnostic/guardrail lane, not a pricing lane in waiting.** The lanes are not independent in practice — corr(stance, fact_signal) 0.905 on `is_occurrence=false` rows (n=2,645), 0.73 whole-pool, with exactly one resolved prediction showing >0.1 divergence — so the paired-Brier gate structurally cannot accumulate discriminating data and the estimator-cutover framing is retired (`Daatan/docs/decisions.md` 2026-08-15). What the lane actually does, and keeps doing: precursor cap (retro#367), the decider-intent stance cap's key (retro#518), settlement gating, and the per-claim audit surface. Re-opening bar: ≥30 resolved predictions with lane divergence >0.1, arising organically from a future extractor change — then re-run the harness. Extraction-side, the FACT_SIGNAL prompt carries a **decider-statement exception** (2026-07-29, A/B-gated): an on-record statement by the actor/authority whose own act would resolve the claim — announcement or denial alike — enters the fact lane as a capped precursor instead of being nulled as opinion; assertions *about* the decider's intent by opponents or analysts stay claimed-and-unverified. A companion **negative-precursor ladder** (2026-07-29, WS5b, A/B-gated) generalizes the negative side beyond decider statements: any contrary reported fact — an obstacle emerging, a preparation reversed, an opposing development, a measured indicator moving against the event — enters the fact lane as a graded negative precursor instead of null, with the extreme negative reserved for established impossibility; this closes the measured null asymmetry (negative-stance rows nulled 33.0% vs 22.6% for positive, fact-era pool) while the mobilization regression keeps deflating (see `test_extractor_prompt.py::test_negative_precursor_ladder_present` for the A/B record). Wording is numeral-free by design (magnitude policy belongs in estimator config); see `test_extractor_prompt.py::test_decider_statements_exception_present` for the A/B evidence and re-run bar. Its facets `event_actors`/`event_target`/`is_occurrence`/`verified` ride from the **dominant** (max \|fact_signal\|) claim so they stay internally coherent. Magnitude for a **precursor** is enforced in code, not by the prompt (retro#367): `enforce_precursor_cap` clamps per-claim \|`fact_signal`\| to `fact_signal_precursor_cap` (`tm/config.py`, **0.3**) whenever the extractor set `is_occurrence=false`, in the `enforce_*` chain immediately before this fusion — so both the mean and the dominant-claim selection see the capped value. |
| `author_lean`, `author_lean_certainty` (shadow) | passed through from `ExtractionOutput` (retro #308/#309) — the byline author's OWN forecast | author-accuracy scoring lane; **not read by aggregation** |
| `consensus_view` (shadow) | passed through from `ExtractionOutput` (retro#686) — `expects_yes` \| `expects_no` \| `divided`, what the **article says OTHERS expect** | **EXPERIMENTAL, shadow**: populated and on the wire, read by nothing — and, unlike `report_kind`, **not yet stored**. `claims_detail` is a `Json` column in daatan, so a new per-claim field is persisted the moment the Oracle emits it; a per-source field needs its own column (`EvidencePoolArticle.authorLean` is the precedent) and that column does not exist yet. Until it does, the only place this field can be observed is a live `/forecast` response and the A/B harness's article-level fill. Sits beside `author_lean` because it is the same shape of question one step out — `author_lean` is what the byline thinks, this is what the byline says everyone else thinks — and the two must be able to disagree (*"analysts expect a cut, but this is wishful thinking"* is `expects_yes` with a negative `author_lean`). Its kill criterion is exactly that confusion: >20% of non-null rows carrying the model's own view rather than the article's. Phase 3 S2's shared-information detector is the consumer — the **gap** between a source's stated consensus and the pool is the shared component (Palley & Satopää 2023), and a pool of sources all reciting the same consensus is one observation, not N |
| `claims_detail` | no reduction — the article's claims themselves, projected onto `ClaimDetail` (`build_claims_detail()`) | **F1/F15, retro#364.** Every other row in this table is a reduction; this is the layer they reduce *from* — literally: `build_claims_detail()` runs first and `reduce_article()` takes its output as input, so the persisted claims are the reduction's argument rather than a copy taken alongside it. Until this layer existed the inputs were discarded at the wire, so no reduction here was checkable and no history was re-scorable. Recorded POST-resolution — after the `enforce_*` chain and `resolve_stance_certainty()` — i.e. the values the fusion actually consumed. `test_claims_detail.py` pins each derivation individually *and* replays `reduce_article()` over the persisted claims alone, asserting it reproduces every scalar of the signal it produced (ordinary, settlement, and demoted-settlement paths). Per claim: `claim`, `quote`, `stance`, `certainty`, `specificity`, `prediction_type`, `evidence_class`, `quantitative_estimate`, `settled`, `event_date`, `fact_signal` + its four facets. Two collapses become visible only here: `evidence_class` is per-claim (the article carries only the most common one) and the fact facets are per-claim (the article carries only the **dominant** claim's), which is why an over-cap interested-party claim diluted by in-contract siblings is invisible above this layer (retro#378). Unlike `claims`, nothing is filtered — a claim with an empty summary still voted, so it is still kept. **Read by nothing in aggregation**; persistence surface only (daatan#1235), same shadow-field rollout as `author_lean` and `fact_signal`. |

### 2.4 Pool level (`aggregation.py`, `forecaster.py:799-932`)

| variable | role |
|---|---|
| `mean, std, ci_low, ci_high` | weighted-mean logit pool + dispersion, stance scale; SEM divides by Kish `n_eff = (Σw)²/Σw²`, and the width is floored at `1.96·pool_dispersion_floor/√min(n_eff, k)` so a unanimous pool cannot publish a point (F16). The `k = Σ min(wᵢ, decisiveness_floor)/decisiveness_floor` term (retro#382) keeps equal-weight row volume — N identical low-mass rows, where Kish `n_eff` = N exactly — from shrinking the floor on multiplicity alone; each statistic overcounts in the failure mode the other catches (`k` alone overpays a one-heavy-row pool ~22%), so the divisor takes the min. Matrix case C15 pins it |
| `evidence_mass = Σ weight` | thin-evidence CI widening (floor 0.5, inflation 0.45) |
| `n_eff`, `age_adjusted_mass` (retro#458 Phase 2, reporting-only) | `n_eff` is Kish's `effective_sample_size(weight)` — the *exact* call the CI floor's `floor_n` divisor already uses, computed once and reused rather than twice. `age_adjusted_mass` is `evidence_mass` recomputed with `rweight` forced to 1.0 (`credibility × avg_evidence_weight × relevance_weight(relevance)`, no recency term) — "how much would this pool weigh if nothing had aged," always ≥ `evidence_mass`. Both are exposed on `ForecastResponse`/`PoolAggregateResponse` alongside `evidence_mass` (previously computed internally but never returned to callers); **neither feeds back into `mean`/`std`/`ci_low`/`ci_high`** — visibility only. Still populated on an abstained/off-topic result, mirroring `evidence_mass`/`thin_evidence`. |
| `relevance_mass = Σ relevance²` | off-topic abstention (floor 0.05). **Deliberately still the raw square, not `relevance_weight`** (retro#394): this asks a different question — *is the whole set off-topic* — and its 0.05 floor was tuned against Σ`relevance²`. Routing it through the band table would silently retune the floor the moment those weights are changed. If the band weights are ever retuned, revisit this floor in the same commit. |
| `settled_directions → settled` | settlement pin ±0.94 when ≥2 valid votes agree — the count is over **independent clusters**, not rows, whenever a cluster assignment exists (retro#372: two syndicated copies of one report are one observation; rows without claim text stay singletons, so a missing claim layer never costs a vote; matrix case C19). **Revalidated per vote** (`settlement_vote_validity`, default on): an occurrence-direction vote needs a parseable `settlement_event_date` within `[claim_created_at, claim_deadline]` (creation lower bound on every archetype since 2026-08-16) and ≤ its article's date; a non-occurrence vote needs a closed window (dated anchors at most `settlement_post_deadline_grace_days` past it, or an undated vote from an article published within that grace — else `stale_undated_foreclosure`) or a dated in-window foreclosure. Valid votes in BOTH directions ⇒ pin suppressed (`settlement_conflict`) — unanimity, not majority. Count alone isn't enough either: `settlement_quality_floor` (**0.20** since 2026-08-02, retro#279/#372 — calibrated against every pin production had published; 0 disables) additionally requires the winning direction's combined per-source weight to clear a bar, else the pin is suppressed (`settlement_quality_floor`). Kill switch `SETTLEMENT_REVALIDATE=false` restores flag-trusting majority vote + `settlement_direction_allowed`. |
| `cluster_ids → weight × k^-exponent` | correlated-evidence discount (retro#355). Pool rows echoing one development are grouped by shingle-Jaccard over their `claims_detail` text (`clustering.py`), and each member of a cluster of size `k` is scaled by `k^-exponent`, so the cluster carries `k^(1-exponent)` rows' worth instead of `k`. **`cluster_downweight_exponent=0.0` ships it inert** — the identity — so nothing has moved. Applied FIRST, before `evidence_mass`, so the decisiveness floor is judged on independent mass rather than on echo. `relevance_mass` is deliberately NOT discounted (same reasoning as the band table above). |
| `insufficient_data, reason, placeholder, articles_used/found` | abstention encoding |

Config constants (16): `recency_half_life_days=7`, `recency_floor=0.02`,
`logit_clamp=0.01`, `relevance_weight_floor=0.05`, `forecast_relevance_bar=0.0`,
`cluster_downweight_exponent=0.0`, `cluster_jaccard_threshold=0.40`,
`cluster_shingle_size=3`,
`syndication_title_similarity=0.8`,
`decisiveness_floor=0.5`, `thin_evidence_ci_inflation=0.45`,
`defer_on_thin_evidence=False`, `pool_dispersion_floor=0.05`,
`settlement_min_sources=2`,
`settlement_stance=0.94`, `min_certainty=0.9`.
`logit_clamp` is not only the per-source log-odds guard: it also bounds the
pooled CI endpoints and (since F16) the thin-evidence widening term.

`forecast_relevance_bar=0.0` **means no per-article relevance bar on `/forecast`, which is
exactly what this path has always done** — an article was dropped only on `not is_prediction`,
and its graded score then went straight into `weight`. That made the bar an *entry-path*
property rather than a verdict property: news-indexer's rescue path requires
`relevance_score >= 0.7` before delivering, so an article the **same judge, same model, same
prompt** scores 0.30 was retired permanently if it arrived via rescue and **voted** if it
arrived via a cosine push, a retry, or on-demand search. Measured on daatan prod voting rows:
**1,186 of 5,827 (20.4%) below 0.7**, 220 at ≤0.40; by origin news-indexer 18.1%, retry
**45.5%**, analyze 42.1%. The gatekeeper prompt even delegates explicitly — *"When in doubt,
PASS — the graded relevance_score below handles weak or loose signal"* — to a threshold that
existed only in the other repo.

Making it a setting **defaulted to current behaviour** changes no forecast, and buys three
things: the number lives in one repo instead of none, raising it is a config change rather
than a code change, and the effective bar is returned as `relevance_bar` on every response so
a caller persisting the sources can record which admission regime produced each row and filter
its pool retroactively (retro#393 option (b)). Raising it to 0.7 would cut 20.4% of the voting
corpus and is **deliberately not done here**: the backtest that would justify it is not
powered — as of 2026-08-04 only **6** resolved BINARY forecasts have a usable evidence pool.
Note also that the score has zero mass in (0.60, 0.70], so every bar in that interval is the
identical filter (retro#394).
`cluster_downweight_exponent=0.0` **means correlated evidence is not discounted, which is
what pooling has always done** — `aggregate_pool` treats every row as independent, so twenty
outlets writing up one wire report read as twenty facts. Per-article classification quality
cannot fix that: however good the extractor gets, N echoes of one development are still one
development. The mobilization pool (`cmrazsvhd000701nsyiyzm2i7`) is 20/22 positive-stance
largely on a single *"sources say the Kremlin is considering a new wave"* reporting wave.

Clustering is **lexical, not semantic, and deliberately so.** retro#355 sketched reusing
news-indexer's pgvector near-dup machinery, but that lives in another service: retro has no
embedding dependency, and adding one buys a per-row API call on every recompute plus a model
whose drift would silently re-cluster history. The gate harness (#350) measures this change by
*replaying past pools*, which a non-deterministic clusterer cannot support. Shingle Jaccard
(`cluster_jaccard_threshold=0.40`, `cluster_shingle_size=3`) is free, exact and reproducible.
Single-linkage, so A~B and B~C puts all three together. A row with no usable text is always
its own singleton — missing text can never *cost* a source its vote.

**Threshold tuned 2026-08-09 (retro#414):** the original 0.5 was never once reached live (max
observed 0.457 across 24 pools), so the discount above could not fire regardless of the
exponent. `api/scripts/eyeball_cluster_pairs.py` pulled the actual `claims_detail` text behind
every pairwise score in prod and printed it for review — the aggregate stats alone couldn't
distinguish real echo from coincidence. Most pairs down to ~0.30 turned out to be genuine
syndication of one report (the same Reuters/AP wire write-up run by several outlets, a Michael
Burry quote picked up by three headlines) — but two independent findings ruled out going that
low: (1) three pairs at 0.22–0.27 shared boilerplate lead-in text while reporting
**contradictory** claims (one said a Patriot-missile production license was granted, another
that it was refused) — clustering those together would suppress real disagreement, not just
double-counting; (2) `test_aggregation_matrix.py`'s own synthetic fixtures, whose default
per-source claim/quote template exists specifically so *different* sources don't cluster
(the docstring on `_prediction()` documents an earlier, worse version of this exact bug — see
retro#372), score an exact, source-name-independent **0.3333** against each other, a hard
structural floor a lower threshold would cross. `0.40` clears both: real margin above the
0.3333 fixture ceiling, and it still catches 4 unambiguous echo pairs (0.404–0.492, one
Hormuz-deal wire story plus an identical Trump quote) with zero contradiction or
fixture-collision risk. Not re-derived after a `claims_detail`-quality change (e.g. the
`title` fix sketched in retro#408); re-eyeball if one lands.

**Coverage is bounded by `claims_detail` today.** Neither `SourceSignal` nor
`PoolSourceInput` carries a title, so that field is the only cluster text either caller can
supply: a row without it is unclusterable and contributes nothing to
`event=evidence_clusters`. Coverage therefore grows only as the pool re-extracts, and the
measurement currently under-reports how much echo the corpus really holds — read early
cluster counts as a **lower bound**, not a rate. Adding a title to those two models is the
one change that would make legacy rows clusterable.

**Both weight sites derive the cluster text through the same `cluster_text_for_claims`**, or a
recompute would re-cluster rows `/forecast` already clustered — the same failure mode the band
table above guards against. This is the first estimator use of `claims_detail`, which
`run_pool_aggregate`'s whitelist comment reserved for exactly this issue; the recompute path
receives it via daatan#1264.

**Deliberately not enabled**, for the reason `forecast_relevance_bar` is not raised: the
verification path (#350 — Brier with vs without) needs resolved forecasts with a usable
evidence pool, and as of 2026-08-05 there are **6** — of which **0 carry a claim layer at
all**, so the backtest corpus and the resolved set do not yet intersect (#403). What ships
instead is the *measurement* — `event=evidence_clusters`, one line per pool:

| field | meaning |
|---|---|
| `rows` | pool rows handed to the clusterer |
| `textful` | rows with usable cluster text — **the real denominator** |
| `pairs` | comparisons performed, `C(textful, 2)` |
| `clusters` / `largest` / `echoed_rows` | the grouping that resulted; `echoed_rows=0` means no echo |
| `max_jaccard` | highest similarity seen, **threshold notwithstanding** |
| `hist` | comma-separated counts per 0.1-wide similarity band, summing to `pairs` |
| `threshold` / `exponent` | the settings in force, so a line stays readable after a retune |

**The line is emitted for every pool, including pools with no echo.** It originally fired
only when some cluster reached size ≥2, which made a zero unreadable — a pool too small to
compare, a pool of text-less legacy rows, and a pool of genuinely independent reporting all
wrote the same nothing. Measured 2026-08-05, that produced exactly **one** line (a synthetic
probe) across 180 `/pool/aggregate` and 374 `/forecast` requests.

The threshold **was** untuned — `max_jaccard`/`hist` exist exactly so it could be, and retro#414
did that tuning (2026-08-09, `cluster_jaccard_threshold` 0.5 → 0.40, see §2.4 above): the
conditional log could only ever show echo that *already* cleared the bar, so there was no data
below it to lower the bar onto until the unconditional line landed. Note this measurement
accrues at **traffic rate**, not at resolution rate: it observes pool structure rather than
forecast accuracy, so unlike enabling the discount it is not gated on the resolved-forecast
backlog.
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
then thin-evidence widening, then the F16 `pool_dispersion_floor` widening;
settlement pins substitute a fixed narrow band and are **exempt-by-design**
from the dispersion floor — `_settlement_pin` runs after it and replaces the
interval outright, deliberately, per its own docstring (retro#383). The
alternative (move the floor after the pin) would shift `ci_high` on 85.4% of
settled snapshots in prod for no epistemic gain, since the pinned mean is
already inside the pool clamp. Matrix case C11 pins the exemption as an
invariant: at that case's `n_eff ≈ 2`, the floor would otherwise demand a
published width of ~0.277 (stance scale); the settlement pin instead
publishes ~0.17.
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
  program in 2019") was elicited as an accomplished-fact settlement; daatan
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
  **Item 3 of the fix — the scalars are now *derived*, not parallel** — landed
  next: `reduce_article()` is the single named reduction from `claims_detail`
  to the article's stance / certainty / evidence_weight / evidence_class /
  settlement / fact lane, and the loop calls `build_claims_detail()` first and
  feeds it in. Same formulas, same order, same floats — a refactor, verified by
  zero movement across the 57 matrix cases; an implementation of "derived" that
  moved a number would have quietly imported R1 (claim-level weighting), which
  is Phase 2 and gated on the shadow pool. Because the function is pure, it is
  also the replay path: a stored pool row can be re-reduced offline, which is
  what makes backtesting, R1 fitting and F3 attribution possible on history.

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

### An article we cannot date is dropped (retro#705)

`forecaster` used to derive `article_date` twice in the same flow: the pooling
layer took `result.published_date or None` and let `recency_weight` fail open,
while the extraction layer took `result.published_date or datetime.now()`. An
undated article was therefore presented to the gatekeeper and the extractor as
**today's news**.

That value is not a display field. It is the calendar anchor
`_apply_relative_date_override` walks "on Friday" against, so an undated old piece
could hand `enforce_settlement_event_date` a fresh, plausible, wrong `event_date` —
the one thing a positive settlement requires. It also made that guard's
future-dated check vacuous, since nothing can be after today. And it contradicted
the extractor prompt one level up, which says *"Never substitute the article's own
publication date for an event the article does not actually date."*

Both call sites now go through `_resolve_article_date`: provider date, then the
date in the URL path (`/2024/03/15/`), then **drop the article** — logged as
`event=article_outcome outcome=no_date` with an `ArticleDebug(outcome="no_date")`,
checked before the fetch so a dropped article costs no request and no per-host
throttle slot. This is the rule the batch path has always applied
(`web_search_ingest.py`'s `skipped_no_date`); the live path now matches it.

Blast radius, measured on prod before the change (13,196 COMPLETE evidence-pool
rows): **20 rows (0.15%)** carry no `published_date`, and **0 of 618 settled rows**
do. Latent, not live — filed and fixed as the guard it is. `ArticleInput`'s
`published_date` is consequently optional only for callers whose URLs carry the
date in the path.

R8 protocol: one matrix case moved, declared — **C6**, which is no longer about an
undated article tying a stale dated one at the floor: the undated row does not
reach aggregation at all, so the pool is the dated row alone. The floor-decay
behaviour stays in `aggregation.recency_weight` as defence in depth for callers
that reach it directly.

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
`fact_signal → aggregate_pool` cutover (at the time still gated on the offline
harness; the cutover framing was retired 2026-08-15 — retro#533, decisions.md), and
the observation that in the audited over-cap rows `fact_signal` tracks `stance`
almost exactly (0.90/0.80, −0.82/−0.82, 0.75/0.75) — the prompt's "never let
fact_signal pull stance, or stance pull fact_signal" is not holding on this
population. That observation, recorded here rather than acted on, was later
measured pool-wide and became retro#533's finding (corr 0.905 on precursors).

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

## 2026-08-08 — settlement-pin ledger, Phase 1 (retro#361)

Settlement pin quality is measurable at exactly the moment each error is
freshest: of the first resolved questions that had been settlement-pinned,
two contradicted the pin (retro#360's England-Argentina sign flip; the
USA-bombs-Iran-2025 pin, arguably correct against stretched question
semantics). retro#361 proposes a ledger + report to make "pins contradicted
by resolution" queryable, plus a classification heuristic (extraction error
/ question semantics / genuine miss / pin-correct) for each contradiction.

**This entry ships Phase 1 only — the ledger + report.** The classification
heuristic is deferred; it needs its own review of real contradicted pins
once enough accumulate, tracked separately from this shipment.

- `IngestResolutionRequest` gains an optional `settlement_snapshot`
  (`SettlementSnapshotInput`: `settled`, `mean`, `ci_low`, `ci_high`,
  `settled_sources`, `settlement_suppressed`, `settlement_suppression_reason`
  — daatan's own copy of the last `/forecast`/`/pool/aggregate` response's
  settlement fields for this claim). Omitted entirely by callers that
  haven't wired it up yet — `POST /leaderboard/ingest` keeps working exactly
  as before, the ledger simply records nothing for that `prediction_id`.
- `settlement_pin_ledger.py` (`api/src/forecast_api/`): on ingest, when the
  snapshot is present and `settled=True`, records `{prediction_id, outcome,
  pin_direction (sign of pin_mean), pin_mean/ci_low/ci_high,
  settled_sources, settlement_suppressed, settlement_suppression_reason,
  contradicted (pin_direction != outcome)}` to
  `data/settlement_pin_ledger.jsonl`. Same diskcache-backed, cross-worker-
  safe dedup shape as `resolution_feedback.py` (retro#434/PR#450) — a
  separate ledger file and dedup store, independently idempotent on
  `prediction_id` so a retry that adds a snapshot the first push omitted
  still lands it even after `resolution_feedback`'s own store has already
  marked that `prediction_id` ingested. A snapshot with `settled=False` (the
  pool never pinned) is not recorded — nothing to post-mortem.
- `GET /leaderboard/settlement-pin-report` — new read-only endpoint, same
  shape as `/leaderboard/resolution-shadow` (recomputed from the ledger file
  on every call). Defaults to contradicted pins only;
  `?include_confirmed=true` returns the full ledger for a precision
  denominator.

## 2026-08-21 — shadow hazard prior for diffuse arrival claims (retro#356)

No article-driven extractor will ever emit *"the deadline is approaching and
nothing has happened"* — yet for a by-deadline claim, sustained absence of
occurrence evidence **is** evidence against. The pool only ever accumulates
positive article signals, so a rumor-heavy claim holds its elevated P right up
to the deadline. This adds a shadow re-drift of the pooled mean toward the
resolved base rate as the claim's window elapses.

**Shadow only, off by default.** `PoolAggregateResult.hazard_shadow_mean` sits
under the same compute-but-don't-use contract as `n_eff` / `age_adjusted_mass`
(retro#458 Phase 2): nothing in `aggregation.py` or its callers reads it back
into the pooled estimate. `test_hazard_never_moves_the_published_mean` asserts
exactly that — hazard off vs on, every published field byte-identical.

### Scope: `diffuse` only

`claim_archetype` (`scheduled | diffuse | threshold | none`) already existed and
was already threaded into `aggregate_pool`, where it gated settlement votes. The
hazard reuses it:

- **`diffuse`** — the pure "X happens by deadline, no scheduled date" arrival
  claim. The only archetype the hazard applies to.
- **`scheduled`** — must NOT decay. The event has a known date, so the open
  question is the *outcome*, not the *arrival*.
- **`threshold`** — arguably hazard-shaped, but a threshold can be crossed and
  un-crossed. Revisit against shadow Briers.
- **`none` / absent** — off, per design rule R3 (missing data never increases
  influence).

Also skipped whenever `any(settled_flags)`: the hazard exists to price
*absence*, and a settlement-grade row **is** occurrence evidence. The extractor's
`is_occurrence` would be the sharper signal, but it is itself still an
EXPERIMENTAL shadow field — building one shadow on top of another would compound
uncertainty rather than measure it.

### Decay target: the resolved base rate, shrunk toward a prior

`archetype_base_rate()` (`resolution_scorer.py`) computes
`(successes + prior_n * prior_p) / (n + prior_n)` over resolved records of that
archetype — the same shrinkage construction as `resolution_shadow_brier_prior_n`,
and for the same reason: it degrades smoothly rather than at a minimum-n cliff.

The target is deliberately **not** 0.5 (maximum uncertainty is not a base rate,
and would *raise* P on any claim currently below it — the exact inverse of the
point), and not the claim's own P at creation (which anchors on the very
estimate this issue suspects is rumor-inflated).

Note that a strict Poisson arrival model would decay toward **0**, not toward a
base rate: if nothing has arrived by `t`, `P(arrive before T) = 1 - exp(-λ(T-t))`,
which vanishes as `t → T`. Shrinking toward the resolved base rate is the
deliberately more conservative choice. **The functional form is a dial to re-fit
against shadow Briers, not a claim to have been calibrated.**

### The ingest dependency

`IngestResolutionRequest` gained optional `claim_archetype`, persisted into
`resolution_feedback.jsonl`. Until **daatan** sends it, no record matches any
archetype and the base rate *is* `hazard_shadow_prior_p` by construction — the
correct behaviour, not a stall. As of 2026-08-21 the file holds 13 resolutions
(7 True / 6 False), none carrying an archetype.

### Config (`api/src/forecast_api/config.py`)

| Variable | Default | Meaning |
|---|---|---|
| `hazard_shadow_enabled` | `False` | Master switch. While off, the resolution-feedback file is never read on the forecast path. |
| `hazard_shadow_prior_p` | `0.15` | Base rate before shrinkage — "most rumored by-deadline events do not happen by the deadline". |
| `hazard_shadow_prior_n` | `10.0` | Pseudo-resolutions of shrinkage toward `prior_p`. |
| `hazard_shadow_half_life_fraction` | `0.5` | Fraction of the `[created_at, deadline]` window at which half the excess over the base rate has decayed. |

Decay is exponential in elapsed **fraction** of the claim's own window, reusing
`recency_weight`'s `0.5 ** (x / half_life)` idiom. Fraction rather than absolute
days because the deadline is what the claim is *about*: a 3-day and a 2-year
window are both fully elapsed at their deadline, and absolute-time decay would
leave the short claim essentially untouched.

## 2026-08-22 — premise verifier, shadow/log-only (retro#575 slice 1 of 3)

**Shadow-only. `premise_verifier_enabled` defaults `False`; `premise_verifier_enforce`
exists but is unread this slice — a documented placeholder, not a live knob.**

retro#575 observed the pool sometimes prices an already-resolved or
structurally impossible premise with a confident number instead of
abstaining, because nothing in the pipeline ever asks whether the question
itself is still open — it only prices whatever evidence the topical search
returns. A premise that already resolved usually has no fresh coverage (news
moves on once something settles), so the topical search either returns stale
pre-resolution articles that read as live, or nothing at all (falling
through to the generic `no_search_results` reason, which reads as "couldn't
find evidence," not "the premise itself is dead").

`premise_verifier.py` asks one grounded LLM call — "is this question's
premise already dead: resolved as an accomplished fact, or structurally
impossible to still occur?" — over whatever `search_results` Step 1 of
`_run_forecast_inner` already fetched (title/snippet/date only, no extra
fetch). Same shape as `settlement_verifier.py`: a frozen `Verdict(dead,
reason, errored)`, the same "announced/scheduled/planned is not carried out"
discipline in the prompt, and the same fail-open contract — an unavailable
or unparseable verifier returns `dead=False, errored=True`, never a false
claim that a live premise is dead.

**Log-only, unlike `settlement_verifier`.** The verdict is logged
(`event=premise_verifier`) and never changes the response — no new field, no
`reason` value, no mutation of `mean`/`ci`/`settled`. Promotion to an
enforcing check (e.g. a new `insufficient_data` reason) is a follow-up once
real trigger/precision data from this shadow period justifies it, the same
rollout shape retro#545 slice (ii) (PR #586) and the Gate-0 evidence-window
shadow (PR #558) used before their own promotion decisions.

### Trigger gate

Every `/forecast` call reaches this point, so firing unconditionally would
double LLM cost on every ordinary request. `premise_check_triggered` fires
only when:

- `claim_archetype` is `scheduled` or `threshold` (elections, court dates —
  the shape retro#575's own examples are), **or**
- `claim_deadline` is present and has already passed (`<=` today).

No archetype and no deadline (older callers that don't classify claims) →
never triggers, zero added cost — the same additive/fail-open framing
`ForecastRequest.claim_archetype`'s own docstring already promises.

### Config (`api/src/forecast_api/config.py`)

| Variable | Default | Meaning |
|---|---|---|
| `premise_verifier_enabled` | `False` | Master switch. While off, the check never runs and never spends a token. |
| `premise_verifier_enforce` | `False` | Unread this slice — reserved for the follow-up that acts on the verdict. |
| `premise_verifier_model` | `None` | Falls back to `extractor_model` (Claude Haiku 4.5 in prod) when unset. |
| `premise_verifier_timeout_seconds` | `12` | Per-call timeout; a timeout is fail-open, not a `dead=True`. |

### Explicitly out of scope for this slice

retro#575 also proposed (2) an outside-view base-rate node keyed on
`claimArchetype` × tag, and (3) scheduled-date anchors as first-class facts
at the stance stage. (2) needs a brand-new `calibration_records` store — it
does not exist anywhere in the codebase yet, only named in a `Daatan/docs`
planning note. (3) already exists downstream in **daatan**
(`temporal-clock.ts`'s impossibility pin) but nothing analogous exists at
retro's extraction/stance stage. Both are separate, larger follow-up issues,
not covered here.

## 2026-08-23 — precursor candidate-match, shadow/log-only (retro#608)

**Shadow-only. `precursor_match_enabled` defaults `False`; `precursor_match_enforce`
exists but is unread this slice — a documented placeholder, not a live knob.**

The v2 playground's `_decompose` (`api/src/forecast_api/v2_playground.py`) proposes
precursor sub-questions and prices every one fresh via a full news-search pipeline —
even when an existing forecast on the same real-world event already sits in Daatan's
own bank or on a live Polymarket market. A partial check already existed
(`_anchor`/`_same_question`), but it only looks at Polymarket, runs *after* pricing (so
it never saves the cost), and only records a binary same-question verdict — anything
narrower/broader/complementary is written to `node["anchor_candidate"]` and never read
again anywhere in the file. retro#571's design hints already name the goal: "existing
questions first, latent nodes second."

This slice adds a **candidate-match step**, fired concurrently alongside pricing (same
shape as `forecaster.py`'s `premise_task`), that checks two sources for each node —
Daatan's own public `/api/forecasts/similar` bank search, and the same Polymarket Gamma
lookup `_anchor` already makes — and, for any candidate found, asks one LLM call to
type the relation (`alias`/`nested`/`complement`/`implies`/`independent`), the same
`_same_question`-shaped call primitive `_anchor` already uses. **It changes nothing
about pricing, recursion, or the propagated result** — it only writes
`node["precursor_match"]` and a structured `event=precursor_match` log line, giving a
retro#601-style follow-up the precision data needed before anything is allowed to gate
on it.

### Status disambiguation

Each source in `node["precursor_match"]` (`{"daatan": ..., "polymarket": ...}`) is one
of three shapes, not a bare candidate-or-`None` — collapsing "no candidate" and "the
lookup itself failed" into one signal would make a real outage of Daatan's API log
identically to "Daatan genuinely has no matching forecasts," which defeats the
precision review this slice exists to enable:

- `{"status": "not_found"}` — source reached, no candidate cleared the bar
- `{"status": "error", "detail": "..."}` — the lookup itself failed
- `{"status": "ok", "candidate": {...}, "relation": {...} | None}` — candidate found;
  `relation` is `None` only when the relation-classifier call itself failed/was
  unparseable, distinct from both cases above

### Polymarket fetch is shared with `_anchor`, not duplicated

`_match_polymarket` caches the fetched Gamma market onto
`node["_polymarket_cache"]` (transient — popped before the node is ever persisted,
either by `_anchor`'s cache-check or, for nodes `_anchor` never reaches, by the
per-depth/end-of-job cleanup in `run_job`). `_anchor` checks that cache before making
its own fetch, so enabling this slice does not double the Oracle's Gamma HTTP volume —
one fetch per node either way.

### Config (`api/src/forecast_api/config.py`)

| Variable | Default | Meaning |
|---|---|---|
| `precursor_match_enabled` | `False` | Master switch. While off, the check never runs and never spends a token or an HTTP call. |
| `precursor_match_enforce` | `False` | Unread this slice — reserved for the follow-up that acts on the verdict. |
| `precursor_match_model` | `None` | Falls back to `extractor_model` (Claude Haiku 4.5 in prod) when unset. |
| `precursor_match_timeout_seconds` | `12` | Per-call timeout for the Daatan bank lookup. |

### Explicitly out of scope for this slice

**Metaculus** as a third grounding source — retro#608's original proposal named it, but
nothing in this repo does full-text search over Metaculus questions today (the existing
`metaculus/` module is Oracle→Metaculus submission for benchmarking, the opposite
direction, and isn't fully wired up yet). Deferred to a follow-up once a
read/search-capable Metaculus client exists.

**Grounding a node on its own settled/confidence signal** (`node["flat"]["settled"]`,
an existing computed-but-unused per-node field) is a separate concern, tracked as
retro#609 — coordinate with retro#575/#601's `premise_verifier` before building it, since
both would independently want a say in the same "is this node's premise still live"
question.

**Promoting any relation type to actually gate `_price_flat`** — that is the
retro#601-style follow-up this slice exists to produce data for, not something decided
here.

## 2026-08-23 — settled-grounding, shadow/log-only (retro#609)

**Shadow-only. `settled_grounding_enabled` defaults `False`; `settled_grounding_enforce`
exists but is unread this slice — a documented placeholder, not a live knob.**

`_price_flat` already computes a usable grounding signal on every node it prices and
discards it: `_flat_summary` (`v2_playground.py:140-157`) captures `settled: bool` — true
when the pool's aggregate came from a majority of claims the extractor itself marked as
already-decided fact (`settlement_grade`), not a forecast — plus `std`/`evidence_mass`/
`n_eff`/`ci`, a softer "how confidently was this estimated" signal distinct from `settled`.
Before this slice, `node["flat"]["settled"]` was written once and never read again
anywhere in the file; it had no effect on `_anchor`, pruning, or recursion.

`_settled_ground` (`v2_playground.py`) runs right after `_price_flat`, no LLM call and no
network call — everything it reads is already sitting in `node["flat"]`. It logs two
things into `node["settled_grounding"]` and an `event=settled_grounding` line:

- `would_lock` — what a hard lock on `settled=True` would have done (the same
  stop-recursion-and-lock treatment `_anchor` gives a same-question Polymarket match).
- The raw `std`/`evidence_mass`/`n_eff`/`ci_width` values — the softer "confidently
  estimated, further decomposition unlikely to move this number" signal the issue
  proposes should weight pruning rather than stop recursion outright.

**This slice deliberately stops at logging the raw soft-signal values, not a derived
prune-weight formula.** No threshold for "tight CI" or "high evidence_mass" exists
anywhere in this codebase to calibrate against, and inventing one without real
distribution data would just be encoding a guess as if it were validated — exactly what
the shadow-first pattern exists to avoid. The follow-up review (same retro#601 shape) is
where that threshold gets picked, from real logged values, not before.

**Coordinate with `premise_verifier`, don't duplicate it (retro#575/#601).** `settled` is
a post-extraction, claim-level signal; `premise_verifier` is a dedicated pre-extraction LLM
call over raw search snippets asking essentially the same "is this premise already dead"
question. Both now log independently (`event=settled_grounding` here,
`event=premise_verifier` in `forecaster.py`), tagged with the same `_question_hash`, so a
future review can check whether they agree before either is promoted — if they're highly
correlated, prefer the free one (`settled`) and don't pay for both.

### Config (`api/src/forecast_api/config.py`)

| Variable | Default | Meaning |
|---|---|---|
| `settled_grounding_enabled` | `False` | Master switch. While off, `_settled_ground` never runs — it's a pure function of data `_price_flat` already computed, so there's no cost to skip. |
| `settled_grounding_enforce` | `False` | Unread this slice — reserved for the follow-up that acts on the verdict. |

### Explicitly out of scope for this slice

**Any actual pruning/locking behavior change** — this slice only logs what a hard lock or
a soft prune-weight *would* do; nothing about `_anchor`, the prune step, or recursion
changes yet. That is the retro#601-style follow-up.

## 2026-08-25 — retry-relaxed-search fallback ladder rung 1, shadow/log-only (retro#621)

**Shadow-only. `retry_relaxed_search_enabled` defaults `False`; `retry_relaxed_search_enforce`
exists but is unread while the shadow log is thin — the same documented-placeholder shape
`precursor_match_enforce`/`settled_grounding_enforce` use.**

`/forecast` returns `insufficient_data` (a.k.a. `outcome=no_usable_predictions`) on
~30% of daatan's own production traffic (retro#621's own measurement; the rate on
Metaculus-style tournament questions is unmeasured, blocked on retro#619's Bot
Benchmarking Tier access). That's fatal for a FutureEval bot: a question is open 1.5h,
scored by spot peer score, and no submission scores nothing.

This is rung 1 of the fallback ladder retro#621 asks for — the cheapest, lowest
honesty-cost rung: when the primary pass comes back insufficient, retry once with a
wider article limit (`retry_relaxed_search_limit_multiplier`, default `2.0`, capped at
the caller's per-key `max_articles` ceiling if one applies). No query rewording in this
slice — see "Explicitly out of scope" below.

`_maybe_retry_relaxed_search` (`api/src/forecast_api/forecaster.py`) runs after
`run_forecast`'s primary `_run_forecast_inner` call returns `insufficient_data=True`,
and only when the caller used live search (`req.articles` unset — a caller who supplied
articles directly has nothing for a wider limit to search). It logs
`event=retry_relaxed_search` with the primary's `reason`, both limits, whether the
retry recovered a usable forecast, and the live `enforce` value — so the shadow log
alone answers "how often would this rung have helped" before it's ever allowed to
change a response.

**The recovered response is tagged, never silently swapped in.** `ForecastResponse`
gained `fallback_path: "primary" | "retry-relaxed"` (`models.py`), surfaced in the MCP
tool's default (non-verbose) payload too — retro#621 ask item 4 is explicit that a
fallback must never masquerade as an ordinary Oracle forecast, and a Metaculus
rationale comment built off this field can say plainly which rung produced the number.

### Config (`api/src/forecast_api/config.py`)

| Variable | Default | Meaning |
|---|---|---|
| `retry_relaxed_search_enabled` | `False` | Master switch. While off, a primary `insufficient_data` response returns unchanged and `_run_forecast_inner` runs exactly once. |
| `retry_relaxed_search_enforce` | `False` | Unread while off — reserved for the follow-up that lets a recovered retry replace the empty response. While `enabled=True` and `enforce=False`, the retry still runs (for the shadow log) but its result never reaches the caller. |
| `retry_relaxed_search_limit_multiplier` | `2.0` | Retry's article limit = primary limit × this, capped at the caller's per-key `max_articles` if one applies. If the primary limit is already at that cap, the retry is skipped entirely (logged as `skipped=at_limit_cap`). |

### Explicitly out of scope for this slice

**Query rewording/decomposition** — the issue's ask lists "relaxed search parameters
**or** a reworded/decomposed query" as alternative levers for this rung; only the
limit lever shipped here. `_distill_query` (keyword distillation) and
`v2_playground._decompose` (question decomposition, different pipeline) both exist and
are reusable for a future query-rewording variant of this rung if the shadow log shows
the limit lever alone doesn't recover much.

**The rest of the ladder** — base-rate/outside-view prior (#589, blocked on
`calibration_records` not existing yet) and the Metaculus community-prediction anchor
(blocked on CP not being exposed by the standard API, likely the same access gate as
#619) are separate follow-ups, not this slice.

**Whether a fallback is worth submitting at all under spot peer score** — see the
scoring-EV analysis on retro#621 itself. Short version: an *uninformative* 50% guess is
expected-negative EV relative to just forfeiting whenever the peer field is informed
(the normal case) — it does not belong on this ladder as a safe last resort. This
rung's retry, by contrast, is a genuine (if narrower) Oracle forecast, so it carries no
special scoring risk beyond Oracle's ordinary accuracy.

**The premise_verifier correlation check itself** — the issue's point 3 names this as a
prerequisite for *promoting* settled-grounding, not for shipping the shadow log. Both
retro#601 (premise_verifier's own review) and this feature's equivalent follow-up are
still open; the correlation check happens once both have real accumulated volume.

## 2026-08-28 — a settlement cannot predate the question (retro#704)

`aggregation.settlement_vote_validity` has bounded settlement votes to the window
after the claim was created since 2026-08-16: an event dated before
`claim_created_at` is *"a dated fact from before the claim existed: the
2021/2022-article class"*, and its vote is discarded with
`reason="event_before_claim_window"`. That check was applied on every archetype and
it works.

What it never did was reach the row. `tm.extractor.enforce_settlement_event_date` —
the extraction-time guard, running one layer up — knew only `article_date`, so it
kept writing `settled=true` on exactly the rows the pooling layer would then throw
away. **144 of the 215 adjacent settlements in the retro#691 labelled set are of
that shape.** The founding case is a China Daily piece on the 2022 Israeli election,
correctly dated `2022-11-04` with a correct `event_date` of `2022-11-01`, which the
extractor marked as settling six *2026* forecasts.

The stored bit is not cosmetic. It is read by the settlement-pin ledger, by
`logs.sh settlement`, by every backtest that counts settlements, and by anything
downstream that trusts a boolean rather than re-deriving the vote — so the two
layers disagreeing about the same rule is a reporting defect even where the estimate
is safe.

`enforce_settlement_event_date` now takes `claim_created_at` and demotes on
`event_date < claim_created_at`, emitting the **same** `event_before_claim_window`
reason string aggregation uses: one grep should find both layers, and the two must
never be allowed to drift apart on the rule. Comparison is at date granularity and
strictly `<`, so an event on the creation day still settles; an absent or
unparseable value fails open, matching aggregation. `forecaster` threads
`req.claim_created_at` through `_process_article_bounded` → `_process_article` to
the enforcement call. The batch path (`tm.runner`) deliberately passes two
arguments: a retroactive event has no claim and therefore no window, so failing open
there is the intended behaviour, not an oversight.

Demotion clears the settlement bit only — stance, claim strength and the rest of the
row survive, and the article still votes as ordinary evidence. That is what makes
this safe to apply at extraction time.

R8 protocol: **no matrix case moved, and that is the acceptance test.** The bound is
already applied downstream, so every affected vote is already being discarded; this
change only makes the stored row agree with the vote already cast. A moved case
would have meant the two layers were not in fact enforcing the same rule. Verified
before the fix on prod as well: the six 2026 forecasts the 2022 article "settled"
are all `settled = f` and ACTIVE.

## 2026-08-29 — deterministic confusion flags, log-only (retro#687, Oracle 1.5 P1 item 1.9)

**Reporting only. Nothing here changes a number.** No LLM call, no network call, no
weight change, no master switch — the rules are pure functions over fields the
extractor has already emitted, so there is nothing to gate. Phase 3 is where flagged
rows leave the credibility bill; this slice only counts them.

The Phase 1 goal is to measure the extractor **as an observer**, and the honest
instrument for that is between-rater disagreement — which needs a second extraction
per row. These rules are the tier below it that costs nothing: each names an
*internal* inconsistency, visible on a single row with nothing to compare it against.

A flagged row is **not known to be wrong.** It is a row whose own fields disagree about
how much confidence anyone should place in it, which is exactly the sampling filter
daatan#1636's second-family re-read wants — and a cheap prior on where between-rater
disagreement will turn up when Phase 1 pays for it.

`api/src/forecast_api/confusion_flags.py`, called from both live paths after the
estimate is computed: `_run_forecast_inner` (live `/forecast`) and `run_pool_aggregate`
(recompute over a stored pool).

| rule | fires when | status |
|---|---|---|
| `trapped_strong_claim` | `reader_confidence.trap` is set **and** `claim_strength >= 0.8` | **live** — retro#681 landed `reader_confidence` on 08-28 |
| `stance_vs_quantity` | the claim's own number and its stance sign disagree about a threshold question | **inert** — both inputs missing, see below |
| `unsure_settlement` | `settled is True` **and** `reader_confidence.level != "high"` | **live** |

`trapped_strong_claim` is keyed on `trap`, not on `level`. A trap is a *named* class with
an independent detector behind it (`negation` → retro#657, `numeric_comparison` → the
PR#671 A/B cases), which is what makes a self-report checkable against something other
than itself. `ReaderConfidence`'s own docstring notes a trap does not imply a low level;
that asymmetry is the rule, not an objection to it. The two fields being separable at all
is what retro#680 bought — before the split, one `certainty` field carried both the
source's commitment and the reader's, and this rule could not have been written.

`unsure_settlement` is the one place where "the reader could plausibly read this
differently" is not an acceptable margin, since a settlement can pin a forecast outright.
Phase 4 turns it into settlement's second bar (`level == "high"` required); here it counts.

**`stance_vs_quantity` ships inert, by construction and not as a placeholder.** It needs
a claim-side number (`quantity`, retro#683) and a question-side threshold (Phase 2's
`question_quantity`); neither exists. The claim side is read via `getattr`, so the rule
starts firing the moment #683 lands with no edit here. `ClaimDetail.quantitative_estimate`
is **not** that number and must not be substituted for it — it is a probability in [0,1]
cited *for the event*, so comparing it against a question threshold ($100, 61 seats) would
compare two different quantities and flag on noise. The comparison direction is a
parameter rather than inferred from question wording: guessing "exceeds" vs "falls below"
is the kind of judgement this module exists to avoid making.

**Null-safety is the design constraint, not a defensive habit.** Two of the three rules
read a shadow field that a row extracted before retro#681 does not have, and rule 2's
inputs do not exist at all — so "input missing" is the common case in production right
now, not an edge case. A rule with a missing input yields **no flag**: never a default,
never a guess. The module therefore activates rule by rule as the fields land.

### The lines

One `event=confusion_flag` per firing (`rule`, `url`, `claim_index`), plus one
`event=confusion_flags` summary **per pool, always** — including pools where nothing
fired. Same rationale as `event=evidence_clusters` (§2.4) and Gate-0: without a
denominator a zero is unreadable, since "no confusions" and "the path never ran" write
the same nothing. `rows` vs `evaluable` separates those two — a pool of pre-#364 legacy
rows carrying no claim layer is *unevaluable*, not clean — and every rule reports its
count even at zero, so a rule that never fires stays distinguishable from one that was
never evaluated.

Both lines carry `question` (`_question_hash`), `prediction_id` and `extractor_model`, so
per-rule counts group **per rater** — which is the Phase 1 exit report's unit.

**`run_pool_aggregate` logs `extractor_model=unknown`, deliberately.** `PoolSourceInput`
carries no extractor model, and the rows in a stored pool were extracted at different
times by whatever model was configured then. Stamping today's `settings.extractor_model`
on them would silently attribute one rater's confusions to another and corrupt the
per-rater grouping that is the entire point of the measurement. `unknown` is recoverable;
a plausible wrong value is not.

Flags are computed over **post-resolution** `claims_detail` — after the `enforce_*` chain
and `resolve_stance_certainty()` — because that is what `ClaimDetail` records. A row
demoted by an `enforce_*` guard is scored as it ended up, not as the extractor first wrote
it.

### Config (`api/src/forecast_api/config.py`)

| Variable | Default | Meaning |
|---|---|---|
| `confusion_flag_claim_strength_min` | `0.8` | The "flat enough to be worth pairing with a trap" bar for `trapped_strong_claim`. Inclusive — `claim_strength` clusters on round values, and an exclusive bar would silently drop the modal one. |

### Explicitly out of scope for this slice

**Any weight or estimate change.** Phase 3 is where flagged rows are excluded from the
credibility bill, and the direction is one-way — flagged rows come *out*, never get
re-weighted up. Nothing here reads a flag back.

**A rate threshold.** No "flag rate above X means the rater is bad" number is picked here.
There is no logged distribution to calibrate one against yet; that is what these lines
produce, and picking the threshold first would encode a guess as if it were measured —
the same reason retro#609 stopped at logging raw values.

## 2026-08-29 — a search-engine link wrapper is not an article (retro#709)

Some pool rows store a redirector URL rather than the publisher's:
`https://google.com/goto?url=<opaque token>`. Measured on prod before the change —
**72 of 21,848 rows (0.33%)**, spanning 2026-07-29 to 2026-08-26, touching 21
predictions, 7 of them carrying `settled`.

Three of the four harms this looked like it should cause turned out not to happen,
and saying so is the point of writing the numbers down:

- **Dating** — unharmed. All 72 carry a `published_date`; the provider supplies one.
- **Fetching** — unharmed, in fact slightly better than baseline: 33.3% of the
  wrapper rows are FAILED against **39.3%** across all rows. They resolve.
- **Outlet identity** — this is the real one. **52 of the 72 store `google.com` as
  their source, and every row in the entire pool whose source reads `google.com` is
  one of these.** Not one of the 72 ever resolved an `outlet_name`.
- **Dedup** — structurally true (two wrappers around one article are two strings)
  but not separately measured.

An outlet is not a cosmetic label: `settlement_min_sources` counts **distinct
outlets**, so a wrong one can manufacture corroboration between two copies of the
same article, and credibility weighting is per-outlet.

### It is not unwrapped, because it cannot be

The `url=CAES…` parameter is Google's **encrypted** article id, not an encoded URL.
Decoding a live sample yields 188 bytes of ciphertext with no URL in it. The only way
to resolve one is to follow the real 30x hop, which is what news-indexer does at
ingestion (news-indexer#306, merged 2026-08-20).

### Where it actually comes from

Upstream, and already fixed there. news-indexer holds **19** articles whose stored
`canonical_url` is a wrapper, **last indexed 2026-08-20** — the day #306 merged — and
**zero since**. Those 19 stale rows keep being served by `/search`, which is why
retro's pool still gained **24 wrapper rows between 2026-08-21 and 2026-08-26** long
after the ingestion leak closed. Re-resolving those 19 rows is
news-indexer#404; this change is retro's own guard, and retro is where a wrong
outlet costs something.

`tm.web_search_ingest.is_redirector_url` is host-and-path, never a substring match —
a real article that quotes a redirector in its query string, and a Google property
that serves its own content, both survive. `_process_article` drops a match before
the fetch (`event=article_outcome outcome=redirector_url`,
`ArticleDebug(outcome="redirector_url")`), the same shape and the same reasoning as
the retro#705 undatable drop it sits beside: an article we cannot attribute is not
evidence. Expected cost once upstream is cleaned: zero.

R8 protocol: no aggregation code touched, no matrix case moved.

## 2026-08-29 — a provider's date is normalised, not trusted (retro#714)

`_resolve_article_date` took the provider's string as gospel:

```python
provider = (result.published_date or "").strip()[:10]
return provider or _date_from_url(result.url)
```

Any non-empty string won, so the URL-path leg behind it was unreachable whenever the
provider said *anything at all*. Nothing raised. The string was stored,
`aggregation._parse_date` returned None on it, and `recency_weight` applied the **floor**
— 0.02 instead of ~1.0, by design rule R3 (missing data must never increase influence).
A correctly dated article in an unexpected format therefore lost **50×** its weight,
silently, and was read as maximally stale. The `[:10]` slice manufactured garbage of its
own: `"Feb 24, 2026"` was stored as `"Feb 24, 20"`.

`tm.web_search_ingest.normalise_published_date` now converts before accepting: ISO (with
or without a time suffix), English month-name forms in either order, unambiguous numeric
forms, and the relative grammar (`"2 days ago"`) delegated to
`web_search._absolutize_relative_date` so retro#562 keeps one copy of those rules.
Unicode format characters (bidi marks, the BOM) are stripped first — invisible, carrying
no date information, and fatal to every parser.

**Recovery, not stricter rejection, is the point.** A row with an unparseable date keeps
its vote today at floor weight; validating alone would have converted that into an
outright drop (retro#705), which is worse. Only 1 of the 13 non-ISO rows in the live pool
had a datable URL, so a naive validate-then-fall-through would have dropped 12 of 13.

**Ambiguous numeric dates are refused on purpose.** `05/09/2026` is 5 September to most
of the world and 9 May in the US, and nothing in a SERP payload says which. A guess would
be *believed*: `article_date` is what `_apply_relative_date_override` walks the calendar
against, so a wrong one propagates into `event_date`. `16/09/2026` is accepted — 16
cannot be a month, so it resolves itself. Rejections log
`event=provider_date_rejected` with the raw value, keeping "a format we do not parse yet"
separate from "the provider sent no date" — the distinction `event_date_state` (retro#554)
draws for settlement dates.

**Measured footprint — the stored rows were not the live vector.** 13 of 13,170 voting
pool rows carried a non-ISO `published_date` (0.099%). Origin analysis dissolves them:
6 are `origin=retry` google.com rows that retro#709 now drops as redirectors, and 7 are
`origin=backfill` from a single run on 2026-08-16 (7 of only 11 backfill rows in the whole
pool). The `news-indexer` origin — 10,238 rows, 78% of the pool — has **zero**. The live
vector is in the code rather than the table: `web_search.py` assigns Brave's `age` field
raw (`published_date=item.get("age", "")`), and that field is relative ("2 days ago") as
often as it is a date. Brave is step 4 of the chain, so it rarely wins, which is why the
pool shows almost nothing. Its assignment site is left alone here — `_filter_by_date`
absolutizes transiently for filtering and changing what is *written* has its own
consequences for the batch path.

## 2026-08-29 — same development, not same words: the event key (retro#682)

`clustering.cluster_texts` asks whether two pool rows used the same wording. Measured
over the `event=evidence_clusters` log in prod — **13,035 pools, 19,926,967 pairwise
comparisons** — that question almost never gets a yes:

| band | pairs | share |
|---|---|---|
| [0.0,0.1) | 19,871,389 | **99.721%** |
| [0.1,0.2) | 44,932 | 0.225% |
| [0.2,0.3) | 5,787 | 0.029% |
| [0.3,0.4) | 2,401 | 0.012% |
| ≥0.40 (fires today) | 2,458 | 0.012% |

`max_jaccard` is exactly **0.0 in 8,658 of 13,035 pools**. Pool rows are LLM paraphrases
of twenty different outlets' prose, so one development routinely shares almost no
trigram with itself.

**Lowering `cluster_jaccard_threshold` is not the alternative.** The [0.3,0.4) band holds
2,401 pairs against 2,458 already firing — dropping the bar to 0.30 moves the firing rate
from 0.012% of pairs to 0.024%. That is noise either way, and `config.py`'s instruction to
tune the threshold "against the logged cluster structure" is now answered: there is no
mass sitting just under it. Do not re-propose a threshold change on this evidence.

`event_key_for_row` asks a paraphrase-invariant question instead — same `(actor, target,
day)`? — off the retro#313 facets, which were elicited but barely consumed before this.

**The day is not optional, and neither is the `published_date` fallback.** One key per
row (highest `claim_strength` claim, ties by array position):

| key | keyed rows | rows collapsed | pools with echo | largest cluster | clusters >20 |
|---|---|---|---|---|---|
| Jaccard ≥0.40 (today) | 7,376 clusterable | 0.57% echoed | 19.2% | — | — |
| dyad only | 5,524 | 40.4% | 78.4% | **171** | 20 |
| dyad + `event_date` only | **793** | 24.4% | 41.9% | — | — |
| dyad + day, `event_date` ?? `published_date` | 5,497 | 17.7% | 62.1% | 24 | 2 |
| **+ row-level facet fallback (shipped)** | **6,886** | **18.8%** | **62.7%** | **24** | 3 |

Two failure modes bracket the design. Requiring `event_date` keys 6% of the pool — the
field's own instruction is "omit entirely when the article states no date", and most
articles state none. Dropping the date instead produces a **171-row** cluster on
`united states -> iran`, because a dyad is a *relationship*, not an event: months of
coverage collapse into one "story", and with the discount enabled that pool would go to
`n_eff ≈ 1`. Adding the publication day splits that same group into 34 sub-clusters,
largest 24. The row-level `event_actors`/`event_target` fallback then buys +27% coverage
(5,403 → 6,886 rows) with the largest cluster unchanged.

**Reporting only. R8: no case moved.** `_cluster_ids` returns the same Jaccard ids it
always did; the event key is logged *beside* the Jaccard numbers
(`event_keyed`, `event_clusters`, `event_echoed_rows`, `event_largest`) so the two can be
compared on identical pools before anything is switched over.
`cluster_downweight_exponent` stays 0.0 and enabling the discount remains gated on #355's
December backtest (#403). This changes the key the seam will use *when* it turns on, not
whether it turns on.

Normalisation is stdlib and deterministic — lowercase, punctuation stripped, leading
article removed, multi-actor strings split on commas and **sorted** so "United States,
Israel" and "Israel, the United States" are one key. The alias table is deliberately tiny
and holds only nation-state synonyms whose merge is not a judgement call; metonyms
("Washington", "Number 10") are left alone because their correctness depends on context
and a wrong merge silently fuses two developments. news-indexer entity ids are the Phase 2
escalation, and the 18.8% collapse rate says they are not needed yet. The date is taken by
**slice, not parse**: `published_date` is free text that holds non-ISO junk, and an
exception inside the clusterer would fail a `/forecast` request over a reporting-only
measurement.
