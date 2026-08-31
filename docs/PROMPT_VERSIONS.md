# Prompt version changelog

Tracks version bumps to the live-path gatekeeper/extractor prompts
(`api/src/forecast_api/forecaster.py`'s `GATEKEEPER_PROMPT_VERSION` /
`EXTRACTOR_PROMPT_VERSION`, sourced from `tm.gatekeeper`/`tm.extractor`'s
`PROMPT_PREFIX`/`PROMPT_SUFFIX`). Shipped via `ProvenanceModels` on every
`/forecast` response (`provenance.models.*_prompt_version`/`*_prompt_hash`,
daatan#1604/retro#627), so a caller persisting extraction results can tell
which prompt produced a given row.

The version string is hand-bumped and only a human-readable label — the
`*_prompt_hash` field (SHA-256 of the actual rendered prompt) is the source
of truth for whether the prompt text changed, independent of whether this
file or the version constant was updated. Bump the version and add a row
here whenever a prompt edit changes model behavior materially (wording that
could change extraction/gating outcomes) — not for comment-only or
formatting-only edits.

**Enforced, not just documented (retro#632):** `docs/prompt_versions.lock.json`
records the hash each version label is supposed to correspond to.
`api/tests/test_prompt_version_enforcement.py` fails CI if the currently
computed `*_PROMPT_HASH` doesn't match the lock file — i.e. if a prompt edit
landed without a version bump. Update the lock file (and this table) in the
same PR as the prompt edit.

The lock covers **three** kinds of text, added in that order as each was found to
be unwatched: the hand-written prompt (retro#632), the serialised response schema
(retro#700), and the blocks appended at call time (retro#731). Each section below
explains what the previous one missed.

## The prompt is the prose **and** the schema (retro#700)

`PROMPT_PREFIX`/`PROMPT_SUFFIX` are not all the text the model reads. Both the
gatekeeper and the extractor are structured calls through `instructor` in
`Mode.MD_JSON`, which serialises the response model's JSON schema into a system
message on every call — every `Field(description=...)`, every enum member, and
every Pydantic **model docstring** (Pydantic copies those into the schema's
`description` key). Measured on today's models:

| Component | Hand-written prompt | Rendered schema | Schema share |
|---|---|---|---|
| extractor (`ExtractionOutput`) | 58,215 chars | 21,339 chars | **27%** |
| gatekeeper (`GatekeeperOutput`) | 4,603 chars | 1,178 chars | **20%** |

(Extractor row measured at v8; it read 55,116 / 20,287 at v7, 62,564 / 23,795 at
v9 and 66,718 / 25,950 at v10. The share has been stable at ~27% across every
bump, which is the point — prose and schema grow together because a new field
needs both.)

Until retro#700 none of it was hashed, so the schema could change arbitrarily
with no hash movement, no lock diff and no version bump — a prompt-version lock
that did not lock. It was found the expensive way: retro#681 shipped a 1,283-char
docstring to the model on every call, changed Nova Lite's `fact_signal` fill
measurably, and every prompt-version test stayed green.

So the lock now pins **four** things per component — `version`, `hash`,
`schema_hash`, and `schema_chars` — and a schema change is a prompt change:
bump the version, add a row to the table, update the lock. In particular,
**adding a field to `ExtractionOutput` is a prompt edit**, because it is.

`schema_chars` is recorded next to the hash rather than implied by it on purpose.
A hash mismatch says *something* moved; the retro#681 failure was not an
unnoticed edit but unnoticed **growth**, and only a number in the diff catches
that. The test reports the delta both ways (`20287 -> 20451 chars (+164, +0.8%)`).

The hashed string is byte-for-byte what `instructor` sends —
`json.dumps(schema, indent=2, ensure_ascii=False)`, reproduced in
`tm.llm.rendered_response_schema` and pinned against instructor's real output by
`pipeline/tests/test_rendered_schema.py`. The two obvious shortcuts were both
rejected and are pinned as rejected: `sort_keys=True` is blind to field
**reordering** (Pydantic emits `properties` in declaration order), and the
compact form under-counts the prompt cost by ~43%. If an `instructor` upgrade
changes that serialisation the pin fails — correctly, since the model's input
changed with it.

Both hashes ride on the wire: `provenance.models.{gatekeeper,extractor}_schema_hash`
on `/forecast`, and `gatekeeper_schema_hash` on `/relevance`. They are kept
separate from `*_prompt_hash` rather than folded in so a stored row says *which*
half moved — "the instructions were reworded" and "a field was added" are
different events with different explanations for a shift in results.

## And a third half: the appended blocks (retro#731)

`PROMPT_PREFIX + PROMPT_SUFFIX` plus the schema still is not everything. Five
instruction blocks are appended to `prompt` **after** `PROMPT_SUFFIX.format(...)`
returns — at the tails of `extractor.extract_predictions` and
`gatekeeper.check_is_prediction` — so they fall outside the reconstructed
`*_PROMPT` in `forecaster.py` and, until retro#731, outside every hash in the lock:

| Block | Chars | Reaches an article when |
|---|---:|---|
| `extractor._CONDITIONAL_BLOCK` | 4,233 | `has_conditional_language()` matches the lexicon |
| `extractor._SHORT_FORM_OVERRIDE` | 604 | the host is on the short-form allowlist |
| `extractor._LANGUAGE_HINT` | 189 | the article is not in English |
| `gatekeeper._SHORT_FORM_OVERRIDE` | 1,354 | the host is on the short-form allowlist |
| `gatekeeper._LANGUAGE_HINT` | 196 | the article is not in English |
| **total** | **6,576** | |

`_CONDITIONAL_BLOCK` alone is larger than the entire gatekeeper prompt and its
schema together, and it carries three worked examples. retro#720 is what an
unwatched block of worked examples costs: `## Multi-stage / bracket events` — a
section inside the *hashed* prefix — suppressed Nova Lite on **unrelated**
numeric-threshold claims, 8/20 → 18/20 when removed, because its seven examples
were uniformly low-stance. Not a length effect: a 1,702-char cut changed nothing
and an 1,886-char cut changed everything. That is the retro#700 mechanism running
forwards, and `_CONDITIONAL_BLOCK` is the same shape of hazard.

These blocks also carry a hazard the prefix does not: they are **conditionally
injected**. A lexical pre-filter decides who gets the conditional block; a host
allowlist decides who gets short-form. So a regression in one lands on a subset of
traffic and presents as unexplained drift rather than as a change with a date.

The lock now carries a `tail_blocks` section, **one entry per block**, each with a
`hash` and a `chars`. One folded hash over the concatenation would have been less
code and would report every change as "the tail moved" — which is not a review.
Five entries mean CI names the block.

**Editing one of these is a prompt edit.** Bump the owning prompt's version, add a
row to the table below, update the lock. And if the block you touched carries worked
examples, run the A/B harness before you touch it — that is the whole lesson of
retro#720.

`test_every_appended_block_is_locked` holds the partition: a sixth block appended at
call time but never registered would otherwise be covered by nothing, which is
exactly how this gap was born — `_CONDITIONAL_BLOCK` was added, appended and shipped,
and every enforcement test in the file stayed green for its entire life.

| Component | Version | Hash | Effective from | PR | Summary |
|---|---|---|---|---|---|
| gatekeeper | v1 | `a09cdb5ecda0ce5e` | 2026-08-24 | retro#627 | Initial versioned baseline — no prior version existed on the live path. |
| extractor | v1 | `6371300bb3b89b8c` | 2026-08-24 | retro#627 | Initial versioned baseline — no prior version existed on the live path. |
| extractor | v2 | `6813d6184d14568b` | 2026-08-25 | retro#649 | Added "tone is not stance direction" rule + worked example (retro#545) — alarming/critical tone getting read as negative stance regardless of the claim's actual content. |
| extractor | v3 | `5a8082ab250463eb` | 2026-08-25 | retro#326 | Added an author_lean self-consistency cross-check against the article's own extracted claims — alarm/criticism about a downstream consequence or a related event must not flip author_lean opposite to what the byline's own claims about the event itself already established. A/B'd against the live model on 13 synthetic regression cases (zero regressions); real-corpus validation was inconclusive — see PR description. |
| extractor | v4 | `fc550c6255ecaa31` | 2026-08-27 | retro#680 | Renamed the elicited field `certainty` → `claim_strength` (Oracle 1.5 Phase 1). No rule, threshold or worked-example VALUE changed — only the identifier, at every site where it names the output slot; the four places the prompt uses "certainty" as ordinary English (e.g. "LATE is certainty the claim is FALSE") keep the word. The rename separates the SOURCE's commitment from the READER's confidence in its own interpretation, which becomes its own field in retro#681. `certainty` stays populated as a wire alias for one schema cycle. |
| extractor | v5 | `5007a1ace1558407` | 2026-08-28 | retro#681 | Added the elicited `reader_confidence` object `{level: high\|medium\|low, trap: null\|negation\|numeric_comparison\|entity_or_event_mismatch\|tone_vs_content\|inference_needed\|conflicting_signals}` (Oracle 1.5 Phase 1) — the READER's confidence in its own reading of a span, the other half of the retro#680 split from the SOURCE's `claim_strength`. New `## READER_CONFIDENCE` section after `## FACT_SIGNAL`, a paragraph in the output instructions, and the field on three worked examples; no existing rule, threshold or example VALUE changed. Deliberately not a scalar — verbalised LLM confidence clusters at 0.8-0.9 whatever the input; each `trap` name matches a detector that already exists, so the self-flag is checkable. Shadow: populated and persisted, read by nothing. |
| extractor | v6 | `6fda05b38efe0ed5` | 2026-08-28 | retro#681 | Recalibrated `reader_confidence.level` and countered the Nova-Lite optional-field crowding measured on v5. (1) `level` is now set by COUNTING resolution steps between span and related event (zero / one / two-or-more, plus "the link is wholly inferred" and "two stance signs are defensible" as `low`), replacing the v5 self-assessment framing ("would another careful reader read it differently?"). v5 returned `low` **once in 346 predictions** across both extractor models — the three-level scale collapsed to a binary for the same reason plan §4.1 rejects a scalar, so Phase 4's `low` down-weighting had no rows to act on. (2) The sole `low` worked example used `entity_or_event_mismatch`, a trap neither model ever returned; a second `low` example now uses `inference_needed` (the most-used trap, 29-40%). (3) `level`/`trap` independence stated in both directions. (4) The output-instruction paragraph no longer re-argues the field, only names its shape. (5) Two worked examples now carry the fact block — `fact_signal` + facets on one, `fact_signal_absent_reason` on another — because v5's examples demonstrated an output shape with NO fact block, and Nova Lite's `fact_signal` fill fell 30% → 0-6% while omitting BOTH `fact_signal` and its absent-reason, violating the retro#471 never-omit-both invariant. |
| extractor | v7 | `8470e187c3b7d30b` | 2026-08-28 | retro#681 | Fixes a `facet` collapse that v6 caused and, in causing it, proved the mechanism behind the whole retro#680/#681 Nova regression story. v6 added the fact block to a worked example but listed only the five fields the output instructions enumerate — and `facet` is **not** among them: the instruction paragraph says "the four facets above", naming event_actors / event_target / is_occurrence / verified, while `facet` is specified solely in the `## FACT_SIGNAL` prefix section. Until v6 that latent inconsistency was harmless (Haiku filled `facet` 68% from the prefix rule alone); once a worked example showed a *complete-looking* fact block without it, `facet` went to **0% on both models** and took two Haiku gate cases with it. v7 adds `facet` to that example AND to the output-instruction enumeration. The general finding: a worked example is read as the definitive enumeration of a block — omission from it is a far stronger signal than presence in it. |
| extractor | v8 | `a0ce599eff01356d` | 2026-08-29 | retro#686 | Added two elicited fields unparked from retro#673 (Oracle 1.5 Phase 1), both shadow — populated and persisted, read by nothing. (1) Per-claim `report_kind: level\|change` — whether the span reports the standing situation or a step in it, so a Phase 4 consumer can stop reading "rose to 61" and "is at 61" as the same evidence. (2) Article-level `consensus_view: expects_yes\|expects_no\|divided` — what the ARTICLE says OTHERS expect, which is not `author_lean` (the byline's own view) and rides beside it. New `## REPORT_KIND` and `## CONSENSUS_VIEW` prefix sections before `## Output`; both fields added to the output-instruction enumeration AND to every worked example, per the v7 finding that a worked example is read as the definitive enumeration of its block. Example VALUES are deliberately varied (`change`/`change`/`level`; `expects_yes`/`expects_no`) rather than repeated — `report_kind`'s own kill criterion is ">90% one value", so examples that agreed with each other would manufacture the failure they are meant to detect. Both are flat enums, not scalars: #673's caveat is that every graded field is a fresh site for the #394 pathology where a scalar collapses onto its band labels, and one bit cannot collapse. Schema growth held to 20,287 → 21,339 chars (+1,052, +5.2%) by writing both `Field(description=...)` strings as pointers to the prose blocks instead of restating the rule — v5's +8.8% is what moved Nova Lite's `fact_signal` fill. No existing rule, threshold or example VALUE changed. |
| extractor | v9 | `79eed80894b8df7f` | 2026-08-30 | retro#683 | Added the elicited `quantity` object `{value, unit, comparator: =\|<\|<=\|>\|>=\|between, value_hi, as_of}` (Oracle 1.5 Phase 1, unparked from retro#664's P2), shadow — populated and persisted, read by nothing. The model is asked for the NUMBER and never for the verdict: whether it clears the question's bar is arithmetic, done in code (`tm.threshold_compare`), because PR#671 measured Nova Lite returning stance +0.00 on every between-bounds case and inverting both tone traps — three prompt sections already tell it to compare and it still does not, which is why #664's P2 was resolved as *the field, not another prompt fix*. New `## QUANTITY` prefix section after `## CONSENSUS_VIEW` and before `## Output`, added to the output-instruction enumeration, plus a THIRD worked example (Airline A / daily departures) carrying it. Unlike v8's two fields this one is legitimately optional, so it is shown on one worked prediction rather than all four — the v7 rule is that an example must SHOW a field, and showing it everywhere would teach that a figure is always available. Worked example and prose deliberately use domains the retro#664 A/B corpus does not (containers, departures), so the corpus can still measure the prompt. `quantity` is kept explicitly apart from `quantitative_estimate`, which retro#362 narrowed to a cited PROBABILITY — a share, count, rate or tonnage goes here and leaves that field null. Schema 21,339 → 23,795 chars (+2,456, +11.5%), the largest bump since v5: a nested object costs five `Field(description=...)` strings and a `$defs` entry, and the descriptions are already pointers to the prose block rather than restatements of it. No existing rule, threshold or example VALUE changed. **Measured on the numeric corpus, both raters, exact (value, unit, comparator) on the case's own number: Haiku 4.5 50/50 runs (100%), zero comparator errors; Nova Lite 120/150 (80%), 27 comparator errors (18%).** Haiku clears retro#683's ≥0.9 bar, Nova Lite does not, and the gap is entirely `comparator`: a verb of movement is written as a bound ("accelerated to 4.1 percent" → `> 4.1`, 15/15; "collapsed to just 35 percent" → `< 35`), with value and unit right in every failing run. Two follow-up prompt edits aimed squarely at that (v9b, v9c: an explicit movement-verb rule, using verbs the corpus does not) made it WORSE — target 80% → 72%, fill 86% → 74% — and were reverted, so the shipped v9 is the original edit. The lesson is v9's own: this rater substitutes tone for the number at whatever level you ask it, and more prose about it costs fill without buying accuracy. Consequence recorded in `tm/threshold_compare.py`: Nova Lite's `quantity` is unfit for any consumer, and Phase 2 must gate on the rater rather than on the field being filled. |
| extractor | v10 | `f4c9cbe799f4c93f` | 2026-08-30 | retro#684 | Added two elicited fields (Oracle 1.5 Phase 1, unparked from retro#673), both shadow — populated, projected, persisted, read by nothing. (1) `tone` (`approve`\|`neutral`\|`alarm`), the quote's own register. The leak retro#326 and retro#657 keep patching by prompt is a *projection* problem — an evaluation read as a direction — and the v8 section `## Alarming or critical tone is not stance direction` can only tell the model where NOT to put it. `tone` is where it goes instead, and the new `## TONE` block says so explicitly rather than repeating the prohibition. (2) `voice` (`{kind: byline\|quoted_person\|institution\|wire\|unattributed, attributed_to?}`), whose assertion the quote is: a wire carried by thirty outlets is ONE observation, and `attributed_to` is the name Phase 3 S2's reception matrix keys its column on. Both at the schema TAIL after `quantity` (retro#680), both on all FOUR worked predictions — unlike `quantity`, neither is ever legitimately absent, so the v7 rule cuts the other way here. Prose and examples use a domain the A/B corpus does not (a regulator revoking an operator's licence). `Voice` deliberately carries **no cross-field validator** where `Quantity` has one: rejecting an `attributed_to` beside a `byline` would hand `_drop_malformed_voice` a raise, and the guard nulls the WHOLE object — trading a stray string for a lost `kind`, which is a lost observation. Schema 23,795 → 25,950 chars (+2,155, +9.1%); hand-written 62,564 → 66,718. **Measured, all five corpus files, 5 runs/case, both raters: `gate_exit_code` 0 on BOTH arms — zero regressions, and Haiku additionally FIXED `bracket-favorite-one-stage-of-several`. `token_usage` +7.9% (Haiku 2,867,733 → 3,093,963) and +7.7% (Nova Lite 2,644,601 → 2,847,019), well inside the phase's +30% cap.** Distributions split the two raters completely, exactly as v9 did: **Haiku 4.5 fills both fields on 315/315** — `tone` neutral 84% / approve 10% / alarm 6%, `voice` byline 60% / institution 21% / unattributed 11% / quoted_person 6% / wire 2%, with `attributed_to` set on exactly the 90 non-`byline`/non-`unattributed` predictions and no others — so **neither kill criterion is anywhere near tripping**, and the tone × stance cross-tab shows 6.3% DISCORDANT pairs landing on PR#671's own trap class (`threshold-tone-negative-number-satisfies`: "support collapsed to just 35 percent … a humiliating fall", where 35 satisfies the question → `tone` alarm at stance **+0.2**). **Nova Lite answers `neutral` on 145/145**, including all three deliberately tonal cases, which **trips the >90%-neutral kill criterion outright**. Not a flat corpus: on those same cases its register surfaces in `stance` instead (−0.5 on an article whose number satisfies the question), so this rater reads the tone and writes it on the wrong axis, and handing it the right axis did not move it — the same shape as v9's comparator finding, and recorded beside `_TONE_VALUES` in `tm/models.py`. Consequence: **`tone` is a Haiku-only field and any Phase 3 S4 consumer must gate on the RATER**, not on the field being filled; a batch row's `neutral` means "Nova Lite", not "even-handed". `voice` clears the bar on BOTH raters (Nova Lite byline 83%, under the 90% bar) and is the rater-agnostic half of #684. No existing rule, threshold or example VALUE changed. |
| extractor | v12 | `2debf9b406f26d01` | 2026-08-31 | retro#763 | Added the elicited `grounds` object `{kind, basis}` (Oracle 1.5 Phase 1, unparked from retro#673 §1), shadow — populated, projected, persisted, read by nothing. `kind` is what the quote's position RESTS ON — the reason, not the direction — so the pool can count *reasons* rather than articles (three outlets repeating one ministry statement are ONE ground; three citing a milestone, a poll and a precedent are three) and recognise same-grounds-same-stance duplicates; `basis` is the phrase that lets two same-`kind` rows be recognised as the SAME statement. Explicitly NOT an `evidence_class` extension: class is the route the information took (reported/cited/opinion), grounds is what was seen at the far end — the two vary independently and are documented as never sharing a value. Schema tail after `claim_scope`; `provenance.schema_version` 1.5 → 1.6. **Full-corpus A/B on v12c** (all five case files, 5 runs/case, both raters): gate 9/10 (Haiku FAILs `poll_facet` on one already-disabled-gate case, Nova passes it — rater-specific, not a prompt-text property); `grounds` fill Haiku 290/290 (100%) / Nova 128/144 (89%), `basis` 100% on both, kind-share max Haiku 35.5% / Nova 60.2% (both well under the >90% kill bar); tokens +6.1% Haiku / +5.5% Nova, inside the +30% cap. **Blocker the gate could not see:** the GROUNDS kind `official_statement` leaked into the sibling `evidence_class` field 21 times — a real, load-bearing field that keys `evidence_class_weight` — pushing `evidence_class is None` from a 0.6% prod baseline to 3.4% (Haiku) / 7.6% (Nova), each unclassified claim silently down-weighted to a 0.25 certainty cap (4-16x below its true class). Saying "the two enums never share a value" in the prompt text held across three wording variants (v12a-c) and did not stop it — a naming problem, not a wording one. **v12d renames the `grounds.kind` vocabulary to share no lexical stem with the five `evidence_class` values** (`observed_milestone`→`event_observed`, `official_statement`→`authority_asserted`, `market_or_poll_figure`→`market_or_poll_number`, `analyst_inference`→`expert_inference`, `precedent_or_base_rate`→`historical_base_rate`, `authors_judgement`→`writer_assertion`); the `_drop_out_of_enum` guard under `evidence_class` stays as the floor regardless. **v12d's own success criterion — `evidence_class is None` back to ~0% on both raters, `grounds` fill/kind-spread held — has not yet been measured by a fresh A/B.** Do not treat v12d as shippable until that run exists. |
| extractor | v11 | `0c56ccd03d50132f` | 2026-08-30 | retro#697 | Added three article-level shadow fields carrying the WHO / WHAT / SCOPE decomposition that `## MATCH THE EVENT` has required since v1 and never given a field to: `claim_actor` `{name, type}`, `claim_predicate`, `claim_scope` — the QUESTION's own event, so two articles on one forecast should answer identically (the harness now reports that consistency). **Question-level, not per-claim** as the issue wrote it: the decomposition is invariant across every claim, and `settlement_semantic.claim_subject_from_fields` (new, the only `proxy=False` constructor) wants one per question. Schema TAIL after `consensus_view`; all three worked examples carry them; `ClaimActor` deliberately has NO docstring (a docstring is schema — retro#700 — and cost 378 chars before it was moved to a comment; the guard test now walks `$defs`). **No prose paragraph in `PROMPT_PREFIX`.** One was written and measured through five variants — placed mid-section (inverted Nova's stance 5/5 on the conflation control), moved to the section end, cut to 445 chars, reworded ("signed against it" read as *opposing* and broke the register-trap case on BOTH raters) — and then deleted entirely: `claim_actor` still fills 15/15 on Nova with nothing but the schema descriptions, the `PROMPT_SUFFIX` contract and the examples. Every byte of it had only moved damage between cases. Prompt 66,718 → 67,770 (+1.6%); schema 25,950 → 28,022 chars (+8.0%). **Measured, all five corpus files, 5 runs/case, both raters: Haiku 4.5 `gate_exit_code` 0 on all five, 130/130 fill on all three fields, consistency 1.00 (every case answers identically across runs). Nova Lite: 3 improvements (`bracket-series-lead-not-the-series`, `bracket-final-stage-cleared`, `stance-tone-conflation-hazard-persists`) and ONE regression, `threshold-at-or-below-satisfied`.** That case was then measured at 15 runs per cell: baseline 8/15 and 13/15 on two samples, every v11 variant 0–4/15 — and, decisively, the **untouched v10 prompt with the decomposition merely appended to the event description as input** also scores 0/15. It is retro#683's comparator failure (Nova cannot do the number comparison and any extra event text tips it), routed by retro#688's `threshold_extractor_model`, not something this prompt can hold; the baseline pass was a coin flip. `token_usage` **+12.7% Haiku** (2,857,125 → 3,221,003), **+4.1% Nova** (2,846,992 → 2,963,334), inside the phase cap. Nova fills `claim_predicate`/`claim_scope` 130/130 but `claim_actor` 117/130: it answers `type: "team"` on the sports brackets and the drop guard nulls the object (Haiku says `other`, 15×); its consistency is 1.10 / 1.29 / 1.45 against Haiku's 1.00, and on `claim_scope` it sometimes pastes the whole question in. Consumers: Haiku's decomposition is usable as-is; Nova's is usable for `claim_actor.name` and `claim_predicate`, and `claim_scope` should be gated on the rater. Two spin-outs: retro#757 (the 5-run gate flipped this one case's verdict four times — voting, per retro#532) and retro#758 (injecting the decomposition as INPUT took the decider-denial sentinel 11 → 15/15 on Nova with zero prompt change). Acceptance criterion 3 — re-scoring `gate_predicate_echo` — is **not met here and cannot be**: every settled pool row predates v11; it needs a pool extracted after this ships. |

## ⚠️ This table does NOT cover the batch pipeline

`pipeline/src/tm/orchestrator.py`'s `EXTRACTION_PROMPT_VERSION` constant is a
**separate, unrelated version scheme** — do not cross-reference it against the
table above, and do not assume a version/hash bump here also applies to batch
output.

- It versions the batch pipeline's own extraction cache (used for
  cache-invalidation inside `_negative_marker_is_current`), not anything
  exposed on an API response.
- It has no SHA-256 prompt-hash equivalent and is **not enforced by
  `test_prompt_version_enforcement.py`** — a batch prompt edit can land
  without any version bump and CI will not catch it.
- It is bumped independently of this table and is **not guaranteed to stay in
  sync** with the live-path versions above, even though both track prompts in
  the same file family.
- Because daatan's evidence pool ingestion only ever calls the live API (not
  the batch pipeline), stored `EvidencePoolArticle` provenance
  (`extractorPromptVersion`/`extractorPromptHash`) always refers to *this*
  table's scheme, never to `EXTRACTION_PROMPT_VERSION`.

This is a known, accepted gap (retro#631) — not something planned to be
unified in the near term, since the batch lane isn't currently read by
daatan's evidence pool. If that changes, the batch pipeline will need its own
hash-based provenance and lock-file enforcement, mirroring retro#627/#632.
