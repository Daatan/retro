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

## The prompt is the prose **and** the schema (retro#700)

`PROMPT_PREFIX`/`PROMPT_SUFFIX` are not all the text the model reads. Both the
gatekeeper and the extractor are structured calls through `instructor` in
`Mode.MD_JSON`, which serialises the response model's JSON schema into a system
message on every call — every `Field(description=...)`, every enum member, and
every Pydantic **model docstring** (Pydantic copies those into the schema's
`description` key). Measured on today's models:

| Component | Hand-written prompt | Rendered schema | Schema share |
|---|---|---|---|
| extractor (`ExtractionOutput`) | 55,116 chars | 20,287 chars | **27%** |
| gatekeeper (`GatekeeperOutput`) | 4,603 chars | 1,178 chars | **20%** |

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

| Component | Version | Hash | Effective from | PR | Summary |
|---|---|---|---|---|---|
| gatekeeper | v1 | `a09cdb5ecda0ce5e` | 2026-08-24 | retro#627 | Initial versioned baseline — no prior version existed on the live path. |
| extractor | v1 | `6371300bb3b89b8c` | 2026-08-24 | retro#627 | Initial versioned baseline — no prior version existed on the live path. |
| extractor | v2 | `6813d6184d14568b` | 2026-08-25 | retro#649 | Added "tone is not stance direction" rule + worked example (retro#545) — alarming/critical tone getting read as negative stance regardless of the claim's actual content. |
| extractor | v3 | `5a8082ab250463eb` | 2026-08-25 | retro#326 | Added an author_lean self-consistency cross-check against the article's own extracted claims — alarm/criticism about a downstream consequence or a related event must not flip author_lean opposite to what the byline's own claims about the event itself already established. A/B'd against the live model on 13 synthetic regression cases (zero regressions); real-corpus validation was inconclusive — see PR description. |
| extractor | v4 | `fc550c6255ecaa31` | 2026-08-27 | retro#680 | Renamed the elicited field `certainty` → `claim_strength` (Oracle 1.5 Phase 1). No rule, threshold or worked-example VALUE changed — only the identifier, at every site where it names the output slot; the four places the prompt uses "certainty" as ordinary English (e.g. "LATE is certainty the claim is FALSE") keep the word. The rename separates the SOURCE's commitment from the READER's confidence in its own interpretation, which becomes its own field in retro#681. `certainty` stays populated as a wire alias for one schema cycle. |
| extractor | v5 | `5007a1ace1558407` | 2026-08-28 | retro#681 | Added the elicited `reader_confidence` object `{level: high\|medium\|low, trap: null\|negation\|numeric_comparison\|entity_or_event_mismatch\|tone_vs_content\|inference_needed\|conflicting_signals}` (Oracle 1.5 Phase 1) — the READER's confidence in its own reading of a span, the other half of the retro#680 split from the SOURCE's `claim_strength`. New `## READER_CONFIDENCE` section after `## FACT_SIGNAL`, a paragraph in the output instructions, and the field on three worked examples; no existing rule, threshold or example VALUE changed. Deliberately not a scalar — verbalised LLM confidence clusters at 0.8-0.9 whatever the input; each `trap` name matches a detector that already exists, so the self-flag is checkable. Shadow: populated and persisted, read by nothing. |

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
| extractor | v6 | `6fda05b38efe0ed5` | 2026-08-28 | retro#681 | Recalibrated `reader_confidence.level` and countered the Nova-Lite optional-field crowding measured on v5. (1) `level` is now set by COUNTING resolution steps between span and related event (zero / one / two-or-more, plus "the link is wholly inferred" and "two stance signs are defensible" as `low`), replacing the v5 self-assessment framing ("would another careful reader read it differently?"). v5 returned `low` **once in 346 predictions** across both extractor models — the three-level scale collapsed to a binary for the same reason plan §4.1 rejects a scalar, so Phase 4's `low` down-weighting had no rows to act on. (2) The sole `low` worked example used `entity_or_event_mismatch`, a trap neither model ever returned; a second `low` example now uses `inference_needed` (the most-used trap, 29-40%). (3) `level`/`trap` independence stated in both directions. (4) The output-instruction paragraph no longer re-argues the field, only names its shape. (5) Two worked examples now carry the fact block — `fact_signal` + facets on one, `fact_signal_absent_reason` on another — because v5's examples demonstrated an output shape with NO fact block, and Nova Lite's `fact_signal` fill fell 30% → 0-6% while omitting BOTH `fact_signal` and its absent-reason, violating the retro#471 never-omit-both invariant. |
| extractor | v7 | `8470e187c3b7d30b` | 2026-08-28 | retro#681 | Fixes a `facet` collapse that v6 caused and, in causing it, proved the mechanism behind the whole retro#680/#681 Nova regression story. v6 added the fact block to a worked example but listed only the five fields the output instructions enumerate — and `facet` is **not** among them: the instruction paragraph says "the four facets above", naming event_actors / event_target / is_occurrence / verified, while `facet` is specified solely in the `## FACT_SIGNAL` prefix section. Until v6 that latent inconsistency was harmless (Haiku filled `facet` 68% from the prefix rule alone); once a worked example showed a *complete-looking* fact block without it, `facet` went to **0% on both models** and took two Haiku gate cases with it. v7 adds `facet` to that example AND to the output-instruction enumeration. The general finding: a worked example is read as the definitive enumeration of a block — omission from it is a far stronger signal than presence in it. |
