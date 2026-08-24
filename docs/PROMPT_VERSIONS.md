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

| Component | Version | Hash | Effective from | PR | Summary |
|---|---|---|---|---|---|
| gatekeeper | v1 | `a09cdb5ecda0ce5e` | 2026-08-24 | retro#627 | Initial versioned baseline — no prior version existed on the live path. |
| extractor | v1 | `6371300bb3b89b8c` | 2026-08-24 | retro#627 | Initial versioned baseline — no prior version existed on the live path. |

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
