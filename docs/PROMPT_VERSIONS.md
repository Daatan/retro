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

| Component | Version | Effective from | PR | Summary |
|---|---|---|---|---|
| gatekeeper | v1 | 2026-08-24 | retro#627 | Initial versioned baseline — no prior version existed on the live path. |
| extractor | v1 | 2026-08-24 | retro#627 | Initial versioned baseline — no prior version existed on the live path. |

Note: `pipeline/src/tm/orchestrator.py`'s `EXTRACTION_PROMPT_VERSION` is a
separate, older constant scoped to the batch pipeline's own extraction
cache (used for cache-invalidation, not exposed on any API response). It
tracks the same prompt files but is bumped independently and is not
guaranteed to stay in sync with the versions in this table — a known
inconsistency, not something this change fixes.
