"""Enforces the docs/PROMPT_VERSIONS.md convention (repo CLAUDE.md, "A new elicited
field is not done when the A/B passes", item 2): a version bump's changelog row must
carry the measured per-rater numbers, not just a description.

Scope note (retro#807, re-scoped 2026-09-08): the issue as filed asked for the
extractor A/B harness (`scripts/ab_extractor_prompt.py`) to run against every
configured model in CI on any prompt/schema PR. `tests.yml` has no Bedrock/AWS
credentials — it mocks all LLM calls by design (see the workflow's own header
comment) — so running the harness live in CI is a spend decision blocked on Mark
(memory: LLM cost runway, credits ~end Nov). This file ships the deterministic
slice that needs no LLM access: it fails CI if a version bump's own changelog row
never names both live raters (Haiku 4.5 — the live `oracle-api` extractor override,
and Nova Lite — the batch `truthmachine.service` default, see repo CLAUDE.md's
infra cheat-sheet) by name. That is the literal failure mode the issue describes —
retro#778's row (v14) *did* document Nova's regression, but a PR that skipped
measuring one rater entirely would previously sail through with zero enforcement.
Silently syncing a known-regressed version to the batch lane (the rest of #778) is
a deployment-pinning gap, not a merge-gate gap, and is out of scope here.
"""
from __future__ import annotations

import re
from pathlib import Path

from forecast_api import forecaster

PROMPT_VERSIONS_PATH = (
    Path(__file__).resolve().parents[2] / "docs" / "PROMPT_VERSIONS.md"
)

# The two raters that actually run this prompt in a live lane (repo CLAUDE.md
# infra cheat-sheet): Haiku 4.5 on the live oracle-api extractor override, Nova
# Lite on the batch truthmachine.service default. Named literally so a future
# roster change (a new lane, a swapped model) has to touch this list on purpose.
LIVE_RATERS = ("Haiku", "Nova")


def _changelog_text() -> str:
    return PROMPT_VERSIONS_PATH.read_text()


def _find_row(component: str, version: str) -> str:
    """Return the full changelog-table line for `component`'s current version.

    Each row in docs/PROMPT_VERSIONS.md's changelog table is one (long) physical
    line, so matching the whole line sidesteps splitting on '|' — several rows
    contain escaped literal pipes inside inline code (e.g. `` `\\|stance\\|` ``)
    that would misparse a naive column split.
    """
    pattern = re.compile(
        rf"^\|\s*{re.escape(component)}\s*\|\s*{re.escape(version)}\s*\|.*$",
        re.MULTILINE,
    )
    match = pattern.search(_changelog_text())
    assert match is not None, (
        f"No docs/PROMPT_VERSIONS.md changelog row found for '{component} {version}' "
        f"(forecaster.{component.upper()}_PROMPT_VERSION). Every version bump needs a "
        "row in the table, added in the same PR that bumps the constant."
    )
    return match.group(0)


def test_current_extractor_row_names_both_live_raters():
    row = _find_row("extractor", forecaster.EXTRACTOR_PROMPT_VERSION)
    missing = [rater for rater in LIVE_RATERS if rater.lower() not in row.lower()]
    assert not missing, (
        f"docs/PROMPT_VERSIONS.md's row for extractor {forecaster.EXTRACTOR_PROMPT_VERSION} "
        f"never mentions {', '.join(missing)}. Both live raters (Haiku 4.5 on the live "
        "oracle-api extractor, Nova Lite on the batch pipeline) must be measured and "
        "reported per repo CLAUDE.md's elicited-field checklist item 2 — 'assume the "
        "raters disagree until measured'. If a bump is a baseline/removal that "
        "genuinely needs no fresh A/B (e.g. withdrawing a field), say so explicitly "
        "for both raters rather than omitting one."
    )
