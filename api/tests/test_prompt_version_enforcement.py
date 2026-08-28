"""Enforces the docs/PROMPT_VERSIONS.md convention: if a prompt edit changes the
rendered gatekeeper/extractor prompt text, *_PROMPT_VERSION must be bumped and
docs/prompt_versions.lock.json updated to match — otherwise provenance.models'
prompt_version label goes stale relative to what actually ran (retro#632).
"""
import json
from pathlib import Path

from forecast_api import forecaster

LOCK_PATH = Path(__file__).resolve().parents[2] / "docs" / "prompt_versions.lock.json"


def _lock():
    return json.loads(LOCK_PATH.read_text())


def test_gatekeeper_hash_matches_lock_file():
    lock = _lock()["gatekeeper"]
    assert lock["version"] == forecaster.GATEKEEPER_PROMPT_VERSION, (
        "GATEKEEPER_PROMPT_VERSION changed but docs/prompt_versions.lock.json wasn't updated to match"
    )
    assert lock["hash"] == forecaster.GATEKEEPER_PROMPT_HASH, (
        "Gatekeeper prompt text changed (hash mismatch) without bumping GATEKEEPER_PROMPT_VERSION. "
        "Bump the version, add a row to docs/PROMPT_VERSIONS.md, and update docs/prompt_versions.lock.json."
    )


def test_extractor_hash_matches_lock_file():
    lock = _lock()["extractor"]
    assert lock["version"] == forecaster.EXTRACTOR_PROMPT_VERSION, (
        "EXTRACTOR_PROMPT_VERSION changed but docs/prompt_versions.lock.json wasn't updated to match"
    )
    assert lock["hash"] == forecaster.EXTRACTOR_PROMPT_HASH, (
        "Extractor prompt text changed (hash mismatch) without bumping EXTRACTOR_PROMPT_VERSION. "
        "Bump the version, add a row to docs/PROMPT_VERSIONS.md, and update docs/prompt_versions.lock.json."
    )


# ── the schema half of the prompt (retro#700) ────────────────────────────────
#
# The two tests above cover the hand-written prompt. They are not the whole
# guarantee: both calls are structured calls through instructor's MD_JSON mode,
# which serialises the response model's JSON schema into a system message, so
# field descriptions, enum members and Pydantic docstrings are prompt text too —
# 27% of the extractor's and 20% of the gatekeeper's. Before this, that half
# could change arbitrarily with no hash movement and no version bump.


def _delta(actual: int, locked: int) -> str:
    diff = actual - locked
    pct = (diff / locked * 100) if locked else float("inf")
    return f"{locked} -> {actual} chars ({diff:+d}, {pct:+.1f}%)"


def test_gatekeeper_schema_hash_matches_lock_file():
    lock = _lock()["gatekeeper"]
    assert lock["schema_hash"] == forecaster.GATEKEEPER_SCHEMA_HASH, (
        "GatekeeperOutput's rendered JSON schema changed — that text goes to the model on "
        f"every call ({_delta(len(forecaster.GATEKEEPER_SCHEMA), lock['schema_chars'])}). "
        "Bump GATEKEEPER_PROMPT_VERSION, add a row to docs/PROMPT_VERSIONS.md, and update "
        "docs/prompt_versions.lock.json."
    )


def test_extractor_schema_hash_matches_lock_file():
    lock = _lock()["extractor"]
    assert lock["schema_hash"] == forecaster.EXTRACTOR_SCHEMA_HASH, (
        "ExtractionOutput's rendered JSON schema changed — that text goes to the model on "
        f"every call ({_delta(len(forecaster.EXTRACTOR_SCHEMA), lock['schema_chars'])}). "
        "Bump EXTRACTOR_PROMPT_VERSION, add a row to docs/PROMPT_VERSIONS.md, and update "
        "docs/prompt_versions.lock.json."
    )


def test_schema_sizes_match_lock_file():
    """The size ratchet, separate from the hash on purpose.

    A hash mismatch says *something* moved. The retro#681 regression was not an
    unnoticed edit — it was a 1,283-char docstring nobody costed, and what would
    have caught it the same day is a number in the diff. Any PR that grows the
    schema has to write down by how much.
    """
    lock = _lock()
    assert lock["gatekeeper"]["schema_chars"] == len(forecaster.GATEKEEPER_SCHEMA), (
        "Gatekeeper schema size moved: "
        + _delta(len(forecaster.GATEKEEPER_SCHEMA), lock["gatekeeper"]["schema_chars"])
    )
    assert lock["extractor"]["schema_chars"] == len(forecaster.EXTRACTOR_SCHEMA), (
        "Extractor schema size moved: "
        + _delta(len(forecaster.EXTRACTOR_SCHEMA), lock["extractor"]["schema_chars"])
    )
