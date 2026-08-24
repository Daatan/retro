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
