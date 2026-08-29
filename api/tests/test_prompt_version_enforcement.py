"""Enforces the docs/PROMPT_VERSIONS.md convention: if a prompt edit changes the
rendered gatekeeper/extractor prompt text, *_PROMPT_VERSION must be bumped and
docs/prompt_versions.lock.json updated to match — otherwise provenance.models'
prompt_version label goes stale relative to what actually ran (retro#632).
"""
import json
from pathlib import Path

import pytest

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


# ── the conditionally-appended tail blocks (retro#731) ───────────────────────
#
# The four tests above cover `PROMPT_PREFIX + PROMPT_SUFFIX` and the serialised
# schema. Neither call sends only that: five instruction blocks are appended to
# `prompt` AFTER `PROMPT_SUFFIX.format(...)` returns, so they fall outside the
# reconstructed *_PROMPT and were outside every hash here. 6,576 chars of live
# prompt text, 4,233 of it in `_CONDITIONAL_BLOCK` alone — larger than the whole
# gatekeeper prompt and its schema together, and carrying three worked examples.
#
# retro#720 is the standing demonstration of what an unwatched block of worked
# examples does: `## Multi-stage / bracket events`, inside the *hashed* prefix,
# suppressed Nova Lite on unrelated numeric-threshold claims (8/20 -> 18/20 when
# removed), because its examples all pointed one way. Same hazard here, and until
# now none of the protection.
#
# One assertion per block. A single hash over the concatenation would be cheaper
# and would report every change as "the tail moved", which tells a reviewer
# nothing about what to review.


def _tail_lock(name: str) -> dict:
    entry = _lock().get("tail_blocks", {}).get(name)
    assert entry is not None, (
        f"docs/prompt_versions.lock.json has no tail_blocks entry for '{name}'. "
        "A block reaching the model with no lock entry is exactly the gap retro#731 closed."
    )
    return entry


@pytest.mark.parametrize("name", sorted(forecaster.PROMPT_TAIL_BLOCKS))
def test_tail_block_hash_matches_lock_file(name: str):
    entry = _tail_lock(name)
    text = forecaster.PROMPT_TAIL_BLOCKS[name]
    assert entry["hash"] == forecaster.PROMPT_TAIL_BLOCK_HASHES[name], (
        f"The '{name}' prompt block changed ({_delta(len(text), entry['chars'])}). "
        "It is appended to the prompt at call time, so this is a PROMPT EDIT, not a "
        "code change: bump the owning prompt's *_PROMPT_VERSION, add a row to "
        "docs/PROMPT_VERSIONS.md, and update docs/prompt_versions.lock.json. If the "
        "block carries worked examples, run the A/B harness first — retro#720."
    )


@pytest.mark.parametrize("name", sorted(forecaster.PROMPT_TAIL_BLOCKS))
def test_tail_block_size_matches_lock_file(name: str):
    """The size ratchet again, per block — see test_schema_sizes_match_lock_file.

    Same reasoning: a hash mismatch says *something* moved, and the cheapest way
    to smuggle a regression past review is growth nobody costed.
    """
    entry = _tail_lock(name)
    text = forecaster.PROMPT_TAIL_BLOCKS[name]
    assert entry["chars"] == len(text), (
        f"The '{name}' prompt block's size moved: {_delta(len(text), entry['chars'])}"
    )


def test_every_appended_block_is_locked():
    """The partition guard: no block reaches the model unlocked.

    The two tests above iterate `PROMPT_TAIL_BLOCKS`, so a block added to
    extractor.py/gatekeeper.py and appended at call time — but never registered
    here — would be covered by nothing and no test would notice. That is precisely
    how this gap was born: `_CONDITIONAL_BLOCK` was added, appended, and shipped,
    and every enforcement test in this file stayed green for its whole life.

    Kept as a hard-coded expectation rather than derived from the same dict the
    other tests iterate: a guard that reads its own answer off the thing it guards
    cannot fail. Adding a sixth block should make this fail and force a decision.
    """
    assert set(forecaster.PROMPT_TAIL_BLOCKS) == {
        "extractor.conditional",
        "extractor.short_form",
        "extractor.language_hint",
        "gatekeeper.short_form",
        "gatekeeper.language_hint",
    }, (
        "The set of prompt blocks appended after PROMPT_SUFFIX.format() changed. If you added "
        "one, register it in forecaster.PROMPT_TAIL_BLOCKS, add a docs/prompt_versions.lock.json "
        "entry, and add it here. If you removed one, drop it from all three."
    )
    assert set(forecaster.PROMPT_TAIL_BLOCKS) == set(_lock()["tail_blocks"]), (
        "forecaster.PROMPT_TAIL_BLOCKS and the lock file's tail_blocks disagree about which "
        "blocks exist."
    )
