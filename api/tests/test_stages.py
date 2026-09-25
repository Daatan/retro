"""The Stage registry is complete and reproduces the legacy flag behaviour (retro#866).

Two kinds of test here, and the distinction matters:

* **Characterization** — ``mode()`` must agree with the raw booleans for every
  combination. Four of these stages are enforcing or shadow-logging on prod right
  now, so the registry is only safe to introduce if it provably describes what the
  existing flags already do.
* **Completeness** — every ``*_enabled`` field in ``ApiSettings`` that belongs to a
  shadow-then-promote feature has a ``Stage``. This is the part that stops the drift
  ``stages.py`` describes, modelled on
  ``pipeline/tests/test_extraction_field_consumers.py`` (retro#808).
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("ORACLE_API_KEY", "dummy")

from forecast_api.config import ApiSettings  # noqa: E402
from forecast_api.stages import STAGES, STAGES_BY_NAME, Mode, Stage, describe_stages  # noqa: E402


# ── Completeness ─────────────────────────────────────────────────────────────

# `*_enabled` settings deliberately kept out of STAGES. Listing them here is the
# decision the completeness test forces: a new flag lands in STAGES or lands here,
# never silently in neither. Two distinct reasons, not one:
#
#   caches / kill switches — no shadow lane and no promotion path at all.
#
#   board cutovers — `resolution_shadow_credibility_enabled` and
#   `hazard_shadow_enabled` ARE shadow-then-promote in spirit ("compute but don't
#   use"), but they promote on a scored board rather than a log line and each has
#   its own dedicated gate script (check_resolution_shadow_gate.py). retro#806
#   excluded them from the telemetry registry for exactly this reason and that
#   judgement is carried over rather than re-litigated here. Folding them into
#   STAGES would mean giving them event= tokens they do not emit.
NON_STAGE_ENABLED_FLAGS = frozenset(
    {
        # board cutovers with dedicated gate scripts
        "resolution_shadow_credibility_enabled",
        "hazard_shadow_enabled",
        # caches / kill switches
        "settlement_verdict_cache_enabled",
        "event_decomposition_cache_enabled",
        "subject_gate_cache_enabled",
    }
)


def _enabled_flags() -> set[str]:
    return {f for f in ApiSettings.model_fields if f.endswith("_enabled")}


def test_every_enabled_flag_is_either_a_stage_or_explicitly_not_one():
    registered = {s.enabled_attr for s in STAGES}
    unclassified = _enabled_flags() - registered - NON_STAGE_ENABLED_FLAGS
    assert not unclassified, (
        "New *_enabled flag(s) with no decision recorded: "
        f"{sorted(unclassified)}. Add a Stage to forecast_api/stages.py, or list it in "
        "NON_STAGE_ENABLED_FLAGS if it is a plain switch with no shadow lane."
    )


def test_registry_has_no_stale_entries():
    fields = set(ApiSettings.model_fields)
    for stage in STAGES:
        assert stage.enabled_attr in fields, f"{stage.name}: no such setting {stage.enabled_attr}"
        if stage.enforce_attr is not None:
            assert stage.enforce_attr in fields, (
                f"{stage.name}: no such setting {stage.enforce_attr}"
            )


def test_non_stage_list_has_no_stale_entries():
    assert NON_STAGE_ENABLED_FLAGS <= _enabled_flags(), (
        "NON_STAGE_ENABLED_FLAGS names a setting that no longer exists: "
        f"{sorted(NON_STAGE_ENABLED_FLAGS - _enabled_flags())}"
    )


def test_every_enforce_flag_belongs_to_a_stage():
    enforce_fields = {f for f in ApiSettings.model_fields if f.endswith("_enforce")}
    registered = {s.enforce_attr for s in STAGES if s.enforce_attr}
    assert enforce_fields == registered, (
        f"unregistered enforce flag(s): {sorted(enforce_fields - registered)}"
    )


def test_stage_names_and_events_are_unique():
    names = [s.name for s in STAGES]
    assert len(names) == len(set(names))
    seen: set[str] = set()
    for stage in STAGES:
        assert stage.events, f"{stage.name} declares no event= token"
        for event in stage.events:
            assert event.startswith("event="), f"{stage.name}: {event!r} is not an event= token"
            assert event not in seen, f"{event} claimed by two stages"
            seen.add(event)


def test_every_stage_cites_an_issue():
    for stage in STAGES:
        assert stage.issue.startswith("retro#"), f"{stage.name} has no issue reference"


# ── Characterization: mode() vs the raw booleans ─────────────────────────────


def _settings_with(**flags) -> SimpleNamespace:
    return SimpleNamespace(**flags)


@pytest.mark.parametrize("stage", STAGES, ids=lambda s: s.name)
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("enforce", [False, True])
def test_mode_matches_the_legacy_boolean_pair(stage: Stage, enabled: bool, enforce: bool):
    flags = {stage.enabled_attr: enabled}
    if stage.enforce_attr is not None:
        flags[stage.enforce_attr] = enforce
    settings = _settings_with(**flags)

    if not enabled:
        expected = Mode.OFF
    elif stage.enforce_attr is not None and enforce:
        expected = Mode.ENFORCE
    else:
        expected = Mode.SHADOW

    assert stage.mode(settings) is expected
    assert stage.is_enabled(settings) is enabled
    assert stage.is_enforcing(settings) is (expected is Mode.ENFORCE)


@pytest.mark.parametrize("stage", [s for s in STAGES if s.enforce_attr is None], ids=lambda s: s.name)
def test_a_stage_without_an_enforce_flag_can_never_enforce(stage: Stage):
    """A measurement-only lane is a different thing from one whose enforce flag is off."""
    settings = _settings_with(**{stage.enabled_attr: True})
    assert stage.mode(settings) is Mode.SHADOW
    assert stage.is_enforcing(settings) is False


def test_enforce_without_enabled_reads_as_off():
    """Enforce alone is not enforcement: the stage never runs, so it decides nothing.

    Pinned because it is the one combination the raw boolean pair lets a caller
    express and no call site honours.
    """
    stage = STAGES_BY_NAME["jev_gate"]
    settings = _settings_with(jev_gate_enabled=False, jev_gate_enforce=True)
    assert stage.mode(settings) is Mode.OFF
    assert stage.is_enforcing(settings) is False


# ── The live defaults, pinned ────────────────────────────────────────────────


def test_shipped_defaults_are_unchanged_by_this_refactor():
    """Guards the stages that are live on prod. Update only with a deliberate rollout.

    Reads the *declared class defaults*, not an instantiated ``ApiSettings``:
    ``tests/conftest.py`` sets ``SETTLEMENT_VERIFIER_ENABLED=false`` (and others)
    for the suite, so an instance here would pin the test environment rather than
    what actually ships.
    """
    declared = SimpleNamespace(
        **{name: field.default for name, field in ApiSettings.model_fields.items()}
    )
    assert describe_stages(declared) == {
        "conditional_attenuation": "shadow",
        "settlement_semantic_gates": "shadow",
        "settlement_verifier": "enforce",
        "subject_gate": "shadow",
        "premise_verifier": "off",
        "precursor_match": "off",
        "settled_grounding": "off",
        "retry_relaxed_search": "off",
        "jev_shadow": "off",
        "jev_gate": "off",
        "jev_gate_ab": "off",
    }


def test_the_suite_runs_with_settlement_verifier_disabled():
    """conftest.py turns it off; pinned so the divergence from prod stays visible."""
    settings = ApiSettings(oracle_api_key="dummy")
    assert STAGES_BY_NAME["settlement_verifier"].mode(settings) is Mode.OFF
    assert ApiSettings.model_fields["settlement_verifier_enabled"].default is True


# ── The telemetry registry cannot fork away from STAGES again ────────────────


def test_telemetry_registry_is_derived_from_stages():
    """Pinned so a future edit that reintroduces a hand-kept list there fails loudly.

    That is what happened before retro#866: the script kept its own registry and it
    drifted to four of the eleven stages.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from check_shadow_flag_telemetry import FLAG_REGISTRY  # noqa: PLC0415

    assert FLAG_REGISTRY == tuple(s for s in STAGES if s.expect_telemetry)


def test_telemetry_coverage_is_the_stages_that_log_per_request():
    """The one literal pin of who is checked — see `Stage.expect_telemetry` for the rule.

    Both halves are asserted: a stage silently dropping out of the daily check is as
    much a regression as one silently joining it.
    """
    assert {s.name for s in STAGES if s.expect_telemetry} == {
        # already covered before retro#866
        "premise_verifier",
        "precursor_match",
        "settled_grounding",
        "retry_relaxed_search",
        # added by retro#866 — these were the blind spots
        "subject_gate",
        "jev_shadow",
        "jev_gate",
        "jev_gate_ab",
    }
    assert {s.name for s in STAGES if not s.expect_telemetry} == {
        "conditional_attenuation",
        "settlement_semantic_gates",
        "settlement_verifier",
    }
