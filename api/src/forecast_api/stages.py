"""Single source of truth for the shadow-then-promote stages (retro#866).

Eleven features in :mod:`forecast_api.config` ship behind the same rollout shape:
compute something alongside the live path, log it, and only later let it change
an outcome. Each one grew its own ``<name>_enabled`` flag (nine of them with an
``<name>_enforce`` beside it),
its own ``event=<name>`` log line, and its own prose comment re-explaining the
contract — several literally say "same shadow-then-promote shape as
``premise_verifier_enforce``".

That convention had already forked into two hand-maintained lists: the settings
fields here, and ``FLAG_REGISTRY`` in ``scripts/check_shadow_flag_telemetry.py``
(retro#806), which knew about four of them. The check written to catch silently
inert shadow flags was itself blind to six. This module is the one list both
read, and :mod:`tests.test_stages` fails when a new pair appears in the config
without an entry — the mechanism retro#808 uses for extractor fields, which is
the only thing that has reliably stopped this drift before.

What a Stage is NOT
-------------------
A Stage **describes** a feature's configured rollout state; it does not execute
it. :meth:`Stage.mode` is derived from the existing settings attributes rather
than replacing them, so every field, env override and systemd drop-in keeps
working exactly as before.

Deliberately, call sites keep their own enforcement branches. They share a naming
convention, not a behaviour: ``settlement_semantic_gates_fallback_enforce``
is a fail-open fallback scoped to one all-samples-errored branch,
``subject_gate_enforce`` turns a fired gate into a dropped article,
``conditional_attenuation_enforce`` substitutes a shadow-computed number, and
``settlement_verifier_enforce`` vetoes settlement pins. Routing those through
one ``is_enforcing()`` accessor would read as uniformity that does not exist and
would flatten exactly the differences a reader needs to see. ``mode()`` is for
reporting — the startup ``event=stage_modes`` line — not for gating. The daily audit
reads ``enabled_attr`` and ``events``, not ``mode()``. Deliberately not exposed on ``/health``, which is unauthenticated.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


class Mode(str, Enum):
    """A stage's configured rollout state.

    ``OFF`` the stage does not run · ``SHADOW`` it runs and logs but changes no
    outcome · ``ENFORCE`` its verdict is allowed to change an outcome.
    """

    OFF = "off"
    SHADOW = "shadow"
    ENFORCE = "enforce"


@dataclass(frozen=True)
class Stage:
    """One shadow-then-promote feature.

    ``enabled_attr`` / ``enforce_attr`` name existing ``ApiSettings`` fields —
    this class reads them, it does not own them. ``enforce_attr`` is None for a
    stage with no promotion path at all (a pure measurement lane), which is a
    different thing from a stage whose enforce flag exists and is off.

    ``events`` are the ``event=`` tokens the stage emits when it fires.

    ``expect_telemetry`` opts a stage into the silent-flag check
    (``scripts/check_shadow_flag_telemetry.py``). The rule is whether the stage
    logs on every request it sees, or only on a rare property of the input:

    * True — the event fires for every article/forecast the enabled stage
      handles, so silence over a 48h window really does mean a dead flag.
    * False — the event is gated on something rare, and silence is a quiet
      window rather than evidence. ``conditional_attenuation`` needs a claim
      the extractor marked conditional (~5% of claims); ``settlement_verifier``
      and ``settlement_semantic_gates`` only run on settlement candidates.
      Widening these needs firing-rate data first, or the daily audit starts
      failing on false alarms.
    """

    name: str
    issue: str
    enabled_attr: str
    enforce_attr: Optional[str] = None
    events: tuple[str, ...] = ()
    expect_telemetry: bool = False

    def is_enabled(self, settings: Any) -> bool:
        return bool(getattr(settings, self.enabled_attr))

    def is_enforcing(self, settings: Any) -> bool:
        """True only when the stage is enabled AND its enforce flag is set.

        A stage with no ``enforce_attr`` can never enforce. Note that this is the
        *configured* state; a call site may scope enforcement more narrowly than
        this (see the module docstring).
        """
        if self.enforce_attr is None:
            return False
        return self.is_enabled(settings) and bool(getattr(settings, self.enforce_attr))

    def mode(self, settings: Any) -> Mode:
        if not self.is_enabled(settings):
            return Mode.OFF
        return Mode.ENFORCE if self.is_enforcing(settings) else Mode.SHADOW


# One entry per shadow-then-promote feature. Adding a `*_enabled` / `*_enforce`
# pair to config.py without adding it here fails tests/test_stages.py.
STAGES: tuple[Stage, ...] = (
    Stage(
        name="conditional_attenuation",
        issue="retro#568",
        enabled_attr="conditional_attenuation_enabled",
        enforce_attr="conditional_attenuation_enforce",
        events=("event=conditional_attenuation_shadow",),
    ),
    Stage(
        name="settlement_semantic_gates",
        issue="retro#691",
        enabled_attr="settlement_semantic_gates_enabled",
        enforce_attr="settlement_semantic_gates_fallback_enforce",
        events=("event=settlement_semantic_gates",),
    ),
    Stage(
        name="settlement_verifier",
        issue="retro#388",
        enabled_attr="settlement_verifier_enabled",
        enforce_attr="settlement_verifier_enforce",
        events=("event=settlement_verifier", "event=settlement_verifier_error"),
    ),
    Stage(
        name="subject_gate",
        issue="retro#805",
        enabled_attr="subject_gate_enabled",
        enforce_attr="subject_gate_enforce",
        events=("event=subject_gate", "event=subject_gate_error"),
        expect_telemetry=True,
    ),
    Stage(
        name="premise_verifier",
        issue="retro#575 / retro#601",
        enabled_attr="premise_verifier_enabled",
        enforce_attr="premise_verifier_enforce",
        events=("event=premise_verifier", "event=premise_verifier_error"),
        expect_telemetry=True,
    ),
    Stage(
        name="precursor_match",
        issue="retro#608",
        enabled_attr="precursor_match_enabled",
        enforce_attr="precursor_match_enforce",
        events=("event=precursor_match", "event=precursor_match_crash"),
        expect_telemetry=True,
    ),
    Stage(
        name="settled_grounding",
        issue="retro#609",
        enabled_attr="settled_grounding_enabled",
        enforce_attr="settled_grounding_enforce",
        events=("event=settled_grounding", "event=settled_grounding_crash"),
        expect_telemetry=True,
    ),
    Stage(
        name="retry_relaxed_search",
        issue="retro#621",
        enabled_attr="retry_relaxed_search_enabled",
        enforce_attr="retry_relaxed_search_enforce",
        events=("event=retry_relaxed_search",),
        expect_telemetry=True,
    ),
    Stage(
        name="jev_shadow",
        issue="retro#840",
        enabled_attr="jev_shadow_enabled",
        events=("event=jev_shadow",),
        expect_telemetry=True,
    ),
    Stage(
        name="jev_gate",
        issue="retro#850",
        enabled_attr="jev_gate_enabled",
        enforce_attr="jev_gate_enforce",
        events=("event=jev_gate",),
        expect_telemetry=True,
    ),
    Stage(
        name="jev_gate_ab",
        issue="retro#849",
        enabled_attr="jev_gate_ab_enabled",
        events=("event=jev_gate_ab",),
        expect_telemetry=True,
    ),
)

STAGES_BY_NAME: dict[str, Stage] = {s.name: s for s in STAGES}


def describe_stages(settings: Any) -> dict[str, str]:
    """``{stage name: mode}`` for every stage — the one-glance shadow/enforce readout.

    Consumed by the ``event=stage_modes`` line ``main.py`` logs at startup, so the
    live rollout state is answerable from the log without reading two booleans per
    feature out of the running config. Deliberately not on ``/health``, which is
    unauthenticated.
    """
    return {s.name: s.mode(settings).value for s in STAGES}
