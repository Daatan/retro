"""Shared logic for the extractor-prompt A/B harness (retro#470).

Pure, network-free: case loading, per-facet expectation checking, the
temporal-leakage flag, and the baseline-vs-patched regression report. The
live-model driver that actually calls Bedrock is scripts/ab_extractor_prompt.py
(NOT a CI test) — this module is what that script, and its own unit tests,
both import, so the scoring logic itself IS covered by the fast suite even
though no test here ever calls a model.

Formalizes the ad hoc methodology already run twice by hand (PR#309, PR#314)
and once as a standalone script (eval_extractor_adjacent_events.py): a fixed
case sample, baseline prompt vs patched prompt, same live model, diffed on
the facets that matter. See docs/AB_HARNESS.md for how to add a case and read
the output.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Optional

from .models import PredictionExtraction


def _sign(x: Optional[float]) -> Optional[int]:
    if x is None:
        return None
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


# Facets the harness knows how to check. Each reads one value off a single
# PredictionExtraction; a case's `expect` dict names which facets it cares
# about and what value each must equal on at least one extracted prediction.
_FACET_READERS = {
    "stance_sign": lambda p: _sign(p.stance),
    "fact_signal_sign": lambda p: _sign(p.fact_signal) if p.fact_signal is not None else None,
    "fact_signal_null": lambda p: p.fact_signal is None,
    "is_occurrence": lambda p: p.is_occurrence,
    "facet": lambda p: p.facet,
    "verified": lambda p: p.verified,
    "settled": lambda p: bool(p.settled),
    "evidence_class": lambda p: p.evidence_class,
}


@dataclass(frozen=True)
class Case:
    """One A/B case: a fixed article + the facets it is expected to get right.

    ``event_description`` is the resolution-rules text handed to the
    extractor — varying it against ``control_event_description`` is exactly
    the axis retro#353 needs (the live path currently passes the bare
    question; the batch path already passes the real rules).
    """
    id: str
    event_name: str
    event_description: str
    claim_deadline: str
    article_date: str
    article_text: str
    expect: dict[str, Any]
    source_name: str = "Test"
    journalist: str = "unknown"
    control_event_description: Optional[str] = None
    tags: tuple[str, ...] = ()

    @property
    def is_temporal_leakage(self) -> bool:
        """True when the article postdates the claim's own deadline — hindsight
        the live system would never have had at forecast time. AVeriTeC prior
        art (cited in retro#352/#470): evidence dated after the claim can make
        a patched prompt look better than it is. Excluded from the
        zero-regression gate by default; always reported separately."""
        try:
            article = date.fromisoformat(self.article_date[:10])
            deadline = date.fromisoformat(self.claim_deadline[:10])
        except ValueError:
            return False
        return article > deadline


def load_cases(path: Path) -> list[Case]:
    raw = json.loads(Path(path).read_text())
    return [Case(**{**c, "tags": tuple(c.get("tags", ()))}) for c in raw]


def unmet_facets(prediction_runs: list[list[PredictionExtraction]], expect: dict[str, Any]) -> set[str]:
    """Facets in `expect` NOT satisfied by any prediction in any run.

    A facet is met if ANY extracted prediction, in ANY run, matches the
    expected value — one correct signal among several extracted predictions
    (or one correct run among several, given LLM non-determinism) is enough,
    mirroring how a single dominant claim can carry a fact for the pool.
    """
    unmet = set(expect)
    for preds in prediction_runs:
        if not unmet:
            break
        for p in preds:
            for facet in list(unmet):
                reader = _FACET_READERS.get(facet)
                if reader is None:
                    raise KeyError(f"unknown facet {facet!r} in case expectation")
                if reader(p) == expect[facet]:
                    unmet.discard(facet)
    return unmet


@dataclass
class CaseResult:
    case: Case
    baseline_unmet: set[str]
    patched_unmet: set[str]
    control_unmet: Optional[set[str]] = None

    @property
    def regressions(self) -> set[str]:
        """Facets baseline satisfied that patched now fails — the only thing
        the zero-regression gate refuses."""
        baseline_met = set(self.case.expect) - self.baseline_unmet
        return baseline_met & self.patched_unmet

    @property
    def improvements(self) -> set[str]:
        """Facets baseline failed that patched now satisfies — reported, not gated."""
        patched_met = set(self.case.expect) - self.patched_unmet
        return self.baseline_unmet & patched_met


def build_case_results(
    cases: list[Case],
    baseline: dict[str, list[list[PredictionExtraction]]],
    patched: dict[str, list[list[PredictionExtraction]]],
    control: Optional[dict[str, list[list[PredictionExtraction]]]] = None,
) -> list[CaseResult]:
    results = []
    for case in cases:
        if case.id not in baseline or case.id not in patched:
            raise KeyError(f"case {case.id!r} missing from baseline or patched results")
        results.append(CaseResult(
            case=case,
            baseline_unmet=unmet_facets(baseline[case.id], case.expect),
            patched_unmet=unmet_facets(patched[case.id], case.expect),
            control_unmet=(
                unmet_facets(control[case.id], case.expect)
                if control is not None and case.id in control else None
            ),
        ))
    return results


def gate_exit_code(results: list[CaseResult], *, allow_leakage: bool = False) -> int:
    """The zero-regression gate, expressed as an exit code rather than a
    judgement call: 0 only if no in-scope case regressed. Temporal-leakage
    cases are out of scope unless `allow_leakage` is set explicitly."""
    for r in results:
        if r.case.is_temporal_leakage and not allow_leakage:
            continue
        if r.regressions:
            return 1
    return 0
