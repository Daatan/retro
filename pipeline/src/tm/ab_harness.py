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
from typing import Any, Mapping, Optional

from .models import PredictionExtraction, Quantity
from .threshold_compare import QuestionThreshold, compare, normalise_unit


def _sign(x: Optional[float]) -> Optional[int]:
    if x is None:
        return None
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


# Magnitude buckets, deliberately coarse (retro#720). `_sign` answers "which way",
# which is the only question most of this corpus needs — but some prompt sections
# exist to control HOW STRONGLY, not which way, and for those a sign reader is blind.
#
# The `## Multi-stage / bracket events` section is the case in point: every one of
# its seven worked examples is POSITIVE (+0.2 … +0.6), so deleting the section
# entirely cannot move `stance_sign` on a single bracket case. It moves the
# magnitude — a single-stage "strong favourite" framing read as full support
# instead of weak support — and until this reader existed the harness had no way
# to see that. A bracket corpus scored on sign alone would have passed the deletion
# as clean, which is exactly the false comfort retro#720 warns about.
#
# Four buckets, not a threshold predicate, because `unmet_facets` compares with
# `==`: a bucket is the only shape that fits without changing that contract.
# Boundaries are read off the section's own numbers (0.2/0.3/0.4 for "one stage of
# several remaining", 0.6 for "one stage left"), so `weak` vs `moderate` splits
# exactly where the prompt tells the model to split. Magnitude only — sign stays
# `stance_sign`'s job, so a case can assert both independently.
_MAGNITUDE_BANDS: tuple[tuple[float, str], ...] = (
    (0.15, "none"),
    (0.50, "weak"),
    (0.80, "moderate"),
)


def _band(x: Optional[float]) -> Optional[str]:
    if x is None:
        return None
    magnitude = abs(x)
    for upper, name in _MAGNITUDE_BANDS:
        if magnitude < upper:
            return name
    return "strong"


# Facets the harness knows how to check. Each reads one value off a single
# PredictionExtraction; a case's `expect` dict names which facets it cares
# about and what value each must equal on at least one extracted prediction.
_FACET_READERS = {
    "stance_sign": lambda p: _sign(p.stance),
    # Magnitude, for the sections that exist to control strength rather than
    # direction — see _band above. "none" | "weak" | "moderate" | "strong".
    "stance_band": lambda p: _band(p.stance),
    "claim_strength_band": lambda p: _band(p.claim_strength),
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
    # --- retro#683, diagnostic only ---
    # Deliberately NOT inside `expect`, which is the zero-regression gate. `quantity` is a
    # shadow field whose validator (exact match on 100 hand-labelled claims, accuracy >= 0.9
    # per rater) has not been run yet, and gating every future prompt PR on an unvalidated
    # field would make it a requirement before it is a measurement. These two feed the
    # quantity report and nothing else.
    #
    # `question_threshold` is the bar the QUESTION sets, hand-declared rather than parsed:
    # the live parse is Oracle 1.5 Phase 2 (`question_quantity`), and a second, unmeasured
    # parser here would be something for Phase 2 to disagree with. `expect_quantity` is what
    # a correct reader should have extracted from THIS article — the "PR#671 known values"
    # half of the validator.
    question_threshold: Optional[dict[str, Any]] = None
    expect_quantity: Optional[dict[str, Any]] = None

    @property
    def threshold(self) -> Optional[QuestionThreshold]:
        if self.question_threshold is None:
            return None
        return QuestionThreshold.from_mapping(self.question_threshold)

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


# ── quantity vs the question's threshold (retro#683 item 1.6) ────────────────
#
# The comparison the extractor demonstrably cannot do in its head, done in code and
# reported beside the stance the model produced anyway. Per rater by construction: an
# arm is one model, so an arm's report is that model's row.
#
# What the issue asks this to show is a GAP, not a pass: agreement is expected to be
# high on Haiku and low on Nova Lite. So the report carries both halves separately —
# whether the code-side comparison got the case right, and whether the model's stance
# did. On the retro#664 corpus the code is right by construction on all ten, which is
# the point: the numbers were never the hard part, reading them was.
#
# Nothing here gates anything. `unmet_facets` and `gate_exit_code` are untouched.

def _quantity_matches(extracted: Quantity, expected: Mapping[str, Any]) -> bool:
    """Exact match on (value, unit, comparator) — the retro#683 validator's own test.

    `value_hi` joins them when the expectation names one; `as_of` never does. The
    validator is defined on the three fields that decide the comparison, and a date
    the article states loosely is a different measurement with a different bar.
    """
    if extracted.comparator != expected["comparator"]:
        return False
    if extracted.value != float(expected["value"]):
        return False
    if normalise_unit(extracted.unit) != normalise_unit(str(expected["unit"])):
        return False
    if expected.get("value_hi") is not None:
        return extracted.value_hi == float(expected["value_hi"])
    return True


@dataclass(frozen=True)
class QuantityDiagnostic:
    """One case's quantity row. Counts are over every prediction of every run."""
    case_id: str
    predictions: int
    filled: int
    exact: int              # matched expect_quantity (0 when the case declares none)
    labelled: int           # predictions scoreable against expect_quantity
    agree: int              # code-side sign == the model's own stance sign
    disagree: int
    undecidable: int        # code abstained: straddles, or units did not match
    code_correct: int       # code-side sign == the case's expected stance_sign
    stance_correct: int     # the model's stance sign == the case's expected stance_sign
    # `exact` is per-prediction, so it is diluted by predictions that correctly report a
    # DIFFERENT number from the same article ("the other party won 24 seats"). Those are
    # right, not wrong, and on a rater that quotes generously they drag `exact` down far
    # enough to invert the ranking between two raters. The three counts below are per RUN
    # and about the one number the case is a test of, which is what the validator means.
    runs: int = 0
    target_hit: int = 0             # some prediction matched expect_quantity exactly
    target_miscomparated: int = 0   # right value+unit, wrong comparator — the retro#664 failure


def quantity_diagnostics(
    cases: list[Case], predictions: dict[str, list[list[PredictionExtraction]]],
) -> list[QuantityDiagnostic]:
    """Per-case quantity fill, validator match, and code-vs-stance agreement.

    Only cases carrying a `question_threshold` or an `expect_quantity` are reported —
    the rest of the corpus has no number for this to be about.
    """
    out: list[QuantityDiagnostic] = []
    for case in cases:
        if case.question_threshold is None and case.expect_quantity is None:
            continue
        threshold = case.threshold
        truth = case.expect.get("stance_sign")
        n = filled = exact = labelled = agree = disagree = undecidable = 0
        code_correct = stance_correct = 0
        runs = target_hit = target_miscomparated = 0
        for run in predictions.get(case.id, []):
            if case.expect_quantity is not None:
                runs += 1
                same_number = [
                    p.quantity for p in run
                    if p.quantity is not None
                    and p.quantity.value == float(case.expect_quantity["value"])
                    and normalise_unit(p.quantity.unit)
                    == normalise_unit(str(case.expect_quantity["unit"]))
                ]
                if any(_quantity_matches(q, case.expect_quantity) for q in same_number):
                    target_hit += 1
                elif same_number:
                    target_miscomparated += 1
            for p in run:
                n += 1
                if truth is not None and _sign(p.stance) == truth:
                    stance_correct += 1
                if p.quantity is None:
                    continue
                filled += 1
                if case.expect_quantity is not None:
                    labelled += 1
                    if _quantity_matches(p.quantity, case.expect_quantity):
                        exact += 1
                if threshold is None:
                    continue
                sign = compare(p.quantity, threshold).sign
                if sign is None:
                    undecidable += 1
                    continue
                if sign == _sign(p.stance):
                    agree += 1
                else:
                    disagree += 1
                if truth is not None and sign == truth:
                    code_correct += 1
        out.append(QuantityDiagnostic(
            case_id=case.id, predictions=n, filled=filled, exact=exact, labelled=labelled,
            agree=agree, disagree=disagree, undecidable=undecidable,
            code_correct=code_correct, stance_correct=stance_correct,
            runs=runs, target_hit=target_hit, target_miscomparated=target_miscomparated,
        ))
    return out
