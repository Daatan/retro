"""Code-side numeric comparison for `quantity` (Oracle 1.5 Phase 1, retro#683 item 1.6).

WHY THIS EXISTS
---------------
retro#664 P1 (PR#671) measured the extractor on ten synthetic numeric-threshold cases.
Nova Lite returned stance ``+0.00`` on every between-bounds case and inverted both tone
traps: it follows how positive the sentence sounds, not what the number says. Three
separate prompt sections already instruct it to compare the numbers, and #664's P2 was
resolved as *the field, not another prompt fix*.

So the model is asked for the number (``PredictionExtraction.quantity``) and the
comparison happens here, in code — deterministic, auditable, and identical for every
rater. This module is the whole of that comparison.

WHAT IT IS NOT
--------------
* **Not a parser.** Nothing here reads a threshold out of an event's prose. The live
  question parse (``question_quantity``) is Oracle 1.5 **Phase 2**; until it exists the
  only thresholds available are the hand-declared ones on the A/B corpus, which is
  exactly the scope retro#683 asks for ("Starts on the A/B corpus ... moves to live
  traffic when Phase 2's ``question_quantity`` parse exists"). Adding a regex parser
  here would create a second, unmeasured source of thresholds for Phase 2 to disagree
  with.
* **Not a consumer.** The result is logged beside ``stance`` as a per-rater diagnostic.
  Agreement was *expected* to be high on Haiku and low on Nova Lite; the gap is the
  finding, not a gate. Nothing reads this to move a forecast. The only caller in the
  tree today is retro#687's confusion flag rule 2, which is log-only.

MEASURED, AND WHY THE RATER GATE IS NOT OPTIONAL
------------------------------------------------
The gap the issue predicted is real and larger than the headline suggests
(numeric corpus, prompt v9, exact (value, unit, comparator) on the case's own number):

    Haiku 4.5   50/50 runs  (100%)   comparator wrong: 0
    Nova Lite  120/150 runs  (80%)   comparator wrong: 27  (18% of runs)

Every Nova Lite failure is ``comparator``, and always the same one: a verb of movement
is encoded as a bound. "inflation accelerated to 4.1 percent" comes back ``> 4.1``
15/15; "support collapsed to just 35 percent" comes back ``< 35``. Value and unit are
right in every one of them — the model reads the number and then overwrites the relation
with the direction the sentence travelled. That is retro#664 one level down: the same
tone-over-number substitution, moved from ``stance`` into ``comparator``.

Two targeted prompt edits against it (a movement-verb rule, v9b and v9c) moved nothing
— 80% -> 72%, with fill falling 86% -> 74% — and were reverted. This is not reachable
from the prompt on this rater.

Consequence, and it holds until the validator is re-run: **Nova Lite's ``quantity`` is
not fit for any consumer.** A ``<`` where the article said ``=`` does not merely lose
the comparison, it flips it — ``< 35`` against "at least 30 percent" contradicts where
the stated level satisfies. Phase 2, which wires this to live traffic, must gate on the
rater and route threshold-archetype questions to Haiku; it must not treat a filled
``quantity`` as trustworthy because it is filled.

WHAT IT ANSWERS
---------------
Both sides are intervals, so the comparison is set containment rather than a chain of
if/elses over comparator pairs:

    reported interval  ⊆  satisfying interval   ->  +1  satisfies
    reported interval  ∩  satisfying interval = ∅ ->  -1  contradicts
    otherwise                                     -> None straddles

Containment is what makes a *bounded* report answerable at all. "The rate stayed below
5%" against "is the rate at or below 9%" is +1 without any value ever being stated: every
point the article allows is inside the question's satisfying set. The same report against
"is the rate at or below 3%" straddles, and abstaining there is the correct answer, not a
gap — a sign invented from an interval that contains both answers is worse than no sign.

Unit mismatch abstains too, and deliberately never contradicts: "40 launchers" against a
threshold in kilometres is a comparison that was never made, not one that failed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Optional, Protocol, runtime_checkable

Comparator = Literal["=", "<", "<=", ">", ">=", "between"]


@runtime_checkable
class QuantityLike(Protocol):
    """The four attributes the comparison reads.

    Structural rather than `tm.models.Quantity`, because there are two of that
    model on purpose: the elicited one here and `forecast_api.models.Quantity`,
    the wire contract, which is a separate declaration for the same reason
    `evidence_class` and `facet` are spelled out on both sides. Naming one of
    them would make this module work on one path and raise on the other, which
    is precisely how retro#687's rule 2 was about to break.
    """
    value: float
    unit: str
    comparator: str
    value_hi: Optional[float]

_COMPARATORS: frozenset[str] = frozenset({"=", "<", "<=", ">", ">=", "between"})


@dataclass(frozen=True)
class QuestionThreshold:
    """The bar a threshold-shaped question sets, declared rather than parsed.

    Same field names and same comparator vocabulary as `Quantity` on purpose: the
    question and the article's report are the same kind of object, and the whole
    comparison below is between two of them.
    """
    comparator: Comparator
    value: float
    unit: str
    value_hi: Optional[float] = None

    def __post_init__(self) -> None:
        if self.comparator not in _COMPARATORS:
            raise ValueError(f"unknown comparator {self.comparator!r}")
        if self.comparator == "between":
            if self.value_hi is None:
                raise ValueError("comparator 'between' requires value_hi")
            if self.value_hi <= self.value:
                raise ValueError("value_hi must be greater than value")
        elif self.value_hi is not None:
            raise ValueError("value_hi is only meaningful with comparator 'between'")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "QuestionThreshold":
        """Build one from a case file's `question_threshold` block."""
        return cls(
            comparator=raw["comparator"],
            value=float(raw["value"]),
            unit=str(raw["unit"]),
            value_hi=None if raw.get("value_hi") is None else float(raw["value_hi"]),
        )


@dataclass(frozen=True)
class Comparison:
    """The verdict, plus why — an abstention has two very different causes and a
    diagnostic that cannot tell them apart would hide the interesting one."""
    sign: Optional[int]
    reason: Literal["satisfies", "contradicts", "straddles", "unit_mismatch"]


# ── units ────────────────────────────────────────────────────────────────────
# Matching is exact after normalisation, never by substring: "percentage point" and
# "percent" share a prefix and are different units, and a threshold in the one compared
# against a report in the other is a wrong answer, not a near miss.
_UNIT_ALIASES: dict[str, str] = {
    "%": "percent",
    "pct": "percent",
    "percentage": "percent",
    "per cent": "percent",
    "percentage point": "percentage_point",
    "basis point": "basis_point",
    "bp": "basis_point",
    "bps": "basis_point",
}


def normalise_unit(unit: str) -> str:
    """Lowercase, de-punctuate and singularise a unit for exact comparison."""
    text = " ".join(unit.strip().lower().replace("_", " ").split())
    text = text.rstrip(".")
    # Singularise before the alias lookup so "percentage points" and "basis points"
    # reach their aliases; "percent" and "bps" are handled by the table itself.
    if text not in _UNIT_ALIASES and text.endswith("s") and not text.endswith("ss"):
        singular = text[:-1]
        if singular:
            text = singular
    return _UNIT_ALIASES.get(text, text).replace(" ", "_")


def _interval(
    comparator: str, value: float, value_hi: Optional[float],
) -> tuple[float, bool, float, bool]:
    """(lo, lo_closed, hi, hi_closed) — the set of values this assertion allows."""
    if comparator == "=":
        return value, True, value, True
    if comparator == "<":
        return -math.inf, False, value, False
    if comparator == "<=":
        return -math.inf, False, value, True
    if comparator == ">":
        return value, False, math.inf, False
    if comparator == ">=":
        return value, True, math.inf, False
    # between — inclusive on both ends, matching how "between 2% and 3%" resolves.
    assert value_hi is not None  # guaranteed by Quantity/QuestionThreshold validation
    return value, True, value_hi, True


def _contains(outer: tuple[float, bool, float, bool], inner: tuple[float, bool, float, bool]) -> bool:
    o_lo, o_lo_closed, o_hi, o_hi_closed = outer
    i_lo, i_lo_closed, i_hi, i_hi_closed = inner
    lo_ok = o_lo < i_lo or (o_lo == i_lo and (o_lo_closed or not i_lo_closed))
    hi_ok = i_hi < o_hi or (i_hi == o_hi and (o_hi_closed or not i_hi_closed))
    return lo_ok and hi_ok


def _disjoint(a: tuple[float, bool, float, bool], b: tuple[float, bool, float, bool]) -> bool:
    a_lo, a_lo_closed, a_hi, a_hi_closed = a
    b_lo, b_lo_closed, b_hi, b_hi_closed = b
    if a_hi < b_lo or b_hi < a_lo:
        return True
    # Touching endpoints overlap only when BOTH sides include the shared point.
    if a_hi == b_lo:
        return not (a_hi_closed and b_lo_closed)
    if b_hi == a_lo:
        return not (b_hi_closed and a_lo_closed)
    return False


def compare(quantity: QuantityLike, threshold: QuestionThreshold) -> Comparison:
    """Does the article's reported quantity satisfy the question's threshold?

    +1 when every value the report allows also satisfies the question, -1 when none
    does, and None when the report spans both answers or the units do not match.
    """
    if normalise_unit(quantity.unit) != normalise_unit(threshold.unit):
        return Comparison(None, "unit_mismatch")
    reported = _interval(quantity.comparator, quantity.value, quantity.value_hi)
    satisfying = _interval(threshold.comparator, threshold.value, threshold.value_hi)
    if _contains(satisfying, reported):
        return Comparison(1, "satisfies")
    if _disjoint(reported, satisfying):
        return Comparison(-1, "contradicts")
    return Comparison(None, "straddles")
