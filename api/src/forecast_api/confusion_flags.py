"""Deterministic confusion flags (retro#687, Oracle 1.5 Phase 1 item 1.9).

The zero-cost first tier of "measure the extractor as an observer". Between-rater
disagreement is a *lower bound* on extractor noise — LLM raters share their errors
(plan §3.5), so two models agreeing is weaker evidence than it looks — and running
a second rater costs a second extraction. These rules cost nothing: they are pure
functions over fields the extractor already emitted, and they fire on a single row
without anything to compare it against.

Each rule names an *internal* inconsistency, not a wrong answer. A flagged row is
not known to be wrong; it is a row whose own fields disagree about how confident
anyone should be in it. That is exactly the sampling filter daatan#1636's
second-family re-read wants, and Phase 3 uses flagged rows one way only —
excluded from the credibility bill, never re-weighted up.

**Reporting only. Nothing here changes a number**, in this module or its callers.

Null-safety is the design constraint, not a defensive habit: two of the three
inputs are shadow fields that a row extracted before retro#681 simply does not
have, and rule 2's inputs do not exist at all yet (see below). A rule with a
missing input yields no flag — never a default, never a guess. So this module
activates rule by rule as the fields land, with no follow-up edit here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal, Optional, Sequence

from .models import ClaimDetail

logger = logging.getLogger(__name__)


# Rule ids are the join key between the per-row log lines, the per-pool summary
# and the Phase 1 exit report, so they are frozen strings rather than an enum
# whose repr could drift.
RULE_TRAPPED_STRONG_CLAIM = "trapped_strong_claim"
RULE_STANCE_VS_QUANTITY = "stance_vs_quantity"
RULE_UNSURE_SETTLEMENT = "unsure_settlement"

CONFUSION_RULES: tuple[str, ...] = (
    RULE_TRAPPED_STRONG_CLAIM,
    RULE_STANCE_VS_QUANTITY,
    RULE_UNSURE_SETTLEMENT,
)
"""Every rule this module can emit, in issue order. The summary line reports a
count for each one on every pool — including the zeros, so a rule that never
fires is distinguishable from a rule that was never evaluated."""


@dataclass(frozen=True)
class ConfusionFlag:
    """One rule firing on one claim of one row."""

    rule: str
    url: Optional[str]
    claim_index: int


def _trapped_strong_claim(claim: ClaimDetail, *, claim_strength_min: float) -> bool:
    """Rule 1 — the source is flat, the reader says it stumbled.

    ``claim_strength`` is the SOURCE's commitment; ``reader_confidence.trap`` is
    the READER saying a known reading trap applied to the same span (retro#680
    split those two apart precisely because one field was carrying both). A span
    that is both maximally unhedged and admittedly tricky to read is the
    combination worth sampling: whichever of the two is right, the other is
    miscalibrated.

    Deliberately keyed on ``trap``, not on ``level``. A trap is a *named* class
    with an independent detector behind it (negation → retro#657,
    numeric_comparison → the PR#671 A/B cases), which is what makes the
    self-flag checkable against something other than itself. `ReaderConfidence`'s
    own docstring notes a trap does not imply a low level — that asymmetry is the
    point of this rule rather than an objection to it.
    """
    rc = claim.reader_confidence
    if rc is None or rc.trap is None:
        return False
    return claim.claim_strength >= claim_strength_min


def _stance_vs_quantity(
    claim: ClaimDetail,
    *,
    question_quantity: Optional[float],
    comparison: Optional[Literal["at_least", "at_most"]],
) -> bool:
    """Rule 2 — the stance sign contradicts the arithmetic.

    On a threshold question ("Brent crosses $100"), the claim's own number
    settles the direction without a model's opinion: a claim reporting $93 that
    nonetheless leans positive is internally inconsistent, and that inconsistency
    is checkable in code with no LLM call.

    **Inert today, by construction.** It needs two inputs that do not exist yet:
    the claim-side number (``quantity``, retro#683) and the question-side
    threshold (Phase 2's ``question_quantity``). The claim side is read with
    ``getattr`` so this rule starts firing the moment #683 adds the field, with
    no edit here; the question side is a parameter no caller passes yet. Both
    absent ⇒ no flag, which is the null-safe path the acceptance criteria ask to
    be tested rather than a placeholder.

    The comparison direction is a parameter rather than something inferred from
    the question text: guessing "exceeds" vs "falls below" from wording is the
    kind of judgement this module exists to avoid making.

    ``ClaimDetail.quantitative_estimate`` is NOT this number and must not be
    substituted for it: it is a probability in [0, 1] the source cited *for the
    event*, so comparing it against a question threshold ($100, 61 seats) would
    compare two different quantities and flag on noise.
    """
    if question_quantity is None or comparison is None:
        return False
    quantity = getattr(claim, "quantity", None)
    if quantity is None:
        return False
    # A claim sitting exactly on the threshold decides nothing either way, and a
    # zero stance is not a direction — neither can contradict anything.
    if quantity == question_quantity or claim.stance == 0.0:
        return False
    meets = quantity > question_quantity if comparison == "at_least" else quantity < question_quantity
    return meets != (claim.stance > 0)


def _unsure_settlement(claim: ClaimDetail) -> bool:
    """Rule 3 — a settlement claim the reader was not sure of.

    A settlement is the one claim type that can pin a forecast outright, so it is
    the one place where "the reader could plausibly read this differently" is not
    an acceptable margin. Phase 4 turns this into settlement's second bar
    (``level == "high"``); here it only counts.

    ``level`` is required whenever ``reader_confidence`` is present, so the only
    null path is the whole object being absent on a pre-#681 row.
    """
    if claim.settled is not True:
        return False
    rc = claim.reader_confidence
    if rc is None:
        return False
    return rc.level != "high"


def flags_for_claim(
    claim: ClaimDetail,
    *,
    claim_strength_min: float,
    question_quantity: Optional[float] = None,
    comparison: Optional[Literal["at_least", "at_most"]] = None,
) -> list[str]:
    """Every rule this claim trips, in ``CONFUSION_RULES`` order.

    A claim can trip more than one rule and each is reported: they name different
    inconsistencies, and collapsing them to "flagged" would throw away the only
    thing the exit report is grouping by.
    """
    fired: list[str] = []
    if _trapped_strong_claim(claim, claim_strength_min=claim_strength_min):
        fired.append(RULE_TRAPPED_STRONG_CLAIM)
    if _stance_vs_quantity(claim, question_quantity=question_quantity, comparison=comparison):
        fired.append(RULE_STANCE_VS_QUANTITY)
    if _unsure_settlement(claim):
        fired.append(RULE_UNSURE_SETTLEMENT)
    return fired


def flags_for_rows(
    rows: Sequence[object],
    *,
    claim_strength_min: float,
    question_quantity: Optional[float] = None,
    comparison: Optional[Literal["at_least", "at_most"]] = None,
) -> list[ConfusionFlag]:
    """Flags over any sequence of rows carrying ``claims_detail`` and ``url``.

    Typed against the structural minimum rather than a union of `SourceSignal`
    and `PoolSourceInput`: both live paths hand in one of those two, they agree
    on the two attributes this reads, and naming the union here would couple a
    pure module to which callers happen to exist.
    """
    out: list[ConfusionFlag] = []
    for row in rows:
        claims = getattr(row, "claims_detail", None)
        if not claims:
            continue
        url = getattr(row, "url", None)
        for i, claim in enumerate(claims):
            for rule in flags_for_claim(
                claim,
                claim_strength_min=claim_strength_min,
                question_quantity=question_quantity,
                comparison=comparison,
            ):
                out.append(ConfusionFlag(rule=rule, url=url, claim_index=i))
    return out


def counts_by_rule(flags: Sequence[ConfusionFlag]) -> dict[str, int]:
    """Per-rule totals, every rule present — zeros included (see CONFUSION_RULES)."""
    counts = {rule: 0 for rule in CONFUSION_RULES}
    for flag in flags:
        counts[flag.rule] += 1
    return counts


def log_confusion_flags(
    rows: Sequence[object],
    *,
    question_hash: str,
    extractor_model: str,
    claim_strength_min: float,
    prediction_id: Optional[str] = None,
    question_quantity: Optional[float] = None,
    comparison: Optional[Literal["at_least", "at_most"]] = None,
) -> list[ConfusionFlag]:
    """Emit the per-row lines and the per-pool summary; return the flags.

    The summary fires on EVERY pool, flagged or not — the same rationale as
    ``event=evidence_clusters`` (retro#412) and Gate-0's per-pool line: without a
    denominator, a zero cannot be told apart from a path that never ran. Rows
    without ``claims_detail`` are counted separately for the same reason, since a
    pool of pre-#364 legacy rows is unevaluable rather than clean.
    """
    flags = flags_for_rows(
        rows,
        claim_strength_min=claim_strength_min,
        question_quantity=question_quantity,
        comparison=comparison,
    )
    for flag in flags:
        logger.info(
            "event=confusion_flag rule=%s question=%s prediction_id=%s "
            "extractor_model=%s url=%s claim_index=%d",
            flag.rule, question_hash, prediction_id, extractor_model,
            flag.url, flag.claim_index,
        )
    counts = counts_by_rule(flags)
    evaluable = sum(1 for row in rows if getattr(row, "claims_detail", None))
    logger.info(
        "event=confusion_flags question=%s prediction_id=%s extractor_model=%s "
        "rows=%d evaluable=%d claims=%d flagged=%d %s",
        question_hash, prediction_id, extractor_model, len(rows), evaluable,
        sum(len(getattr(r, "claims_detail", None) or []) for r in rows),
        len(flags),
        " ".join(f"{rule}={counts[rule]}" for rule in CONFUSION_RULES),
    )
    return flags
