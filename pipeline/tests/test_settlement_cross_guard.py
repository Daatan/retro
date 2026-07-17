"""A dated foreclosing negative must survive the full guard chain.

The guards are unit-tested in isolation elsewhere; this file pins their
INTERACTION, in the exact order the forecaster runs them
(forecast_api/forecaster.py::_process_article):

    enforce_relative_date_resolution -> enforce_deadline_arithmetic
        -> enforce_settlement_event_date

The incident behind it: France was eliminated from the 2026 World Cup by Spain
on 2026-07-14 (deadline 2026-07-19, arrival claim). The correct extraction is a
settled negative dated with the FORECLOSING event (the semi-final result).
Before the arrival-negative exemption, enforce_deadline_arithmetic read that
date as the claim-event's own occurrence within the deadline and flipped the
verdict to +1.0 — and, now positive, it would then have sailed through
enforce_settlement_event_date's positive checks with a valid date: a false
YES pin built from a correct NO extraction.
"""
from tm.extractor import (
    enforce_deadline_arithmetic,
    enforce_relative_date_resolution,
    enforce_settlement_event_date,
)
from tm.models import PredictionExtraction

ARTICLE_DATE = "2026-07-14"
DEADLINE = "2026-07-19"


def run_guard_chain(
    p: PredictionExtraction,
    article_date: str = ARTICLE_DATE,
    claim_deadline: str = DEADLINE,
    claim_direction: str = "arrival",
) -> PredictionExtraction:
    preds = enforce_relative_date_resolution([p], article_date)
    preds = enforce_deadline_arithmetic(preds, claim_deadline, claim_direction)
    [out] = enforce_settlement_event_date(preds, article_date)
    return out


def test_the_france_case_a_dated_foreclosing_negative_survives_the_chain():
    out = run_guard_chain(PredictionExtraction(
        quote="Spain beat France 2-0 in Tuesday's semi-final to reach the final",
        claim="France was eliminated from the 2026 World Cup on 2026-07-14",
        stance=-1.0, certainty=0.95, settled=True, event_date="2026-07-14",
    ))
    assert out.stance == -1.0
    assert out.settled is True
    assert out.event_date == "2026-07-14"


def test_relative_reference_resolution_still_applies_to_a_foreclosing_negative():
    """The date-resolution guard runs first and corrects the calendar walk; the
    corrected date must then ride through the two later guards untouched."""
    out = run_guard_chain(PredictionExtraction(
        quote="q", claim="c", stance=-1.0, certainty=0.95, settled=True,
        # Extractor mis-resolved "yesterday" (2026-07-13) to the article's own date.
        event_date="2026-07-14", event_date_reference="yesterday",
    ))
    assert out.event_date == "2026-07-13"
    assert out.stance == -1.0
    assert out.settled is True


def test_a_settled_positive_still_gets_deadline_arithmetic_through_the_chain():
    """The exemption is negatives-on-arrival only: the Knesset-class flip on a
    positive dated after the deadline must keep working end-to-end. The flipped
    (now negative) settlement keeps its occurrence date and stays settled."""
    out = run_guard_chain(PredictionExtraction(
        quote="q", claim="c", stance=1.0, certainty=0.95, settled=True,
        event_date="2026-07-21",  # after the 07-19 deadline
    ), article_date="2026-07-22")
    assert out.stance == -1.0
    assert out.settled is True
    assert out.event_date == "2026-07-21"


def test_a_survival_negative_is_still_arithmetic_corrected_through_the_chain():
    """On a survival claim a settled negative dates the OCCURRENCE, so the
    arithmetic remains valid: occurrence after the deadline means the claim
    held, and the sign is corrected to +."""
    out = run_guard_chain(PredictionExtraction(
        quote="q", claim="c", stance=-1.0, certainty=0.95, settled=True,
        event_date="2026-07-21",
    ), article_date="2026-07-22", claim_direction="survival")
    assert out.stance == 1.0
    assert out.settled is True
