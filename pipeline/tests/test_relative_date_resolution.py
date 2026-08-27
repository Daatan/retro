"""Code, not the model, resolves a relative date expression to a calendar date.

The residual bug these pin down: after #267 taught the extractor to report
``event_date`` and let Python compare it to the deadline, the model still could not
resolve WEEKDAYS — asked about the Knesset dissolving "on Friday" (article dated
Monday 2026-07-13) it offered 2026-07-18, a Saturday. The sign only survived because
both candidate dates fell after the July 15 deadline; a ±1-day miss against a date
sitting ON the deadline still inverts the answer. So the prompt now also asks for the
verbatim expression (``event_date_reference``) and enforce_relative_date_resolution
redoes the calendar walk in code, overriding a disagreeing model date.
"""
from tm.extractor import _resolve_relative_reference, enforce_relative_date_resolution
from tm.models import PredictionExtraction

from datetime import date

MONDAY = "2026-07-13"  # the Knesset incident's article date


def pred(event_date: str | None, reference: str | None, stance: float = -1.0):
    return PredictionExtraction(
        quote="q", claim="c", stance=stance, certainty=0.95,
        event_date=event_date, event_date_reference=reference,
    )


# ── the incident ──────────────────────────────────────────────────────────────


def test_the_knesset_case_a_misresolved_weekday_is_corrected():
    """Model said "Friday" = 2026-07-18 (a Saturday). The Friday after Monday the
    13th is the 17th — code wins."""
    [out] = enforce_relative_date_resolution([pred("2026-07-18", "on Friday")], MONDAY)
    assert out.event_date == "2026-07-17"


def test_a_correctly_resolved_weekday_is_left_alone():
    [out] = enforce_relative_date_resolution([pred("2026-07-17", "on Friday")], MONDAY)
    assert out.event_date == "2026-07-17"


def test_stance_and_certainty_are_never_touched():
    [out] = enforce_relative_date_resolution([pred("2026-07-18", "on Friday", stance=1.0)], MONDAY)
    assert out.stance == 1.0
    assert out.claim_strength == 0.95


# ── vocabulary ────────────────────────────────────────────────────────────────


def test_day_words():
    assert _resolve_relative_reference("today", date(2026, 7, 13)) == date(2026, 7, 13)
    assert _resolve_relative_reference("tonight", date(2026, 7, 13)) == date(2026, 7, 13)
    assert _resolve_relative_reference("yesterday", date(2026, 7, 13)) == date(2026, 7, 12)
    assert _resolve_relative_reference("tomorrow", date(2026, 7, 13)) == date(2026, 7, 14)


def test_bare_and_modified_weekdays_mean_the_coming_occurrence():
    monday = date(2026, 7, 13)
    assert _resolve_relative_reference("friday", monday) == date(2026, 7, 17)
    assert _resolve_relative_reference("on Friday", monday) == date(2026, 7, 17)
    assert _resolve_relative_reference("this Friday", monday) == date(2026, 7, 17)
    assert _resolve_relative_reference("coming Friday", monday) == date(2026, 7, 17)
    assert _resolve_relative_reference("the coming Friday", monday) == date(2026, 7, 17)


def test_same_weekday_means_the_same_day():
    friday = date(2026, 7, 17)
    assert _resolve_relative_reference("on Friday", friday) == friday


def test_last_weekday_is_the_previous_occurrence():
    monday = date(2026, 7, 13)
    assert _resolve_relative_reference("last Friday", monday) == date(2026, 7, 10)
    # "last Monday" from a Monday is a week back, not today
    assert _resolve_relative_reference("last Monday", monday) == date(2026, 7, 6)


def test_punctuation_and_case_are_normalized():
    assert _resolve_relative_reference('"on Friday,"', date(2026, 7, 13)) == date(2026, 7, 17)


def test_next_weekday_is_deliberately_out_of_vocabulary():
    """Speakers disagree on what "next Friday" means; an ambiguous guard is worse
    than none."""
    assert _resolve_relative_reference("next Friday", date(2026, 7, 13)) is None


def test_out_of_vocabulary_expressions_resolve_to_none():
    monday = date(2026, 7, 13)
    assert _resolve_relative_reference("next week", monday) is None
    assert _resolve_relative_reference("in three days", monday) is None
    assert _resolve_relative_reference("ביום שישי", monday) is None  # non-English fails open


# ── fail-open behaviour ───────────────────────────────────────────────────────


def test_no_reference_leaves_the_prediction_untouched():
    [out] = enforce_relative_date_resolution([pred("2026-07-18", None)], MONDAY)
    assert out.event_date == "2026-07-18"


def test_out_of_vocabulary_reference_leaves_the_model_date():
    [out] = enforce_relative_date_resolution([pred("2026-07-20", "next week")], MONDAY)
    assert out.event_date == "2026-07-20"


def test_a_reference_never_creates_a_missing_event_date():
    """Settlement gating must stay anchored to a date the model itself committed to."""
    [out] = enforce_relative_date_resolution([pred(None, "on Friday")], MONDAY)
    assert out.event_date is None


def test_unparseable_article_date_fails_open():
    [out] = enforce_relative_date_resolution([pred("2026-07-18", "on Friday")], "no date")
    assert out.event_date == "2026-07-18"


def test_missing_article_date_fails_open():
    [out] = enforce_relative_date_resolution([pred("2026-07-18", "on Friday")], None)
    assert out.event_date == "2026-07-18"
