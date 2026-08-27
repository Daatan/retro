"""A settlement vote must be anchored to a date the outcome occurred.

The incident these pin down: on 2026-07-15 the forecast "Netanyahu will win the 2026
Israeli general election and be appointed PM" was pinned to 97% / settled by exactly
two settlement-grade votes — a jns.org opinion piece ("Netanyahu secured a 64-seat
Likud-led coalition, confirming his electoral victory") and a Guardian claim
("Netanyahu will serve out his full term") — both describing the SITTING government
formed after the PREVIOUS election, while the election the claim asks about was
scheduled for 2026-10-27 and hadn't happened. Neither article dated the "outcome",
because the outcome they described wasn't this question's. The prompt's "historical
background is not settlement" rule is advisory; enforce_settlement_event_date is the
enforcement: no event_date, no settled.
"""
from tm.extractor import enforce_settlement_event_date
from tm.models import PredictionExtraction

ARTICLE_DATE = "2026-07-11"


def pred(stance: float, event_date: str | None = None, settled: bool | None = None, certainty: float = 0.95):
    return PredictionExtraction(
        quote="q", claim="c", stance=stance, certainty=certainty,
        settled=settled, event_date=event_date,
    )


# ── the incident ──────────────────────────────────────────────────────────────


def test_the_netanyahu_case_an_undated_settlement_is_demoted():
    """"secured a 64-seat coalition" — accomplished-fact language with no date the
    article can anchor it to, because the coalition predates the question."""
    [out] = enforce_settlement_event_date([pred(1.0, None, settled=True)], ARTICLE_DATE)
    assert out.settled is False
    assert out.stance == 1.0      # still votes as ordinary evidence
    assert out.claim_strength == 0.95  # only the settlement bit is cleared


def test_a_dated_settlement_is_left_alone():
    [out] = enforce_settlement_event_date([pred(1.0, "2026-07-09", settled=True)], ARTICLE_DATE)
    assert out.settled is True


def test_an_event_dated_on_the_article_day_itself_counts_as_accomplished():
    """Same-day reporting ("the Knesset dissolved today") is legitimate settlement."""
    [out] = enforce_settlement_event_date([pred(1.0, ARTICLE_DATE, settled=True)], ARTICLE_DATE)
    assert out.settled is True


# ── future-dated "settlements" are scheduled events, not accomplished facts ──


def test_an_event_dated_after_the_article_is_demoted():
    """"France will play Spain in the semifinal on Tuesday" cannot settle anything —
    the article was written before the event it "reports"."""
    [out] = enforce_settlement_event_date([pred(1.0, "2026-07-15", settled=True)], ARTICLE_DATE)
    assert out.settled is False
    assert out.stance == 1.0


def test_an_undated_negative_settlement_survives():
    """An impossibility that comes only from time expiring has nothing to date —
    the missing-date demotion applies to positive settlements only. Premature
    negative pins are guarded by settlement_direction_allowed and per-vote
    revalidation at aggregation time."""
    [out] = enforce_settlement_event_date([pred(-1.0, None, settled=True)], ARTICLE_DATE)
    assert out.settled is True
    assert out.stance == -1.0


def test_a_dated_foreclosing_negative_survives_with_its_date():
    """"Spain beat France in Tuesday's semi-final" — the foreclosing event's
    date anchors the negative settlement and must be preserved for
    aggregation-time revalidation."""
    [out] = enforce_settlement_event_date([pred(-1.0, "2026-07-09", settled=True)], ARTICLE_DATE)
    assert out.settled is True
    assert out.event_date == "2026-07-09"


def test_a_future_dated_negative_settlement_is_demoted():
    """A "foreclosure" dated after the article is a schedule, not a fact —
    same reasoning as the positive future-dated check."""
    [out] = enforce_settlement_event_date([pred(-1.0, "2026-07-15", settled=True)], ARTICLE_DATE)
    assert out.settled is False
    assert out.stance == -1.0  # still votes as ordinary evidence


# ── only settlement claims are touched ────────────────────────────────────────


def test_unsettled_claims_are_never_touched():
    [a, b] = enforce_settlement_event_date(
        [pred(0.4, None, settled=False), pred(0.7, None, settled=None)], ARTICLE_DATE,
    )
    assert a.settled is False
    assert b.settled is None
    assert a.stance == 0.4 and b.stance == 0.7


# ── degraded inputs ───────────────────────────────────────────────────────────


def test_an_unparseable_event_date_is_treated_as_missing():
    [out] = enforce_settlement_event_date([pred(1.0, "next Friday", settled=True)], ARTICLE_DATE)
    assert out.settled is False


def test_a_full_timestamp_event_date_still_parses():
    [out] = enforce_settlement_event_date([pred(1.0, "2026-07-09T15:30:00+00:00", settled=True)], ARTICLE_DATE)
    assert out.settled is True


def test_missing_article_date_skips_only_the_future_check():
    """Without an article date we can't call an event "future" — but the
    date-presence requirement is unconditional."""
    [dated] = enforce_settlement_event_date([pred(1.0, "2027-01-01", settled=True)], None)
    assert dated.settled is True  # can't prove it's future-dated
    [undated] = enforce_settlement_event_date([pred(1.0, None, settled=True)], None)
    assert undated.settled is False


def test_an_unparseable_article_date_behaves_like_a_missing_one():
    [out] = enforce_settlement_event_date([pred(1.0, "2027-01-01", settled=True)], "circa July")
    assert out.settled is True
