"""retro#873 — code-side date candidates: found and resolved against the publication date,
ambiguous ones offered both ways for Jev to choose."""
from datetime import date

from forecast_api.jev_dates import intervals

PUB = date(2026, 9, 16)  # a Wednesday


def _by_expr(text):
    return {e: (a, b) for a, b, e in intervals(text, PUB)}


def test_weekday_is_offered_as_both_last_and_next_occurrence():
    iv = _by_expr("The Fed hiked on Monday.")
    assert iv["Monday (past)"] == (date(2026, 9, 14),) * 2
    assert iv["Monday (upcoming)"] == (date(2026, 9, 21),) * 2


def test_yearless_month_day_both_sides_and_abbreviation_with_dot():
    iv = _by_expr("Talks resume Oct. 23; the deal was signed Feb. 28.")
    assert iv["Oct. 23 (past)"][0] == date(2025, 10, 23) and iv["Oct. 23 (upcoming)"][0] == date(2026, 10, 23)
    assert iv["Feb. 28 (past)"][0] == date(2026, 2, 28)
    assert not any(e.startswith("Oct.") and a.day == 1 for e, (a, _) in iv.items())  # no month-only duplicate


def test_relative_words_weeks_and_hebrew():
    iv = _by_expr("אתמול; last week; ב-14 בספטמבר; end of 2026")
    assert iv["אתמול"] == (date(2026, 9, 15),) * 2
    assert iv["last week"] == (date(2026, 9, 7), date(2026, 9, 13))
    assert date(2026, 9, 14) in {a for a, _ in iv.values()}
    assert iv["end of 2026"] == (date(2026, 12, 1), date(2026, 12, 31))


def test_modal_may_is_not_a_month():
    assert intervals("Rates may fall.", PUB) == []


# --- review fixes (retro#873 follow-up) ------------------------------------------------

def test_levantine_two_word_months_are_one_name():
    iv = _by_expr("في 5 كانون الأول")
    assert iv["5 كانون الأول (past)"] == (date(2025, 12, 5),) * 2
    assert intervals("المرحلة الثانية", PUB) == []           # "the second", not January


def test_lowercase_month_abbreviations_and_word_prefixes_are_not_dates():
    for t in ("about 30 may be killed", "5 mayors met", "10 decades ago", "3 marines"):
        assert intervals(t, PUB) == [], t
    assert "5 de mayo (past)" in _by_expr("5 de mayo")


def test_hebrew_prefix_tries_every_strip_length():
    assert _by_expr("ומחר")["ומחר"] == (date(2026, 9, 17),) * 2
    assert _by_expr("והיום")["והיום"] == (PUB, PUB)
    assert "בשבת (past)" in _by_expr("בשבת")
    assert intervals("בשני שלבים", PUB) == []                  # "in two stages"


def test_possessive_and_a_following_number_that_is_not_a_year():
    assert "Friday (past)" in _by_expr("Friday's vote")
    assert _by_expr("On May 5, 2000 soldiers marched")["May 5 (past)"] == (date(2026, 5, 5),) * 2
    assert _by_expr("May 5, 2026")["May 5, 2026"] == (date(2026, 5, 5),) * 2


def test_same_weekday_also_offers_a_week_earlier():
    iv = _by_expr("last Wednesday the vote passed")          # PUB is a Wednesday
    assert iv["Wednesday (a week earlier)"] == (date(2026, 9, 9),) * 2


def test_german_hier_and_capitalised_morgen_are_not_relative_days():
    assert intervals("hier ist es", PUB) == []
    assert [e for *_, e in intervals("heute Morgen", PUB)] == ["heute"]
    assert _by_expr("morgen früh")["morgen"] == (date(2026, 9, 17),) * 2
