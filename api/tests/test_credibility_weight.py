"""Regression: a legitimate conservative score of 0.0 must be treated as present
(weight 1.0), not fall through the old falsy-`or` to the ELO/Brier path."""

from forecast_api.leaderboard import _conservative_or_zero, _conservative_score


def test_zero_is_a_valid_score_not_missing():
    assert _conservative_score({"skill_conservative": 0.0}) == 0.0
    assert _conservative_or_zero({"skill_conservative": 0.0}) == 0.0


def test_falls_back_to_legacy_field():
    assert _conservative_score({"trueskill_conservative": 5.0}) == 5.0
    # A present 0.0 on the new field wins over the legacy field.
    assert _conservative_score({"skill_conservative": 0.0, "trueskill_conservative": 5.0}) == 0.0


def test_none_only_when_neither_present():
    assert _conservative_score({}) is None
    assert _conservative_or_zero({}) == 0.0
