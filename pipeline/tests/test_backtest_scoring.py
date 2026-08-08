"""Unit tests for the Brier scoring/calibration math in backtest.py:
brier_score, weighted_average_prediction, _pava/fit_isotonic — untested
until now (retro#437)."""

import pytest

from tm.backtest import (
    brier_score,
    weighted_average_prediction,
    fit_isotonic,
    _pava,
)


# --- brier_score ---

def test_brier_score_perfect_prediction_is_zero():
    assert brier_score(1.0, True) == pytest.approx(0.0)
    assert brier_score(0.0, False) == pytest.approx(0.0)


def test_brier_score_worst_prediction_is_one():
    assert brier_score(0.0, True) == pytest.approx(1.0)
    assert brier_score(1.0, False) == pytest.approx(1.0)


def test_brier_score_random_guess_is_quarter():
    assert brier_score(0.5, True) == pytest.approx(0.25)
    assert brier_score(0.5, False) == pytest.approx(0.25)


# --- weighted_average_prediction ---

def _entry(source_id, *stances):
    return {"source_id": source_id, "predictions": [{"stance": s, "certainty": 0.5} for s in stances]}


def test_weighted_average_maps_stance_to_probability():
    # Single entry, stance=1.0 (max positive) -> probability should map to 1.0.
    result = weighted_average_prediction([_entry("a", 1.0)], {"a": 0.1})
    assert result == pytest.approx(1.0)


def test_weighted_average_negative_stance_maps_near_zero():
    result = weighted_average_prediction([_entry("a", -1.0)], {"a": 0.1})
    assert result == pytest.approx(0.0)


def test_weighted_average_more_accurate_source_dominates():
    # source "good" (brier=0.05 -> weight 0.95) says +1; source "bad" (brier=0.9 -> weight
    # clamped to 0.01) says -1. Result should be dominated by the accurate source.
    entries = [_entry("good", 1.0), _entry("bad", -1.0)]
    result = weighted_average_prediction(entries, {"good": 0.05, "bad": 0.9})
    assert result > 0.9


def test_weighted_average_unranked_source_uses_default_brier():
    # No source_briers entry -> DEFAULT_SOURCE_BRIER (0.25) is used, weight 0.75.
    result = weighted_average_prediction([_entry("unknown", 1.0)], {})
    assert result == pytest.approx(1.0)


def test_weighted_average_empty_entries_returns_neutral():
    assert weighted_average_prediction([], {}) == pytest.approx(0.5)


def test_weighted_average_skips_entries_with_no_predictions():
    entries = [{"source_id": "a", "predictions": []}, _entry("b", 1.0)]
    result = weighted_average_prediction(entries, {"b": 0.1})
    assert result == pytest.approx(1.0)


def test_weighted_average_skips_malformed_predictions_without_neutral_default():
    # A malformed prediction (non-numeric stance) must be dropped, not scored as 0.
    entries = [
        {"source_id": "a", "predictions": [{"stance": "not-a-number", "certainty": 0.5}]},
        _entry("b", 1.0),
    ]
    result = weighted_average_prediction(entries, {"b": 0.1})
    assert result == pytest.approx(1.0)


# --- _pava / fit_isotonic ---

def test_pava_already_monotonic_is_unchanged():
    assert _pava([0.1, 0.3, 0.5, 0.9]) == pytest.approx([0.1, 0.3, 0.5, 0.9])


def test_pava_pools_a_single_violation():
    # 0.5 > 0.2 violates monotonicity -> pooled into their mean (0.35).
    assert _pava([0.1, 0.5, 0.2, 0.9]) == pytest.approx([0.1, 0.35, 0.35, 0.9])


def test_pava_empty_input_is_noop():
    assert _pava([]) == []


def test_pava_single_value():
    assert _pava([0.7]) == pytest.approx([0.7])


def test_fit_isotonic_transform_is_monotonic_and_bounded():
    predictions = [0.1, 0.9, 0.3, 0.7, 0.5]
    outcomes = [False, True, False, True, True]
    transform = fit_isotonic(predictions, outcomes)
    lo, hi = transform(min(predictions)), transform(max(predictions))
    assert 0.0 <= lo <= hi <= 1.0


def test_fit_isotonic_perfect_calibration_is_identity_at_training_points():
    # Predictions already equal outcomes exactly and are sorted -> PAV changes nothing.
    predictions = [0.0, 1.0]
    outcomes = [False, True]
    transform = fit_isotonic(predictions, outcomes)
    assert transform(0.0) == pytest.approx(0.0)
    assert transform(1.0) == pytest.approx(1.0)
