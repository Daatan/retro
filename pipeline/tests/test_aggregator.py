"""aggregate_predictions() — cell-level certainty×specificity-weighted mean/median.
Pure math, no LLM/network (unlike aggregate_article_predictions, mocked elsewhere)."""

import pytest

from tm.aggregator import aggregate_predictions
from tm.models import PredictionExtraction


def _pred(stance, certainty, specificity=None, **overrides):
    overrides.setdefault("quote", "q")
    overrides.setdefault("claim", "c")
    return PredictionExtraction(
        stance=stance, certainty=certainty, specificity=specificity, **overrides,
    )


def test_empty_predictions_raises():
    with pytest.raises(ValueError):
        aggregate_predictions([])


def test_single_prediction_passes_through():
    sig = aggregate_predictions([_pred(0.6, 0.8, specificity=0.5)])
    assert sig.claim_count == 1
    assert sig.stance == pytest.approx(0.6)
    assert sig.certainty == pytest.approx(0.8)
    assert sig.specificity == pytest.approx(0.5)
    assert sig.quotes == ["q"] and sig.claims == ["c"]


def test_weighted_mean_favors_higher_certainty_specificity():
    # p1: weight 0.9*1.0=0.9, stance=1.0. p2: weight 0.1*1.0=0.1, stance=-1.0.
    sig = aggregate_predictions([
        _pred(1.0, 0.9, specificity=1.0),
        _pred(-1.0, 0.1, specificity=1.0),
    ])
    expected = (1.0 * 0.9 + -1.0 * 0.1) / (0.9 + 0.1)
    assert sig.stance == pytest.approx(expected)
    assert sig.stance > 0  # dominated by the higher-weight prediction


def test_missing_specificity_defaults_weight_to_certainty():
    # No specificity given -> weight = certainty * 1.0.
    sig = aggregate_predictions([_pred(0.4, 0.5), _pred(0.8, 0.5)])
    assert sig.stance == pytest.approx((0.4 + 0.8) / 2)
    assert sig.specificity is None


def test_zero_total_weight_falls_back_to_plain_mean():
    # certainty=0 for both -> weight sum is 0 -> _weighted_mean falls back to mean().
    sig = aggregate_predictions([_pred(1.0, 0.0), _pred(-1.0, 0.0)])
    assert sig.stance == pytest.approx(0.0)
    assert sig.certainty == pytest.approx(0.0)


def test_optional_field_none_when_all_predictions_lack_it():
    sig = aggregate_predictions([_pred(0.2, 0.5), _pred(0.3, 0.5)])
    assert sig.sentiment is None
    assert sig.hedge_ratio is None
    assert sig.magnitude is None
    assert sig.source_authority is None


def test_optional_field_skips_predictions_missing_it():
    sig = aggregate_predictions([
        _pred(0.2, 0.5, sentiment=0.8),
        _pred(0.3, 0.5, sentiment=None),
    ])
    # Only the first prediction contributes -> weighted mean of a single value is itself.
    assert sig.sentiment == pytest.approx(0.8)


def test_time_horizon_majority_vote():
    sig = aggregate_predictions([
        _pred(0.1, 0.5, time_horizon="short"),
        _pred(0.2, 0.5, time_horizon="short"),
        _pred(0.3, 0.5, time_horizon="long"),
    ])
    assert sig.time_horizon == "short"


def test_time_horizon_none_when_all_missing():
    sig = aggregate_predictions([_pred(0.1, 0.5), _pred(0.2, 0.5)])
    assert sig.time_horizon is None


def test_prediction_type_majority_vote():
    sig = aggregate_predictions([
        _pred(0.1, 0.5, prediction_type="binary"),
        _pred(0.2, 0.5, prediction_type="continuous"),
        _pred(0.3, 0.5, prediction_type="binary"),
    ])
    assert sig.prediction_type == "binary"


def test_time_horizon_days_weighted_median():
    # Weights: 0.9, 0.05, 0.05 (certainty, no specificity) -> cumulative hits 50% at the
    # first (highest-weight) value once sorted by value, i.e. the dominant contributor wins.
    sig = aggregate_predictions([
        _pred(0.1, 0.9, time_horizon_days=30),
        _pred(0.1, 0.05, time_horizon_days=60),
        _pred(0.1, 0.05, time_horizon_days=90),
    ])
    assert sig.time_horizon_days == 30


def test_time_horizon_days_none_when_all_missing():
    sig = aggregate_predictions([_pred(0.1, 0.5), _pred(0.2, 0.5)])
    assert sig.time_horizon_days is None


def test_time_horizon_days_zero_weight_falls_back_to_plain_median():
    sig = aggregate_predictions([
        _pred(0.1, 0.0, time_horizon_days=10),
        _pred(0.1, 0.0, time_horizon_days=20),
        _pred(0.1, 0.0, time_horizon_days=30),
    ])
    assert sig.time_horizon_days == 20


def test_claim_count_and_quotes_claims_preserve_order():
    sig = aggregate_predictions([
        _pred(0.1, 0.5, quote="q1", claim="c1"),
        _pred(0.2, 0.5, quote="q2", claim="c2"),
    ])
    assert sig.claim_count == 2
    assert sig.quotes == ["q1", "q2"]
    assert sig.claims == ["c1", "c2"]
