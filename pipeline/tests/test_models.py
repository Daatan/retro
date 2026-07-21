"""Basic model validation tests — no LLM calls."""

import pytest
from pydantic import ValidationError

from tm.models import (
    GatekeeperOutput,
    PredictionExtraction,
    ExtractionOutput,
    PredictionType,
    MatrixState,
    CellStatus,
)


def test_gatekeeper_output():
    out = GatekeeperOutput(is_prediction=True, reason="Contains forecast", prediction_count_estimate=2)
    assert out.is_prediction is True
    assert out.prediction_count_estimate == 2


def test_prediction_extraction_clamps():
    pred = PredictionExtraction(
        quote="test",
        claim="test claim",
        stance=0.8,
        sentiment=0.5,
        certainty=0.9,
        specificity=0.7,
        hedge_ratio=0.1,
        conditionality=0.0,
        magnitude=0.6,
        time_horizon="months",
        time_horizon_days=90,
        prediction_type=PredictionType.binary,
        source_authority=0.8,
    )
    assert pred.stance == 0.8
    assert pred.time_horizon_days == 90


def test_prediction_extraction_quantitative_estimate_defaults_to_none():
    pred = PredictionExtraction(quote="q", claim="c", stance=0.0, certainty=0.5)
    assert pred.quantitative_estimate is None


def test_prediction_extraction_quantitative_estimate_accepts_valid_probability():
    pred = PredictionExtraction(
        quote="q", claim="c", stance=-0.62, certainty=0.85, quantitative_estimate=0.1883,
    )
    assert pred.quantitative_estimate == 0.1883


def test_prediction_extraction_quantitative_estimate_rejects_out_of_range():
    with pytest.raises(ValidationError):
        PredictionExtraction(quote="q", claim="c", stance=0.0, certainty=0.5, quantitative_estimate=1.5)


def test_prediction_extraction_fact_signal_facets_default_to_none():
    pred = PredictionExtraction(quote="q", claim="c", stance=0.0, certainty=0.5)
    assert pred.fact_signal is None
    assert pred.event_actors is None
    assert pred.event_target is None
    assert pred.is_occurrence is None
    assert pred.verified is None


def test_prediction_extraction_fact_signal_facets_accept_valid_values():
    pred = PredictionExtraction(
        quote="q", claim="c", stance=0.8, certainty=0.9,
        fact_signal=0.3, event_actors="United States", event_target="Iran",
        is_occurrence=False, verified=True,
    )
    assert pred.fact_signal == 0.3
    assert pred.event_actors == "United States"
    assert pred.event_target == "Iran"
    assert pred.is_occurrence is False
    assert pred.verified is True


def test_prediction_extraction_fact_signal_rejects_out_of_range():
    with pytest.raises(ValidationError):
        PredictionExtraction(quote="q", claim="c", stance=0.0, certainty=0.5, fact_signal=1.5)
    with pytest.raises(ValidationError):
        PredictionExtraction(quote="q", claim="c", stance=0.0, certainty=0.5, fact_signal=-1.5)


def test_extraction_output_author_lean_defaults_to_none():
    out = ExtractionOutput(predictions=[])
    assert out.author_lean is None
    assert out.author_lean_certainty is None


def test_extraction_output_author_lean_accepts_valid_values():
    out = ExtractionOutput(predictions=[], author_lean=-0.6, author_lean_certainty=0.5)
    assert out.author_lean == -0.6
    assert out.author_lean_certainty == 0.5


def test_extraction_output_author_lean_rejects_out_of_range():
    with pytest.raises(ValidationError):
        ExtractionOutput(predictions=[], author_lean=1.5)
    with pytest.raises(ValidationError):
        ExtractionOutput(predictions=[], author_lean_certainty=-0.1)


def test_matrix_state_tracking():
    state = MatrixState()

    # Default is pending
    cell = state.get("A01", "ynet")
    assert cell.status == CellStatus.pending

    # Update status
    state.set_status("A01", "ynet", CellStatus.done, prediction_count=3)
    assert state.get("A01", "ynet").status == CellStatus.done
    assert state.get("A01", "ynet").prediction_count == 3

    # Stats
    stats = state.stats()
    assert stats["done"] == 1


def test_matrix_state_key():
    state = MatrixState()
    assert state.key("B01", "haaretz") == "B01:haaretz"
