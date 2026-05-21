"""Basic model validation tests — no LLM calls."""

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
