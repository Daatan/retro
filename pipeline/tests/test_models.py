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


def test_gatekeeper_output_unwraps_properties_envelope():
    """Nova Lite (MD_JSON mode) intermittently wraps output as {"properties": {...}}
    instead of the flat shape — retro#306."""
    out = GatekeeperOutput.model_validate({
        "properties": {"is_prediction": True, "reason": "Contains forecast", "prediction_count_estimate": 2}
    })
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


def test_prediction_extraction_unwraps_properties_envelope():
    pred = PredictionExtraction.model_validate({
        "properties": {"quote": "q", "claim": "c", "stance": 0.4, "certainty": 0.6}
    })
    assert pred.quote == "q"
    assert pred.stance == 0.4


def test_extraction_output_unwraps_properties_envelope():
    out = ExtractionOutput.model_validate({
        "properties": {"predictions": [], "author_lean": 0.3, "author_lean_certainty": 0.7}
    })
    assert out.author_lean == 0.3
    assert out.author_lean_certainty == 0.7


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


class TestClaimStrengthCertaintyAlias:
    """`certainty` was renamed to `claim_strength` in Oracle 1.5 Phase 1 (retro#680).

    The old name stays live as a WIRE alias for one schema cycle: inbound it is
    accepted, outbound it is still emitted. That keeps every already-persisted
    atlas row and every reader that indexes the literal key ``certainty``
    (`tm.utils.split_scored_predictions`, `tm.scorer`, `tm.backtest`,
    `tm.render_atlas`) working unchanged across the deploy, rather than having
    them silently reclassify new rows as malformed.

    These tests are the contract. When the alias is dropped next cycle, the ones
    asserting the old name are what should fail first.
    """

    def test_accepts_the_old_name_inbound(self):
        pred = PredictionExtraction(quote="q", claim="c", stance=0.1, certainty=0.65)
        assert pred.claim_strength == 0.65

    def test_accepts_the_new_name_inbound(self):
        pred = PredictionExtraction(quote="q", claim="c", stance=0.1, claim_strength=0.65)
        assert pred.claim_strength == 0.65

    def test_emits_both_names_with_the_same_value(self):
        dumped = PredictionExtraction(
            quote="q", claim="c", stance=0.1, claim_strength=0.65
        ).model_dump()
        assert dumped["claim_strength"] == 0.65
        assert dumped["certainty"] == 0.65

    def test_the_old_name_survives_a_dump_validate_round_trip(self):
        pred = PredictionExtraction(quote="q", claim="c", stance=0.1, certainty=0.65)
        assert PredictionExtraction.model_validate(pred.model_dump()).claim_strength == 0.65

    def test_the_alias_is_not_offered_to_the_llm(self):
        """The alias must live on the wire only, never in the elicitation schema.

        A `@computed_field` would also emit both names — but it would additionally
        publish `certainty` into the JSON schema instructor sends to the model,
        re-teaching the model the name this rename exists to retire, and inviting
        it to fill in two fields that must never disagree. `@model_serializer`
        keeps the alias strictly outbound; this test is what tells the two apart.
        """
        props = PredictionExtraction.model_json_schema()["properties"]
        assert "claim_strength" in props
        assert "certainty" not in props
