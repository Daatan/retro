"""retro#428: the batch pipeline (runner.run_article, which orchestrator.py calls
for every article) must apply the same enforce_*/flag_* safety chain
forecast_api/forecaster.py already runs on the live Oracle path. Without it, bugs
already found and fixed for the live path (24.4% of precursor rows, 30.3% of
interested-party rows over-cap) reproduce silently in the batch/atlas path and get
cached indefinitely by (article_hash, event_id, prompt_version)."""

from unittest.mock import AsyncMock, patch

import pytest

from tm.models import ExtractionOutput, GatekeeperOutput, PredictionExtraction
from tm.runner import ArticleInput, run_article

pytestmark = pytest.mark.asyncio


def _article(**overrides) -> ArticleInput:
    defaults = dict(
        text="some article text",
        source_id="src1",
        source_name="Source One",
        article_date="2026-08-01",
        event_id="E1",
        event_name="Event happens",
        event_description="Will the event happen?",
    )
    defaults.update(overrides)
    return ArticleInput(**defaults)


async def _run_with_predictions(preds: list[PredictionExtraction]) -> ExtractionOutput:
    gate = GatekeeperOutput(is_prediction=True, reason="looks predictive")
    extraction = ExtractionOutput(predictions=preds)
    with patch("tm.runner.check_is_prediction", new=AsyncMock(return_value=(gate, {}))), \
         patch("tm.runner.extract_predictions", new=AsyncMock(return_value=(extraction, {}))), \
         patch("tm.runner.update_cell"):
        result = await run_article(_article())
    assert result.extraction is not None
    return result.extraction


async def test_precursor_cap_is_clamped():
    preds = [PredictionExtraction(
        quote="q", claim="c", stance=0.5, certainty=0.6,
        fact_signal=0.9, is_occurrence=False,
    )]
    out = await _run_with_predictions(preds)
    assert out.predictions[0].fact_signal == pytest.approx(0.3)


async def test_interested_party_stance_and_certainty_are_clamped():
    preds = [PredictionExtraction(
        quote="q", claim="c", stance=0.95, certainty=0.9, verified=False,
    )]
    out = await _run_with_predictions(preds)
    assert out.predictions[0].stance == pytest.approx(0.3)
    assert out.predictions[0].certainty == pytest.approx(0.5)


async def test_decider_intent_stance_is_clamped():
    preds = [PredictionExtraction(
        quote="q", claim="c", stance=0.9, certainty=0.8,
        fact_signal=0.25, is_occurrence=False, facet="announcement", verified=True,
    )]
    out = await _run_with_predictions(preds)
    assert out.predictions[0].stance == pytest.approx(0.3)


async def test_unanchored_cited_probability_is_demoted():
    preds = [PredictionExtraction(
        quote="a poll-aggregator model gives it 80%", claim="c",
        stance=0.8, certainty=0.8, evidence_class="cited_probability",
    )]
    out = await _run_with_predictions(preds)
    assert out.predictions[0].evidence_class != "cited_probability"


async def test_unanchored_positive_settlement_is_demoted():
    preds = [PredictionExtraction(
        quote="q", claim="c", stance=0.9, certainty=0.9, settled=True,
        event_date=None,
    )]
    out = await _run_with_predictions(preds)
    assert out.predictions[0].settled is False


async def test_relative_date_reference_is_resolved_against_article_date():
    preds = [PredictionExtraction(
        quote="q", claim="c", stance=0.5, certainty=0.5,
        event_date="2026-08-05", event_date_reference="this Friday",
    )]
    out = await _run_with_predictions(preds)
    # article_date=2026-08-01 (Saturday); "this Friday" resolves to 2026-08-07,
    # overriding the model's wrong 2026-08-05.
    assert out.predictions[0].event_date == "2026-08-07"


async def test_well_formed_prediction_is_left_untouched():
    preds = [PredictionExtraction(
        quote="q", claim="c", stance=0.6, certainty=0.7, verified=True,
    )]
    out = await _run_with_predictions(preds)
    assert out.predictions[0].stance == 0.6
    assert out.predictions[0].certainty == 0.7


async def test_winner_entity_conflict_is_neutralised_using_article_event_name():
    """retro#401: the batch path must run enforce_winner_entity_consistency
    exactly like the live Oracle path, keyed on article.event_name (runner.py
    has no separate `question` field — event_name IS the question here)."""
    gate = GatekeeperOutput(is_prediction=True, reason="looks predictive")
    preds = [PredictionExtraction(
        quote="q", claim="Argentina beat England", stance=1.0, certainty=0.95,
        settled=True, event_date="2026-07-30", fact_signal=1.0,
        event_actors="Argentina", event_target="England",
        is_occurrence=True, verified=True,
    )]
    extraction = ExtractionOutput(predictions=preds)
    article = _article(
        event_name=(
            "England will win their FIFA World Cup 2026 semi-final match "
            "against Argentina on July 15"
        ),
    )
    with patch("tm.runner.check_is_prediction", new=AsyncMock(return_value=(gate, {}))), \
         patch("tm.runner.extract_predictions", new=AsyncMock(return_value=(extraction, {}))), \
         patch("tm.runner.update_cell"):
        result = await run_article(article)
    assert result.extraction is not None
    out = result.extraction.predictions[0]
    assert out.stance == 0.0
    assert out.settled is False


async def test_settlement_contradicting_its_own_fact_lane_is_neutralised():
    """retro#545: a `settled` claim whose `fact_signal` opposes its own stance is
    self-contradictory — the live 41-row Burnham cluster, `stance=-1.00 settled`
    off articles reporting he took office. Neutralised, not inverted, and the
    batch path must run it like the live Oracle path (the batch feeds the atlas
    and Brier/ELO scoring, and caches its extraction indefinitely)."""
    preds = [PredictionExtraction(
        quote="q", claim="Burnham will remain PM", stance=-1.0, certainty=0.95,
        settled=True, event_date="2026-08-01", fact_signal=1.0, is_occurrence=True,
    )]
    out = await _run_with_predictions(preds)
    assert out.predictions[0].stance == 0.0
    assert out.predictions[0].settled is False
    assert out.predictions[0].certainty == pytest.approx(0.95)
