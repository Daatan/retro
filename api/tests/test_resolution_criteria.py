"""retro#353: the extractor must see the caller's resolution rules, not the
bare question, whenever the caller sends them — mirroring the batch
pipeline's llm_referee_criteria (pipeline/src/tm/orchestrator.py). Additive
and fail-open: a caller that doesn't send resolution_criteria gets today's
behavior (event_description=question) byte-for-byte unchanged.
"""
from __future__ import annotations

import pytest

from forecast_api import forecaster
from forecast_api.models import ArticleInput, ForecastRequest
from tm.models import ExtractionOutput, GatekeeperOutput, PredictionExtraction

_BODY = (
    "Fixture article body for the resolution-criteria suite. The gatekeeper "
    "and extractor are stubbed, so no model ever reads this text; it exists "
    "to clear the pipeline's non-empty-body checks. "
) * 3


def _article() -> ArticleInput:
    return ArticleInput(
        url="https://fixture.example.test/resolution-criteria",
        title="Resolution criteria fixture dispatch",
        snippet="Fixture snippet, long enough to be usable by the pipeline.",
        source="fixture",
        published_date="2026-07-28",
        text=_BODY,
    )


def _patch(monkeypatch, captured: list[str]):
    async def fake_gate(**kwargs):
        return (
            GatekeeperOutput(
                is_prediction=True, reason="fixture gate",
                prediction_count_estimate=1, relevance_score=1.0,
            ),
            {"total_tokens": 0},
        )

    async def fake_extract(**kwargs):
        captured.append(kwargs["event_description"])
        return (
            ExtractionOutput(predictions=[
                PredictionExtraction(quote="q", claim="c", stance=0.1, certainty=0.5),
            ]),
            {"total_tokens": 0},
        )

    monkeypatch.setattr(forecaster, "check_is_prediction", fake_gate)
    monkeypatch.setattr(forecaster, "extract_predictions", fake_extract)
    monkeypatch.setattr(forecaster, "get_credibility_weight", lambda sid: 1.0)


class TestResolutionCriteriaReachesTheExtractor:
    async def test_falls_back_to_the_bare_question_when_absent(self, monkeypatch):
        """No resolution_criteria sent — behavior must be byte-for-byte the
        pre-#353 default, not merely 'similar'."""
        captured: list[str] = []
        _patch(monkeypatch, captured)

        question = "[resolution-criteria-fallback] Will the event occur?"
        await forecaster.run_forecast(ForecastRequest(question=question, articles=[_article()]))

        assert captured == [question]

    async def test_uses_resolution_criteria_when_present(self, monkeypatch):
        captured: list[str] = []
        _patch(monkeypatch, captured)

        rules = "Only a formal, on-record government announcement counts."
        await forecaster.run_forecast(ForecastRequest(
            question="[resolution-criteria-present] Will the event occur?",
            articles=[_article()],
            resolution_criteria=rules,
        ))

        assert captured == [rules]

    async def test_empty_string_resolution_criteria_falls_back_too(self, monkeypatch):
        """An empty string is falsy, same as None — a caller that sends '' by
        accident (e.g. an un-set form field) must not blank the extractor's
        event context."""
        captured: list[str] = []
        _patch(monkeypatch, captured)

        question = "[resolution-criteria-empty] Will the event occur?"
        await forecaster.run_forecast(ForecastRequest(
            question=question, articles=[_article()], resolution_criteria="",
        ))

        assert captured == [question]
