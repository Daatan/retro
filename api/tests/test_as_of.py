"""retro#1077: ``ForecastRequest.as_of`` answers a question as of a past date.

Every "today" the /forecast path reads must follow it: recency decay, the
premise verifier (its trigger and its prompt's TODAY line), the settlement
timing guards and the glide. Live requests (no ``as_of``) are unchanged.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from forecast_api import forecaster
from forecast_api.aggregation import settlement_decision
from forecast_api.config import settings as api_settings
from forecast_api.forecaster import build_claim_meta
from forecast_api.models import ArticleInput, ForecastRequest
from tm.models import ExtractionOutput, GatekeeperOutput, PredictionExtraction

_AS_OF = "2026-11-08"
_Q = "Will the fixture event happen by November 30, 2026?"
_ARTICLE = ArticleInput(
    url="https://source.example.test/as-of-story",
    title="Fixture headline",
    snippet="Fixture snippet long enough to be usable.",
    source="source",
    published_date="2026-11-01",
    text="Fixture body for the as-of wiring suite; gate and extractor are stubbed. " * 3,
)


class TestRequestField:
    def test_normalises_timestamp_to_date(self):
        r = ForecastRequest(question=_Q, articles=[_ARTICLE], as_of="2026-11-08T13:00:00Z")
        assert r.as_of == _AS_OF

    def test_rejects_garbage(self):
        with pytest.raises(ValidationError):
            ForecastRequest(question=_Q, articles=[_ARTICLE], as_of="last tuesday")

    def test_requires_supplied_articles(self):
        # Oracul's own search is not date-bounded: as_of without articles would
        # silently mix post-as_of evidence into a "pre-as_of" answer.
        with pytest.raises(ValidationError):
            ForecastRequest(question=_Q, as_of=_AS_OF)

    def test_absent_by_default(self):
        assert ForecastRequest(question=_Q).as_of is None


class TestCacheKey:
    def test_as_of_discriminates(self):
        live = ForecastRequest(question=_Q, articles=[_ARTICLE], claim_deadline="2026-11-30")
        retro = ForecastRequest(question=_Q, articles=[_ARTICLE], claim_deadline="2026-11-30", as_of=_AS_OF)
        assert build_claim_meta(live) == "|2026-11-30"  # unchanged for live callers
        assert build_claim_meta(retro) == f"|2026-11-30|as_of={_AS_OF}"


class TestSettlementTiming:
    """A NO pin on an arrival claim is only coherent after its deadline."""

    def _decide(self, today):
        return settlement_decision(
            [-0.9, -0.9], [1.0, 1.0], [True, True],
            settlement_min_sources=2,
            claim_direction="arrival",
            claim_deadline="2026-11-30",
            today=today,
        )

    def test_before_deadline_as_of_suppresses_no_pin(self):
        assert self._decide("2026-11-08").suppression_reason == "settlement_direction"

    def test_after_deadline_as_of_allows_no_pin(self):
        d = self._decide("2026-12-05")
        assert d.direction == -1.0 and not d.suppressed


async def test_run_forecast_threads_as_of(monkeypatch):
    async def fake_gate(**kwargs):
        return (GatekeeperOutput(is_prediction=True, reason="fixture", relevance_score=1.0), {"total_tokens": 0})

    async def fake_extract(**kwargs):
        return (ExtractionOutput(predictions=[PredictionExtraction(
            quote="Officials said the step is planned.", claim="The step is planned.",
            stance=0.4, certainty=0.6, settled=False,
        )]), {"total_tokens": 0})

    premise_today: list = []

    async def fake_verify(*a, **kw):
        premise_today.append(kw.get("today"))
        from forecast_api.premise_verifier import Verdict
        return Verdict(dead=False, reason="fixture")

    ref_dates: list = []
    real_recency = forecaster.recency_weight

    def spy_recency(published, ref, *a, **kw):
        ref_dates.append(ref)
        return real_recency(published, ref, *a, **kw)

    monkeypatch.setattr(forecaster, "check_is_prediction", fake_gate)
    monkeypatch.setattr(forecaster, "extract_predictions", fake_extract)
    monkeypatch.setattr(forecaster, "verify_premise", fake_verify)
    monkeypatch.setattr(forecaster, "recency_weight", spy_recency)
    monkeypatch.setattr(api_settings, "premise_verifier_enabled", True)

    await forecaster.run_forecast(ForecastRequest(
        question="Will the as-of wiring step happen by November 30, 2026?",
        articles=[_ARTICLE], claim_deadline="2026-11-30", claim_archetype="scheduled",
        claim_created_at=_AS_OF, as_of=_AS_OF,
    ))
    assert premise_today == [_AS_OF]
    assert ref_dates and set(ref_dates) == {_AS_OF}
