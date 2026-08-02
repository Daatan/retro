"""Settlement-override tests for ``run_forecast``.

Pooling in log-odds space is bounded by its most confident member, so a decided
event could never read as decided from averaging alone — the Knicks forecast sat
at 82% the day after the title was won because the extractor's stances topped
out at 0.92 and a pre-clincher article dragged the pool down. The settlement
override pins the estimate to ±settlement_stance and sets ``settled=True`` when
at least ``settlement_min_sources`` independent sources carry settlement claims
(the outcome reported as an accomplished fact) agreeing in direction.

These tests drive the full ``_run_forecast_inner`` flow through the
caller-supplied-articles path with the gatekeeper and extractor mocked, so no
network or LLM is involved.
"""

from datetime import date, timedelta

import pytest

from forecast_api import forecaster
from forecast_api.config import settings as api_settings
from forecast_api.models import ArticleInput, ForecastRequest
from tm.models import ExtractionOutput, GatekeeperOutput, PredictionExtraction


_TITLES = [
    "Champions crowned after decisive final game five",
    "Analysts weigh title odds ahead of pivotal matchup",
    "Fans celebrate across the city following historic victory parade",
    "League officials review season records amid ongoing debate",
]


# Settlement is weighted evidence, not just a flag: with
# settlement_quality_floor enabled (retro#372) the settling votes must carry
# real mass, and mass includes recency. A fixture with a hard-coded past date
# decays a little further every day the suite is run, so a test written to
# check the pin mechanics silently becomes a test of recency decay. These
# dates are therefore relative — "yesterday's coverage" — which is what every
# test in this file actually means by a settlement.
_FRESH = (date.today() - timedelta(days=1)).isoformat()
_FRESH_EVENT = (date.today() - timedelta(days=2)).isoformat()


def _article(i: int) -> ArticleInput:
    # Titles must be genuinely distinct or the syndication dedupe collapses the
    # set to one source (title-token Jaccard >= 0.8 merges near-duplicates).
    return ArticleInput(
        url=f"https://source-{i}.example.com/story-{i}",
        title=_TITLES[i - 1],
        snippet=f"A snippet with enough length to pass the fallback minimum, variant {i}.",
        source=f"source-{i}",
        published_date=_FRESH,
        text="A long enough prefetched article body about the event in question." * 3,
    )


def _gate_ok():
    return GatekeeperOutput(
        is_prediction=True, reason="on topic", prediction_count_estimate=2,
        relevance_score=1.0,
    ), {"total_tokens": 10}


def _prediction(stance: float, certainty: float, settled: bool | None = None) -> PredictionExtraction:
    # Positive settlements must be dated (enforce_settlement_event_date) — on/before
    # the article date. Negative ones carry no date by definition.
    return PredictionExtraction(
        quote="quote", claim="claim", stance=stance, certainty=certainty, settled=settled,
        event_date=_FRESH_EVENT if settled and stance > 0 else None,
    )


def _patch_pipeline(monkeypatch, extractions_by_url: dict[str, list[PredictionExtraction]]):
    async def fake_gate(**_kwargs):
        return _gate_ok()

    async def fake_extract(**kwargs):
        # source_name is "source-N"; match by index embedded in the name.
        preds = extractions_by_url[kwargs["source_name"]]
        return ExtractionOutput(predictions=preds), {"total_tokens": 20}

    monkeypatch.setattr(forecaster, "check_is_prediction", fake_gate)
    monkeypatch.setattr(forecaster, "extract_predictions", fake_extract)
    # Isolate from cross-test cache hits.
    monkeypatch.setattr(api_settings, "cache_ttl_seconds", 0)


class TestSettlementOverride:
    async def test_two_settled_sources_pin_the_estimate(self, monkeypatch):
        _patch_pipeline(monkeypatch, {
            # Two sources report the outcome as fact; one stale hedge dissents.
            "source-1": [_prediction(1.0, 0.95, settled=True), _prediction(0.4, 0.5)],
            "source-2": [_prediction(0.95, 0.9, settled=True)],
            "source-3": [_prediction(0.13, 0.6)],
        })
        resp = await forecaster.run_forecast(ForecastRequest(
            question="settlement pin — unique question A",
            articles=[_article(1), _article(2), _article(3)],
        ))

        assert resp.settled is True
        assert resp.mean == pytest.approx(api_settings.settlement_stance)
        assert resp.ci_low < resp.mean < resp.ci_high
        assert resp.ci_high <= 1.0
        settled_flags = {s.source_id: s.settled for s in resp.sources}
        assert sum(1 for v in settled_flags.values() if v) == 2

    async def test_single_settled_source_does_not_pin(self, monkeypatch):
        _patch_pipeline(monkeypatch, {
            "source-1": [_prediction(1.0, 0.95, settled=True)],
            "source-2": [_prediction(0.5, 0.6)],
        })
        resp = await forecaster.run_forecast(ForecastRequest(
            question="settlement pin — unique question B",
            articles=[_article(1), _article(2)],
        ))

        assert resp.settled is False
        assert resp.mean != pytest.approx(api_settings.settlement_stance)

    async def test_negative_settlement_pins_to_negative_boundary(self, monkeypatch):
        # Dated with the FORECLOSING event (the PR-#291 extractor contract) —
        # under aggregation-time revalidation an undated negative without a
        # closed claim window no longer pins (see test_settlement_revalidation).
        def neg(stance: float, certainty: float) -> PredictionExtraction:
            return PredictionExtraction(
                quote="quote", claim="claim", stance=stance, certainty=certainty,
                settled=True, event_date="2026-06-15",
            )
        _patch_pipeline(monkeypatch, {
            "source-1": [neg(-1.0, 0.95)],
            "source-2": [neg(-0.9, 0.9)],
        })
        resp = await forecaster.run_forecast(ForecastRequest(
            question="settlement pin — unique question C",
            articles=[_article(1), _article(2)],
        ))

        assert resp.settled is True
        assert resp.mean == pytest.approx(-api_settings.settlement_stance)
        assert -1.0 <= resp.ci_low < resp.mean < resp.ci_high

    async def test_conflicting_settlements_tie_leaves_pool_untouched(self, monkeypatch):
        _patch_pipeline(monkeypatch, {
            "source-1": [_prediction(1.0, 0.95, settled=True)],
            "source-2": [_prediction(-1.0, 0.95, settled=True)],
        })
        resp = await forecaster.run_forecast(ForecastRequest(
            question="settlement pin — unique question D",
            articles=[_article(1), _article(2)],
        ))

        assert resp.settled is False

    async def test_settled_claims_override_article_color_quotes(self, monkeypatch):
        # One article: a settlement claim plus three mid-stance color quotes.
        # The article's stance must come from the settled claim alone, not the
        # certainty-weighted soup (the "Brunson's mental toughness" dilution).
        _patch_pipeline(monkeypatch, {
            "source-1": [
                _prediction(1.0, 0.95, settled=True),
                _prediction(0.5, 0.7),
                _prediction(0.4, 0.6),
                _prediction(0.6, 0.5),
            ],
        })
        resp = await forecaster.run_forecast(ForecastRequest(
            question="settlement pin — unique question E",
            articles=[_article(1)],
        ))

        assert resp.sources[0].stance == pytest.approx(1.0)
        assert resp.sources[0].settled is True
        # Only one settled source → no response-level pin.
        assert resp.settled is False

    async def test_override_disabled_via_settings(self, monkeypatch):
        _patch_pipeline(monkeypatch, {
            "source-1": [_prediction(1.0, 0.95, settled=True)],
            "source-2": [_prediction(1.0, 0.95, settled=True)],
        })
        monkeypatch.setattr(api_settings, "settlement_min_sources", 0)
        resp = await forecaster.run_forecast(ForecastRequest(
            question="settlement pin — unique question F",
            articles=[_article(1), _article(2)],
        ))

        assert resp.settled is False
