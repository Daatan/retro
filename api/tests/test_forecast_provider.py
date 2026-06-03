"""The /forecast response must expose the search provider, fallback chain, and
distilled query at the top level (not only under debug=true), so daatan can log
which engine served a forecast call.

Mirrors test_searcher.py's provider-attribution coverage for the forecaster.
"""

import asyncio

from forecast_api import forecaster
from forecast_api.forecaster import _empty_response, run_forecast
from forecast_api.models import ArticleInput, ForecastRequest


class TestEmptyResponseForwardsProvider:
    def test_forwards_provider_chain_and_distilled(self):
        r = _empty_response(
            "q?",
            reason="all_articles_off_topic",
            provider="gdelt_bq",
            provider_chain=["gdelt", "brightdata", "dataforseo", "ddg", "gdelt_bq"],
            distilled_query="arab minister israel",
        )
        assert r.provider == "gdelt_bq"
        assert r.provider_chain == ["gdelt", "brightdata", "dataforseo", "ddg", "gdelt_bq"]
        assert r.distilled_query == "arab minister israel"

    def test_defaults_safe(self):
        r = _empty_response("q?")
        assert r.provider == "" and r.provider_chain == [] and r.distilled_query is None


class TestForecastResponseProvider:
    def test_empty_path_carries_provider(self, monkeypatch):
        # Search returns nothing; provider attribution still flows to the top level.
        monkeypatch.setattr(forecaster, "search_articles", lambda q, limit: [])
        monkeypatch.setattr(forecaster, "get_last_search_provider", lambda: "ddg")
        monkeypatch.setattr(forecaster, "get_last_search_provider_chain",
                            lambda: ["gdelt", "brightdata", "dataforseo", "ddg", "gdelt_bq"])

        async def _no_distill(q):
            return q
        monkeypatch.setattr(forecaster, "_distill_query", _no_distill)

        resp = asyncio.run(run_forecast(ForecastRequest(question="Empty-probe 7x — will Z happen?")))
        assert resp.insufficient_data and resp.reason == "no_search_results"
        assert resp.provider == "ddg"
        assert resp.provider_chain == ["gdelt", "brightdata", "dataforseo", "ddg", "gdelt_bq"]
        assert resp.distilled_query is None  # distillation was a no-op

    def test_distilled_query_is_exposed(self, monkeypatch):
        monkeypatch.setattr(forecaster, "search_articles", lambda q, limit: [])
        monkeypatch.setattr(forecaster, "get_last_search_provider", lambda: "none")
        monkeypatch.setattr(forecaster, "get_last_search_provider_chain", lambda: [])

        async def _distill(q):
            return "russia ukraine ceasefire"
        monkeypatch.setattr(forecaster, "_distill_query", _distill)

        resp = asyncio.run(run_forecast(ForecastRequest(question="Distill-probe 4q2 — ceasefire soon?")))
        assert resp.distilled_query == "russia ukraine ceasefire"

    def test_caller_supplied_articles_report_caller(self, monkeypatch):
        # Caller passes pre-fetched articles → no search, no network. Mock the
        # gatekeeper/extractor so the success path builds a real ForecastResponse.
        from tm.models import GatekeeperOutput, ExtractionOutput, PredictionExtraction

        async def _gate(**kw):
            return GatekeeperOutput(is_prediction=True, reason="on topic", prediction_count_estimate=1), {}

        async def _extract(**kw):
            return ExtractionOutput(predictions=[
                PredictionExtraction(quote="q", claim="c", stance=0.6, certainty=0.8)
            ]), {}

        monkeypatch.setattr(forecaster, "check_is_prediction", _gate)
        monkeypatch.setattr(forecaster, "extract_predictions", _extract)

        req = ForecastRequest(
            question="Caller-probe 9z — will the measure pass?",
            articles=[ArticleInput(
                url="https://example.com/a",
                title="Detailed analysis of whether the measure will pass this session",
                snippet="Analysts weigh in on the vote.",
                text="The measure is widely expected to pass given the coalition's majority. " * 4,
            )],
        )
        resp = asyncio.run(run_forecast(req))
        assert not resp.insufficient_data
        assert resp.articles_used >= 1
        assert resp.provider == "caller"
        assert resp.provider_chain == ["caller"]
        assert resp.distilled_query is None
