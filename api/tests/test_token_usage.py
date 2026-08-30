"""token_usage on API responses (Daatan/docs#57 item 3).

The pipeline has computed per-call token usage since the llm.py consolidation,
then dropped it on the production path — spend was only visible via Cost
Explorer, or by sending debug=true. These tests pin the new contract:

    "token_usage": {"prompt_tokens", "completion_tokens", "total_tokens",
                    "cache_read_tokens", "cache_write_tokens"}

on ForecastResponse (summed over the run's LLM calls, NOT gated on debug),
RelevanceResponse and LlmResponse (their single call). The whole object is
null when no call reported usage. LLMs are mocked per repo convention.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from forecast_api import forecaster
from forecast_api.main import app
from forecast_api.models import ForecastRequest, TokenUsage
from tm.models import GatekeeperOutput, PredictionExtraction

client = TestClient(app)
HEADERS = {"x-api-key": "test-key"}

GATE_USAGE = {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}
EXTRACT_USAGE = {
    "prompt_tokens": 200, "completion_tokens": 30, "total_tokens": 230,
    "cache_read_input_tokens": 50, "cache_creation_input_tokens": 20,
}


def _article(url, title="A clear title about the event"):
    # Distinct titles per call site where two articles must survive — identical
    # titles get collapsed by the syndication dedupe before processing.
    return {
        "url": url,
        "title": title,
        "snippet": "A snippet long enough to clear the twenty-char fallback guard.",
        "source": "x.com",
        "published_date": "2026-07-15",
        "text": "Full article body, already fetched so no network call happens here.",
    }


def _gatekeeper(is_prediction=True):
    return AsyncMock(return_value=(
        GatekeeperOutput(is_prediction=is_prediction, reason="judged", relevance_score=0.9,
                         prediction_count_estimate=1),
        dict(GATE_USAGE),
    ))


def _extractor():
    async def _extract(**kwargs):
        return SimpleNamespace(predictions=[PredictionExtraction(
            quote="q", claim="c", stance=0.6, certainty=0.8, specificity=1.0, settled=None,
        )], author_lean=None, author_lean_certainty=None, consensus_view=None,
            claim_actor=None, claim_predicate=None, claim_scope=None), dict(EXTRACT_USAGE)
    return _extract


class TestTokenUsageModel:
    def test_sums_usage_dicts_and_maps_cache_keys(self):
        tu = TokenUsage.from_usages([GATE_USAGE, EXTRACT_USAGE])
        assert tu == TokenUsage(
            prompt_tokens=300, completion_tokens=40, total_tokens=340,
            cache_read_tokens=50, cache_write_tokens=20,
        )

    def test_all_empty_yields_none_not_zeros(self):
        # A null field says "nothing known"; an all-zeros object would claim
        # the run verifiably cost nothing — different statements on the wire.
        assert TokenUsage.from_usages([]) is None
        assert TokenUsage.from_usages([{}, {}]) is None


class TestForecastTokenUsage:
    async def test_forecast_sums_gate_and_extract_over_all_articles_without_debug(self, monkeypatch):
        monkeypatch.setattr(forecaster, "check_is_prediction", _gatekeeper())
        monkeypatch.setattr(forecaster, "extract_predictions", _extractor())
        monkeypatch.setattr(forecaster, "get_credibility_weight", lambda sid: 1.0)

        req = ForecastRequest(
            question="Token-usage probe 41a — will the event happen by 2026-12-31?",
            articles=[
                _article("http://x.com/tu-1", title="Minister says vote likely before deadline"),
                _article("http://y.com/tu-2", title="Analysts doubt the coalition holds through autumn"),
            ],
            debug=False,  # the point of docs#57 item 3: NOT gated on debug
        )
        resp = await forecaster.run_forecast(req)

        assert resp.insufficient_data is False
        assert resp.debug is None  # debug stays opt-in, untouched
        assert resp.token_usage is not None
        # 2 articles × (gate + extract)
        assert resp.token_usage.prompt_tokens == 2 * (100 + 200)
        assert resp.token_usage.completion_tokens == 2 * (10 + 30)
        assert resp.token_usage.total_tokens == 2 * (110 + 230)
        assert resp.token_usage.cache_read_tokens == 2 * 50
        assert resp.token_usage.cache_write_tokens == 2 * 20

    async def test_rejected_articles_still_report_their_spend(self, monkeypatch):
        # A gate-rejected run produced no forecast, but the gate calls were made
        # and paid for — an empty answer must still say what it cost.
        monkeypatch.setattr(forecaster, "check_is_prediction", _gatekeeper(is_prediction=False))
        monkeypatch.setattr(forecaster, "extract_predictions", _extractor())

        req = ForecastRequest(
            question="Token-usage probe 41b — all articles rejected?",
            articles=[_article("http://x.com/tu-3")],
        )
        resp = await forecaster.run_forecast(req)

        assert resp.insufficient_data is True
        assert resp.token_usage is not None
        assert resp.token_usage.total_tokens == 110  # one gate call, no extract

    async def test_no_usage_reported_leaves_the_field_null(self, monkeypatch):
        gk = AsyncMock(return_value=(
            GatekeeperOutput(is_prediction=False, reason="r", relevance_score=0.0), {},
        ))
        monkeypatch.setattr(forecaster, "check_is_prediction", gk)

        req = ForecastRequest(
            question="Token-usage probe 41c — backend reported nothing?",
            articles=[_article("http://x.com/tu-4")],
        )
        resp = await forecaster.run_forecast(req)
        assert resp.token_usage is None


class TestRelevanceTokenUsage:
    BODY = {
        "claim": "The event will happen by 2026-12-31",
        "article_text": "Enough text to judge.",
    }

    def test_relevance_carries_the_gate_calls_usage(self):
        verdict = GatekeeperOutput(is_prediction=True, reason="r", relevance_score=0.8,
                                   prediction_count_estimate=1)
        with patch("forecast_api.main.check_is_prediction",
                   new=AsyncMock(return_value=(verdict, dict(GATE_USAGE)))):
            r = client.post("/relevance", json=self.BODY, headers=HEADERS)
        assert r.status_code == 200
        assert r.json()["token_usage"] == {
            "prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110,
            "cache_read_tokens": 0, "cache_write_tokens": 0,
        }

    def test_relevance_without_usage_is_null(self):
        verdict = GatekeeperOutput(is_prediction=True, reason="r", relevance_score=0.8)
        with patch("forecast_api.main.check_is_prediction",
                   new=AsyncMock(return_value=(verdict, {}))):
            r = client.post("/relevance", json=self.BODY, headers=HEADERS)
        assert r.json()["token_usage"] is None


class TestLlmProxyTokenUsage:
    BODY = {"messages": [{"role": "user", "content": "hi"}]}

    def test_llm_carries_the_calls_usage(self):
        with patch("forecast_api.main.complete_text_once_with_usage",
                   new=AsyncMock(return_value=("hello", dict(EXTRACT_USAGE)))):
            r = client.post("/llm", json=self.BODY, headers=HEADERS)
        assert r.status_code == 200
        body = r.json()
        assert body["content"] == "hello"
        assert body["token_usage"] == {
            "prompt_tokens": 200, "completion_tokens": 30, "total_tokens": 230,
            "cache_read_tokens": 50, "cache_write_tokens": 20,
        }

    def test_llm_without_usage_is_null(self):
        with patch("forecast_api.main.complete_text_once_with_usage",
                   new=AsyncMock(return_value=("hello", {}))):
            r = client.post("/llm", json=self.BODY, headers=HEADERS)
        assert r.json()["token_usage"] is None
