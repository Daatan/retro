"""Retrieval relevance gate: the gatekeeper emits a graded ``relevance_score``
that down-weights off-topic articles convexly (relevance²) in the pool, and an
all-off-topic set short-circuits to ``insufficient_data`` instead of pooling junk.

The LLM is never called here — ``_process_article_bounded`` is mocked to return
controlled ``(result, relevance, predictions)`` tuples, so these assertions are
fully deterministic.
"""

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from forecast_api import forecaster
from forecast_api.config import settings as api_settings
from forecast_api.models import ForecastRequest
from tm.config import settings as pipeline_settings
from tm.models import GatekeeperOutput
from tm.web_search import SearchResult


# --- GatekeeperOutput.relevance_score ---------------------------------------

class TestGatekeeperRelevanceScore:
    def test_defaults_to_neutral_when_omitted(self):
        g = GatekeeperOutput(is_prediction=True, reason="r")
        assert g.relevance_score == 1.0

    def test_accepts_a_graded_value(self):
        g = GatekeeperOutput(is_prediction=True, reason="r", relevance_score=0.3)
        assert g.relevance_score == 0.3

    def test_rejects_out_of_range(self):
        with pytest.raises(ValidationError):
            GatekeeperOutput(is_prediction=True, reason="r", relevance_score=1.5)
        with pytest.raises(ValidationError):
            GatekeeperOutput(is_prediction=True, reason="r", relevance_score=-0.1)


# --- run_forecast wiring ----------------------------------------------------

def _preds(stance: float):
    return [SimpleNamespace(stance=stance, certainty=0.8, specificity=1.0, claim="c")]


def _wire(monkeypatch, articles):
    """Stub search + per-article processing. ``articles`` is a list of
    ``(url, relevance, stance)``; each yields one prediction with that stance."""
    results = [SearchResult(title="t", url=url, snippet="s", source=url) for url, _, _ in articles]
    monkeypatch.setattr(forecaster, "search_articles", lambda q, limit: list(results))

    async def _no_distill(question):
        return question
    monkeypatch.setattr(forecaster, "_distill_query", _no_distill)
    # Neutralise credibility so the test isolates the relevance term.
    monkeypatch.setattr(forecaster, "get_credibility_weight", lambda sid: 1.0)

    by_url = {url: (rel, stance) for url, rel, stance in articles}

    async def _bounded(result, question, *, max_article_chars, timings, article_debugs, timeout_s):
        rel, stance = by_url[result.url]
        timings.append({"url": result.url, "outcome": "ok"})
        return (result, rel, _preds(stance))
    monkeypatch.setattr(forecaster, "_process_article_bounded", _bounded)


class TestAllOffTopicShortCircuits:
    async def test_returns_insufficient_data(self, monkeypatch):
        # Two articles, both near-zero relevance → Σ relevance² = 0.02 < 0.05 floor.
        _wire(monkeypatch, [
            ("http://a.com/1", 0.1, 0.9),
            ("http://b.com/2", 0.1, -0.9),
        ])
        resp = await forecaster.run_forecast(ForecastRequest(question="off-topic probe 7c1?"))

        assert resp.insufficient_data is True
        assert resp.reason == "all_articles_off_topic"
        assert resp.articles_used == 0
        assert resp.outcome_counts.get("all_low_relevance") == 2


class TestConvexRelevanceWeighting:
    async def test_on_topic_source_dominates_off_topic(self, monkeypatch):
        # A high-relevance "+0.9" source and a tangential "-0.9" source. With
        # relevance² weighting (0.81 vs 0.16) the on-topic stance dominates, so
        # the pooled mean is clearly positive rather than ~0.
        _wire(monkeypatch, [
            ("http://news.com/on", 0.9, 0.9),
            ("http://blog.com/off", 0.4, -0.9),
        ])
        resp = await forecaster.run_forecast(ForecastRequest(question="convex weight probe 4d2?"))

        assert resp.insufficient_data is False
        assert resp.articles_used == 2
        assert resp.mean > 0.5  # on-topic +0.9 wins; without weighting this would be ~0
        # Per-source relevance is surfaced for the admin dashboard.
        by_url = {s.url: s for s in resp.sources}
        assert by_url["http://news.com/on"].relevance_score == 0.9
        assert by_url["http://blog.com/off"].relevance_score == 0.4

    async def test_floor_is_configurable(self, monkeypatch):
        # Same low-relevance set passes when the floor is lowered below the mass.
        _wire(monkeypatch, [("http://a.com/1", 0.1, 0.9), ("http://b.com/2", 0.1, 0.9)])
        monkeypatch.setattr(api_settings, "relevance_weight_floor", 0.001)
        resp = await forecaster.run_forecast(ForecastRequest(question="floor toggle probe 9a?"))

        assert resp.insufficient_data is False
        assert resp.articles_used == 2
        assert resp.outcome_counts.get("low_relevance") == 2


def test_pipeline_settings_unaffected():
    # Guard: the gatekeeper/extractor model fields still live on pipeline settings.
    assert pipeline_settings.gatekeeper_model
    assert pipeline_settings.extractor_model
