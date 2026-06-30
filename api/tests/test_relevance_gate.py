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

def _preds(stance: float, certainty: float = 0.8):
    return [SimpleNamespace(stance=stance, certainty=certainty, specificity=1.0, claim="c")]


def _wire(monkeypatch, articles, certainty: float = 0.8):
    """Stub search + per-article processing. ``articles`` is a list of
    ``(url, relevance, stance)``; each yields one prediction with that stance
    and the given certainty (drives the evidence-mass / decisiveness floor)."""
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
        return (result, rel, _preds(stance, certainty))
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
        # This tiny set would also trip the decisiveness floor; lower it too so the
        # test isolates the relevance floor.
        monkeypatch.setattr(api_settings, "decisiveness_floor", 0.001)
        resp = await forecaster.run_forecast(ForecastRequest(question="floor toggle probe 9a?"))

        assert resp.insufficient_data is False
        assert resp.articles_used == 2
        assert resp.outcome_counts.get("low_relevance") == 2


class TestDecisivenessFloor:
    async def test_thin_low_certainty_pool_defers(self, monkeypatch):
        # On-subject articles (relevance 0.6, so NOT off-topic) but modest certainty
        # 0.3 (above the per-source certainty gate) → evidence mass =
        # 2 × (1.0·0.3·1.0·0.6²) = 0.216 < 0.5 floor. The Oracle should defer
        # (insufficient_data) instead of emitting a coin-flip, so the caller keeps
        # its base rate. (The "Elon will tweet about X" case.)
        _wire(monkeypatch, [
            ("http://a.com/1", 0.6, 0.4),
            ("http://b.com/2", 0.6, -0.4),
        ], certainty=0.3)
        resp = await forecaster.run_forecast(ForecastRequest(question="decisiveness probe e1?"))

        assert resp.insufficient_data is True
        assert resp.reason == "no_decisive_signal"
        assert resp.articles_used == 0
        assert resp.outcome_counts.get("low_evidence_mass") == 2

    async def test_strong_pool_is_not_blocked(self, monkeypatch):
        # Same articles at full certainty clear the floor and produce a forecast.
        _wire(monkeypatch, [
            ("http://a.com/1", 0.9, 0.4),
            ("http://b.com/2", 0.9, 0.5),
        ], certainty=0.9)
        resp = await forecaster.run_forecast(ForecastRequest(question="decisiveness probe e2?"))

        assert resp.insufficient_data is False
        assert resp.articles_used == 2


class TestCertaintyGate:
    async def test_all_hedged_sources_dropped_defers(self, monkeypatch):
        # On-topic (relevance 0.9, clears the relevance floor) but every claim is
        # hedged speculation (certainty 0.1 < 0.2 gate). Each source is dropped
        # before aggregation, so the pool is empty and the Oracle defers rather
        # than pooling junk into a confident-looking estimate. This is the Fable
        # case: a search match on a common word yields tangential "could/implies"
        # claims that should not produce a number.
        _wire(monkeypatch, [
            ("http://a.com/1", 0.9, 0.6),
            ("http://b.com/2", 0.9, 0.4),
        ], certainty=0.1)
        resp = await forecaster.run_forecast(ForecastRequest(question="certainty gate probe a1?"))

        assert resp.insufficient_data is True
        assert resp.reason == "all_low_certainty"
        assert resp.articles_used == 0
        assert resp.outcome_counts.get("low_certainty") == 2

    async def test_gate_disabled_when_floor_zero(self, monkeypatch):
        # certainty_floor=0 disables the gate; the hedged sources survive (and then
        # the decisiveness floor governs, lowered here to isolate the gate).
        _wire(monkeypatch, [
            ("http://a.com/1", 0.9, 0.6),
            ("http://b.com/2", 0.9, 0.4),
        ], certainty=0.1)
        monkeypatch.setattr(api_settings, "certainty_floor", 0.0)
        monkeypatch.setattr(api_settings, "decisiveness_floor", 0.001)
        resp = await forecaster.run_forecast(ForecastRequest(question="certainty gate probe a2?"))

        assert resp.insufficient_data is False
        assert resp.articles_used == 2

    async def test_confident_source_kept_hedged_dropped(self, monkeypatch):
        # Mixed pool: one confident on-topic source survives, one hedged source is
        # dropped, so a single decisive article still yields a forecast.
        results = [
            SearchResult(title="t", url=u, snippet="s", source=u)
            for u in ("http://keep.com/1", "http://drop.com/2")
        ]
        monkeypatch.setattr(forecaster, "search_articles", lambda q, limit: list(results))

        async def _no_distill(question):
            return question
        monkeypatch.setattr(forecaster, "_distill_query", _no_distill)
        monkeypatch.setattr(forecaster, "get_credibility_weight", lambda sid: 1.0)

        per_url = {"http://keep.com/1": 0.9, "http://drop.com/2": 0.1}

        async def _bounded(result, question, *, max_article_chars, timings, article_debugs, timeout_s):
            timings.append({"url": result.url, "outcome": "ok"})
            return (result, 0.9, _preds(0.8, per_url[result.url]))
        monkeypatch.setattr(forecaster, "_process_article_bounded", _bounded)

        resp = await forecaster.run_forecast(ForecastRequest(question="certainty gate probe a3?"))

        assert resp.insufficient_data is False
        assert resp.articles_used == 1
        assert [s.url for s in resp.sources] == ["http://keep.com/1"]
        assert resp.outcome_counts.get("low_certainty") == 1


def test_pipeline_settings_unaffected():
    # Guard: the gatekeeper/extractor model fields still live on pipeline settings.
    assert pipeline_settings.gatekeeper_model
    assert pipeline_settings.extractor_model
