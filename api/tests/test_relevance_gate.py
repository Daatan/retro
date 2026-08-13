"""Retrieval relevance gate: the gatekeeper emits a graded ``relevance_score``
that down-weights off-topic articles convexly (relevance²) in the pool, and an
all-off-topic set short-circuits to ``insufficient_data`` instead of pooling junk.

The LLM is never called here — ``_process_article_bounded`` is mocked to return
controlled ``(result, relevance, predictions, author_lean, author_lean_certainty)``
tuples, so these assertions are
fully deterministic.
"""

from datetime import datetime

import pytest
from pydantic import ValidationError

from forecast_api import forecaster
from forecast_api.config import settings as api_settings
from forecast_api.models import ForecastRequest
from tm import web_search
from tm.config import settings as pipeline_settings
from tm.models import GatekeeperOutput, PredictionExtraction
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

    def test_unscored_rejection_defaults_relevance_to_zero(self):
        # A rejected article that the model left unscored must never read as
        # "relevant" to a caller that only checks relevance_score (e.g. the
        # /relevance endpoint response) without also checking is_prediction —
        # the 1.0 "neutral" default is for the PASSING case only.
        g = GatekeeperOutput(is_prediction=False, reason="off-topic")
        assert g.relevance_score == 0.0

    def test_rejection_preserves_an_explicit_graded_score(self):
        # A graded near-miss on a rejection (e.g. 0.1) is meaningful — a rescue
        # path that can't hear a low-but-nonzero "no" would push everything it
        # looked at. Only the OMITTED case gets the is_prediction-aware default.
        g = GatekeeperOutput(is_prediction=False, reason="off-topic", relevance_score=0.1)
        assert g.relevance_score == 0.1


# --- run_forecast wiring ----------------------------------------------------

def _preds(stance: float, certainty: float = 0.8, evidence_class=None):
    return [PredictionExtraction(
        quote="q", claim="c", stance=stance, certainty=certainty, specificity=1.0, settled=None,
        evidence_class=evidence_class,
    )]


def _wire(monkeypatch, articles, certainty: float = 0.8, evidence_class=None):
    """Stub search + per-article processing. ``articles`` is a list of
    ``(url, relevance, stance)``; each yields one prediction with that stance
    and the given certainty. Evidence mass (hence the decisiveness floor) comes
    from ``evidence_class`` when set, and otherwise from certainty capped at
    ``evidence_class_weight_unclassified_cap`` (R3/F10)."""
    # Dated "today": recency is a neutral 1.0 for every article, so these tests
    # isolate the relevance and certainty/class terms. (Leaving the date empty
    # would now decay every article to recency_floor — R3/F13.)
    today = datetime.now().strftime("%Y-%m-%d")
    results = [
        SearchResult(title="t", url=url, snippet="s", source=url, published_date=today)
        for url, _, _ in articles
    ]
    monkeypatch.setattr(web_search, "search_articles", lambda q, limit, date_from=None, date_to=None: list(results))

    async def _no_distill(question):
        return question, {}
    monkeypatch.setattr(forecaster, "_distill_query", _no_distill)
    # Neutralise credibility so the test isolates the relevance term.
    monkeypatch.setattr(forecaster, "get_credibility_weight", lambda sid: 1.0)

    by_url = {url: (rel, stance) for url, rel, stance in articles}

    async def _bounded(
        result, question, *, max_article_chars, timings, article_debugs, timeout_s,
        claim_deadline=None, claim_direction=None, prediction_id=None,
        resolution_criteria=None, usage_events=None,
    ):
        rel, stance = by_url[result.url]
        timings.append({"url": result.url, "outcome": "ok"})
        return (result, rel, _preds(stance, certainty, evidence_class), None, None)
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


class TestThinEvidenceWidensCI:
    async def test_thin_pool_widens_ci_instead_of_deferring(self, monkeypatch):
        # On-subject articles (relevance 0.6, so NOT off-topic) but modest certainty
        # 0.3 → evidence mass = 2 × (1.0·0.3·1.0·0.6²) = 0.216 < 0.5 floor. The pool
        # is thin, but it IS on-topic, so the Oracle no longer abstains — it emits an
        # estimate with a CI widened toward maximal uncertainty (a wide band reads as
        # "low confidence" honestly, instead of a deceptively tight one or a "no AI
        # estimate" hole). This is the Mythos case.
        _wire(monkeypatch, [
            ("http://a.com/1", 0.6, 0.4),
            ("http://b.com/2", 0.6, -0.4),
        ], certainty=0.3)
        resp = await forecaster.run_forecast(ForecastRequest(question="thin probe e1?"))

        assert resp.insufficient_data is False
        assert resp.articles_used == 2
        assert resp.outcome_counts.get("thin_evidence") == 2
        # Thin evidence ⇒ a wide CI (here the inflation drives it to (near) full span).
        assert (resp.ci_high - resp.ci_low) > 1.5

    async def test_hedged_on_topic_pool_is_kept_not_dropped(self, monkeypatch):
        # Every claim is hedged speculation (certainty 0.1) but on-topic (relevance
        # 0.9). Previously the certainty gate dropped them all and forced an
        # abstention; now they are down-weighted but kept, so an on-topic pool yields
        # a (very) wide-CI estimate rather than "insufficient data".
        _wire(monkeypatch, [
            ("http://a.com/1", 0.9, 0.6),
            ("http://b.com/2", 0.9, 0.4),
        ], certainty=0.1)
        resp = await forecaster.run_forecast(ForecastRequest(question="hedged probe a1?"))

        assert resp.insufficient_data is False
        assert resp.articles_used == 2
        # Wide band (the mean sits high here, so the upper bound clamps at 1.0; the
        # widening still roughly quadruples the raw pooled width).
        assert (resp.ci_high - resp.ci_low) > 1.0

    async def test_strong_pool_keeps_tight_ci(self, monkeypatch):
        # Classified reported_fact (weight 1.0) + high relevance clears the floor:
        # no widening, a forecast with a normal (tight) CI. The class is what makes
        # the pool strong post-S2-cutover — an equally confident UNCLASSIFIED pool is
        # capped at 0.25 per claim (R3/F10) and lands under the floor, which is the
        # intended reading of "we do not know what kind of evidence this is".
        _wire(monkeypatch, [
            ("http://a.com/1", 0.9, 0.4),
            ("http://b.com/2", 0.9, 0.5),
        ], certainty=0.9, evidence_class="reported_fact")
        resp = await forecaster.run_forecast(ForecastRequest(question="strong probe e2?"))

        assert resp.insufficient_data is False
        assert resp.articles_used == 2
        assert resp.outcome_counts.get("thin_evidence") is None
        assert (resp.ci_high - resp.ci_low) < 1.0

    async def test_defer_flag_restores_abstention(self, monkeypatch):
        # The escape hatch: with defer_on_thin_evidence set, a thin pool abstains
        # exactly as before (insufficient_data, reason=no_decisive_signal).
        _wire(monkeypatch, [
            ("http://a.com/1", 0.6, 0.4),
            ("http://b.com/2", 0.6, -0.4),
        ], certainty=0.3)
        monkeypatch.setattr(api_settings, "defer_on_thin_evidence", True)
        resp = await forecaster.run_forecast(ForecastRequest(question="defer probe e3?"))

        assert resp.insufficient_data is True
        assert resp.reason == "no_decisive_signal"
        assert resp.articles_used == 0
        assert resp.outcome_counts.get("low_evidence_mass") == 2

    async def test_inflation_disabled_keeps_pool_ci(self, monkeypatch):
        # thin_evidence_ci_inflation=0 turns off widening: a thin pool still produces
        # an estimate (no abstention) but with the raw pooled CI.
        _wire(monkeypatch, [
            ("http://a.com/1", 0.6, 0.4),
            ("http://b.com/2", 0.6, -0.4),
        ], certainty=0.3)
        monkeypatch.setattr(api_settings, "thin_evidence_ci_inflation", 0.0)
        resp = await forecaster.run_forecast(ForecastRequest(question="no-inflate probe e4?"))

        assert resp.insufficient_data is False
        assert resp.articles_used == 2
        assert (resp.ci_high - resp.ci_low) < 1.5


def test_pipeline_settings_unaffected():
    # Guard: the gatekeeper/extractor model fields still live on pipeline settings.
    assert pipeline_settings.gatekeeper_model
    assert pipeline_settings.extractor_model


class TestForecastRelevanceBarIsInert:
    """retro#393, the /forecast half. The 0.7 bar was an ENTRY-PATH property, not a verdict
    property: news-indexer's rescue path retired an article the same judge scored 0.30 while
    /forecast let it vote. The bar now exists on this side too — shipped at 0.0, i.e. off."""

    async def test_a_low_relevance_article_still_votes_at_the_shipped_default(self, monkeypatch):
        # The default MUST reproduce today's behaviour exactly. 20.4% of live voting rows sit
        # below 0.7 and the backtest that would justify cutting them is not powered (only 6
        # resolved BINARY forecasts have a usable pool, retro#393), so shipping a real bar
        # would be an unmeasured change to every forecast.
        assert api_settings.forecast_relevance_bar == 0.0
        _wire(monkeypatch, [
            ("http://news.com/on", 0.9, 0.9),
            ("http://blog.com/weak", 0.3, 0.9),
        ])
        resp = await forecaster.run_forecast(ForecastRequest(question="inert bar probe 1f?"))

        assert resp.articles_used == 2  # the 0.3 article still votes
        assert resp.relevance_bar == 0.0

    async def test_the_effective_bar_is_reported_so_a_caller_can_record_the_regime(self, monkeypatch):
        # This is the whole of option (b): a caller persisting these sources can record WHICH
        # admission regime produced each row and filter the pool retroactively, instead of
        # re-deriving which entry path admitted what.
        monkeypatch.setattr(api_settings, "forecast_relevance_bar", 0.5)
        _wire(monkeypatch, [("http://news.com/on", 0.9, 0.9)])
        resp = await forecaster.run_forecast(ForecastRequest(question="regime probe 3f?"))

        assert resp.relevance_bar == 0.5
