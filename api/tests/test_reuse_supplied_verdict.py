"""Phase 1.1 — the Oracle reuses a caller-supplied gatekeeper verdict instead of
re-judging (kills the double-judge: the SAME judge already ran once in news-indexer's
POST /relevance). Only the relevance call is skipped; the extractor still runs.

Gated behind settings.reuse_supplied_relevance and fail-open: an article without a
verdict, or the flag off, judges exactly as before. The LLM is never called here —
check_is_prediction / extract_predictions are mocked, so these are deterministic.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

from forecast_api import forecaster
from forecast_api.config import settings as api_settings
from tm.models import GatekeeperOutput, PredictionExtraction
from tm.web_search import SearchResult


def _sr(url="http://x.com/1", *, relevance=None, is_prediction=None):
    return SearchResult(
        title="A clear title about the event",
        url=url,
        snippet="A snippet long enough to clear the twenty-char fallback guard.",
        source="x.com",
        published_date="2026-07-15",
        _prefetched_text="Full article body, already fetched so no network call happens here.",
        _supplied_relevance=relevance,
        _supplied_is_prediction=is_prediction,
    )


def _extractor_stub():
    async def _extract(**kwargs):
        # author_lean / author_lean_certainty mirror ExtractionOutput's shadow fields
        # (retro #308/#309) so _process_article can surface them per-source.
        return SimpleNamespace(predictions=[PredictionExtraction(
            quote="q", claim="c", stance=0.6, certainty=0.8, specificity=1.0, settled=None,
        )], author_lean=0.5, author_lean_certainty=0.4), {}
    return _extract


def _gk_spy(is_prediction=True, relevance=0.42):
    return AsyncMock(return_value=(
        GatekeeperOutput(is_prediction=is_prediction, reason="judged", relevance_score=relevance), {},
    ))


async def _process(monkeypatch, result, *, flag, gk, prediction_id=None):
    monkeypatch.setattr(api_settings, "reuse_supplied_relevance", flag)
    monkeypatch.setattr(forecaster, "check_is_prediction", gk)
    monkeypatch.setattr(forecaster, "extract_predictions", _extractor_stub())
    monkeypatch.setattr(forecaster, "enforce_deadline_arithmetic", lambda preds, dl, direction: preds)
    return await forecaster._process_article(
        result, "Will the event happen by 2026-12-31?",
        max_article_chars=4000, timings=[], article_debugs=[],
        prediction_id=prediction_id,
    )


class TestReuseSuppliedVerdict:
    async def test_supplied_verdict_skips_the_gatekeeper(self, monkeypatch):
        gk = _gk_spy()
        out = await _process(monkeypatch, _sr(relevance=0.83, is_prediction=True), flag=True, gk=gk)
        gk.assert_not_awaited()                 # the double-judge is gone
        assert out is not None
        _, relevance, preds, author_lean, author_lean_certainty = out
        assert relevance == 0.83                 # the supplied verdict, not a re-judge
        assert preds                             # extractor still ran
        assert author_lean == 0.5                # the byline author's own forecast, surfaced per-source
        assert author_lean_certainty == 0.4      # (shadow — never enters the estimate)

    async def test_supplied_reject_drops_without_judging(self, monkeypatch):
        gk = _gk_spy()
        out = await _process(monkeypatch, _sr(relevance=0.0, is_prediction=False), flag=True, gk=gk)
        gk.assert_not_awaited()
        assert out is None                       # gate_rejected, same as a real reject

    async def test_flag_off_ignores_the_supplied_verdict(self, monkeypatch):
        gk = _gk_spy(relevance=0.42)
        out = await _process(monkeypatch, _sr(relevance=0.83, is_prediction=True), flag=False, gk=gk)
        gk.assert_awaited_once()                 # re-judges exactly as before
        _, relevance, *_ = out
        assert relevance == 0.42                  # the judge's value, not the supplied 0.83

    async def test_article_without_a_verdict_is_judged(self, monkeypatch):
        gk = _gk_spy(relevance=0.42)
        out = await _process(monkeypatch, _sr(relevance=None, is_prediction=None), flag=True, gk=gk)
        gk.assert_awaited_once()                 # SERP/GDELT articles carry no verdict
        _, relevance, *_ = out
        assert relevance == 0.42

    async def test_partial_verdict_is_not_reused(self, monkeypatch):
        # relevance without is_prediction is not a verdict — judge, don't guess.
        gk = _gk_spy(relevance=0.42)
        out = await _process(monkeypatch, _sr(relevance=0.83, is_prediction=None), flag=True, gk=gk)
        gk.assert_awaited_once()
        _, relevance, *_ = out
        assert relevance == 0.42

    async def test_gate_reused_log_carries_the_caller_prediction_id(self, monkeypatch, caplog):
        # So a future check can join daatan's context_snapshots rows to this log line
        # directly instead of correlating by timestamp.
        import logging
        caplog.set_level(logging.INFO, logger=forecaster.logger.name)
        gk = _gk_spy()
        await _process(
            monkeypatch, _sr(relevance=0.83, is_prediction=True), flag=True, gk=gk,
            prediction_id="cmrm4byby02nw01qtvnoqenc3",
        )
        assert any(
            "outcome=gate_reused" in r.message and "prediction_id=cmrm4byby02nw01qtvnoqenc3" in r.message
            for r in caplog.records
        )


class TestForecastRelevanceBar:
    """The bar itself, tested where it LIVES — inside ``_process_article``. The
    ``test_relevance_gate`` harness mocks ``_process_article_bounded``, one level above,
    so a bar asserted there would pass no matter what this branch did."""

    async def test_off_by_default_a_weak_article_is_still_extracted(self, monkeypatch):
        monkeypatch.setattr(api_settings, "forecast_relevance_bar", 0.0)
        out = await _process(
            monkeypatch, _sr(), flag=False, gk=_gk_spy(is_prediction=True, relevance=0.3),
        )
        assert out is not None
        assert out[1] == 0.3

    async def test_raised_bar_drops_the_article_and_never_calls_the_extractor(self, monkeypatch):
        # Position matters: the gatekeeper call is already paid for by this point, so the
        # only thing dropping here saves is the EXTRACTOR — which is the expensive one.
        # Asserting the extractor is untouched is the difference between a real cost guard
        # and a filter that merely discards work it already paid for.
        extractor = AsyncMock(side_effect=AssertionError("extractor must not run"))
        monkeypatch.setattr(api_settings, "forecast_relevance_bar", 0.7)
        monkeypatch.setattr(api_settings, "reuse_supplied_relevance", False)
        monkeypatch.setattr(forecaster, "check_is_prediction", _gk_spy(is_prediction=True, relevance=0.3))
        monkeypatch.setattr(forecaster, "extract_predictions", extractor)
        debugs: list = []
        timings: list = []

        out = await forecaster._process_article(
            _sr(), "Will the event happen by 2026-12-31?",
            max_article_chars=4000, timings=timings, article_debugs=debugs,
        )

        assert out is None
        extractor.assert_not_awaited()
        assert [d.outcome for d in debugs] == ["low_relevance"]
        assert [t["outcome"] for t in timings] == ["low_relevance"]

    async def test_an_article_at_the_bar_is_kept(self, monkeypatch):
        # `<` not `<=`: the bar is the lowest ACCEPTABLE score, matching news-indexer's
        # `relevance_score >= 0.7`. An off-by-one here would silently discard the single
        # most populated band — 51.9% of live voting rows sit at exactly 0.70.
        monkeypatch.setattr(api_settings, "forecast_relevance_bar", 0.7)
        out = await _process(
            monkeypatch, _sr(), flag=False, gk=_gk_spy(is_prediction=True, relevance=0.7),
        )
        assert out is not None
        assert out[1] == 0.7
