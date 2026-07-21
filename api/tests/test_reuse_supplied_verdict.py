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
