"""Phase 1.1 — the Oracul reuses a caller-supplied gatekeeper verdict instead of
re-judging (kills the double-judge: the SAME judge already ran once in news-indexer's
POST /relevance). Only the relevance call is skipped; the extractor still runs.

Gated behind settings.reuse_supplied_relevance and fail-open: an article without a
verdict, or the flag off, judges exactly as before. The LLM is never called here —
check_is_prediction / extract_predictions are mocked, so these are deterministic.

``TestSuppliedVerdictAllowlist`` (retro#536) covers the second gate: WHOSE verdict may
be reused. The flag above only ever said "reuse is on", so any holder of any valid API
key could skip claim-aware judging for its own requests by setting relevance/is_prediction
on the request body. Those tests run the real ``run_forecast`` so the strip happens where
production strips it — at the API trust boundary, not inside ``_process_article``.
"""

import itertools
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from forecast_api import forecaster
from forecast_api.auth import ApiKeyClient
from forecast_api.config import settings as api_settings
from forecast_api.models import ArticleInput, ForecastRequest
from tm.models import ExtractionOutput, GatekeeperOutput, PredictionExtraction
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
        # (retro #308/#309) so _process_article can surface them per-source;
        # consensus_view (retro#686) is the third of them and rides the same outcome,
        # and claim_actor/claim_predicate/claim_scope (retro#697) the next three.
        return SimpleNamespace(predictions=[PredictionExtraction(
            quote="q", claim="c", stance=0.6, certainty=0.8, specificity=1.0, settled=None,
        )], author_lean=0.5, author_lean_certainty=0.4, consensus_view="divided",
            claim_actor=None, claim_predicate=None, claim_scope=None), {}
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
        # By name, not by position: this unpack read six of the outcome's fields
        # positionally until retro#697 made it nine, at which point it broke without
        # anything about the assertions below having changed.
        assert out.relevance == 0.83             # the supplied verdict, not a re-judge
        assert out.predictions                   # extractor still ran
        assert out.author_lean == 0.5            # the byline author's own forecast, surfaced per-source
        assert out.author_lean_certainty == 0.4  # (shadow — never enters the estimate)
        assert out.consensus_view == "divided"   # what the article says OTHERS expect (retro#686)

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


_QUESTION_SEQ = itertools.count()

_ALLOWLIST_BODY = (
    "Fixture article body for the caller-allowlist suite. The gatekeeper and extractor "
    "are stubbed, so no model ever reads this text; it exists to clear the pipeline's "
    "non-empty-body checks. "
) * 3


class TestSuppliedVerdictAllowlist:
    """retro#536 — ``reuse_supplied_relevance`` said reuse is on but never said whose
    verdict may be reused, so ANY authenticated key could hand itself a gatekeeper pass.

    These go through ``run_forecast`` (not ``_process_article``) on purpose: the strip
    lives at the trust boundary, where the request first meets the caller's identity, and
    a test one level below it would pass no matter which client made the call.
    """

    @pytest.fixture
    def gk(self, monkeypatch):
        """Gatekeeper spy + the minimum stubs to run the real pipeline over one article."""
        spy = AsyncMock(return_value=(
            GatekeeperOutput(is_prediction=True, reason="judged", relevance_score=0.42), {},
        ))

        async def _extract(**kwargs):
            return ExtractionOutput(predictions=[PredictionExtraction(
                quote="q", claim="c", stance=0.6, certainty=0.8, specificity=1.0, settled=None,
            )]), {}

        monkeypatch.setattr(api_settings, "reuse_supplied_relevance", True)
        monkeypatch.setattr(forecaster, "check_is_prediction", spy)
        monkeypatch.setattr(forecaster, "extract_predictions", _extract)
        monkeypatch.setattr(forecaster, "get_credibility_weight", lambda sid: 1.0)
        return spy

    async def _forecast(self, *, client, relevance=0.83, is_prediction=True):
        # Unique question per call: run_forecast's response cache is keyed on it, and a
        # hit would return another test's answer without ever reaching the allowlist.
        resp = await forecaster.run_forecast(
            ForecastRequest(
                question=f"[536-allowlist-{next(_QUESTION_SEQ)}] Will the event occur?",
                articles=[ArticleInput(
                    url="https://fixture.example.test/536",
                    title="Vellum sonata dispatch",
                    snippet="Fixture snippet, long enough to clear the fallback guard.",
                    source="fixture",
                    published_date="2026-07-28",
                    text=_ALLOWLIST_BODY,
                    relevance=relevance,
                    is_prediction=is_prediction,
                )],
            ),
            client=client,
        )
        assert resp.sources, "the fixture article should always produce a source signal"
        return resp.sources[0]

    async def test_allowlisted_client_verdict_is_still_reused(self, gk):
        # Regression: the daatan backend holds the primary key and threads news-indexer's
        # POST /relevance verdict. Breaking this would restore the double-judge #536's
        # parent change removed.
        source = await self._forecast(client=ApiKeyClient(name="default"))
        gk.assert_not_awaited()
        assert source.relevance_score == 0.83      # the supplied verdict, not a re-judge

    async def test_non_allowlisted_client_verdict_is_ignored_and_judged(self, gk):
        # The defect: a named key (staging, or any future third-party key) could set
        # relevance/is_prediction and skip claim-aware judging entirely.
        source = await self._forecast(client=ApiKeyClient(name="staging", max_articles=3))
        gk.assert_awaited_once()                   # the real judge ran
        assert source.relevance_score == 0.42      # the judge's value, not the supplied 0.83

    async def test_non_allowlisted_reject_verdict_cannot_drop_an_article(self, gk):
        # The mirror abuse: is_prediction=False would silently delete an article from
        # someone else's evidence pool without a single LLM call.
        source = await self._forecast(
            client=ApiKeyClient(name="staging"), relevance=0.0, is_prediction=False,
        )
        gk.assert_awaited_once()
        assert source.relevance_score == 0.42

    async def test_clientless_call_still_reuses(self, gk):
        # The MCP lane and the pipeline's in-process callers carry no API key; they are
        # treated as the primary client, exactly as the per-key cap treats them.
        source = await self._forecast(client=None)
        gk.assert_not_awaited()
        assert source.relevance_score == 0.83

    async def test_allowlist_is_configurable(self, gk, monkeypatch):
        # Provisioning a dedicated news-indexer key must not need a code change.
        monkeypatch.setattr(api_settings, "relevance_reuse_allowed_clients", "default,news-indexer")
        source = await self._forecast(client=ApiKeyClient(name="news-indexer"))
        gk.assert_not_awaited()
        assert source.relevance_score == 0.83

    async def test_empty_allowlist_disables_reuse_for_everyone(self, gk, monkeypatch):
        monkeypatch.setattr(api_settings, "relevance_reuse_allowed_clients", "")
        source = await self._forecast(client=ApiKeyClient(name="default"))
        gk.assert_awaited_once()
        assert source.relevance_score == 0.42

    async def test_dropping_a_verdict_is_logged_with_the_client_name(self, gk, caplog):
        # An un-allowlisted caller degrades silently by design (no 4xx), so the log line
        # is the only way to notice a client trying to self-certify.
        import logging
        caplog.set_level(logging.WARNING, logger=forecaster.logger.name)
        await self._forecast(client=ApiKeyClient(name="staging"))
        assert any(
            "event=supplied_verdict_dropped" in r.message and "client=staging" in r.message
            for r in caplog.records
        )

    async def test_an_article_without_a_verdict_is_untouched(self, gk, caplog):
        # Nothing to drop => no warning, and the article judges exactly as it always did.
        import logging
        caplog.set_level(logging.WARNING, logger=forecaster.logger.name)
        source = await self._forecast(
            client=ApiKeyClient(name="staging"), relevance=None, is_prediction=None,
        )
        gk.assert_awaited_once()
        assert source.relevance_score == 0.42
        assert not any("event=supplied_verdict_dropped" in r.message for r in caplog.records)


class TestMaySupplyVerdict:
    """The predicate itself — the allowlist is a set membership test on the resolved
    ``ApiKeyClient.name``, with ``None`` (in-process caller) reading as the primary key."""

    def test_default_config_allows_only_the_primary_key(self):
        assert forecaster._may_supply_verdict(ApiKeyClient(name="default")) is True
        assert forecaster._may_supply_verdict(None) is True
        assert forecaster._may_supply_verdict(ApiKeyClient(name="staging")) is False

    def test_whitespace_in_the_config_is_tolerated(self, monkeypatch):
        monkeypatch.setattr(api_settings, "relevance_reuse_allowed_clients", " default , ni ")
        assert forecaster._may_supply_verdict(ApiKeyClient(name="ni")) is True
        assert forecaster._may_supply_verdict(ApiKeyClient(name="staging")) is False
