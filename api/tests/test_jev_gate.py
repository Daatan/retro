"""retro#850 — the Jev skip-gate inside _process_article.

Pass 1 is faked at the `fire_jev_pass1` seam (the HTTP layer has its own tests in
test_jev_shadow.py); what these tests pin is the wiring: shadow never touches the
extractor and logs its verdict next to Haiku's count, enforce skips the extractor only
on a real below-threshold verdict, and every failure path — Jev down, timeout, flag
off — extracts exactly as before.
"""

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from forecast_api import forecaster
from forecast_api.config import settings as api_settings
from tm.models import GatekeeperOutput, PredictionExtraction
from tm.web_search import SearchResult

QUESTION = "Will the event happen by 2026-12-31?"


def _sr(url="http://x.com/1"):
    return SearchResult(
        title="A clear title about the event", url=url,
        snippet="A snippet long enough to clear the twenty-char fallback guard.",
        source="x.com", published_date="2026-07-15",
        _prefetched_text="Analysts expect the event to happen in October. The weather was sunny.",
    )


def _extractor_spy(n_preds=1):
    async def _extract(**kwargs):
        preds = [PredictionExtraction(quote="q", claim="c", stance=0.6, certainty=0.8,
                                      specificity=1.0, settled=None) for _ in range(n_preds)]
        return SimpleNamespace(predictions=preds, author_lean=None, author_lean_certainty=None,
                               consensus_view=None, claim_actor=None, claim_predicate=None,
                               claim_scope=None), {}
    return AsyncMock(side_effect=_extract)


def _fake_pass1(result, *, delay=0.0):
    """A fire_jev_pass1 stand-in: schedules a task that resolves to `result`."""
    seen: list = []

    def fire(**kwargs):
        seen.append(kwargs)

        async def go():
            if delay:
                await asyncio.sleep(delay)
            return result
        return asyncio.get_running_loop().create_task(go())
    fire.seen = seen
    return fire


async def _process(monkeypatch, *, enabled, enforce, pass1, extractor, threshold=0.15,
                   timeout=8.0, timings=None, debugs=None):
    monkeypatch.setattr(api_settings, "jev_gate_enabled", enabled)
    monkeypatch.setattr(api_settings, "jev_gate_enforce", enforce)
    monkeypatch.setattr(api_settings, "jev_gate_threshold", threshold)
    monkeypatch.setattr(api_settings, "jev_gate_timeout_seconds", timeout)
    monkeypatch.setattr(api_settings, "jev_shadow_enabled", False)
    monkeypatch.setattr(forecaster, "fire_jev_pass1", pass1)
    monkeypatch.setattr(forecaster, "check_is_prediction", AsyncMock(return_value=(
        GatekeeperOutput(is_prediction=True, reason="judged", relevance_score=0.9), {})))
    monkeypatch.setattr(forecaster, "extract_predictions", extractor)
    monkeypatch.setattr(forecaster, "enforce_deadline_arithmetic", lambda preds, dl, direction: preds)
    return await forecaster._process_article(
        _sr(), QUESTION, max_article_chars=4000,
        timings=[] if timings is None else timings,
        article_debugs=[] if debugs is None else debugs,
        prediction_id="pid-1",
    )


LOW = {"nouls": [0.05, 0.02], "max_noul": 0.05, "sentences": ["a", "b"], "ms": 40, "neg": 0.03, "tok_in": 50}
HIGH = {"nouls": [0.9, 0.02], "max_noul": 0.9, "sentences": ["a", "b"], "ms": 40, "neg": 0.03, "tok_in": 50}


def _gate_line(caplog):
    return next(r.getMessage() for r in caplog.records if "event=jev_gate " in r.getMessage())


class TestJevGate:
    async def test_flag_off_never_fires_pass1(self, monkeypatch, caplog):
        fire = _fake_pass1(LOW)
        ex = _extractor_spy()
        out = await _process(monkeypatch, enabled=False, enforce=False, pass1=fire, extractor=ex)
        assert out is not None and fire.seen == []
        ex.assert_awaited_once()
        assert not any("event=jev_gate" in r.getMessage() for r in caplog.records)

    async def test_shadow_extracts_and_logs_would_skip(self, monkeypatch, caplog):
        fire = _fake_pass1(LOW)
        ex = _extractor_spy(n_preds=0)
        with caplog.at_level(logging.INFO, logger="forecast_api.forecaster"):
            out = await _process(monkeypatch, enabled=True, enforce=False, pass1=fire, extractor=ex)
        ex.assert_awaited_once()                     # shadow: Haiku still runs
        assert out is None                           # no_predictions, as before
        line = _gate_line(caplog)
        assert "would_skip=True enforce=False max_noul=0.050 threshold=0.15 n_preds=0" in line
        assert "status=ok script=latin" in line and "prediction_id=pid-1" in line
        assert fire.seen[0]["question"] == QUESTION and fire.seen[0]["timeout_s"] == 8.0

    async def test_shadow_logs_keep_next_to_haiku_count(self, monkeypatch, caplog):
        fire = _fake_pass1(HIGH)
        with caplog.at_level(logging.INFO, logger="forecast_api.forecaster"):
            out = await _process(monkeypatch, enabled=True, enforce=False, pass1=fire, extractor=_extractor_spy(2))
        assert out is not None and out.predictions
        assert "would_skip=False enforce=False max_noul=0.900 threshold=0.15 n_preds=2" in _gate_line(caplog)

    async def test_enforce_skips_the_extractor_below_threshold(self, monkeypatch, caplog):
        fire = _fake_pass1(LOW)
        ex = _extractor_spy()
        timings, debugs = [], []
        with caplog.at_level(logging.INFO, logger="forecast_api.forecaster"):
            out = await _process(monkeypatch, enabled=True, enforce=True, pass1=fire, extractor=ex,
                                 timings=timings, debugs=debugs)
        ex.assert_not_awaited()                      # the whole point: no Haiku call
        assert out is None
        assert timings[-1]["outcome"] == "jev_gated" and timings[-1]["jev_ms"] == 40
        assert debugs[-1].outcome == "jev_gated" and debugs[-1].gate_passed is True
        assert "would_skip=True enforce=True max_noul=0.050 threshold=0.15 n_preds=skipped" in _gate_line(caplog)
        assert any("event=article_outcome outcome=jev_gated" in r.getMessage() for r in caplog.records)

    async def test_enforce_extracts_above_threshold(self, monkeypatch, caplog):
        ex = _extractor_spy()
        with caplog.at_level(logging.INFO, logger="forecast_api.forecaster"):
            out = await _process(monkeypatch, enabled=True, enforce=True, pass1=_fake_pass1(HIGH), extractor=ex)
        ex.assert_awaited_once()
        assert out is not None
        assert "would_skip=False enforce=True max_noul=0.900" in _gate_line(caplog)

    async def test_enforce_fails_open_when_jev_errors(self, monkeypatch, caplog):
        ex = _extractor_spy()
        with caplog.at_level(logging.INFO, logger="forecast_api.forecaster"):
            out = await _process(monkeypatch, enabled=True, enforce=True,
                                 pass1=_fake_pass1({"err": "HTTPStatusError(500)", "sentences": ["a"], "ms": 12}),
                                 extractor=ex)
        ex.assert_awaited_once()
        assert out is not None
        assert "would_skip=False enforce=True max_noul=none" in _gate_line(caplog)
        assert "status=HTTPStatusError(500)" in _gate_line(caplog)

    async def test_enforce_fails_open_on_timeout(self, monkeypatch, caplog):
        ex = _extractor_spy()
        with caplog.at_level(logging.INFO, logger="forecast_api.forecaster"):
            out = await _process(monkeypatch, enabled=True, enforce=True, timeout=0.01,
                                 pass1=_fake_pass1(LOW, delay=0.5), extractor=ex)
        ex.assert_awaited_once()                     # a slow Jev never blocks extraction
        assert out is not None
        assert "status=gate_timeout" in _gate_line(caplog)

    async def test_enforce_fails_open_without_a_key(self, monkeypatch):
        ex = _extractor_spy()
        out = await _process(monkeypatch, enabled=True, enforce=True,
                             pass1=_fake_pass1({"skip": "no_key", "sentences": ["a"], "ms": 0}), extractor=ex)
        ex.assert_awaited_once()
        assert out is not None

    async def test_threshold_is_configurable(self, monkeypatch):
        ex = _extractor_spy()
        out = await _process(monkeypatch, enabled=True, enforce=True, threshold=0.95,
                             pass1=_fake_pass1(HIGH), extractor=ex)
        ex.assert_not_awaited()                      # 0.9 < 0.95 → skipped
        assert out is None

    async def test_shadow_passes_pass1_into_the_jev_shadow(self, monkeypatch):
        # Both flags on: the shadow must reuse pass 1, not re-ask the selection questions.
        captured = {}
        monkeypatch.setattr(forecaster, "fire_jev_shadow", lambda **kw: captured.update(kw))
        monkeypatch.setattr(api_settings, "jev_shadow_enabled", True)
        monkeypatch.setattr(api_settings, "jev_gate_enabled", True)
        monkeypatch.setattr(api_settings, "jev_gate_enforce", False)
        monkeypatch.setattr(forecaster, "fire_jev_pass1", _fake_pass1(HIGH))
        monkeypatch.setattr(forecaster, "check_is_prediction", AsyncMock(return_value=(
            GatekeeperOutput(is_prediction=True, reason="judged", relevance_score=0.9), {})))
        monkeypatch.setattr(forecaster, "extract_predictions", _extractor_spy())
        monkeypatch.setattr(forecaster, "enforce_deadline_arithmetic", lambda preds, dl, direction: preds)
        await forecaster._process_article(_sr(), QUESTION, max_article_chars=4000, timings=[], article_debugs=[])
        assert captured["pass1"] is HIGH
