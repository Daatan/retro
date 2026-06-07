"""Unit tests for the per-article wall-clock ceiling in ``_process_article_bounded``.

The wrapper bounds each article's gatekeeper+extractor work so a single slow
LLM call can't stall the parallel batch. ``_process_article`` itself is mocked
here — we only exercise the timeout/pass-through behaviour of the wrapper.
"""

import asyncio
from types import SimpleNamespace

import forecast_api.forecaster as fc


async def test_passes_through_a_fast_article(monkeypatch):
    async def fast(result, question, **kwargs):
        return (result, ["ok"])

    monkeypatch.setattr(fc, "_process_article", fast)
    timings: list = []
    debugs: list = []
    r = SimpleNamespace(url="http://example.com/a")

    out = await fc._process_article_bounded(
        r, "Q?", max_article_chars=3000, timings=timings, article_debugs=debugs, timeout_s=1.0
    )

    assert out == (r, ["ok"])
    assert timings == []  # no timeout entry recorded
    assert debugs == []


async def test_drops_a_slow_article_and_records_a_timeout(monkeypatch):
    async def slow(result, question, **kwargs):
        await asyncio.sleep(0.5)
        return (result, ["too late"])

    monkeypatch.setattr(fc, "_process_article", slow)
    timings: list = []
    debugs: list = []
    r = SimpleNamespace(url="http://example.com/slow")

    out = await fc._process_article_bounded(
        r, "Q?", max_article_chars=3000, timings=timings, article_debugs=debugs, timeout_s=0.05
    )

    assert out is None
    assert timings and timings[-1] == {"url": "http://example.com/slow", "outcome": "timeout"}
    assert debugs and debugs[-1].outcome == "timeout"
    assert debugs[-1].url == "http://example.com/slow"
