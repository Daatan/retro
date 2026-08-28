"""A search-engine link wrapper is dropped before the fetch (retro#709).

The rule itself lives in `pipeline/tests/test_redirector_url.py`. What this pins is
that the live path applies it, drops **before** spending a fetch, and reports the
outcome — the same shape as the retro#705 no_date drop it sits beside.
"""
from unittest.mock import AsyncMock, Mock

import pytest

from forecast_api import forecaster
from tm.web_search import SearchResult

_Q = "Will the coalition hold through the winter session?"
_WRAPPED = "https://google.com/goto?url=CAESmwEB7keqTUrdHsqa6IxQA_vjiznE"


def _result(url, **kw):
    return SearchResult(
        title="Coalition expected to hold through the winter session",
        url=url, snippet="Analysts weigh the numbers.",
        source=kw.pop("source", "google.com"),
        published_date=kw.pop("published_date", "2026-08-26"),
    )


async def _run(monkeypatch, result, *, fetcher=None):
    monkeypatch.setattr(forecaster, "check_is_prediction",
                        AsyncMock(side_effect=AssertionError("gatekeeper must not run")))
    monkeypatch.setattr(forecaster, "_fetch_article_text",
                        fetcher or Mock(side_effect=AssertionError("must not fetch")))
    debugs = []
    out = await forecaster._process_article_bounded(
        result, _Q, max_article_chars=4000, timings=[], article_debugs=debugs,
        timeout_s=30.0,
    )
    return out, debugs


@pytest.mark.asyncio
class TestRedirectorUrlsAreDropped:
    async def test_a_wrapper_is_dropped(self, monkeypatch):
        out, debugs = await _run(monkeypatch, _result(_WRAPPED))
        assert out is None
        assert [d.outcome for d in debugs] == ["redirector_url"]

    async def test_dropped_before_the_fetch(self, monkeypatch):
        """A dropped article must not cost a request or a per-host throttle slot —
        the stubs above raise if either the fetch or the gatekeeper is reached."""
        spy = Mock(side_effect=AssertionError("fetched a wrapper URL"))
        out, _ = await _run(monkeypatch, _result(_WRAPPED), fetcher=spy)
        assert out is None
        assert not spy.called

    async def test_a_valid_date_does_not_rescue_it(self, monkeypatch):
        """These arrive dated — all 72 in the prod sample had a published_date. The
        drop is about attribution, not dating, so a good date must not exempt it."""
        out, debugs = await _run(monkeypatch, _result(_WRAPPED, published_date="2026-08-26"))
        assert out is None
        assert debugs[0].outcome == "redirector_url"

    async def test_a_real_article_from_the_same_host_family_survives(self, monkeypatch):
        """The control: without this, a passing test above proves only that
        _process_article returns None for everything the stubs touch."""
        monkeypatch.setattr(forecaster, "_fetch_article_text", Mock(return_value="Body text. " * 40))
        gate = AsyncMock(side_effect=RuntimeError("reached the gatekeeper"))
        monkeypatch.setattr(forecaster, "check_is_prediction", gate)
        debugs = []
        await forecaster._process_article_bounded(
            _result("https://blog.google/products/search/", source="blog.google"),
            _Q, max_article_chars=4000, timings=[], article_debugs=debugs, timeout_s=30.0,
        )
        assert gate.called, "a non-wrapper URL must reach the gatekeeper"
        assert "redirector_url" not in [d.outcome for d in debugs]
