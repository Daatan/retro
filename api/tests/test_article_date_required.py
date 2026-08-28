"""retro#705 — an article we cannot date is dropped, not dated to today.

`forecaster` used to derive `article_date` two different ways in the same flow:
the aggregation layer took `result.published_date or None` and let recency fail
open, while the extraction layer took `result.published_date or
datetime.now()` — so an undated article was presented to the gatekeeper and the
extractor as today's news.

That value is not a display field. It is the calendar anchor
`_apply_relative_date_override` walks "on Friday" against, so an undated 2019
piece could hand `enforce_settlement_event_date` a fresh, plausible, wrong
`event_date` — the one thing a positive settlement requires. It also made that
guard's future-dated check vacuous, since nothing can be after today.

Network and LLM are mocked throughout.
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from forecast_api import forecaster
from forecast_api.models import ArticleDebug
from tm.models import GatekeeperOutput, PredictionExtraction
from tm.web_search import SearchResult

_QUESTION = "Will Knesset elections be held by 2026-12-31?"
_SNIPPET = "A snippet comfortably over twenty characters long."


def _sr(url, *, published_date=None, snippet=_SNIPPET):
    return SearchResult(
        title="A detailed report on whether the elections will be held",
        url=url, snippet=snippet, source="example.com", published_date=published_date,
    )


class TestResolveArticleDate:
    def test_provider_date_wins(self):
        sr = _sr("https://example.com/2019/03/15/old-piece", published_date="2026-08-01")
        assert forecaster._resolve_article_date(sr) == "2026-08-01"

    def test_url_path_date_is_the_fallback(self):
        assert forecaster._resolve_article_date(_sr("https://example.com/2019/03/15/a")) == "2019-03-15"

    def test_no_date_anywhere_is_none_not_today(self):
        """The whole point. A `datetime.now()` here is indistinguishable from a
        real same-day article everywhere downstream."""
        assert forecaster._resolve_article_date(_sr("https://example.com/some-slug")) is None

    def test_blank_provider_date_is_not_a_date(self):
        assert forecaster._resolve_article_date(_sr("https://example.com/a", published_date="   ")) is None

    def test_a_provider_timestamp_is_truncated_to_the_day(self):
        sr = _sr("https://example.com/a", published_date="2026-08-01T09:31:00Z")
        assert forecaster._resolve_article_date(sr) == "2026-08-01"


@pytest.mark.asyncio
class TestUndatedArticleIsDropped:
    async def _run(self, monkeypatch, sr, *, fetch=None):
        gk = AsyncMock(return_value=(
            GatekeeperOutput(is_prediction=True, reason="judged", relevance_score=0.8), {},
        ))
        ex = AsyncMock(return_value=(SimpleNamespace(
            predictions=[PredictionExtraction(
                quote="q", claim="The elections were held.", stance=0.6, certainty=0.8,
                specificity=1.0, settled=None,
            )],
            author_lean=None, author_lean_certainty=None,
        ), {}))
        monkeypatch.setattr(forecaster, "check_is_prediction", gk)
        monkeypatch.setattr(forecaster, "extract_predictions", ex)
        monkeypatch.setattr(forecaster, "enforce_deadline_arithmetic", lambda p, dl, d: p)
        fetch = fetch or Mock(return_value="A long article body with plenty of substance. " * 4)
        monkeypatch.setattr(forecaster, "_fetch_article_text", fetch)
        debugs: list[ArticleDebug] = []
        out = await forecaster._process_article(
            sr, _QUESTION, max_article_chars=4000, timings=[], article_debugs=debugs,
        )
        return out, debugs, gk, ex, fetch

    async def test_undated_article_is_dropped(self, monkeypatch):
        out, debugs, gk, ex, _ = await self._run(monkeypatch, _sr("https://example.com/slug"))
        assert out is None
        assert [d.outcome for d in debugs] == ["no_date"]
        gk.assert_not_awaited()
        ex.assert_not_awaited()

    async def test_it_is_dropped_before_paying_for_a_fetch(self, monkeypatch):
        """A dropped article must not cost a request and a per-host throttle slot."""
        fetch = Mock(side_effect=AssertionError("must not fetch an article we will drop"))
        out, _, _, _, _ = await self._run(monkeypatch, _sr("https://example.com/slug"), fetch=fetch)
        assert out is None
        fetch.assert_not_called()

    async def test_a_url_path_date_rescues_the_article(self, monkeypatch):
        out, debugs, gk, _, _ = await self._run(monkeypatch, _sr("https://example.com/2026/08/01/a"))
        assert out is not None
        assert [d.outcome for d in debugs] != ["no_date"]
        assert gk.await_args.kwargs["article_date"] == "2026-08-01"

    async def test_the_dated_path_still_passes_the_provider_date_through(self, monkeypatch):
        sr = _sr("https://example.com/a", published_date="2026-07-04")
        out, _, gk, ex, _ = await self._run(monkeypatch, sr)
        assert out is not None
        assert gk.await_args.kwargs["article_date"] == "2026-07-04"
        assert ex.await_args.kwargs["article_date"] == "2026-07-04"

    async def test_prefetched_text_does_not_exempt_an_undated_article(self, monkeypatch):
        """Caller-supplied text asserts relevance, not a date — and the recency,
        relative-date and future-dated arguments all still apply."""
        sr = _sr("https://example.com/slug")
        sr._prefetched_text = "A caller-supplied article body of ample length. " * 4
        out, debugs, _, _, _ = await self._run(monkeypatch, sr)
        assert out is None
        assert [d.outcome for d in debugs] == ["no_date"]
