"""retro#417 — short-form/Telegram evidence on /forecast's article path.

Three Oracul-side losses for Telegram-sourced evidence, three pins:

1. t.me URLs are never origin-fetched. t.me serves a web preview that trafilatura
   extracts to near-nothing, so the fetch always lost the
   ``extracted_len <= len(fallback)`` comparison — a wasted request plus a wasted
   per-host throttle slot.
2. The 20-char fallback floor does not apply to t.me posts. A terse Telegram post IS
   the article; the silent drop bypassed the ``_SHORT_FORM_OVERRIDE`` built for exactly
   that class. Only a truly-empty floor (<5 chars) remains, and the gatekeeper's
   deterministic ``carries_proposition`` guard still rejects content-free posts.
3. A caller-supplied ``language`` hint threads from ArticleInput through to both the
   gatekeeper and extractor prompts.

LLM and network are mocked throughout — deterministic, no Bedrock, no sockets.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from forecast_api import forecaster
from forecast_api.models import ArticleInput
from tm.models import GatekeeperOutput, PredictionExtraction
from tm.web_search import SearchResult

_QUESTION = "Will Knesset elections be held by 2026-12-31?"


def _sr(url, *, title="", snippet="", text=None, language=None):
    return SearchResult(
        title=title,
        url=url,
        snippet=snippet,
        source="",
        published_date="2026-08-01",
        _prefetched_text=text,
        _language=language,
    )


def _gk_spy():
    return AsyncMock(return_value=(
        GatekeeperOutput(is_prediction=True, reason="judged", relevance_score=0.8), {},
    ))


def _extractor_spy():
    return AsyncMock(return_value=(
        SimpleNamespace(
            predictions=[PredictionExtraction(
                quote="q", claim="c", stance=0.6, certainty=0.8, specificity=1.0, settled=None,
            )],
            author_lean=None,
            author_lean_certainty=None,
            consensus_view=None,
            claim_actor=None, claim_predicate=None, claim_scope=None,
        ),
        {},
    ))


async def _process(monkeypatch, result, *, fetch=None):
    """Run _process_article with gatekeeper/extractor spies; returns (out, gk, ex, fetch)."""
    gk, ex = _gk_spy(), _extractor_spy()
    if fetch is None:
        fetch = Mock(side_effect=AssertionError("origin fetch must not happen in this test"))
    monkeypatch.setattr(forecaster, "check_is_prediction", gk)
    monkeypatch.setattr(forecaster, "extract_predictions", ex)
    monkeypatch.setattr(forecaster, "enforce_deadline_arithmetic", lambda preds, dl, direction: preds)
    monkeypatch.setattr(forecaster, "_fetch_article_text", fetch)
    out = await forecaster._process_article(
        result, _QUESTION,
        max_article_chars=4000, timings=[], article_debugs=[],
    )
    return out, gk, ex, fetch


class TestNeverFetchTme:
    async def test_tme_url_without_text_is_not_fetched(self, monkeypatch):
        # The fetch spy raises if called — the t.me article must go straight to fallback.
        sr = _sr("https://t.me/bencaspit/4471", snippet="Elections are now final for Oct 27.")
        out, gk, _, fetch = await _process(monkeypatch, sr)
        assert out is not None
        fetch.assert_not_called()
        assert gk.await_args.kwargs["article_text"] == "Elections are now final for Oct 27."

    async def test_www_tme_is_also_skipped(self, monkeypatch):
        # Same normalization as the short_form detection: www. is stripped before matching.
        sr = _sr("https://www.t.me/bencaspit/4471", snippet="Elections are now final for Oct 27.")
        out, _, _, fetch = await _process(monkeypatch, sr)
        assert out is not None
        fetch.assert_not_called()

    async def test_prefetched_text_still_wins_over_the_fallback(self, monkeypatch):
        sr = _sr(
            "https://t.me/bencaspit/4471",
            snippet="Elections are now final for Oct 27.",
            text="The full post text, as pushed over MTProto by news-indexer.",
        )
        out, gk, _, fetch = await _process(monkeypatch, sr)
        assert out is not None
        fetch.assert_not_called()
        assert gk.await_args.kwargs["article_text"].startswith("The full post text")

    async def test_non_tme_url_is_still_fetched(self, monkeypatch):
        # The guard is t.me-scoped: ordinary articles keep the trafilatura path.
        body = "A long fetched article body with plenty of substance about the elections."
        sr = _sr("https://www.haaretz.co.il/news/1", snippet="A snippet comfortably over twenty chars.")
        out, gk, _, fetch = await _process(monkeypatch, sr, fetch=Mock(return_value=body))
        assert out is not None
        fetch.assert_called_once()
        assert gk.await_args.kwargs["article_text"] == body


class TestShortFormFloorExemption:
    async def test_short_tme_post_reaches_the_gatekeeper(self, monkeypatch):
        # 13 chars — under the 20-char floor that used to drop it before any judge call.
        sr = _sr("https://t.me/bencaspit/4472", snippet="Deal is done.")
        out, gk, _, _ = await _process(monkeypatch, sr)
        assert out is not None                       # judged, not silently dropped
        gk.assert_awaited_once()
        # ...and judged WITH the override the floor exemption exists for: without it the
        # base prompt's "under ~200 meaningful words" rule would reject what the floor spared.
        assert gk.await_args.kwargs["short_form"] is True

    async def test_inn_post_also_gets_the_short_form_override(self, monkeypatch):
        # ni#380 added israelnationalnews.com to news-indexer's short-form host list on
        # 2026-08-24; retro's two inline copies kept checking only t.me, so INN posts were
        # judged against the ~200-word long-form floor. Both sides now read one shared list
        # (tm.gatekeeper.is_short_form).
        sr = _sr("https://www.israelnationalnews.com/news/123456", snippet="Deal is done.")
        # INN IS fetched (see the test below) — the fetch returns the fallback here, which is
        # what _fetch_article_text does when extraction is no better than the snippet.
        out, gk, _, _ = await _process(
            monkeypatch, sr, fetch=Mock(return_value="Deal is done."),
        )
        assert out is not None
        assert gk.await_args.kwargs["short_form"] is True

    async def test_inn_is_still_origin_fetched(self, monkeypatch):
        """`short_form` governs JUDGING, not fetching. INN is terse but has a real article page,
        so the t.me no-fetch shortcut must not follow it: keying that branch on `short_form`
        would silently reduce every INN item to its search snippet."""
        body = "A full INN article body with plenty of substance about the elections."
        sr = _sr("https://www.israelnationalnews.com/news/123456", snippet="Deal is done.")
        out, gk, _, fetch = await _process(monkeypatch, sr, fetch=Mock(return_value=body))
        assert out is not None
        fetch.assert_called_once()
        assert gk.await_args.kwargs["article_text"] == body
        assert gk.await_args.kwargs["short_form"] is True

    def test_forecaster_uses_the_shared_helper(self):
        # No inline host check may come back on this side either — that is what drifted.
        from pathlib import Path

        src = Path(forecaster.__file__).read_text()
        assert "is_short_form(" in src
        assert 'removeprefix("www.") == "t.me"' not in src

    async def test_short_non_tme_article_is_still_dropped(self, monkeypatch):
        sr = _sr("https://x.com/some/1", snippet="Deal is done.")
        out, gk, _, _ = await _process(monkeypatch, sr)
        assert out is None
        gk.assert_not_awaited()

    async def test_truly_empty_tme_post_is_still_dropped(self, monkeypatch):
        # <5 chars: nothing for even the short-form judge to work with.
        sr = _sr("https://t.me/bencaspit/4473", snippet="Ok.")
        out, gk, _, _ = await _process(monkeypatch, sr)
        assert out is None
        gk.assert_not_awaited()

    async def test_regular_article_does_not_get_the_short_form_override(self, monkeypatch):
        body = "A long fetched article body with plenty of substance about the elections."
        sr = _sr("https://www.haaretz.co.il/news/2", snippet="A snippet comfortably over twenty chars.")
        _, gk, ex, _ = await _process(monkeypatch, sr, fetch=Mock(return_value=body))
        assert gk.await_args.kwargs["short_form"] is False
        assert ex.await_args.kwargs["short_form"] is False


class TestLanguageHint:
    async def test_language_threads_to_gatekeeper_and_extractor(self, monkeypatch):
        sr = _sr(
            "https://t.me/bencaspit/4474",
            snippet="הבחירות סופיות — 27 באוקטובר.",
            language="Hebrew",
        )
        out, gk, ex, _ = await _process(monkeypatch, sr)
        assert out is not None
        assert gk.await_args.kwargs["language"] == "Hebrew"
        assert ex.await_args.kwargs["language"] == "Hebrew"

    async def test_no_language_threads_none(self, monkeypatch):
        sr = _sr("https://t.me/bencaspit/4475", snippet="Elections are now final for Oct 27.")
        _, gk, ex, _ = await _process(monkeypatch, sr)
        assert gk.await_args.kwargs["language"] is None
        assert ex.await_args.kwargs["language"] is None


class TestArticleInputModel:
    def test_language_is_optional_and_defaults_to_none(self):
        assert ArticleInput(url="https://t.me/x/1").language is None
        assert ArticleInput(url="https://t.me/x/1", language="he").language == "he"

    def test_unknown_fields_are_still_ignored(self):
        # Senders may run ahead of the API. Adding `language` must not flip the model into
        # rejecting extras — pydantic's default extra="ignore" has to keep holding.
        a = ArticleInput.model_validate(
            {"url": "https://t.me/x/1", "language": "he", "some_future_field": 1}
        )
        assert a.language == "he"

    async def test_request_article_language_reaches_the_search_result(self, monkeypatch):
        # The mapping in _run_forecast_inner, tested at the seam where it happens: the
        # SearchResult handed to _process_article_bounded must carry the caller's language.
        import time

        from forecast_api.models import ForecastRequest

        seen: list = []

        async def _capture(*args, **kwargs):
            seen.append(args[0])
            return None

        monkeypatch.setattr(forecaster, "_process_article_bounded", _capture)
        req = ForecastRequest(
            question=_QUESTION,
            articles=[ArticleInput(
                url="https://t.me/x/1", title="t", snippet="s",
                text="The full post text.", language="ru",
            )],
        )
        await forecaster._run_forecast_inner(
            req, cache_key="test-key-417", limit=5, total_start=time.perf_counter(),
        )
        assert [r._language for r in seen] == ["ru"]
