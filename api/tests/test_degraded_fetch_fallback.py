"""retro#520 — hybrid degraded-domain fetch fallback.

Le Monde (and other publishers on ``degraded_fetch_domains``) fail live re-fetch
almost always in prod (paywalls/bot-challenges), which starved the extractor of
full article text and drove confidence-score variance. Two tiers, both routed
through news-indexer's archived-S3-text lookup (``GET /articles/text``,
Daatan/news-indexer#277):

1. Pre-check — known-degraded domains skip the live fetch and try the archive
   first; a miss falls through to a normal live fetch.
2. Post-check — every other domain keeps live-fetch-first; on failure, try the
   archive before giving up to the title+snippet fallback.

LLM and network are mocked throughout — deterministic, no Bedrock, no sockets.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import httpx

from forecast_api import forecaster
from tm.models import GatekeeperOutput, PredictionExtraction
from tm.web_search import SearchResult

_QUESTION = "Will Knesset elections be held by 2026-12-31?"


def _mock_response(status_code: int, text: str = "") -> Mock:
    resp = Mock(spec=httpx.Response)
    resp.status_code = status_code
    resp.text = text
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "err", request=Mock(), response=resp
        )
    else:
        resp.raise_for_status.return_value = None
    return resp


class TestIsKnownDegradedDomain:
    def test_matches_configured_domain(self):
        assert forecaster._is_known_degraded_domain("https://www.reuters.com/world/1")
        assert forecaster._is_known_degraded_domain("https://lemonde.fr/international/1")

    def test_www_prefix_is_stripped(self):
        assert forecaster._is_known_degraded_domain("https://www.lemonde.fr/a")

    def test_non_degraded_domain_is_false(self):
        assert not forecaster._is_known_degraded_domain("https://www.haaretz.co.il/news/1")


class TestFetchArchivedText:
    def test_returns_none_when_news_indexer_unconfigured(self):
        with patch.object(forecaster, "NEWS_INDEXER_URL", None), \
             patch.object(forecaster, "NEWS_INDEXER_API_KEY", None), \
             patch("forecast_api.forecaster.httpx.get") as mock_get:
            out = forecaster._fetch_archived_text("https://lemonde.fr/a")
        assert out is None
        mock_get.assert_not_called()

    def test_returns_archived_text_on_hit(self):
        with patch.object(forecaster, "NEWS_INDEXER_URL", "https://scrapper.daatan.com"), \
             patch.object(forecaster, "NEWS_INDEXER_API_KEY", "k"), \
             patch("forecast_api.forecaster.httpx.get") as mock_get:
            resp = Mock(spec=httpx.Response)
            resp.raise_for_status.return_value = None
            resp.json.return_value = {"url": "https://lemonde.fr/a", "text": "archived body"}
            mock_get.return_value = resp
            out = forecaster._fetch_archived_text("https://lemonde.fr/a")
        assert out == "archived body"
        mock_get.assert_called_once()
        assert mock_get.call_args.kwargs["params"] == {"url": "https://lemonde.fr/a"}

    def test_returns_none_when_nothing_archived(self):
        with patch.object(forecaster, "NEWS_INDEXER_URL", "https://scrapper.daatan.com"), \
             patch.object(forecaster, "NEWS_INDEXER_API_KEY", "k"), \
             patch("forecast_api.forecaster.httpx.get") as mock_get:
            resp = Mock(spec=httpx.Response)
            resp.raise_for_status.return_value = None
            resp.json.return_value = {"url": "https://lemonde.fr/a", "text": None}
            mock_get.return_value = resp
            out = forecaster._fetch_archived_text("https://lemonde.fr/a")
        assert out is None

    def test_returns_none_on_error(self):
        with patch.object(forecaster, "NEWS_INDEXER_URL", "https://scrapper.daatan.com"), \
             patch.object(forecaster, "NEWS_INDEXER_API_KEY", "k"), \
             patch("forecast_api.forecaster.httpx.get", side_effect=httpx.ConnectError("dns")):
            out = forecaster._fetch_archived_text("https://lemonde.fr/a")
        assert out is None


class TestFetchArticleTextArchivedFallback:
    FALLBACK = "short title — short snippet"

    def test_uses_archived_text_when_live_fetch_fails(self):
        archived = "The archived article body, richer than the snippet fallback."
        with patch("forecast_api.forecaster.safe_get") as mock_get, \
             patch.object(forecaster, "_fetch_archived_text", return_value=archived) as mock_archived:
            mock_get.return_value = _mock_response(403, "<html>Forbidden</html>")
            out = forecaster._fetch_article_text("https://lemonde.fr/a", self.FALLBACK)
        assert out == archived
        mock_archived.assert_called_once_with("https://lemonde.fr/a")

    def test_falls_back_to_snippet_when_archive_also_misses(self):
        with patch("forecast_api.forecaster.safe_get") as mock_get, \
             patch.object(forecaster, "_fetch_archived_text", return_value=None):
            mock_get.return_value = _mock_response(403, "<html>Forbidden</html>")
            out = forecaster._fetch_article_text("https://lemonde.fr/a", self.FALLBACK)
        assert out == self.FALLBACK

    def test_falls_back_to_snippet_when_archived_text_is_not_richer(self):
        with patch("forecast_api.forecaster.safe_get") as mock_get, \
             patch.object(forecaster, "_fetch_archived_text", return_value="hi"):
            mock_get.return_value = _mock_response(403, "<html>Forbidden</html>")
            out = forecaster._fetch_article_text("https://lemonde.fr/a", self.FALLBACK)
        assert out == self.FALLBACK

    def test_try_archived_false_skips_the_lookup(self):
        with patch("forecast_api.forecaster.safe_get") as mock_get, \
             patch.object(forecaster, "_fetch_archived_text") as mock_archived:
            mock_get.return_value = _mock_response(403, "<html>Forbidden</html>")
            out = forecaster._fetch_article_text(
                "https://lemonde.fr/a", self.FALLBACK, try_archived=False
            )
        assert out == self.FALLBACK
        mock_archived.assert_not_called()

    def test_unsafe_url_skips_the_archived_lookup(self):
        from tm.net_guard import UnsafeURLError

        with patch(
            "forecast_api.forecaster.safe_get",
            side_effect=UnsafeURLError("http://169.254.169.254/"),
        ), patch.object(forecaster, "_fetch_archived_text") as mock_archived:
            out = forecaster._fetch_article_text(
                "http://169.254.169.254/latest/meta-data/", self.FALLBACK
            )
        assert out == self.FALLBACK
        mock_archived.assert_not_called()

    def test_successful_live_fetch_does_not_consult_the_archive(self):
        long_article = "A real article body. " * 100
        with patch("forecast_api.forecaster.safe_get") as mock_get, \
             patch("forecast_api.forecaster.trafilatura.extract", return_value=long_article), \
             patch.object(forecaster, "_fetch_archived_text") as mock_archived:
            mock_get.return_value = _mock_response(200, "<html>...</html>")
            out = forecaster._fetch_article_text("https://lemonde.fr/a", self.FALLBACK)
        assert out == long_article
        mock_archived.assert_not_called()


def _sr(url, *, title="", snippet=""):
    return SearchResult(
        title=title, url=url, snippet=snippet, source="", published_date="2026-08-01",
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


async def _process(monkeypatch, result, *, fetch_archived=None, fetch_live=None):
    gk, ex = _gk_spy(), _extractor_spy()
    monkeypatch.setattr(forecaster, "check_is_prediction", gk)
    monkeypatch.setattr(forecaster, "extract_predictions", ex)
    monkeypatch.setattr(forecaster, "enforce_deadline_arithmetic", lambda preds, dl, direction: preds)
    if fetch_archived is not None:
        monkeypatch.setattr(forecaster, "_fetch_archived_text", fetch_archived)
    if fetch_live is not None:
        monkeypatch.setattr(forecaster, "_fetch_article_text", fetch_live)
    out = await forecaster._process_article(
        result, _QUESTION,
        max_article_chars=4000, timings=[], article_debugs=[],
    )
    return out, gk, ex


class TestProcessArticlePreCheck:
    async def test_degraded_domain_uses_archive_and_skips_live_fetch(self, monkeypatch):
        archived = "The archived article body, comfortably longer than the snippet fallback."
        fetch_archived = Mock(return_value=archived)
        fetch_live = Mock(side_effect=AssertionError("live fetch must not happen on an archive hit"))
        sr = _sr("https://lemonde.fr/a", snippet="A snippet comfortably over twenty chars.")
        out, gk, _ = await _process(monkeypatch, sr, fetch_archived=fetch_archived, fetch_live=fetch_live)
        assert out is not None
        fetch_archived.assert_called_once_with("https://lemonde.fr/a")
        fetch_live.assert_not_called()
        assert gk.await_args.kwargs["article_text"] == archived

    async def test_degraded_domain_falls_through_to_live_fetch_on_archive_miss(self, monkeypatch):
        body = "A long live-fetched article body with plenty of substance about the elections."
        fetch_archived = Mock(return_value=None)
        fetch_live = Mock(return_value=body)
        sr = _sr("https://lemonde.fr/a", snippet="A snippet comfortably over twenty chars.")
        out, gk, _ = await _process(monkeypatch, sr, fetch_archived=fetch_archived, fetch_live=fetch_live)
        assert out is not None
        fetch_archived.assert_called_once_with("https://lemonde.fr/a")
        fetch_live.assert_called_once()
        # try_archived=False: the pre-check already tried and missed this exact lookup.
        assert fetch_live.call_args.kwargs.get("try_archived") is False
        assert gk.await_args.kwargs["article_text"] == body

    async def test_non_degraded_domain_skips_the_precheck(self, monkeypatch):
        body = "A long fetched article body with plenty of substance about the elections."
        fetch_archived = Mock(side_effect=AssertionError("pre-check must not run for this domain"))
        fetch_live = Mock(return_value=body)
        sr = _sr("https://www.haaretz.co.il/news/1", snippet="A snippet comfortably over twenty chars.")
        out, gk, _ = await _process(monkeypatch, sr, fetch_archived=fetch_archived, fetch_live=fetch_live)
        assert out is not None
        fetch_archived.assert_not_called()
        fetch_live.assert_called_once()
        assert gk.await_args.kwargs["article_text"] == body
