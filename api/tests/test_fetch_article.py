"""Unit tests for ``_fetch_article_text`` and ``_looks_like_paywall``.

These cover the robustness fixes added in PR #44: HTTP status checking,
paywall-stub detection, and the "extracted must beat the snippet" rule.
They use ``unittest.mock`` to stub ``safe_get`` (the SSRF-guarded fetch) and
``trafilatura.extract`` so no network or real HTML parsing is required.
"""

from unittest.mock import patch, MagicMock

import httpx

from forecast_api.forecaster import (
    _MIN_ARTICLE_CHARS,
    _fetch_article_text,
    _looks_like_paywall,
)


class TestLooksLikePaywall:
    def test_returns_true_for_canonical_phrases(self):
        assert _looks_like_paywall("Subscribe to continue reading")
        assert _looks_like_paywall("SIGN IN TO READ the full article")
        assert _looks_like_paywall("please create a free account to continue")

    def test_returns_false_for_regular_article_prose(self):
        assert not _looks_like_paywall(
            "The central bank is expected to raise rates by 25bp next quarter."
        )

    def test_is_case_insensitive(self):
        assert _looks_like_paywall("BECOME A SUBSCRIBER to our daily newsletter")


def _mock_response(status_code: int, text: str = "") -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.text = text
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "err", request=MagicMock(), response=resp
        )
    else:
        resp.raise_for_status.return_value = None
    return resp


class TestFetchArticleText:
    FALLBACK = "short title — short snippet"

    def test_returns_extracted_when_long_and_healthy(self):
        long_article = "A real article body. " * 100  # ~2000 chars
        with patch("forecast_api.forecaster.safe_get") as mock_get, \
             patch("forecast_api.forecaster.trafilatura.extract", return_value=long_article):
            mock_get.return_value = _mock_response(200, "<html>...</html>")
            out = _fetch_article_text("https://example.com/a", self.FALLBACK)
        assert out == long_article

    def test_falls_back_on_404(self):
        """A 404 page must never be handed to the gatekeeper as article text."""
        with patch("forecast_api.forecaster.safe_get") as mock_get:
            mock_get.return_value = _mock_response(404, "<html>Not found</html>")
            out = _fetch_article_text("https://example.com/gone", self.FALLBACK)
        assert out == self.FALLBACK

    def test_falls_back_on_403(self):
        with patch("forecast_api.forecaster.safe_get") as mock_get:
            mock_get.return_value = _mock_response(403, "<html>Forbidden</html>")
            out = _fetch_article_text("https://example.com/locked", self.FALLBACK)
        assert out == self.FALLBACK

    def test_falls_back_on_500(self):
        with patch("forecast_api.forecaster.safe_get") as mock_get:
            mock_get.return_value = _mock_response(500, "<html>Server error</html>")
            out = _fetch_article_text("https://example.com/boom", self.FALLBACK)
        assert out == self.FALLBACK

    def test_falls_back_on_paywall_stub(self):
        """Trafilatura extracts the paywall CTA — we must not trust it."""
        stub = "Subscribe to read the full article for $4.99/month."
        assert len(stub) < _MIN_ARTICLE_CHARS  # guard: precondition for the check
        with patch("forecast_api.forecaster.safe_get") as mock_get, \
             patch("forecast_api.forecaster.trafilatura.extract", return_value=stub):
            mock_get.return_value = _mock_response(200, "<html>...</html>")
            out = _fetch_article_text("https://ft.com/article", self.FALLBACK)
        assert out == self.FALLBACK

    def test_falls_back_when_extraction_shorter_than_snippet(self):
        """If the fallback (title+snippet) is richer than the extracted body,
        the fallback wins — otherwise we'd *lose* signal to a stub."""
        rich_fallback = (
            "Russia's finance ministry warns of widening 2026 deficit — "
            "officials cite oil revenue shortfall and rising defence costs "
            "that may push the budget gap above last year's level"
        )
        short_body = "Short clip without the detail."
        assert len(short_body) < len(rich_fallback)  # guard
        with patch("forecast_api.forecaster.safe_get") as mock_get, \
             patch("forecast_api.forecaster.trafilatura.extract", return_value=short_body):
            mock_get.return_value = _mock_response(200, "<html>...</html>")
            out = _fetch_article_text("https://example.com/a", rich_fallback)
        assert out == rich_fallback

    def test_falls_back_on_trafilatura_empty(self):
        with patch("forecast_api.forecaster.safe_get") as mock_get, \
             patch("forecast_api.forecaster.trafilatura.extract", return_value=None):
            mock_get.return_value = _mock_response(200, "<html>junk</html>")
            out = _fetch_article_text("https://example.com/a", self.FALLBACK)
        assert out == self.FALLBACK

    def test_falls_back_on_network_error(self):
        with patch("forecast_api.forecaster.safe_get", side_effect=httpx.ConnectError("dns")):
            out = _fetch_article_text("https://example.com/a", self.FALLBACK)
        assert out == self.FALLBACK

    def test_falls_back_on_unsafe_url(self):
        """An SSRF-blocked URL (safe_get raises UnsafeURLError) yields the fallback,
        never a fetched internal response."""
        from forecast_api.net_guard import UnsafeURLError

        with patch(
            "forecast_api.forecaster.safe_get",
            side_effect=UnsafeURLError("http://169.254.169.254/"),
        ):
            out = _fetch_article_text("http://169.254.169.254/latest/meta-data/", self.FALLBACK)
        assert out == self.FALLBACK
