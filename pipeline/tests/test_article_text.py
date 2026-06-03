"""Tests for the shared HTML → article-body extraction (tm.article_text).

This logic was consolidated from four near-identical copies in the batch
ingestors, so these tests pin the behaviour every caller now depends on:
selector precedence (most-specific first), the short-body fall-through, and
boilerplate stripping.
"""

from bs4 import BeautifulSoup

from tm.article_text import BROWSER_HEADERS, extract_article_body


def _soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "html.parser")


class TestExtractArticleBody:
    def test_prefers_article_element(self):
        body = "word " * 100
        html = f"<html><body><div>{'junk ' * 100}</div><article>{body}</article></body></html>"
        out = extract_article_body(_soup(html))
        assert out.strip() == body.strip()

    def test_strips_boilerplate_tags(self):
        body = "real content " * 50
        html = (
            "<html><body>"
            "<nav>menu menu menu</nav><footer>footer footer</footer>"
            f"<article>{body}</article>"
            "<script>var x=1</script>"
            "</body></html>"
        )
        out = extract_article_body(_soup(html))
        assert "menu" not in out and "footer" not in out and "var x" not in out
        assert "real content" in out

    def test_specific_selector_wins_over_generic(self):
        # Both <article> and a generic [class*="content"] are long enough; the
        # most-specific selector (article) is tried first and must win.
        article_body = "article body text " * 40
        content_body = "generic content wrapper " * 40
        html = (
            "<html><body>"
            f'<div class="page-content">{content_body}</div>'
            f"<article>{article_body}</article>"
            "</body></html>"
        )
        out = extract_article_body(_soup(html))
        assert "article body text" in out
        assert "generic content wrapper" not in out

    def test_falls_back_to_full_text_when_no_selector_matches(self):
        # No matching container and the only block is under the 300-char floor →
        # fall back to the whole-page text rather than returning "".
        html = "<html><body><div>short blurb here</div></body></html>"
        out = extract_article_body(_soup(html))
        assert "short blurb here" in out

    def test_short_selector_match_falls_through(self):
        # <article> exists but is too short (<300 chars); the long generic
        # wrapper after it should be picked instead.
        long_body = "the real story continues " * 40
        html = (
            "<html><body>"
            "<article>tiny</article>"
            f'<div class="content">{long_body}</div>'
            "</body></html>"
        )
        out = extract_article_body(_soup(html))
        assert "the real story continues" in out

    def test_empty_page_returns_empty(self):
        assert extract_article_body(_soup("<html><body></body></html>")).strip() == ""


class TestBrowserHeaders:
    def test_has_user_agent(self):
        assert "User-Agent" in BROWSER_HEADERS
        assert "Mozilla" in BROWSER_HEADERS["User-Agent"]
