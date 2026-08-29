"""`is_redirector_url` — a search-engine link wrapper is not an article (retro#709).

The predicate is host-and-path, never a substring match, because the cheap version of
this check has two obvious false positives: a real article whose query string happens
to quote a redirector, and a publisher with "google" somewhere in its path.

Why it does not attempt to unwrap: the `goto?url=CAES…` parameter is Google's
*encrypted* article id, not an encoded URL. A live sample decodes to 188 bytes of
ciphertext containing no URL. news-indexer resolves the ones it can by following the
real 30x hop at ingestion (news-indexer#306); nothing can be recovered offline here.
"""
import pytest

from tm.web_search_ingest import is_redirector_url


class TestWrappersAreDetected:
    @pytest.mark.parametrize("url", [
        # the shape actually observed in the pool: an opaque encrypted token
        "https://google.com/goto?url=CAESmwEB7keqTUrdHsqa6IxQA_vjiznE",
        "https://www.google.com/url?q=https://bbc.com/news/x",
        "https://news.google.com/articles/CBMiK2h0dHBz",   # wrapper at every path
        "https://www.bing.com/ck/a?u=a1aHR0cHM",
        "https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com",
        "https://r.search.yahoo.com/l/_ylt=A",
    ])
    def test_known_wrappers(self, url):
        assert is_redirector_url(url)


class TestArticlesAreNotDetected:
    @pytest.mark.parametrize("url", [
        "https://bbc.com/news/world-middle-east-123",
        # the substring trap: a real article ABOUT the redirector
        "https://theguardian.com/2024/03/15/google-goto-antitrust-ruling",
        # the other substring trap: a redirector quoted inside a real article's params
        "https://example.com/story?ref=https://google.com/goto?url=x",
        # a Google property that serves its own content rather than redirecting
        "https://blog.google/products/search/",
        "https://developers.google.com/search/docs",
    ])
    def test_real_articles_survive(self, url):
        assert not is_redirector_url(url)

    def test_missing_url_is_not_a_wrapper(self):
        """Fail open: an absent URL is someone else's error, not a wrapper."""
        assert not is_redirector_url(None)
        assert not is_redirector_url("")

    def test_unparseable_url_is_not_a_wrapper(self):
        assert not is_redirector_url("http://[::1")
