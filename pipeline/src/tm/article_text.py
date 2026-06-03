"""Shared HTML → article-body extraction for the batch ingestors.

``gdelt_ingest``, ``web_search_ingest`` and ``site_search`` each fetched a page
and pulled the main body out the same way: strip boilerplate elements, try a
list of content selectors, fall back to the whole-page text. That logic lived in
three near-identical copies (``gnews_ingest`` had a fourth, as the BeautifulSoup
fallback after trafilatura). It now lives here once.

The selector and strip-tag lists are the **union** of what those copies used, so
every caller tries at least the selectors it tried before. The lists are ordered
most-specific-first, so where the union adds a selector ahead of one a caller
already had (e.g. gdelt now tries ``.article-body`` before ``[class*="content"]``)
the result may be a *different, more specific* block than before — a deliberate
improvement, not strict byte-for-byte preservation.
"""

from bs4 import BeautifulSoup

# Browser-like headers. Some sites 403 the default httpx User-Agent; this UA +
# Accept pair is what the site scrapers have used in practice.
BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}

# Non-content elements removed before extracting body text.
_BOILERPLATE_TAGS = ["script", "style", "nav", "footer", "header", "aside", "iframe", "figure"]

# Content selectors tried in order; the first match longer than _MIN_BODY_CHARS
# wins. Most-specific selectors come first so we prefer a real article body over
# a generic ``[class*="content"]`` wrapper.
_CONTENT_SELECTORS = [
    "article",
    "main",
    ".article-body",
    ".article-content",
    ".post-content",
    '[class*="article"]',
    '[class*="story"]',
    '[class*="content"]',
]

_MIN_BODY_CHARS = 300


def extract_article_body(soup: BeautifulSoup) -> str:
    """Extract the main article text from a parsed page.

    Strips boilerplate, tries the content selectors in order, and falls back to
    the full page text when no selector yields a long-enough block. Returns ``""``
    only when the page itself has no text.
    """
    for tag in soup(_BOILERPLATE_TAGS):
        tag.extract()
    for sel in _CONTENT_SELECTORS:
        el = soup.select_one(sel)
        if el:
            text = el.get_text(separator=" ", strip=True)
            if len(text) > _MIN_BODY_CHARS:
                return text
    return soup.get_text(separator=" ", strip=True)
