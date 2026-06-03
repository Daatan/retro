"""
Google News RSS ingestor with multi-backend URL resolution.

Step 1: Query Google News RSS with date operators to find article titles
        from the event's predictive window.
Step 2: Resolve each title to a real URL via a chain of search backends
        tried in order until one succeeds:
          1. Direct slug construction (instant, works for TOI/Reuters)
          2. Brave Search API        (BRAVE_API_KEY)
          3. SerpApi                 (SERPAPI_KEY  — serpapi.com)
          4. Serper.dev              (SERPERDEV_KEY — serper.dev)
          5. DuckDuckGo              (free, blocked on EC2 datacenters)
        Backends with no key set are skipped automatically.
Step 3: Scrape full article text via trafilatura (primary) with
        BeautifulSoup fallback, and save to data/raw_ingest/.
        If text < 500 chars (paywall), tries Wayback Machine cached copy.
Step 4: Newsdata.io /archive fallback — if web search yields 0 articles,
        query the paid Newsdata.io archive API with domainurl= filter.
        Returns full_description for paywalled sources on qualifying plans.
Step 5: GDELT fallback — if Newsdata.io yields 0 articles,
        query the free GDELT DOC API (English sources only) for direct URLs.
Step 6: CDX fallback — enumerate Wayback CDX archives (disabled on EC2;
        set ENABLE_CDX=1 to re-enable).

Sources: toi, jpost, haaretz, reuters, globes, ynet, israel_hayom,
         walla, haaretz_he, n12, maariv, ch13, calcalist  (Hebrew sources included)

Usage:
    DATA_DIR=/path/to/data uv run python -m tm.gnews_ingest
    DATA_DIR=/path/to/data uv run python -m tm.gnews_ingest --events C05 E07 --sources toi jpost
"""

import asyncio
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import List, Dict, Optional

import httpx
import trafilatura
from bs4 import BeautifulSoup
from ddgs import DDGS
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn
from rich.table import Table

from .models import CellStatus
from .progress import load_state, update_cell
from .web_search import search_articles as _web_search
from .article_text import BROWSER_HEADERS, extract_article_body
from .utils import existing_articles

console = Console()

GNEWS_BASE = "https://news.google.com/rss/search"

# domain + GNews locale per source
# lang="en" → ASCII keyword, en-US GNews params
# lang="he" → first Hebrew keyword, iw-IL GNews params
SOURCES_CONFIG: Dict[str, Dict] = {
    # English sources
    "toi":          {"domain": "www.timesofisrael.com", "lang": "en"},
    "jpost":        {"domain": "jpost.com",          "lang": "en"},
    "haaretz":      {"domain": "haaretz.com",        "lang": "en"},
    "reuters":      {"domain": "reuters.com",        "lang": "en"},
    "globes":       {"domain": "en.globes.co.il",    "lang": "en"},
    "ynet":         {"domain": "ynetnews.com",       "lang": "en"},
    "israel_hayom": {"domain": "israelhayom.com",    "lang": "en"},
    "aljazeera":    {"domain": "aljazeera.com",      "lang": "en"},
    "bloomberg":    {"domain": "bloomberg.com",       "lang": "en"},
    "bbc":          {"domain": "bbc.com",             "lang": "en"},
    "nyt":          {"domain": "nytimes.com",         "lang": "en"},
    "ft":           {"domain": "ft.com",              "lang": "en"},
    "guardian":     {"domain": "theguardian.com",     "lang": "en"},
    "wapost":       {"domain": "washingtonpost.com",  "lang": "en"},
    "axios":        {"domain": "axios.com",           "lang": "en"},
    # Hebrew-native sources
    "walla":        {"domain": "news.walla.co.il",   "lang": "he"},
    "haaretz_he":   {"domain": "www.haaretz.co.il",  "lang": "he"},  # Hebrew edition — less paywalled than haaretz.com
    "n12":          {"domain": "www.mako.co.il",     "lang": "he"},
    "maariv":       {"domain": "www.maariv.co.il",   "lang": "he"},
    "ch13":         {"domain": "13tv.co.il",         "lang": "he"},
    "calcalist":    {"domain": "www.calcalist.co.il", "lang": "he"},
}

GNEWS_LOCALE = {
    "en": {"hl": "en-US", "gl": "US", "ceid": "US:en"},
    "he": {"hl": "iw-IL", "gl": "IL", "ceid": "IL:iw"},
}

# Keep the old flat dict for backwards-compat callsites
SOURCES = {sid: cfg["domain"] for sid, cfg in SOURCES_CONFIG.items()}

MVP_EVENTS = [
    "C05", "C06", "C07", "C08", "C09",
    "B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B10", "B11", "B12", "B13",
    "A04", "A12", "A13", "A14", "A15", "A19",
    "D02", "D03",
    "E07", "E08",
    "G02", "G05", "G06",
    "F04", "F05",
]

HEADERS = BROWSER_HEADERS

_DDG_LAST_CALL: float = 0.0
DDG_MIN_INTERVAL = 2.0  # seconds between DDG calls

# Search backend API keys — set whichever you have, others are skipped.
# Priority order: Brave → SerpApi → Serper.dev → DDG
BRAVE_API_KEY:  Optional[str] = os.environ.get("BRAVE_API_KEY")   # api.search.brave.com
SERPAPI_KEY:    Optional[str] = os.environ.get("SERPAPI_KEY") \
                              or os.environ.get("SERPER_API_KEY")   # serpapi.com (backwards-compat)
SERPERDEV_KEY:  Optional[str] = os.environ.get("SERPERDEV_KEY")   # serper.dev
NEWSDATA_API_KEY: Optional[str] = os.environ.get("NEWSDATA_API_KEY")  # newsdata.io

# Known paywall/redirect page title fragments — Wayback often returns these
# instead of real article content. Checked case-insensitively against the
# scraped page title (first 120 chars of text).
_STUB_TITLE_FRAGMENTS = (
    "bloomberg mercury",
    "bloomberg data",
    "watch russia invades ukraine",
    "this week in crypto",
    "bloomberg podcasts",
    "bloomberg quicktake",
    "bloomberg live",
)

# Circuit breakers — set to True after quota/persistent errors to skip for rest of session
_BRAVE_QUOTA_EXHAUSTED: bool = False
_SERPER_QUOTA_EXHAUSTED: bool = False
_GDELT_BLOCKED: bool = False
_NEWSDATA_QUOTA_EXHAUSTED: bool = False


def _is_ascii(s: str) -> bool:
    try:
        s.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def _is_stub_page(text: str) -> bool:
    """Return True if scraped text matches a known paywall/redirect stub."""
    preview = text[:120].lower()
    return any(frag in preview for frag in _STUB_TITLE_FRAGMENTS)


def _clean_title(title: str) -> str:
    """Strip publisher suffix like ' - The Times of Israel'."""
    for sep in [" - ", " | ", " — "]:
        if sep in title:
            title = title.rsplit(sep, 1)[0]
    return title.strip()


def search_gnews_rss(
    domain: str,
    keywords: List[str],
    start_date: datetime,
    end_date: datetime,
    max_results: int = 10,
    lang: str = "en",
) -> List[Dict]:
    """
    Query Google News RSS for article titles in the date window.
    Returns list of {title, published_at}.
    lang="en" uses the first ASCII keyword + en-US locale.
    lang="he" uses the first non-ASCII (Hebrew) keyword + iw-IL locale.
    """
    if lang == "he":
        kws = [k.strip('"') for k in keywords if not _is_ascii(k) and k.strip('"')]
    else:
        kws = [k.strip('"') for k in keywords if _is_ascii(k) and k.strip('"')]

    if not kws:
        return []

    after = (start_date - timedelta(days=1)).strftime("%Y-%m-%d")
    before = (end_date + timedelta(days=1)).strftime("%Y-%m-%d")
    locale = GNEWS_LOCALE.get(lang, GNEWS_LOCALE["en"])

    # Try each keyword in order; stop at the first that returns results
    r = None
    for kw_part in kws:
        query = f"{kw_part} after:{after} before:{before} site:{domain}"
        params = {"q": query, **locale}
        try:
            r = httpx.get(GNEWS_BASE, params=params, headers=HEADERS, timeout=20, follow_redirects=True)
            if r.status_code != 200:
                continue
            # Peek at item count before full parse
            if "<item>" in r.text:
                break
        except Exception as e:
            console.print(f"    [dim red]GNews RSS error: {e}[/dim red]")
            r = None

    if r is None or r.status_code != 200:
        return []

    articles = []
    try:
        root = ET.fromstring(r.text)
        channel = root.find("channel")
        if channel is None:
            return []
        for item in channel.findall("item"):
            title_raw = item.findtext("title", "")
            pubdate_str = item.findtext("pubDate", "")

            title = _clean_title(title_raw)
            if not title:
                continue

            try:
                pub_dt = parsedate_to_datetime(pubdate_str).replace(tzinfo=None)
            except Exception:
                pub_dt = None

            # Strict date filter
            if pub_dt and not (start_date <= pub_dt <= end_date):
                continue

            pub_str = pub_dt.strftime("%Y-%m-%d") if pub_dt else start_date.strftime("%Y-%m-%d")
            gnews_link = item.findtext("link", "") or item.findtext("guid", "")
            articles.append({"title": title, "published_at": pub_str, "gnews_link": gnews_link})

            if len(articles) >= max_results:
                break
    except ET.ParseError as e:
        console.print(f"    [dim red]RSS parse error: {e}[/dim red]")
        return []

    return articles


def _title_slug(title: str) -> str:
    """Convert title to URL slug (lowercase, ASCII, hyphen-separated)."""
    t = title.lower()
    t = re.sub(r"['\",.:;!?()\[\]{}&|+=%#@$\\\\/]", "", t)
    t = re.sub(r"[^a-z0-9\s-]", "", t)
    t = re.sub(r"\s+", "-", t.strip())
    return re.sub(r"-+", "-", t)


def _construct_url(title: str, domain: str, expected_date: Optional[datetime] = None) -> Optional[str]:
    """Construct a probable article URL from title for domains with predictable slugs."""
    slug = _title_slug(title)
    if not slug:
        return None
    if domain == "timesofisrael.com":
        return f"https://www.timesofisrael.com/{slug}/"
    # Reuters: caller will try each candidate URL via httpx; return all options
    # as a list is not supported — just return the most likely one
    if domain == "reuters.com" and expected_date:
        date_str = expected_date.strftime("%Y-%m-%d")
        return f"https://www.reuters.com/world/middle-east/{slug}-{date_str}/"
    if domain == "israelhayom.com" and expected_date:
        date_path = expected_date.strftime("%Y/%m/%d")
        return f"https://www.israelhayom.com/{date_path}/{slug}/"
    if domain == "bloomberg.com" and expected_date:
        date_str = expected_date.strftime("%Y-%m-%d")
        return f"https://www.bloomberg.com/news/articles/{date_str}/{slug}"
    return None


def _url_date(url: str) -> Optional[datetime]:
    """Extract date from URL pattern like /2024/04/12/ or -2024-04-12."""
    m = re.search(r"[/\-](20\d\d)[/\-](\d\d)[/\-](\d\d)", url)
    if m:
        try:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass
    return None


def _filter_url(href: str, domain: str, expected_date: Optional[datetime]) -> bool:
    """Return True if the URL is a valid article candidate."""
    _BAD_PATHS = ("/authors/", "/author/", "/topics/", "/tag/", "/category/",
                  "/section/", "/search/", "/video/")
    if domain not in href:
        return False
    if any(p in href for p in _BAD_PATHS):
        return False
    if expected_date:
        url_dt = _url_date(href)
        if url_dt and abs((url_dt - expected_date).days) > 30:
            return False
    return True


def _search_brave(query: str, domain: str, expected_date: Optional[datetime]) -> Optional[str]:
    global _BRAVE_QUOTA_EXHAUSTED
    r = httpx.get(
        "https://api.search.brave.com/res/v1/web/search",
        params={"q": query, "count": 5, "search_lang": "en"},
        headers={"Accept": "application/json", "X-Subscription-Token": BRAVE_API_KEY},
        timeout=10,
    )
    if r.status_code == 402:
        _BRAVE_QUOTA_EXHAUSTED = True
        raise RuntimeError("quota exhausted (402) — disabling Brave for this session")
    r.raise_for_status()
    for res in r.json().get("web", {}).get("results", []):
        if _filter_url(res.get("url", ""), domain, expected_date):
            return res["url"]
    return None


def _search_serpapi(query: str, domain: str, expected_date: Optional[datetime]) -> Optional[str]:
    r = httpx.get(
        "https://serpapi.com/search.json",
        params={"engine": "google", "q": query, "num": 5, "api_key": SERPAPI_KEY},
        timeout=10,
    )
    r.raise_for_status()
    for res in r.json().get("organic_results", []):
        if _filter_url(res.get("link", ""), domain, expected_date):
            return res["link"]
    return None


def _search_serperdev(query: str, domain: str, expected_date: Optional[datetime]) -> Optional[str]:
    global _SERPER_QUOTA_EXHAUSTED
    r = httpx.post(
        "https://google.serper.dev/search",
        json={"q": query, "num": 5},
        headers={"X-API-KEY": SERPERDEV_KEY, "Content-Type": "application/json"},
        timeout=10,
    )
    if r.status_code == 400:
        _SERPER_QUOTA_EXHAUSTED = True
        raise RuntimeError(f"Serper quota/key error (400) — disabling for session")
    r.raise_for_status()
    for res in r.json().get("organic", []):
        if _filter_url(res.get("link", ""), domain, expected_date):
            return res["link"]
    return None


def _search_ddg(query: str, domain: str, expected_date: Optional[datetime]) -> Optional[str]:
    global _DDG_LAST_CALL
    elapsed = time.time() - _DDG_LAST_CALL
    if elapsed < DDG_MIN_INTERVAL:
        time.sleep(DDG_MIN_INTERVAL - elapsed)
    with DDGS() as d:
        results = list(d.text(f'site:{domain} "{query[:60]}"', max_results=5))
    _DDG_LAST_CALL = time.time()
    for r in results:
        if _filter_url(r.get("href", ""), domain, expected_date):
            return r["href"]
    return None


# Registry: (name, key_or_None, search_fn)
# Entries with no key are skipped; DDG has no key so always attempted last.
# SerpApi (serpapi.com) omitted — free tier rate-limits aggressively (429 on
# nearly every call), adding latency with no benefit since Serper.dev works.
# Re-add ("SerpApi", lambda: SERPAPI_KEY, _search_serpapi) if plan is upgraded.
_SEARCH_BACKENDS = [
    ("Brave",  lambda: BRAVE_API_KEY and not _BRAVE_QUOTA_EXHAUSTED, _search_brave),
    ("Serper", lambda: SERPERDEV_KEY and not _SERPER_QUOTA_EXHAUSTED, _search_serperdev),
    # DDG omitted: blocked on EC2 (AWS IPs rate-limited by DuckDuckGo/Yahoo)
]


def resolve_url(
    title: str, domain: str, expected_date: Optional[datetime] = None,
    gnews_link: Optional[str] = None,
) -> Optional[str]:
    """
    Find the real article URL. Tries in order:
      1. Construct URL directly from title slug (TOI, Reuters)
      2. Each configured search backend (Brave → SerpApi → Serper.dev → DDG)
    Backends with no API key are skipped automatically.
    """
    direct = _construct_url(title, domain, expected_date)
    if direct:
        try:
            r = httpx.get(direct, headers=HEADERS, timeout=10, follow_redirects=True)
            if r.status_code == 200 and len(r.text) > 500:
                console.print(f"    [dim green]URL via slug[/dim green]")
                return str(r.url)
            elif r.status_code in (401, 403, 429):
                # Paywalled/blocked — return URL so fetch_article_text can try Wayback.
                console.print(f"    [dim]Slug → HTTP {r.status_code}, will try Wayback[/dim]")
                return direct
            else:
                console.print(f"    [dim]Slug → HTTP {r.status_code}[/dim]")
        except Exception as e:
            console.print(f"    [dim]Slug failed: {type(e).__name__}[/dim]")

    query = f'site:{domain} {title[:80]}'
    for name, has_key, search_fn in _SEARCH_BACKENDS:
        if not has_key():
            continue
        try:
            url = search_fn(query, domain, expected_date)
            if url:
                console.print(f"    [dim green]URL via {name}[/dim green]")
                return url
            console.print(f"    [dim]{name}: no results[/dim]")
        except RuntimeError as e:
            console.print(f"    [yellow]{name}: {e}[/yellow]")
        except Exception as e:
            console.print(f"    [dim red]{name} error: {type(e).__name__}: {e}[/dim red]")
        finally:
            if name == "DDG":
                _DDG_LAST_CALL = time.time()

    console.print(f"    [red]resolve failed: '{title[:50]}' on {domain}[/red]")
    return None



PAYWALL_THRESHOLD = 500  # chars — below this, try Wayback Machine


async def _scrape_html(html: str) -> str:
    """Extract article body: trafilatura primary, BeautifulSoup fallback."""
    text = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=False,
        no_fallback=False,
        favor_precision=False,
    )
    if text and len(text) > 300:
        return text
    return extract_article_body(BeautifulSoup(html, "html.parser"))


async def _fetch_wayback(url: str, client: httpx.AsyncClient) -> str:
    """Try Wayback Machine for a cached copy of a paywalled/blocked URL."""
    try:
        wb_url = f"https://web.archive.org/web/{url}"
        r = await client.get(wb_url, timeout=20, follow_redirects=True)
        if r.status_code != 200:
            return ""
        return await _scrape_html(r.text)
    except Exception as e:
        console.print(f"    [dim]Wayback failed: {e}[/dim]")
        return ""


NEWSDATA_ARCHIVE_API = "https://newsdata.io/api/1/archive"
PAYWALL_SOURCES = frozenset({"bloomberg.com", "ft.com", "nytimes.com", "washingtonpost.com"})


async def search_newsdata(
    domain: str,
    keywords: List[str],
    start_date: datetime,
    end_date: datetime,
    max_results: int = 5,
) -> List[Dict]:
    """
    Query Newsdata.io /archive for articles from *domain* in the date window.
    Returns list of {url, published_at, headline, text}.
    Uses full_description (paid field) when available to bypass paywalls.
    """
    global _NEWSDATA_QUOTA_EXHAUSTED
    if not NEWSDATA_API_KEY or _NEWSDATA_QUOTA_EXHAUSTED:
        return []

    ascii_kws = [k.strip('"') for k in keywords if _is_ascii(k) and k.strip('"')]
    if not ascii_kws:
        return []

    # Strip leading www./news./en. so newsdata domainurl matching works
    bare_domain = re.sub(r"^(?:www|news|en)\.", "", domain)

    params: dict = {
        "apikey": NEWSDATA_API_KEY,
        "q": ascii_kws[0],
        "domainurl": bare_domain,
        "language": "en",
        "from_date": start_date.strftime("%Y-%m-%d"),
        "to_date": end_date.strftime("%Y-%m-%d"),
        "size": min(max_results * 3, 50),
    }

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(NEWSDATA_ARCHIVE_API, params=params)
        if r.status_code == 429:
            _NEWSDATA_QUOTA_EXHAUSTED = True
            console.print(f"    [yellow]Newsdata.io: quota exhausted[/yellow]")
            return []
        if r.status_code != 200:
            console.print(f"    [dim red]Newsdata.io: HTTP {r.status_code}[/dim red]")
            return []
        data = r.json()
        if data.get("status") != "success":
            code = (data.get("results") or {}).get("code", "")
            if code in ("AccessDenied", "RateLimitExceeded"):
                _NEWSDATA_QUOTA_EXHAUSTED = True
            console.print(f"    [dim red]Newsdata.io error: {code or data.get('status')}[/dim red]")
            return []
    except Exception as e:
        console.print(f"    [dim red]Newsdata.io error: {e}[/dim red]")
        return []

    items = data.get("results") or []
    console.print(f"    [dim]Newsdata.io: {len(items)} candidates[/dim]")

    results: List[Dict] = []
    for item in items:
        url = item.get("link") or ""
        if not url:
            continue
        pub_date = (item.get("pubDate") or "")[:10]
        headline = item.get("title") or ""
        # full_description is a paid-plan field with the full article text
        text = item.get("full_description") or item.get("description") or ""
        if len(text) < 250 and domain not in PAYWALL_SOURCES:
            fetched = await fetch_article_text(url)
            if fetched:
                text = fetched
        if len(text) < 250:
            continue
        results.append({
            "url": url,
            "published_at": pub_date or start_date.strftime("%Y-%m-%d"),
            "headline": headline,
            "text": text,
        })
        if len(results) >= max_results:
            break

    return results


GDELT_API = "https://api.gdeltproject.org/api/v2/doc/doc"
_GDELT_LAST_CALL: float = 0.0
_GDELT_CONSECUTIVE_ERRORS: int = 0
GDELT_MIN_INTERVAL = 10.0  # seconds between GDELT calls to avoid 429
GDELT_MAX_CONSECUTIVE_ERRORS = 5  # trip circuit breaker after this many consecutive failures


async def search_gdelt(
    domain: str,
    keywords: List[str],
    start_date: datetime,
    end_date: datetime,
    max_results: int = 5,
) -> List[Dict]:
    """
    Query GDELT DOC API for articles from *domain* in the date window.
    Free, no API key. English coverage only (weak for Hebrew domains).
    Returns list of {url, published_at, headline, text}.
    """
    global _GDELT_LAST_CALL, _GDELT_BLOCKED, _GDELT_CONSECUTIVE_ERRORS
    if _GDELT_BLOCKED:
        return []
    elapsed = time.time() - _GDELT_LAST_CALL
    if elapsed < GDELT_MIN_INTERVAL:
        await asyncio.sleep(GDELT_MIN_INTERVAL - elapsed)
    _GDELT_LAST_CALL = time.time()

    ascii_kws = [k.strip('"') for k in keywords if _is_ascii(k) and k.strip('"')]
    if not ascii_kws:
        return []

    query = f'domain:{domain} "{ascii_kws[0]}"'
    params = {
        "query": query,
        "mode": "artlist",
        "maxrecords": str(max_results * 3),  # fetch extra, filter by date
        "startdatetime": start_date.strftime("%Y%m%d") + "000000",
        "enddatetime": end_date.strftime("%Y%m%d") + "235959",
        "format": "json",
    }

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(GDELT_API, params=params)
        if r.status_code != 200:
            _GDELT_CONSECUTIVE_ERRORS += 1
            if _GDELT_CONSECUTIVE_ERRORS >= GDELT_MAX_CONSECUTIVE_ERRORS:
                _GDELT_BLOCKED = True
                console.print(f"    [yellow]GDELT: {_GDELT_CONSECUTIVE_ERRORS} consecutive errors — disabling for this session[/yellow]")
            else:
                console.print(f"    [dim red]GDELT error: HTTP {r.status_code}[/dim red]")
            return []
        data = r.json()
        _GDELT_CONSECUTIVE_ERRORS = 0  # reset on success
    except Exception as e:
        _GDELT_CONSECUTIVE_ERRORS += 1
        if _GDELT_CONSECUTIVE_ERRORS >= GDELT_MAX_CONSECUTIVE_ERRORS:
            _GDELT_BLOCKED = True
            console.print(f"    [yellow]GDELT: {_GDELT_CONSECUTIVE_ERRORS} consecutive errors — disabling for this session[/yellow]")
        else:
            console.print(f"    [dim red]GDELT error: {type(e).__name__}: {e}[/dim red]")
        return []

    articles_meta = data.get("articles", [])
    if not articles_meta:
        return []

    console.print(f"    [dim]GDELT: {len(articles_meta)} candidates[/dim]")
    results: List[Dict] = []
    async with httpx.AsyncClient(headers=HEADERS, timeout=25, follow_redirects=True) as client:
        for item in articles_meta:
            if len(results) >= max_results:
                break
            url = item.get("url", "")
            if not url or not _filter_url(url, domain, None):
                continue
            seendate = item.get("seendate", "")
            try:
                pub_dt = datetime.strptime(seendate[:8], "%Y%m%d")
            except Exception:
                pub_dt = start_date
            try:
                r = await client.get(url)
                if r.status_code != 200:
                    continue
                text = await _scrape_html(r.text)
                if len(text) < 300:
                    continue
                results.append({
                    "url": url,
                    "published_at": pub_dt.strftime("%Y-%m-%d"),
                    "headline": item.get("title", ""),
                    "text": text,
                })
                await asyncio.sleep(0.5)
            except Exception as e:
                console.print(f"    [dim]GDELT fetch failed: {e}[/dim]")
                continue

    return results


CDX_API = "http://web.archive.org/cdx/search/cdx"
CDX_FETCH_LIMIT = 15   # max candidates to actually fetch per cell
CDX_SCAN_LIMIT  = 150  # max CDX rows to download before filtering


async def search_wayback_cdx(
    domain: str,
    start_date: datetime,
    end_date: datetime,
    keywords: List[str],
    max_results: int = 5,
) -> List[Dict]:
    """
    Enumerate archived article URLs for *domain* in [start_date, end_date]
    via the Wayback CDX API, then fetch up to *max_results* articles from
    Wayback.  Used as a fallback when Google News RSS returns nothing.

    Returns list of {url, published_at, headline, text}.
    """
    # Strip leading subdomain (www., news., en.) so CDX matchType=domain
    # covers all subdomains. E.g. news.walla.co.il → walla.co.il
    cdx_domain = re.sub(r"^(?:www|news|en)\.", "", domain)
    params = [
        ("url",       cdx_domain),
        ("matchType", "domain"),
        ("output",    "json"),
        ("from",      start_date.strftime("%Y%m%d")),
        ("to",        end_date.strftime("%Y%m%d")),
        ("limit",     str(CDX_SCAN_LIMIT)),
        ("filter",    "mimetype:text/html"),
        ("filter",    "statuscode:200"),
        ("fl",        "original,timestamp"),
        ("collapse",  "urlkey"),
    ]

    try:
        async with httpx.AsyncClient(timeout=8) as client:
            r = await client.get(CDX_API, params=params)
        if r.status_code != 200:
            console.print(f"    [dim red]CDX error: HTTP {r.status_code}[/dim red]")
            return []
        rows = r.json()
    except Exception as e:
        console.print(f"    [dim red]CDX error: {type(e).__name__}: {e}[/dim red]")
        return []

    if len(rows) < 2:   # first row is header ["original","timestamp"]
        return []

    # ── Filter to article-like URLs ──────────────────────────────────────────
    candidates: List[tuple] = []
    for original, timestamp in rows[1:]:
        if not _filter_url(original, domain, None):
            continue
        try:
            pub_dt = datetime.strptime(timestamp[:8], "%Y%m%d")
        except ValueError:
            continue
        if not (start_date <= pub_dt <= end_date):
            continue
        candidates.append((original, timestamp, pub_dt))

    if not candidates:
        return []

    # ── Score by keyword presence in URL path ────────────────────────────────
    ascii_kws = [k.lower() for k in keywords if _is_ascii(k)]
    if ascii_kws:
        def _score(url: str) -> int:
            u = url.lower()
            return sum(1 for kw in ascii_kws if kw.replace(" ", "-") in u or kw.replace(" ", "") in u)
        candidates.sort(key=lambda x: -_score(x[0]))

    fetch_pool = candidates[:CDX_FETCH_LIMIT]
    console.print(f"    [dim]CDX: {len(candidates)} candidates → fetching {len(fetch_pool)}[/dim]")

    # ── Fetch from Wayback ───────────────────────────────────────────────────
    results: List[Dict] = []
    async with httpx.AsyncClient(headers=HEADERS, timeout=30, follow_redirects=True) as client:
        for original, timestamp, pub_dt in fetch_pool:
            if len(results) >= max_results:
                break
            wb_url = f"https://web.archive.org/web/{timestamp}/{original}"
            try:
                r = await client.get(wb_url)
                if r.status_code != 200:
                    continue
                html = r.text

                # Extract headline via trafilatura metadata
                meta = trafilatura.extract_metadata(html)
                headline = (meta.title or "") if meta else ""

                text = await _scrape_html(html)
                if len(text) < 300:
                    continue

                results.append({
                    "url":          original,
                    "published_at": pub_dt.strftime("%Y-%m-%d"),
                    "headline":     headline,
                    "text":         text,
                })
                await asyncio.sleep(0.5)
            except Exception as e:
                console.print(f"    [dim]CDX fetch failed: {e}[/dim]")
                continue

    return results


async def fetch_article_text(url: str) -> str:
    """
    Scrape full article text from URL.
    1. trafilatura (primary) + BeautifulSoup fallback
    2. If result < PAYWALL_THRESHOLD chars, or status was 403/401/429
       (paywall/block) → try Wayback Machine cached copy
    """
    try:
        async with httpx.AsyncClient(
            headers=HEADERS, timeout=25, follow_redirects=True
        ) as client:
            r = await client.get(url)
            if r.status_code in (401, 403, 429):
                console.print(f"    [dim]HTTP {r.status_code}, trying Wayback...[/dim]")
                wb_text = await _fetch_wayback(url, client)
                if _is_stub_page(wb_text):
                    console.print(f"    [dim]Wayback returned known stub page, skipping[/dim]")
                    return ""
                if len(wb_text) >= PAYWALL_THRESHOLD:
                    console.print(f"    [dim green]Wayback: {len(wb_text)} chars[/dim green]")
                return wb_text
            if r.status_code != 200:
                return ""
            text = await _scrape_html(r.text)

            if len(text) < PAYWALL_THRESHOLD:
                console.print(f"    [dim]Short article ({len(text)} chars), trying Wayback...[/dim]")
                wb_text = await _fetch_wayback(url, client)
                if len(wb_text) > len(text):
                    console.print(f"    [dim green]Wayback: {len(wb_text)} chars[/dim green]")
                    return wb_text

            return text
    except Exception as e:
        console.print(f"    [dim red]scrape error: {e}[/dim red]")
        return ""


async def ingest_cell(
    event: dict,
    source_id: str,
    raw_ingest_dir: Path,
    force: bool = False,
) -> int:
    """
    Fetch and save articles for one (event, source) cell.
    Returns number of articles saved.
    """
    cell_dir = raw_ingest_dir / source_id / event["id"]
    existing = existing_articles(cell_dir)
    if existing and not force:
        return len(existing)

    cfg = SOURCES_CONFIG.get(source_id)
    if not cfg:
        return 0
    domain = cfg["domain"]
    lang = cfg.get("lang", "en")

    outcome_dt = datetime.strptime(event["outcome_date"], "%Y-%m-%d")
    window = int(event.get("predictive_window_days", 14))
    start_dt = outcome_dt - timedelta(days=window)

    keywords = event.get("search_keywords", [])

    candidates = search_gnews_rss(
        domain=domain,
        keywords=keywords,
        start_date=start_dt,
        end_date=outcome_dt,
        lang=lang,
    )

    cell_dir.mkdir(parents=True, exist_ok=True)
    saved = 0

    for art in candidates[:5]:
        # ── GNews path: resolve URL then scrape ──────────────────────────────
        expected_dt = datetime.strptime(art["published_at"], "%Y-%m-%d")
        loop = asyncio.get_event_loop()
        url = await loop.run_in_executor(
            None, resolve_url, art["title"], domain, expected_dt, art.get("gnews_link")
        )
        if not url:
            console.print(f"    [dim]No URL for '{art['title'][:50]}'[/dim]")
            continue

        await asyncio.sleep(1.0)
        text = await fetch_article_text(url)
        if len(text) < 250:
            continue

        article = {
            "headline": art["title"],
            "text":     text,
            "published_at": art["published_at"],
            "author":   "Unknown",
            "url":      url,
        }
        out = cell_dir / f"article_{saved+1:02d}.json"
        out.write_text(json.dumps(article, indent=2, ensure_ascii=False))
        saved += 1

    if saved == 0:
        # ── Web search fallback: Serper.dev → Brave → DDG ────────────────────
        # Direct news search for the event keywords restricted to the source
        # domain. Runs synchronously (no URL resolution step needed — results
        # include the real article URL already).
        console.print(f"    [dim]Web search fallback for {source_id}/{event['id']}[/dim]")
        ascii_kws = [k.strip('"') for k in keywords if _is_ascii(k) and k.strip('"')]
        if ascii_kws:
            ws_query = f"site:{domain} {ascii_kws[0]}"
            loop = asyncio.get_event_loop()
            ws_results = await loop.run_in_executor(
                None, lambda: _web_search(ws_query, 5, start_dt, outcome_dt)
            )
            for ws in ws_results:
                if not ws.url or not _filter_url(ws.url, domain, None):
                    continue
                await asyncio.sleep(1.0)
                text = await fetch_article_text(ws.url)
                if len(text) < 250:
                    continue
                article = {
                    "headline": ws.title,
                    "text":     text,
                    "published_at": ws.published_date or start_dt.strftime("%Y-%m-%d"),
                    "author":   "Unknown",
                    "url":      ws.url,
                }
                out = cell_dir / f"article_{saved+1:02d}.json"
                out.write_text(json.dumps(article, indent=2, ensure_ascii=False))
                saved += 1

    if saved == 0:
        # ── Newsdata.io /archive fallback: paid, domain-filtered, historical ──
        # Preferred over GDELT for paywalled sources (bloomberg, ft, nyt, wapost)
        # because newsdata may carry full_description bypassing the paywall.
        console.print(f"    [dim]Newsdata.io fallback for {source_id}/{event['id']}[/dim]")
        nd_arts = await search_newsdata(domain, keywords, start_dt, outcome_dt)
        for art in nd_arts:
            article = {
                "headline": art["headline"],
                "text":     art["text"],
                "published_at": art["published_at"],
                "author":   "Unknown",
                "url":      art["url"],
            }
            out = cell_dir / f"article_{saved+1:02d}.json"
            out.write_text(json.dumps(article, indent=2, ensure_ascii=False))
            saved += 1

    if saved == 0:
        # ── GDELT fallback: free DOC API, returns real article URLs directly ──
        # Note: search_gdelt() filters to ASCII keywords internally, so this
        # works for Hebrew-domain sources too (all events have English keywords).
        console.print(f"    [dim]GDELT fallback for {source_id}/{event['id']}[/dim]")
        gdelt_arts = await search_gdelt(domain, keywords, start_dt, outcome_dt)
        for art in gdelt_arts:
            article = {
                "headline": art["headline"],
                "text":     art["text"],
                "published_at": art["published_at"],
                "author":   "Unknown",
                "url":      art["url"],
            }
            out = cell_dir / f"article_{saved+1:02d}.json"
            out.write_text(json.dumps(article, indent=2, ensure_ascii=False))
            saved += 1

    if saved == 0:
        # ── CDX fallback: enumerate Wayback archives ─────────────────────────
        # Disabled: Wayback CDX silently drops connections from AWS EC2 IPs
        # (accepts TCP/HTTP but returns 0 bytes on domain+date queries → 30s timeout per cell).
        # Re-enable by setting ENABLE_CDX=1 in environment.
        if not os.environ.get("ENABLE_CDX"):
            return 0
        console.print(f"    [dim]CDX fallback for {source_id}/{event['id']}[/dim]")
        cdx_arts = await search_wayback_cdx(domain, start_dt, outcome_dt, keywords)
        for art in cdx_arts:
            article = {
                "headline": art["headline"],
                "text":     art["text"],
                "published_at": art["published_at"],
                "author":   "Unknown",
                "url":      art["url"],
            }
            out = cell_dir / f"article_{saved+1:02d}.json"
            out.write_text(json.dumps(article, indent=2, ensure_ascii=False))
            saved += 1

    return saved


async def run_batch(
    data_dir: Path,
    event_ids: List[str],
    source_filter: Optional[List[str]],
    force: bool,
    max_concurrent: int = 3,
):
    events_dir = data_dir / "events"
    raw_ingest_dir = data_dir / "raw_ingest"

    events = {}
    for eid in event_ids:
        p = events_dir / f"{eid}.json"
        if p.exists():
            events[eid] = json.load(open(p))
        else:
            console.print(f"[yellow]Event {eid} not found, skipping[/yellow]")

    sources = [s for s in SOURCES_CONFIG if source_filter is None or s in source_filter]
    total = len(events) * len(sources)

    console.print(
        f"\n[bold]Google News + DDG Batch Ingest[/bold]  "
        f"{len(events)} events × {len(sources)} sources = {total} cells  "
        f"[dim](max_concurrent={max_concurrent})[/dim]\n"
    )

    results: dict[str, dict] = {eid: {} for eid in events}
    state = load_state()
    sem = asyncio.Semaphore(max_concurrent)

    # Build work queue — skip cells that are already done
    work: list[tuple[str, dict, str]] = []
    skipped = 0
    for eid, event in events.items():
        for sid in sources:
            cell = state.get(eid, sid)
            if not force and cell.status == CellStatus.done:
                results[eid][sid] = 0
                skipped += 1
                continue
            if not force and cell.status == CellStatus.no_predictions:
                cell_dir = raw_ingest_dir / sid / eid
                if cell_dir.exists() and any(cell_dir.glob("article_*.json")):
                    results[eid][sid] = 0
                    skipped += 1
                    continue
            work.append((eid, event, sid))

    console.print(f"  {len(work)} cells to fetch, {skipped} skipped (done/no_pred)")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Fetching…", total=len(work))

        async def _fetch(eid: str, event: dict, sid: str) -> None:
            async with sem:
                count = await ingest_cell(event, sid, raw_ingest_dir, force)
                results[eid][sid] = count
                label = f"[green]{count} art[/green]" if count else "[dim]0[/dim]"
                console.print(f"  {eid}/{sid}: {label}")
                cell = state.get(eid, sid)
                if count > 0 and cell.status == CellStatus.no_predictions:
                    update_cell(eid, sid, CellStatus.pending)
                progress.advance(task)

        await asyncio.gather(*[_fetch(eid, ev, sid) for eid, ev, sid in work])

    # Summary table
    table = Table(title="Ingest Summary")
    table.add_column("Event", style="cyan")
    for sid in sources:
        table.add_column(sid, justify="center")
    table.add_column("Total", justify="right", style="bold")

    grand = 0
    for eid in event_ids:
        if eid not in results:
            continue
        row = [eid]
        t = 0
        for sid in sources:
            n = results[eid].get(sid, 0)
            row.append(str(n) if n else "·")
            t += n
        row.append(str(t))
        table.add_row(*row)
        grand += t

    console.print(table)
    console.print(f"\n[bold green]Total articles saved: {grand}[/bold green]")
    console.print(
        f"Next: [bold]DATA_DIR={data_dir} uv run python -m tm.orchestrator local_file[/bold]"
    )


def main():
    import argparse
    p = argparse.ArgumentParser(description="Google News RSS + DDG batch ingestor")
    p.add_argument("--events", nargs="+", default=MVP_EVENTS)
    p.add_argument("--sources", nargs="+", default=None)
    p.add_argument("--force", action="store_true", help="Re-fetch already populated cells")
    p.add_argument("--max-concurrent", type=int, default=3,
                   help="Max concurrent event×source cells (default: 3)")
    args = p.parse_args()

    data_dir = Path(os.environ.get("DATA_DIR", "/app/data"))
    # If no explicit --events, discover all event files on disk (not just MVP_EVENTS)
    event_ids = args.events if args.events != MVP_EVENTS else sorted(
        p.stem for p in (data_dir / "events").glob("*.json")
    ) or MVP_EVENTS
    asyncio.run(run_batch(data_dir, event_ids, args.sources, args.force, args.max_concurrent))


if __name__ == "__main__":
    main()
