"""
Multi-provider news search with fallback chain.
Python equivalent of daatan's webSearch.ts utility.

Fallback order:
  1. GDELT Doc API                  (free, no key — primary; news-only, reliable dates)
  2. SerpAPI (serpapi.com)          SERPAPI_API_KEY
  3. Serper.dev /news endpoint      SERPER_API_KEY
  4. Brave News Search              BRAVE_API_KEY
  5. BrightData SERP API            BRIGHTDATA_API_KEY
  6. Nimbleway SERP API             NIMBLEWAY_API_KEY
  7. ScrapingBee Google Search      SCRAPINGBEE_API_KEY
  8. Newsdata.io                    NEWSDATA_API_KEY
  9. DataForSEO Google News         DATAFORSEO_API_KEY  (last-resort paid fallback)
 10. DuckDuckGo Lite                (free, no key)

All keys are loaded from the environment first, then from AWS Secrets Manager
(openclaw/* namespace) as a fallback. See _secret().

Usage:
    from tm.web_search import search_articles

    results = search_articles(
        "Gaza ceasefire negotiations site:timesofisrael.com",
        limit=5,
        date_from=datetime(2024, 1, 1),
        date_to=datetime(2024, 1, 14),
    )
    for r in results:
        print(r.url, r.title)
"""

import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import List, Optional
from urllib.parse import urlencode

import httpx
from ddgs import DDGS

try:
    import trafilatura as _trafilatura
    from bs4 import BeautifulSoup as _BeautifulSoup
    _SNIPPET_LIBS_AVAILABLE = True
except ImportError:
    _SNIPPET_LIBS_AVAILABLE = False

logger = logging.getLogger(__name__)

# Thread-local: stores the provider name that served the last search_articles()
# call in this thread, and the full chain of providers attempted.
# Read via get_last_search_provider() / get_last_search_provider_chain() after the call returns.
_provider_local = threading.local()


def get_last_search_provider() -> str:
    """Return the provider that served the most recent search_articles() call in this thread."""
    return getattr(_provider_local, "name", "none")


def get_last_search_provider_chain() -> list[str]:
    """Return the ordered list of providers attempted in the most recent search_articles() call."""
    return list(getattr(_provider_local, "chain", []))


def _secret(env_var: str, secret_name: str) -> Optional[str]:
    """Return env var if set, otherwise fetch from AWS Secrets Manager."""
    val = os.environ.get(env_var)
    if val:
        return val
    try:
        import boto3
        client = boto3.client("secretsmanager", region_name="eu-central-1")
        val = client.get_secret_value(SecretId=secret_name)["SecretString"].strip()
        logger.info("Loaded %s from Secrets Manager", secret_name)
        return val
    except Exception as e:
        logger.debug("Could not load %s from Secrets Manager: %s", secret_name, e)
        return None


DATAFORSEO_API_KEY: Optional[str] = _secret("DATAFORSEO_API_KEY", "openclaw/dataforseo-key")
SERPAPI_API_KEY: Optional[str] = _secret("SERPAPI_API_KEY", "openclaw/serpapi-key")
SERPER_API_KEY: Optional[str] = _secret("SERPER_API_KEY", "openclaw/serperdev-key")
BRAVE_API_KEY: Optional[str] = _secret("BRAVE_API_KEY", "openclaw/brave-api-key")
BRIGHTDATA_API_KEY: Optional[str] = _secret("BRIGHTDATA_API_KEY", "openclaw/brightdata-api-key")
NIMBLEWAY_API_KEY: Optional[str] = _secret("NIMBLEWAY_API_KEY", "openclaw/nimbleway-api-key")
SCRAPINGBEE_API_KEY: Optional[str] = _secret("SCRAPINGBEE_API_KEY", "openclaw/scrapingbee-api-key")
NEWSDATA_API_KEY: Optional[str] = _secret("NEWSDATA_API_KEY", "openclaw/newsdata-api-key")

_KEY_LOADED_AT: float = time.time()
_KEY_MAX_AGE_SECONDS: float = 86400.0  # 24h


def _refresh_keys_if_stale() -> None:
    """Re-fetch all search API keys from Secrets Manager if >24h old.

    Keys are loaded once at module import. Long-running processes (the batch
    pipeline can run for days) would use stale keys after rotation. This
    function is called at the top of search_articles() to catch that case.
    """
    global DATAFORSEO_API_KEY, SERPAPI_API_KEY, SERPER_API_KEY, BRAVE_API_KEY
    global BRIGHTDATA_API_KEY, NIMBLEWAY_API_KEY, SCRAPINGBEE_API_KEY, NEWSDATA_API_KEY
    global _KEY_LOADED_AT
    if time.time() - _KEY_LOADED_AT < _KEY_MAX_AGE_SECONDS:
        return
    logger.info("Refreshing search API keys from Secrets Manager (>24h since last fetch)")
    DATAFORSEO_API_KEY = _secret("DATAFORSEO_API_KEY", "openclaw/dataforseo-key")
    SERPAPI_API_KEY = _secret("SERPAPI_API_KEY", "openclaw/serpapi-key")
    SERPER_API_KEY = _secret("SERPER_API_KEY", "openclaw/serperdev-key")
    BRAVE_API_KEY = _secret("BRAVE_API_KEY", "openclaw/brave-api-key")
    BRIGHTDATA_API_KEY = _secret("BRIGHTDATA_API_KEY", "openclaw/brightdata-api-key")
    NIMBLEWAY_API_KEY = _secret("NIMBLEWAY_API_KEY", "openclaw/nimbleway-api-key")
    SCRAPINGBEE_API_KEY = _secret("SCRAPINGBEE_API_KEY", "openclaw/scrapingbee-api-key")
    NEWSDATA_API_KEY = _secret("NEWSDATA_API_KEY", "openclaw/newsdata-api-key")
    _KEY_LOADED_AT = time.time()



_DDG_LAST_CALL: float = 0.0
DDG_MIN_INTERVAL = 2.0
_GDELT_LAST_CALL: float = 0.0
GDELT_MIN_INTERVAL = 10.0  # GDELT rate limit: 1 req / 10s

_DATAFORSEO_QUOTA_EXHAUSTED: bool = False
_SERPAPI_QUOTA_EXHAUSTED: bool = False
_BRAVE_QUOTA_EXHAUSTED: bool = False
_SERPER_QUOTA_EXHAUSTED: bool = False
_BRIGHTDATA_QUOTA_EXHAUSTED: bool = False
_NIMBLEWAY_QUOTA_EXHAUSTED: bool = False
_SCRAPINGBEE_QUOTA_EXHAUSTED: bool = False
_NEWSDATA_QUOTA_EXHAUSTED: bool = False


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    source: str = ""
    published_date: str = ""
    _prefetched_text: Optional[str] = field(default=None)


# ──────────────────────────────────────────────
# Date helpers shared across providers
# ──────────────────────────────────────────────

def _date_query_suffix(date_from: Optional[datetime], date_to: Optional[datetime]) -> str:
    """Append Google after:/before: operators for providers without native date params."""
    parts = []
    if date_from:
        parts.append(f"after:{date_from.strftime('%Y-%m-%d')}")
    if date_to:
        parts.append(f"before:{date_to.strftime('%Y-%m-%d')}")
    return (" " + " ".join(parts)) if parts else ""


def _filter_by_date(
    results: List["SearchResult"],
    date_from: Optional[datetime],
    date_to: Optional[datetime],
) -> List["SearchResult"]:
    """Post-filter results to the requested window using published_date.
    Entries without a parseable date are kept (benefit of the doubt).
    """
    if not date_from and not date_to:
        return results
    out = []
    for r in results:
        if not r.published_date:
            out.append(r)
            continue
        try:
            d = datetime.strptime(r.published_date[:10], "%Y-%m-%d")
            if date_from and d < date_from:
                continue
            if date_to and d > date_to:
                continue
            out.append(r)
        except ValueError:
            out.append(r)
    return out


# ──────────────────────────────────────────────
# Provider: DataForSEO Google News
# ──────────────────────────────────────────────

def _search_dataforseo(
    query: str,
    limit: int,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
) -> List[SearchResult]:
    global _DATAFORSEO_QUOTA_EXHAUSTED
    if not DATAFORSEO_API_KEY:
        raise RuntimeError("DATAFORSEO_API_KEY not set")

    task: dict = {
        "keyword": query,
        "language_code": "en",
        "location_code": 2840,  # United States
        "depth": min(limit, 100),
    }
    if date_from:
        task["date_from"] = date_from.strftime("%Y-%m-%d")
    if date_to:
        task["date_to"] = date_to.strftime("%Y-%m-%d")

    r = httpx.post(
        "https://api.dataforseo.com/v3/serp/google/news/live/advanced",
        headers={
            "Authorization": f"Basic {DATAFORSEO_API_KEY}",
            "Content-Type": "application/json",
        },
        json=[task],
        timeout=20,
    )
    r.raise_for_status()
    data = r.json()
    api_task = (data.get("tasks") or [{}])[0]
    task_code = api_task.get("status_code", 0)
    # 40101 = account suspended, 40201/40202/40203 = insufficient funds / billing
    if task_code in (40101, 40201, 40202, 40203):
        _DATAFORSEO_QUOTA_EXHAUSTED = True
        raise RuntimeError(f"DataForSEO billing/quota error: status_code={task_code}")
    items = ((api_task.get("result") or [{}])[0].get("items")) or []
    results = []
    for item in items[:limit]:
        url = item.get("url", "")
        if not url:
            continue
        source = (item.get("source") or {}).get("name", "") or item.get("domain", "")
        # DataForSEO returns "timestamp" (e.g. "2024-09-20 14:32:00 +00:00"), not "date_published"
        raw_date = item.get("timestamp") or item.get("date_published", "")
        published_date = raw_date[:10] if raw_date else ""
        results.append(SearchResult(
            title=item.get("title", ""),
            url=url,
            snippet=item.get("snippet", ""),
            source=source,
            published_date=published_date,
        ))
    # DataForSEO date_from/date_to params are not reliably enforced by Google News;
    # post-filter to ensure we stay within the requested window.
    return _filter_by_date(results, date_from, date_to)


# ──────────────────────────────────────────────
# Provider: SerpAPI (serpapi.com) news
# ──────────────────────────────────────────────

def _search_serpapi_news(
    query: str,
    limit: int,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
) -> List[SearchResult]:
    global _SERPAPI_QUOTA_EXHAUSTED
    if not SERPAPI_API_KEY:
        raise RuntimeError("SERPAPI_API_KEY not set")

    # SerpAPI news (tbm=nws) doesn't support site: operator — strip it.
    # Results are filtered by domain by the caller anyway.
    import re as _re
    clean_query = _re.sub(r"\bsite:\S+\s*", "", query).strip()

    params: dict = {
        "q": clean_query,
        "tbm": "nws",
        "num": min(limit, 100),
        "api_key": SERPAPI_API_KEY,
    }
    if date_from:
        params["tbs"] = f"cdr:1,cd_min:{date_from.month}/{date_from.day}/{date_from.year}"
        if date_to:
            params["tbs"] += f",cd_max:{date_to.month}/{date_to.day}/{date_to.year}"

    r = httpx.get(
        "https://serpapi.com/search.json",
        params=params,
        timeout=12,
    )
    if r.status_code == 429:
        body = r.text.lower()
        if "run out" in body or "quota" in body or "searches" in body:
            _SERPAPI_QUOTA_EXHAUSTED = True
            raise RuntimeError("SerpAPI quota exhausted")
        raise RuntimeError("SerpAPI rate-limited (429)")
    r.raise_for_status()
    items = r.json().get("news_results", [])

    results = [
        SearchResult(
            title=item.get("title", ""),
            url=item.get("link", ""),
            snippet=item.get("snippet", ""),
            source=item.get("source", _extract_domain(item.get("link", ""))),
            published_date=item.get("date", ""),
        )
        for item in items[:limit]
        if item.get("link")
    ]
    return _filter_by_date(results, date_from, date_to)


# ──────────────────────────────────────────────
# Provider: Serper.dev /news
# ──────────────────────────────────────────────

def _search_serper_news(
    query: str,
    limit: int,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
) -> List[SearchResult]:
    global _SERPER_QUOTA_EXHAUSTED
    if not SERPER_API_KEY:
        raise RuntimeError("SERPER_API_KEY not set")

    body: dict = {"q": query, "num": limit}
    if date_from and date_to:
        # Serper tbs date range: cdr:1,cd_min:M/D/YYYY,cd_max:M/D/YYYY
        def _fmt(dt: datetime) -> str:
            return f"{dt.month}/{dt.day}/{dt.year}"
        body["tbs"] = f"cdr:1,cd_min:{_fmt(date_from)},cd_max:{_fmt(date_to)}"

    r = httpx.post(
        "https://google.serper.dev/news",
        json=body,
        headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
        timeout=10,
    )
    if r.status_code == 400 and "credits" in r.text.lower():
        _SERPER_QUOTA_EXHAUSTED = True
        raise RuntimeError("Serper quota exhausted (no credits)")
    r.raise_for_status()
    data = r.json()
    items = data.get("news", [])

    results = [
        SearchResult(
            title=item.get("title", ""),
            url=item.get("link", ""),
            snippet=item.get("snippet", ""),
            source=_extract_domain(item.get("link", "")),
            published_date=item.get("date", ""),
        )
        for item in items[:limit]
        if item.get("link")
    ]
    return _filter_by_date(results, date_from, date_to)


# ──────────────────────────────────────────────
# Provider: Brave News Search
# ──────────────────────────────────────────────

def _search_brave_news(
    query: str,
    limit: int,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
) -> List[SearchResult]:
    global _BRAVE_QUOTA_EXHAUSTED
    if not BRAVE_API_KEY:
        raise RuntimeError("BRAVE_API_KEY not set")

    # Brave has no arbitrary date-range param; inject Google-style operators into query.
    dated_query = query + _date_query_suffix(date_from, date_to)
    params: dict = {
        "q": dated_query,
        "count": min(limit, 20),
        "search_lang": "en",
        "country": "us",
    }

    r = httpx.get(
        "https://api.search.brave.com/res/v1/news/search",
        params=params,
        headers={
            "Accept": "application/json",
            "X-Subscription-Token": BRAVE_API_KEY,
        },
        timeout=10,
    )
    if r.status_code == 402:
        _BRAVE_QUOTA_EXHAUSTED = True
        raise RuntimeError("Brave quota exhausted (402)")
    r.raise_for_status()

    items = r.json().get("results", [])
    results = [
        SearchResult(
            title=item.get("title", ""),
            url=item.get("url", ""),
            snippet=item.get("description", ""),
            source=item.get("meta_url", {}).get("hostname", _extract_domain(item.get("url", ""))),
            published_date=item.get("age", ""),
        )
        for item in items[:limit]
        if item.get("url")
    ]
    return _filter_by_date(results, date_from, date_to)


# ──────────────────────────────────────────────
# Provider: BrightData SERP API
# ──────────────────────────────────────────────

def _search_brightdata(
    query: str,
    limit: int,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
) -> List[SearchResult]:
    global _BRIGHTDATA_QUOTA_EXHAUSTED
    if not BRIGHTDATA_API_KEY:
        raise RuntimeError("BRIGHTDATA_API_KEY not set")

    # Inject Google date operators since BrightData proxies Google.
    # NOTE: zone "serp_api1" returns empty body (zone not configured); keeping the
    # implementation correct for when the zone is fixed.
    dated_query = query + _date_query_suffix(date_from, date_to)
    search_url = "https://www.google.com/search?" + urlencode({"q": dated_query, "gl": "us", "hl": "en"})
    r = httpx.post(
        "https://api.brightdata.com/request",
        json={"zone": "serp_api1", "url": search_url, "format": "raw"},
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {BRIGHTDATA_API_KEY}"},
        timeout=20,
    )
    if r.status_code in (401, 402):
        _BRIGHTDATA_QUOTA_EXHAUSTED = True
        raise RuntimeError(f"BrightData quota/auth error ({r.status_code})")
    r.raise_for_status()

    html = r.text
    if not html:
        return []

    # Google's CSS classes rotate; try several known variants
    for title_cls in ["LC20lb", "DKV0Md", "vvjwJb"]:
        result_pat = re.compile(
            rf'href="(https://[^"#]+)"[^>]*>[^<]*<h3[^>]*class="{title_cls}[^"]*">([^<]+)</h3>'
        )
        pairs = [(m.group(1), m.group(2)) for m in result_pat.finditer(html)]
        if pairs:
            break

    snippet_pat = re.compile(r'class="(?:VwiC3b|yXK7lf|MUxGbd)[^"]*"[^>]*>(.*?)</div>')
    snippets = [re.sub(r'<[^>]+>', '', m.group(1)).strip() for m in snippet_pat.finditer(html)]

    results = [
        SearchResult(
            title=title,
            url=url,
            snippet=snippets[i] if i < len(snippets) else "",
            source=_extract_domain(url),
        )
        for i, (url, title) in enumerate(pairs[:limit])
    ]
    return _filter_by_date(results, date_from, date_to)


# ──────────────────────────────────────────────
# Provider: Nimbleway SERP API
# ──────────────────────────────────────────────

def _search_nimbleway(
    query: str,
    limit: int,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
) -> List[SearchResult]:
    global _NIMBLEWAY_QUOTA_EXHAUSTED
    if not NIMBLEWAY_API_KEY:
        raise RuntimeError("NIMBLEWAY_API_KEY not set")

    dated_query = query + _date_query_suffix(date_from, date_to)
    r = httpx.post(
        "https://api.webit.live/api/v1/realtime/serp",
        json={"search_engine": "google_search", "country": "US", "query": dated_query, "parse": True},
        headers={"Authorization": f"Bearer {NIMBLEWAY_API_KEY}", "Content-Type": "application/json"},
        timeout=20,
    )
    if r.status_code == 402:
        _NIMBLEWAY_QUOTA_EXHAUSTED = True
        raise RuntimeError("Nimbleway quota exhausted (402)")
    r.raise_for_status()

    data = r.json()
    if data.get("status") != "success":
        raise RuntimeError(f"Nimbleway error: {data.get('status')}")

    items = data.get("parsing", {}).get("entities", {}).get("OrganicResult", [])
    results = [
        SearchResult(
            title=item.get("title", ""),
            url=item.get("url", ""),
            snippet=item.get("snippet", ""),
            source=item.get("cleaned_domain") or _extract_domain(item.get("url", "")),
        )
        for item in items[:limit]
        if item.get("url")
    ]
    return _filter_by_date(results, date_from, date_to)


# ──────────────────────────────────────────────
# Provider: ScrapingBee Google Search
# ──────────────────────────────────────────────

def _search_scrapingbee(
    query: str,
    limit: int,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
) -> List[SearchResult]:
    global _SCRAPINGBEE_QUOTA_EXHAUSTED
    if not SCRAPINGBEE_API_KEY:
        raise RuntimeError("SCRAPINGBEE_API_KEY not set")

    dated_query = query + _date_query_suffix(date_from, date_to)
    r = httpx.get(
        "https://app.scrapingbee.com/api/v1/store/google",
        params={"api_key": SCRAPINGBEE_API_KEY, "search": dated_query, "nb_results": limit},
        timeout=20,
    )
    if r.status_code == 402:
        _SCRAPINGBEE_QUOTA_EXHAUSTED = True
        raise RuntimeError("ScrapingBee quota exhausted (402)")
    if r.status_code == 401:
        _SCRAPINGBEE_QUOTA_EXHAUSTED = True
        raise RuntimeError("ScrapingBee unauthorized (401) — invalid key, disabling for session")
    r.raise_for_status()

    data = r.json()
    items = (
        data.get("news_results")
        or data.get("top_stories")
        or data.get("organic_results")
        or []
    )
    results = [
        SearchResult(
            title=item.get("title", ""),
            url=item.get("url", ""),
            snippet=item.get("description", ""),
            source=item.get("domain") or _extract_domain(item.get("url", "")),
            published_date=item.get("date_utc") or item.get("date") or "",
        )
        for item in items[:limit]
        if item.get("url")
    ]
    return _filter_by_date(results, date_from, date_to)


# ──────────────────────────────────────────────
# Provider: GDELT Doc API (free, reliable dates)
# ──────────────────────────────────────────────

def _search_gdelt(
    query: str,
    limit: int,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
) -> List[SearchResult]:
    global _GDELT_LAST_CALL

    # Translate Google-style site: operator to GDELT domain: filter
    gdelt_query = re.sub(r"site:(\S+)", r"domain:\1", query)

    params: dict = {
        "query": gdelt_query,
        "mode": "artlist",
        "format": "json",
        "maxrecords": min(limit, 25),
        "sort": "DateDesc",
    }
    if date_from:
        params["startdatetime"] = date_from.strftime("%Y%m%d000000")
    if date_to:
        params["enddatetime"] = date_to.strftime("%Y%m%d235959")

    last_err: Exception = RuntimeError("GDELT: no attempts made")
    for attempt in range(3):
        if attempt:
            time.sleep(5 * attempt)  # 5 s, 10 s between retries
        elapsed = time.time() - _GDELT_LAST_CALL
        if elapsed < GDELT_MIN_INTERVAL:
            time.sleep(GDELT_MIN_INTERVAL - elapsed)
        try:
            r = httpx.get(
                "https://api.gdeltproject.org/api/v2/doc/doc",
                params=params,
                # Explicit sub-timeouts: 5 s connect caps SSL handshake (was 20 s
                # scalar which allowed 80+ s hangs and consumed gunicorn budget).
                timeout=httpx.Timeout(connect=5.0, read=10.0, write=5.0, pool=5.0),
            )
        except Exception as e:
            last_err = e
            continue
        finally:
            _GDELT_LAST_CALL = time.time()

        if r.status_code != 200:
            last_err = RuntimeError(f"GDELT returned HTTP {r.status_code}")
            continue
        if not r.text.strip():
            last_err = RuntimeError("GDELT returned empty body")
            continue
        try:
            data = r.json()
        except Exception as e:
            last_err = RuntimeError(f"GDELT returned non-JSON body: {e}")
            continue

        articles = data.get("articles") or []
        results = []
        for art in articles:
            pub = art.get("seendate", "")
            try:
                pub_str = datetime.strptime(pub[:8], "%Y%m%d").strftime("%Y-%m-%d")
            except ValueError:
                pub_str = ""
            results.append(SearchResult(
                title=art.get("title", ""),
                url=art.get("url", ""),
                snippet="",  # GDELT artlist mode returns no body text
                source=art.get("domain", _extract_domain(art.get("url", ""))),
                published_date=pub_str,
            ))
        # GDELT occasionally leaks ±1 day past enddatetime; post-filter to be strict.
        return _filter_by_date(results[:limit], date_from, date_to)

    raise last_err


# ──────────────────────────────────────────────
# Provider: DuckDuckGo Lite (free fallback)
# ──────────────────────────────────────────────

def _search_ddg_news(
    query: str,
    limit: int,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
) -> List[SearchResult]:
    global _DDG_LAST_CALL
    elapsed = time.time() - _DDG_LAST_CALL
    if elapsed < DDG_MIN_INTERVAL:
        time.sleep(DDG_MIN_INTERVAL - elapsed)

    # DDG has no arbitrary date range; inject after:/before: operators (Bing honours them).
    dated_query = query + _date_query_suffix(date_from, date_to)
    results = []
    with DDGS() as d:
        for item in d.text(dated_query, max_results=limit):
            results.append(
                SearchResult(
                    title=item.get("title", ""),
                    url=item.get("href", ""),
                    snippet=item.get("body", ""),
                    source=_extract_domain(item.get("href", "")),
                    published_date=item.get("published", "")[:10] if item.get("published") else "",
                )
            )
    _DDG_LAST_CALL = time.time()
    return _filter_by_date(results[:limit], date_from, date_to)


# ──────────────────────────────────────────────
# Provider: Newsdata.io
# ──────────────────────────────────────────────

def _search_newsdata_io(
    query: str,
    limit: int,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
) -> List[SearchResult]:
    global _NEWSDATA_QUOTA_EXHAUSTED
    if not NEWSDATA_API_KEY:
        raise RuntimeError("NEWSDATA_API_KEY not set")

    params: dict = {
        "apikey": NEWSDATA_API_KEY,
        "q": query,
        "language": "en",
        "size": min(limit, 50),
    }
    if date_from:
        params["from_date"] = date_from.strftime("%Y-%m-%d")
    if date_to:
        params["to_date"] = date_to.strftime("%Y-%m-%d")

    r = httpx.get("https://newsdata.io/api/1/latest", params=params, timeout=10)
    if r.status_code == 429:
        _NEWSDATA_QUOTA_EXHAUSTED = True
        raise RuntimeError("Newsdata.io quota exhausted (rate limit)")
    r.raise_for_status()
    data = r.json()
    if data.get("status") != "success":
        code = (data.get("results") or {}).get("code", "")
        if code in ("AccessDenied", "RateLimitExceeded"):
            _NEWSDATA_QUOTA_EXHAUSTED = True
        raise RuntimeError(f"Newsdata.io error: {data}")

    items = data.get("results") or []
    results = [
        SearchResult(
            title=item.get("title") or "",
            url=item.get("link") or "",
            snippet=item.get("description") or "",
            source=item.get("source_id") or _extract_domain(item.get("link") or ""),
            published_date=(item.get("pubDate") or "")[:10],
        )
        for item in items[:limit]
        if item.get("link")
    ]
    return _filter_by_date(results, date_from, date_to)


# ──────────────────────────────────────────────
# Snippet enrichment (opt-in scraping for GDELT results)
# ──────────────────────────────────────────────

_SNIPPET_UA = "Mozilla/5.0 (compatible; TruthMachine/1.0)"
_SNIPPET_TIMEOUT = httpx.Timeout(connect=3.0, read=7.0, write=3.0, pool=3.0)


def _fetch_snippet(url: str) -> str:
    """Fetch *url* and return first ~600 chars of article text. Returns '' on any failure."""
    if not _SNIPPET_LIBS_AVAILABLE:
        return ""
    try:
        r = httpx.get(url, headers={"User-Agent": _SNIPPET_UA},
                      timeout=_SNIPPET_TIMEOUT, follow_redirects=True)
        if r.status_code != 200 or "text/html" not in r.headers.get("content-type", ""):
            return ""
        text = _trafilatura.extract(r.text, include_comments=False, include_tables=False)
        if not text:
            soup = _BeautifulSoup(r.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "aside", "iframe", "figure"]):
                tag.extract()
            for sel in ["article", "main", ".article-body", ".article-content", ".post-content",
                        '[class*="article"]', '[class*="story"]']:
                el = soup.select_one(sel)
                if el:
                    candidate = el.get_text(separator=" ", strip=True)
                    if len(candidate) > 200:
                        text = candidate
                        break
            if not text:
                text = soup.get_text(separator=" ", strip=True)
        return text[:600].strip() if text else ""
    except Exception:
        return ""


def enrich_snippets(results: List[SearchResult], timeout: float = 8.0) -> List[SearchResult]:
    """Fetch and fill snippets for results that have none, up to *timeout* seconds total.

    Uses up to 8 parallel workers. Articles that timeout or fail keep an empty snippet.
    Only has an effect when trafilatura and beautifulsoup4 are installed.
    """
    to_enrich = [i for i, r in enumerate(results) if not r.snippet]
    if not to_enrich or not _SNIPPET_LIBS_AVAILABLE:
        return results

    fetched: dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_fetch_snippet, results[i].url): i for i in to_enrich}
        try:
            for fut in as_completed(futures, timeout=timeout):
                idx = futures[fut]
                try:
                    fetched[idx] = fut.result()
                except Exception:
                    fetched[idx] = ""
        except TimeoutError:
            logger.warning("enrich_snippets: %ss wall hit, %d URLs incomplete",
                           timeout, sum(1 for f in futures if not f.done()))
            for fut, idx in futures.items():
                if idx not in fetched:
                    fetched[idx] = ""

    out = list(results)
    for idx, snippet in fetched.items():
        out[idx] = replace(out[idx], snippet=snippet)
    return out


# ──────────────────────────────────────────────
# Public API — tries providers in order
# ──────────────────────────────────────────────

def search_articles(
    query: str,
    limit: int = 10,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
) -> List[SearchResult]:
    """
    Search for news articles matching *query*, returning up to *limit* results.

    Tries providers in order, skipping any without a configured key or with
    an exhausted quota flag set for this process lifetime:
      GDELT → SerpAPI → Serper.dev → Brave → BrightData → Nimbleway → ScrapingBee → Newsdata.io → DataForSEO → DDG

    DDG is tried last; if AWS IPs are blocked by DDG/Yahoo it fails and is logged.

    Args:
        query:     Search string. Include `site:domain.com` to restrict to a source.
        limit:     Max results to return.
        date_from: Optional start of date window.
        date_to:   Optional end of date window.

    Returns:
        List of SearchResult(title, url, snippet, source, published_date).
    """
    _refresh_keys_if_stale()
    _provider_local.name = "none"
    _provider_local.chain = []

    # 1. GDELT (free, no key) — primary provider; news-only, reliable dates, no snippets
    _provider_local.chain.append("gdelt")
    try:
        results = _search_gdelt(query, limit, date_from, date_to)
        if results:
            _provider_local.name = "gdelt"
            return results
        logger.warning("GDELT returned 0 results for: %s", query[:60])
    except Exception as e:
        logger.warning("GDELT failed: %s", e)

    # 2. SerpAPI
    if SERPAPI_API_KEY and not _SERPAPI_QUOTA_EXHAUSTED:
        _provider_local.chain.append("serpapi")
        try:
            results = _search_serpapi_news(query, limit, date_from, date_to)
            if results:
                _provider_local.name = "serpapi"
                return results
            logger.warning("SerpAPI returned 0 results for: %s", query[:60])
        except Exception as e:
            logger.warning("SerpAPI failed: %s", e)

    # 3. Serper.dev news
    if SERPER_API_KEY and not _SERPER_QUOTA_EXHAUSTED:
        _provider_local.chain.append("serper")
        try:
            results = _search_serper_news(query, limit, date_from, date_to)
            if results:
                _provider_local.name = "serper"
                return results
            logger.warning("Serper returned 0 results for: %s", query[:60])
        except Exception as e:
            logger.warning("Serper failed: %s", e)

    # 4. Brave News
    if BRAVE_API_KEY and not _BRAVE_QUOTA_EXHAUSTED:
        _provider_local.chain.append("brave")
        try:
            results = _search_brave_news(query, limit, date_from, date_to)
            if results:
                _provider_local.name = "brave"
                return results
            logger.warning("Brave returned 0 results for: %s", query[:60])
        except Exception as e:
            logger.warning("Brave failed: %s", e)

    # 5. BrightData SERP API
    if BRIGHTDATA_API_KEY and not _BRIGHTDATA_QUOTA_EXHAUSTED:
        _provider_local.chain.append("brightdata")
        try:
            results = _search_brightdata(query, limit, date_from, date_to)
            if results:
                _provider_local.name = "brightdata"
                return results
            logger.warning("BrightData returned 0 results for: %s", query[:60])
        except Exception as e:
            logger.warning("BrightData failed: %s", e)

    # 6. Nimbleway SERP API
    if NIMBLEWAY_API_KEY and not _NIMBLEWAY_QUOTA_EXHAUSTED:
        _provider_local.chain.append("nimbleway")
        try:
            results = _search_nimbleway(query, limit, date_from, date_to)
            if results:
                _provider_local.name = "nimbleway"
                return results
            logger.warning("Nimbleway returned 0 results for: %s", query[:60])
        except Exception as e:
            logger.warning("Nimbleway failed: %s", e)

    # 7. ScrapingBee Google Search
    if SCRAPINGBEE_API_KEY and not _SCRAPINGBEE_QUOTA_EXHAUSTED:
        _provider_local.chain.append("scrapingbee")
        try:
            results = _search_scrapingbee(query, limit, date_from, date_to)
            if results:
                _provider_local.name = "scrapingbee"
                return results
            logger.warning("ScrapingBee returned 0 results for: %s", query[:60])
        except Exception as e:
            logger.warning("ScrapingBee failed: %s", e)

    # 8. Newsdata.io
    if NEWSDATA_API_KEY and not _NEWSDATA_QUOTA_EXHAUSTED:
        _provider_local.chain.append("newsdata")
        try:
            results = _search_newsdata_io(query, limit, date_from, date_to)
            if results:
                _provider_local.name = "newsdata"
                return results
            logger.warning("Newsdata.io returned 0 results for: %s", query[:60])
        except Exception as e:
            logger.warning("Newsdata.io failed: %s", e)

    # 9. DataForSEO (paid — last-resort fallback only)
    if DATAFORSEO_API_KEY and not _DATAFORSEO_QUOTA_EXHAUSTED:
        _provider_local.chain.append("dataforseo")
        try:
            results = _search_dataforseo(query, limit, date_from, date_to)
            if results:
                _provider_local.name = "dataforseo"
                return results
            logger.warning("DataForSEO returned 0 results for: %s", query[:60])
        except Exception as e:
            logger.warning("DataForSEO failed: %s", e)

    # 10. DuckDuckGo (free, no key)
    _provider_local.chain.append("ddg")
    try:
        results = _search_ddg_news(query, limit, date_from, date_to)
        if results:
            _provider_local.name = "ddg"
            return results
        logger.warning("DDG returned 0 results for: %s", query[:60])
    except Exception as e:
        logger.warning("DDG failed: %s", e)

    logger.error("All search providers exhausted — no articles found for: %s", query[:60])
    return []


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _extract_domain(url: str) -> str:
    try:
        return re.sub(r"^www\.", "", __import__("urllib.parse", fromlist=["urlparse"]).urlparse(url).netloc)
    except Exception:
        return ""
