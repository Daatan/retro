"""
Search endpoint logic — exposes web_search.py via async wrappers for /search and /search/health.
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional

import httpx

from tm import web_search as _ws

from .models import ProviderStatus, SearchHealthResponse, SearchRequest, SearchResponse, SearchResultItem

logger = logging.getLogger(__name__)


async def run_search(req: SearchRequest) -> SearchResponse:
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    if req.date_from:
        date_from = datetime.fromisoformat(req.date_from)
    if req.date_to:
        date_to = datetime.fromisoformat(req.date_to)

    results = await asyncio.to_thread(
        _ws.search_articles, req.query, req.limit, date_from, date_to
    )
    if req.enrich_snippets:
        results = await asyncio.to_thread(_ws.enrich_snippets, results)
    return SearchResponse(
        query=req.query,
        results=[
            SearchResultItem(
                title=r.title,
                url=r.url,
                snippet=r.snippet or "",
                source=r.source,
                published_date=r.published_date,
            )
            for r in results
        ],
        count=len(results),
    )


# ── Per-provider credit checks ────────────────────────────────────────────────

async def _check_simple(key: Optional[str], exhausted: bool) -> ProviderStatus:
    """For providers without a live credit-check API."""
    if not key:
        return ProviderStatus(configured=False, exhausted=False, status="not_configured")
    if exhausted:
        return ProviderStatus(configured=True, exhausted=True, status="exhausted")
    return ProviderStatus(configured=True, exhausted=False, status="ok")


async def _check_dataforseo() -> ProviderStatus:
    if not _ws.DATAFORSEO_API_KEY:
        return ProviderStatus(configured=False, exhausted=False, status="not_configured")
    if _ws._DATAFORSEO_QUOTA_EXHAUSTED:
        return ProviderStatus(configured=True, exhausted=True, status="exhausted")
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            r = await c.get(
                "https://api.dataforseo.com/v3/appendix/user_data",
                headers={"Authorization": f"Basic {_ws.DATAFORSEO_API_KEY}"},
            )
        if not r.is_success:
            return ProviderStatus(configured=True, exhausted=False, status="error", error=f"HTTP {r.status_code}")
        result = (r.json().get("tasks") or [{}])[0].get("result", [{}])[0]
        balance = (result.get("money_data") or {}).get("balance")
        credits = int(balance) if balance is not None else None
        return ProviderStatus(configured=True, exhausted=False, status="ok", credits=credits)
    except Exception as e:
        return ProviderStatus(configured=True, exhausted=False, status="error", error=str(e))


async def _check_serper() -> ProviderStatus:
    if not _ws.SERPER_API_KEY:
        return ProviderStatus(configured=False, exhausted=False, status="not_configured")
    if _ws._SERPER_QUOTA_EXHAUSTED:
        return ProviderStatus(configured=True, exhausted=True, status="exhausted")
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            r = await c.get(
                "https://google.serper.dev/account",
                headers={"X-API-KEY": _ws.SERPER_API_KEY},
            )
        if not r.is_success:
            return ProviderStatus(configured=True, exhausted=False, status="error", error=f"HTTP {r.status_code}")
        credits = r.json().get("balance")
        return ProviderStatus(configured=True, exhausted=False, status="ok", credits=credits)
    except Exception as e:
        return ProviderStatus(configured=True, exhausted=False, status="error", error=str(e))


async def _check_serpapi() -> ProviderStatus:
    if not _ws.SERPAPI_API_KEY:
        return ProviderStatus(configured=False, exhausted=False, status="not_configured")
    if _ws._SERPAPI_QUOTA_EXHAUSTED:
        return ProviderStatus(configured=True, exhausted=True, status="exhausted")
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            r = await c.get(
                "https://serpapi.com/account.json",
                params={"api_key": _ws.SERPAPI_API_KEY},
            )
        if not r.is_success:
            return ProviderStatus(configured=True, exhausted=False, status="error", error=f"HTTP {r.status_code}")
        data = r.json()
        credits = data.get("total_searches_left")
        exhausted = credits == 0
        if exhausted:
            _ws._SERPAPI_QUOTA_EXHAUSTED = True
        return ProviderStatus(configured=True, exhausted=exhausted,
                              status="exhausted" if exhausted else "ok", credits=credits)
    except Exception as e:
        return ProviderStatus(configured=True, exhausted=False, status="error", error=str(e))


async def _check_scrapingbee() -> ProviderStatus:
    if not _ws.SCRAPINGBEE_API_KEY:
        return ProviderStatus(configured=False, exhausted=False, status="not_configured")
    if _ws._SCRAPINGBEE_QUOTA_EXHAUSTED:
        return ProviderStatus(configured=True, exhausted=True, status="exhausted")
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            r = await c.get(
                "https://app.scrapingbee.com/api/v1/usage",
                params={"api_key": _ws.SCRAPINGBEE_API_KEY},
            )
        if not r.is_success:
            return ProviderStatus(configured=True, exhausted=False, status="error", error=f"HTTP {r.status_code}")
        data = r.json()
        max_c = data.get("max_api_credit")
        used_c = data.get("used_api_credit")
        credits = (max_c - used_c) if (max_c is not None and used_c is not None) else None
        return ProviderStatus(configured=True, exhausted=False, status="ok", credits=credits)
    except Exception as e:
        return ProviderStatus(configured=True, exhausted=False, status="error", error=str(e))


async def _check_gdelt() -> ProviderStatus:
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(
                "https://api.gdeltproject.org/api/v2/doc/doc",
                params={"query": "test", "mode": "artlist", "format": "json", "maxrecords": 1},
            )
        if not r.is_success:
            return ProviderStatus(configured=True, exhausted=False, status="error", error=f"HTTP {r.status_code}")
        return ProviderStatus(configured=True, exhausted=False, status="ok")
    except Exception as e:
        return ProviderStatus(configured=True, exhausted=False, status="error", error=str(e))


async def run_search_health() -> SearchHealthResponse:
    _ws._refresh_keys_if_stale()

    # Each entry: (provider_name, check_coroutine)
    provider_checks = [
        ("dataforseo", _check_dataforseo()),
        ("serpapi",    _check_serpapi()),
        ("serper",     _check_serper()),
        ("brave",      _check_simple(_ws.BRAVE_API_KEY, _ws._BRAVE_QUOTA_EXHAUSTED)),
        ("brightdata", _check_simple(_ws.BRIGHTDATA_API_KEY, _ws._BRIGHTDATA_QUOTA_EXHAUSTED)),
        ("nimbleway",  _check_simple(_ws.NIMBLEWAY_API_KEY, _ws._NIMBLEWAY_QUOTA_EXHAUSTED)),
        ("scrapingbee", _check_scrapingbee()),
        ("newsdata",   _check_simple(_ws.NEWSDATA_API_KEY, _ws._NEWSDATA_QUOTA_EXHAUSTED)),
        ("gdelt",      _check_gdelt()),
    ]
    names, coros = zip(*provider_checks)
    results = await asyncio.gather(*coros)
    providers: dict[str, ProviderStatus] = dict(zip(names, results))
    providers["ddg"] = ProviderStatus(configured=True, exhausted=False, status="ok")

    # Usable = configured + not exhausted + status ok, excluding DDG (blocked on EC2)
    usable = sum(
        1 for k, p in providers.items()
        if k != "ddg" and p.configured and not p.exhausted and p.status == "ok"
    )
    overall = "ok" if usable >= 2 else ("degraded" if usable == 1 else "down")

    return SearchHealthResponse(providers=providers, overall=overall, usable_count=usable)
