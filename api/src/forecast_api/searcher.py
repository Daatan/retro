"""
Search endpoint logic — exposes web_search.py via async wrappers for /search and /search/health.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Optional

import httpx

from tm import web_search as _ws

from .forecaster import _distill_query
from .models import ProviderStatus, SearchHealthResponse, SearchRequest, SearchResponse, SearchResultItem

logger = logging.getLogger(__name__)


async def run_search(req: SearchRequest) -> SearchResponse:
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    if req.date_from:
        date_from = datetime.fromisoformat(req.date_from)
    if req.date_to:
        date_to = datetime.fromisoformat(req.date_to)

    t0 = time.perf_counter()
    # Provider attribution is thread-local: search runs in a worker thread via
    # asyncio.to_thread, so we must read get_last_search_provider() *inside that
    # same thread*. Reading it here in the event-loop thread always returned
    # "none" (same bug the forecaster fixed with _search_capturing).
    def _search_capturing(q: str):
        results = _ws.search_articles(q, req.limit, date_from, date_to)
        return results, _ws.get_last_search_provider(), _ws.get_last_search_provider_chain()

    results, provider, chain = await asyncio.to_thread(_search_capturing, req.query)

    # On 0 verbatim results, distill the query to keywords and retry once. Reuses
    # the same _distill_query the /forecast pipeline uses (Nova Micro; it also
    # translates non-Latin questions to English keywords). This lets /search
    # consumers — e.g. daatan forecast-creation, which sends sentence-style or
    # Hebrew queries — recover without re-implementing distillation. _distill_query
    # is fail-open (returns the original question on error) and non-retrying, so
    # the added cost is bounded to ~one LLM call + one more search.
    distilled_query: Optional[str] = None
    if not results and req.distill:
        distilled = await _distill_query(req.query)
        if distilled and distilled != req.query:
            distilled_query = distilled
            results, provider, chain = await asyncio.to_thread(_search_capturing, distilled)

    duration_ms = round((time.perf_counter() - t0) * 1000, 1)

    if req.enrich_snippets:
        results = await asyncio.to_thread(_ws.enrich_snippets, results)

    logger.info(
        "event=search_done provider=%s chain=%s query=%r distilled=%r count=%d duration_ms=%s",
        provider, chain, req.query[:60], (distilled_query or "")[:60], len(results), duration_ms,
    )

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
        provider=provider,
        provider_chain=chain,
        distilled_query=distilled_query,
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
        if credits is not None and credits <= 0:
            _ws._SCRAPINGBEE_QUOTA_EXHAUSTED = True
            return ProviderStatus(configured=True, exhausted=True, status="exhausted", credits=credits)
        return ProviderStatus(configured=True, exhausted=False, status="ok", credits=credits)
    except Exception as e:
        return ProviderStatus(configured=True, exhausted=False, status="error", error=str(e))


async def _check_gdelt() -> ProviderStatus:
    # GDELT needs no API key and has no credit quota. We deliberately do NOT fire
    # a live probe — that would consume the shared 1-req/10s slot and degrade real
    # searches. Instead we surface the Doc API's runtime circuit/cooldown state,
    # which web_search maintains on 429s and repeated failures. This is the signal
    # that actually mattered ("ok" was misleading while GDELT was throttled and the
    # chain was silently degrading to BigQuery). Note: these flags are per-process
    # (per gunicorn worker), so this reflects the health worker's view of GDELT.
    now = time.time()
    broken_remaining = _ws._GDELT_DOC_BROKEN_UNTIL - now
    cooldown_remaining = _ws._GDELT_COOLDOWN_UNTIL - now
    if broken_remaining > 0:
        return ProviderStatus(configured=True, exhausted=True, status="circuit_open",
                              error=f"Doc API circuit open for {int(broken_remaining)}s after repeated failures")
    if cooldown_remaining > 0:
        return ProviderStatus(configured=True, exhausted=True, status="cooldown",
                              error=f"429 rate-limit cooldown for {int(cooldown_remaining)}s")
    return ProviderStatus(configured=True, exhausted=False, status="ok")


async def _check_gdelt_bq() -> ProviderStatus:
    if not _ws.GCP_SA_KEY_JSON:
        return ProviderStatus(configured=False, exhausted=False, status="not_configured")
    if not _ws._BQ_AVAILABLE:
        return ProviderStatus(configured=False, exhausted=False, status="not_configured",
                              error="google-cloud-bigquery not installed")
    return ProviderStatus(configured=True, exhausted=False, status="ok")


async def _check_google_cse() -> ProviderStatus:
    # Needs BOTH an API key and a search-engine id (cx). No live credit-check API,
    # so we report configured + the in-process quota flag (like _check_simple).
    if not (_ws.GOOGLE_CSE_API_KEY and _ws.GOOGLE_CSE_CX):
        return ProviderStatus(configured=False, exhausted=False, status="not_configured")
    if _ws._GOOGLE_CSE_QUOTA_EXHAUSTED:
        return ProviderStatus(configured=True, exhausted=True, status="exhausted")
    return ProviderStatus(configured=True, exhausted=False, status="ok")


async def run_search_health() -> SearchHealthResponse:
    _ws._refresh_keys_if_stale()

    # Each entry: (provider_name, check_coroutine)
    provider_checks = [
        ("dataforseo", _check_dataforseo()),
        ("google_cse", _check_google_cse()),
        ("serpapi",    _check_serpapi()),
        ("serper",     _check_serper()),
        ("tavily",     _check_simple(_ws.TAVILY_API_KEY, _ws._TAVILY_QUOTA_EXHAUSTED)),
        ("brave",      _check_simple(_ws.BRAVE_API_KEY, _ws._BRAVE_QUOTA_EXHAUSTED)),
        ("brightdata", _check_simple(_ws.BRIGHTDATA_API_KEY, _ws._BRIGHTDATA_QUOTA_EXHAUSTED)),
        ("nimbleway",  _check_simple(_ws.NIMBLEWAY_API_KEY, _ws._NIMBLEWAY_QUOTA_EXHAUSTED)),
        ("scrapingbee", _check_scrapingbee()),
        ("newsdata",   _check_simple(_ws.NEWSDATA_API_KEY, _ws._NEWSDATA_QUOTA_EXHAUSTED)),
        ("gdelt",      _check_gdelt()),
        ("gdelt_bq",   _check_gdelt_bq()),
    ]
    names, coros = zip(*provider_checks)
    results = await asyncio.gather(*coros)
    providers: dict[str, ProviderStatus] = dict(zip(names, results))
    providers["ddg"] = ProviderStatus(configured=True, exhausted=False, status="ok")
    # Last-resort, key-less fallback (batched site: DDG over trusted domains) — always available.
    providers["trusted_sites"] = ProviderStatus(configured=True, exhausted=False, status="ok")

    # Usable = configured + not exhausted + status ok, excluding the key-less
    # last-resort fallbacks (ddg, trusted_sites) so they don't mask a real drought.
    usable = sum(
        1 for k, p in providers.items()
        if k not in ("ddg", "trusted_sites") and p.configured and not p.exhausted and p.status == "ok"
    )
    overall = "ok" if usable >= 2 else ("degraded" if usable == 1 else "down")

    return SearchHealthResponse(providers=providers, overall=overall, usable_count=usable)
