"""SSRF regression tests for gnews_ingest.py's _fetch_wayback/search_wayback_cdx
(retro#429). Both hit a fixed public host (web.archive.org), so the real bug
was never the initial request — it was blindly following redirects
(httpx.AsyncClient(follow_redirects=True), no revalidation). These tests
confirm a redirect to a private/internal address is now rejected instead of
silently followed. web.archive.org needs real DNS (its host isn't
attacker-controlled) — this mirrors the code's own DNS dependency, not a
testing shortcut."""

import httpx
import pytest

from tm.gnews_ingest import _fetch_wayback, search_wayback_cdx
from datetime import datetime


def _redirect_to_private_ip_transport():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "http://127.0.0.1/secret"})
    return httpx.MockTransport(handler)


def _ok_transport(text="article body", content_type="text/html"):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=text, headers={"content-type": content_type})
    return httpx.MockTransport(handler)


async def test_fetch_wayback_rejects_redirect_to_private_ip():
    async with httpx.AsyncClient(transport=_redirect_to_private_ip_transport()) as client:
        result = await _fetch_wayback("https://paywalled-site.example/article", client)
    assert result == ""


async def test_fetch_wayback_follows_safe_response():
    async with httpx.AsyncClient(transport=_ok_transport(text="<p>" + "x" * 400 + "</p>")) as client:
        result = await _fetch_wayback("https://paywalled-site.example/article", client)
    assert isinstance(result, str)


async def test_search_wayback_cdx_rejects_redirect_to_private_ip(monkeypatch):
    import tm.gnews_ingest as gi_mod

    monkeypatch.setattr(gi_mod.httpx, "AsyncClient",
                         lambda *a, **kw: httpx.AsyncClient(transport=_redirect_to_private_ip_transport()))

    result = await search_wayback_cdx(
        "example.com", datetime(2024, 1, 1), datetime(2024, 2, 1), keywords=[],
    )
    assert result == []
