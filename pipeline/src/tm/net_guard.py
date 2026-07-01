"""SSRF guard for the ingest/search pipeline's outbound article fetches.

The pipeline fetches URLs that come from third-party search results, GDELT/GNews
JSON, and scraped sitemaps/result pages — all attacker-influenceable — so every
such fetch must refuse non-http(s) schemes and any host that resolves to a
non-public address (loopback, RFC1918, link-local incl. the cloud metadata IP
169.254.169.254, reserved/multicast/unspecified).

``safe_get`` / ``safe_get_async`` follow redirects manually and re-validate each
hop, because a validated public host can still 30x-redirect to an internal one.

This mirrors ``forecast_api.net_guard`` in the api/ project; the two live in
separate deployables (separate venvs) so the logic is intentionally duplicated
rather than shared across the project boundary.
"""

import ipaddress
import logging
import socket
from urllib.parse import urljoin, urlparse

import httpx

logger = logging.getLogger(__name__)

_ALLOWED_SCHEMES = ("http", "https")
_REDIRECT_STATUSES = (301, 302, 303, 307, 308)


class UnsafeURLError(Exception):
    """Raised when a URL (or a redirect target) is not a public http(s) endpoint."""


def is_safe_url(url: str) -> bool:
    """True iff ``url`` is http(s) and its host resolves entirely to public IPs.

    Requiring every resolved address to be public guards the basic DNS-rebinding
    trick of returning one public and one private A record.
    """
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    if parsed.scheme not in _ALLOWED_SCHEMES:
        return False
    host = parsed.hostname
    if not host:
        return False
    try:
        infos = socket.getaddrinfo(host, None)
    except (socket.gaierror, UnicodeError, ValueError):
        return False
    if not infos:
        return False
    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0])
        except ValueError:
            return False
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            return False
    return True


def _next_hop(resp: httpx.Response, current: str) -> str | None:
    if resp.status_code in _REDIRECT_STATUSES and "location" in resp.headers:
        return urljoin(current, resp.headers["location"])
    return None


def safe_get(
    url: str, *, client: httpx.Client | None = None, max_redirects: int = 5, **kwargs
) -> httpx.Response:
    """Sync GET with SSRF validation on the initial URL and every redirect hop.

    Reuses ``client`` (preserving its headers/timeout) if given — overriding its
    redirect behaviour per request — otherwise opens a short-lived client.
    Raises :class:`UnsafeURLError` on any non-public http(s) hop or after
    ``max_redirects`` hops.
    """
    owned = client is None
    client = client or httpx.Client()
    try:
        current = url
        for _ in range(max_redirects + 1):
            if not is_safe_url(current):
                raise UnsafeURLError(current)
            resp = client.get(current, follow_redirects=False, **kwargs)
            nxt = _next_hop(resp, current)
            if nxt is None:
                return resp
            resp.close()
            current = nxt
        raise UnsafeURLError(f"too many redirects starting from {url}")
    finally:
        if owned:
            client.close()


async def safe_get_async(
    client: httpx.AsyncClient, url: str, *, max_redirects: int = 5, **kwargs
) -> httpx.Response:
    """Async counterpart of :func:`safe_get`, reusing an existing ``AsyncClient``
    (its redirect behaviour is overridden per request)."""
    current = url
    for _ in range(max_redirects + 1):
        if not is_safe_url(current):
            raise UnsafeURLError(current)
        resp = await client.get(current, follow_redirects=False, **kwargs)
        nxt = _next_hop(resp, current)
        if nxt is None:
            return resp
        await resp.aclose()
        current = nxt
    raise UnsafeURLError(f"too many redirects starting from {url}")
