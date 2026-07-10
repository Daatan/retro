"""SSRF guard for every outbound fetch in this repo.

Both the pipeline (third-party search results, GDELT/GNews JSON, scraped
sitemaps) and the api's public ``/fetch-url`` proxy fetch attacker-influenceable
URLs, so every such fetch must refuse non-http(s) schemes and any host that
resolves to a non-public address (loopback, RFC1918, link-local incl. the cloud
metadata IP 169.254.169.254, reserved/multicast/unspecified).

``safe_get`` / ``safe_get_async`` follow redirects manually and re-validate each
hop, because a validated public host can still 30x-redirect to an internal one.

This is the single copy. ``forecast_api`` imports it from here: the api declares
``truthmachine-pipeline`` as a dependency (``api/pyproject.toml`` [tool.uv.sources]
→ ``../pipeline``, editable), so ``tm`` is always present in the api's venv. Keep
it that way — a second copy means a DNS-rebinding fix can land in one and not the
other.
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


def unsafe_reason(url: str) -> str | None:
    """``None`` if ``url`` is a public http(s) endpoint, else why it was rejected.

    Requiring every resolved address to be public guards the basic DNS-rebinding
    trick of returning one public and one private A record.

    The reason exists so callers can log the two rejection classes apart. A host
    that does not resolve is a dead link; a host that resolves into private space
    is an SSRF attempt. Both are refused, but reporting them identically buries
    the second in a stream of the first.
    """
    try:
        parsed = urlparse(url)
    except ValueError:
        return "malformed URL"
    if parsed.scheme not in _ALLOWED_SCHEMES:
        return f"scheme not allowed: {parsed.scheme!r}"
    host = parsed.hostname
    if not host:
        return "no host"
    try:
        infos = socket.getaddrinfo(host, None)
    except (socket.gaierror, UnicodeError, ValueError):
        return f"host does not resolve: {host!r}"
    if not infos:
        return f"host does not resolve: {host!r}"
    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0])
        except ValueError:
            return f"unparsable resolved address for {host!r}"
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            return f"host resolves to non-public address {ip}"
    return None


def is_safe_url(url: str) -> bool:
    """True iff ``url`` is http(s) and its host resolves entirely to public IPs."""
    return unsafe_reason(url) is None


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
            if (reason := unsafe_reason(current)) is not None:
                raise UnsafeURLError(f"{current}: {reason}")
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
        if (reason := unsafe_reason(current)) is not None:
            raise UnsafeURLError(f"{current}: {reason}")
        resp = await client.get(current, follow_redirects=False, **kwargs)
        nxt = _next_hop(resp, current)
        if nxt is None:
            return resp
        await resp.aclose()
        current = nxt
    raise UnsafeURLError(f"too many redirects starting from {url}")
