"""SSRF guards for outbound fetches.

Both the public ``/fetch-url`` proxy and the forecast pipeline fetch URLs that
originate from callers or from third-party search results, so they must refuse
non-http(s) schemes and any host that resolves to a non-public address
(loopback, RFC1918 private ranges, link-local incl. the cloud metadata IP
169.254.169.254, reserved/multicast/unspecified).

``safe_get`` additionally re-validates every redirect hop: a validated public
host can 30x-redirect to an internal address, so following redirects blindly
(httpx ``follow_redirects=True``) reopens the hole.
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


def safe_get(url: str, *, max_redirects: int = 5, **kwargs) -> httpx.Response:
    """Sync GET with SSRF validation on the initial URL and every redirect hop.

    Follows redirects manually (never httpx's automatic ``follow_redirects``) so
    each hop's host is re-resolved and checked. Raises :class:`UnsafeURLError` if
    any hop is not a public http(s) endpoint, or after ``max_redirects`` hops.
    Extra kwargs (``timeout``, ``headers``, …) are forwarded to each request.
    """
    current = url
    with httpx.Client(follow_redirects=False) as client:
        for _ in range(max_redirects + 1):
            if not is_safe_url(current):
                raise UnsafeURLError(current)
            resp = client.get(current, **kwargs)
            if resp.status_code in _REDIRECT_STATUSES and "location" in resp.headers:
                current = urljoin(current, resp.headers["location"])
                resp.close()
                continue
            return resp
    raise UnsafeURLError(f"too many redirects starting from {url}")
