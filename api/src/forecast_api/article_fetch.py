"""Shared article fetch + extraction.

Single implementation behind both the REST `/fetch-url` route (main.py) and the
MCP `fetch_article` tool (mcp_server.py) so the SSRF-sensitive path exists once.
Fetches through tm.net_guard.safe_get (validates the host it actually connects to
and re-checks every redirect hop) rather than trafilatura.fetch_url, which
re-resolves the host independently and leaves a DNS-rebinding (TOCTOU) window.
"""

import asyncio

from tm.net_guard import UnsafeURLError, is_safe_url, safe_get

from .models import FetchUrlResponse


class ArticleFetchError(Exception):
    """Raised when a URL can't be fetched/extracted. `status` mirrors the HTTP
    status the REST route returns (422 for both unsafe and unfetchable, matching
    the pre-refactor behaviour)."""

    def __init__(self, detail: str, status: int = 422):
        super().__init__(detail)
        self.detail = detail
        self.status = status


async def fetch_and_extract(url: str) -> FetchUrlResponse:
    """Fetch an article URL and extract text/title/date/source via trafilatura.

    Raises ArticleFetchError on a non-public URL, a failed fetch, or an empty
    body — callers map .status/.detail to their transport (HTTP 422 / MCP error).
    """
    if not is_safe_url(url):
        raise ArticleFetchError("URL must be a public http(s) address")

    import trafilatura

    # Sync fetch off the event loop; safe_get enforces the SSRF guard on the
    # host it connects to and on every redirect hop.
    try:
        resp = await asyncio.to_thread(
            safe_get,
            url,
            timeout=10.0,
            headers={"User-Agent": "Mozilla/5.0 (compatible; TruthMachine/1.0)"},
        )
    except UnsafeURLError:
        raise ArticleFetchError("URL must be a public http(s) address")
    except Exception:
        raise ArticleFetchError("Could not fetch URL")

    if resp.status_code != 200 or not resp.text:
        raise ArticleFetchError("Could not fetch URL")

    downloaded = resp.text
    metadata = trafilatura.extract_metadata(downloaded)
    text = trafilatura.extract(downloaded, include_comments=False, include_tables=False) or ""
    date_str: str | None = None
    if metadata and metadata.date:
        # trafilatura returns date as a string already
        date_str = str(metadata.date)[:10]
    return FetchUrlResponse(
        text=text,
        title=metadata.title if metadata else None,
        date=date_str,
        source=metadata.sitename if metadata else None,
    )
