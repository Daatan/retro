"""Thin client for Daatan's public similar-forecasts lookup.

Reused by the Oracle's precursor candidate-match step (retro#608) to check whether a
decompose precursor already matches an open forecast in Daatan's own bank before pricing
it fresh. ``GET /api/forecasts/similar`` is public and unauthenticated (Daatan's
``docs/API.md``), so no new secret or auth wiring is needed.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

import httpx

DAATAN_API_BASE = "https://daatan.com"
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; TruthMachine/1.0)"}


class DaatanLookupError(Exception):
    """The lookup itself failed — network error, non-200, or an unparseable body.

    Deliberately distinct from a genuine zero-result response: collapsing both into
    an empty list would make a Daatan outage log identically to "no match exists",
    which corrupts the precision data this shadow-only slice exists to produce.
    """


async def find_similar_forecasts(
    query: str,
    *,
    limit: int = 3,
    timeout: float = 10.0,
    transport: Optional[httpx.AsyncBaseTransport] = None,
) -> list[dict[str, Any]]:
    """Query ``GET /api/forecasts/similar?q=<query>&limit=<limit>``.

    Returns the ``similar`` list as-is — each item shaped like ``{id, slug,
    claimText, status, resolveByDatetime, author, score}`` — empty when Daatan
    genuinely has no match (this also covers AI-features-disabled deployments,
    which the endpoint itself reports as an empty list, not an error). Raises
    ``DaatanLookupError`` on a real failure instead of silently returning ``[]``.
    """
    q = (query or "").strip()[:200]
    if not q:
        return []
    try:
        async with httpx.AsyncClient(timeout=timeout, transport=transport) as client:
            r = await client.get(
                f"{DAATAN_API_BASE}/api/forecasts/similar",
                params={"q": q, "limit": limit},
                headers=_HEADERS,
            )
    except Exception as exc:
        raise DaatanLookupError(f"{type(exc).__name__}: {exc}") from exc
    if r.status_code != 200:
        raise DaatanLookupError(f"HTTP {r.status_code}")
    try:
        data = r.json()
    except Exception as exc:
        raise DaatanLookupError(f"invalid JSON: {exc}") from exc
    similar = data.get("similar") if isinstance(data, dict) else None
    if not isinstance(similar, list):
        raise DaatanLookupError("missing 'similar' list in response")
    return similar


def is_open(candidate: dict[str, Any], now: Optional[datetime] = None) -> bool:
    """True when a candidate forecast is genuinely open/live right now.

    Unused this shadow-only slice — ``/api/forecasts/similar`` already filters to
    ``ACTIVE``/``PENDING_APPROVAL`` server-side, so nothing here gates the log. Kept
    for a future enforcing slice that needs to know a match is actually stakeable,
    not merely returned by the similarity search (e.g. ``PENDING_APPROVAL`` isn't
    live yet).
    """
    if candidate.get("status") != "ACTIVE":
        return False
    deadline_raw = candidate.get("resolveByDatetime")
    if not deadline_raw:
        return True
    try:
        deadline = datetime.fromisoformat(str(deadline_raw).replace("Z", "+00:00"))
    except ValueError:
        return True
    if deadline.tzinfo is None:
        deadline = deadline.replace(tzinfo=timezone.utc)
    return deadline > (now or datetime.now(timezone.utc))
