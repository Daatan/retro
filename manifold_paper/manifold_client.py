"""Read-only client for Manifold Markets' public API (retro#918).

Only GET requests, only endpoints that need no authentication:
``/v0/search-markets`` (candidate discovery) and ``/v0/market/{id}`` (one market
with its description, and re-checking a tracked market for resolution). There is
no API key, no ``/v0/bet``, no POST of any kind, and ``tests/test_manifold_client``
fails if one is added — a paper bot must not be able to place a Mana bet.

Manifold's ToS permit bots through the public API only (500 req/min/IP) and
forbid commercial AI/ML training on API data without a data licence. This
module feeds an *evaluation* ledger; nothing downstream may use it as training
or prompt-tuning data (see README).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import httpx

API_BASE = "https://api.manifold.markets"
_HEADERS = {"User-Agent": "daatan-oracle-paper/1.0 (+https://daatan.com; read-only evaluation bot)"}

# resolution values of a BINARY market: YES / NO are scoreable; MKT is a
# probabilistic resolution and CANCEL voids the market — neither is a Brier row.
SCOREABLE = ("YES", "NO")


@dataclass(frozen=True)
class Market:
    market_id: str
    market_slug: str
    question: str
    description: str
    yes_price: Optional[float]
    closed: bool
    end_date: Optional[str]          # ISO-8601 UTC, from closeTime
    volume: float                    # Mana
    unique_bettors: int
    creator: str
    resolved_outcome: Optional[str]  # "YES" | "NO" only when scoreable
    resolution_raw: Optional[str]    # "YES" | "NO" | "MKT" | "CANCEL" | None
    url: str

    @property
    def has_price(self) -> bool:
        return self.yes_price is not None


def _iso_ms(ms) -> Optional[str]:
    try:
        return datetime.fromtimestamp(float(ms) / 1000.0, tz=timezone.utc).isoformat()
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def _prob(raw) -> Optional[float]:
    try:
        p = float(raw)
    except (TypeError, ValueError):
        return None
    return p if 0.0 <= p <= 1.0 else None


def _description_text(m: dict) -> str:
    """Full-market responses carry ``textDescription`` (plain) next to a rich-text
    JSON ``description``; the lite search rows carry neither."""
    t = m.get("textDescription")
    if isinstance(t, str):
        return t.strip()
    d = m.get("description")
    return d.strip() if isinstance(d, str) else ""


def parse_market(m: dict) -> Optional[Market]:
    """Normalise one Manifold market dict; None unless it is a BINARY market."""
    if m.get("outcomeType") != "BINARY" or not m.get("id"):
        return None
    resolved = bool(m.get("isResolved"))
    raw = m.get("resolution") if resolved else None
    raw = str(raw).upper() if raw else None
    close_ms = m.get("closeTime")
    closed = resolved
    if not closed and close_ms is not None:
        try:
            closed = float(close_ms) <= datetime.now(timezone.utc).timestamp() * 1000.0
        except (TypeError, ValueError):
            pass
    try:
        volume = float(m.get("volume") or 0.0)
    except (TypeError, ValueError):
        volume = 0.0
    try:
        bettors = int(m.get("uniqueBettorCount") or 0)
    except (TypeError, ValueError):
        bettors = 0
    return Market(
        market_id=str(m["id"]),
        market_slug=str(m.get("slug") or ""),
        question=str(m.get("question") or "").strip(),
        description=_description_text(m),
        yes_price=_prob(m.get("probability")),
        closed=closed,
        end_date=_iso_ms(close_ms),
        volume=volume,
        unique_bettors=bettors,
        creator=str(m.get("creatorUsername") or ""),
        resolved_outcome=raw if raw in SCOREABLE else None,
        resolution_raw=raw,
        url=str(m.get("url") or ""),
    )


class ManifoldClient:
    def __init__(self, base_url: str = API_BASE, timeout: float = 20.0,
                 transport: httpx.BaseTransport | None = None):
        self._client = httpx.Client(base_url=base_url, headers=_HEADERS, timeout=timeout, transport=transport)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "ManifoldClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def search_open_binary(self, *, filter: str = "closing-month", sort: str = "score",
                           limit: int = 100, pages: int = 3) -> list[Market]:
        """Open BINARY markets, paged by offset. Duplicates across pages are dropped."""
        seen: dict[str, Market] = {}
        for page in range(pages):
            resp = self._client.get("/v0/search-markets", params={
                "term": "", "filter": filter, "sort": sort, "contractType": "BINARY",
                "limit": limit, "offset": page * limit,
            })
            resp.raise_for_status()
            rows = resp.json() or []
            for raw in rows:
                m = parse_market(raw)
                if m is not None:
                    seen.setdefault(m.market_id, m)
            if len(rows) < limit:
                break
        return list(seen.values())

    def market(self, market_id: str) -> Optional[Market]:
        """One market by id (with description); None when it is gone or not binary."""
        resp = self._client.get(f"/v0/market/{market_id}")
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return parse_market(resp.json())
