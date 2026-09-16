"""Read-only client for Polymarket's public Gamma API.

Only two things are read: an event by slug (with its markets) and a batch of
markets by id. Nothing here can place an order — there is no CLOB write path,
no wallet, no signer, and no dependency that could provide one (retro#620).

Gamma serialises list fields (``outcomes``, ``outcomePrices``) as JSON
*strings*; ``parse_market`` normalises that. A missing/anonymous User-Agent
gets a 403 from Gamma, hence the header (same one ``tm.polymarket`` uses).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

import httpx

GAMMA_BASE = "https://gamma-api.polymarket.com"
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; TruthMachine/1.0)"}


@dataclass(frozen=True)
class Market:
    market_id: str
    event_slug: str
    market_slug: str
    question: str
    description: str
    yes_price: Optional[float]  # None when Gamma publishes no price (placeholder outcomes)
    closed: bool
    end_date: Optional[str]
    volume_usd: float
    resolved_outcome: Optional[str]  # "YES" | "NO" | None while open/unresolved

    @property
    def has_price(self) -> bool:
        return self.yes_price is not None


def _json_list(raw) -> list:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            v = json.loads(raw)
            return v if isinstance(v, list) else []
        except ValueError:
            return []
    return []


def _price(raw) -> Optional[float]:
    try:
        p = float(raw)
    except (TypeError, ValueError):
        return None
    return p if 0.0 <= p <= 1.0 else None


def is_binary_yesno(m: dict) -> bool:
    outcomes = [str(o).strip().lower() for o in _json_list(m.get("outcomes"))]
    return outcomes[:2] == ["yes", "no"]


def resolved_outcome(m: dict) -> Optional[str]:
    """Resolution read off Gamma: a closed market whose YES price collapsed to 1
    or 0. Gamma also carries ``umaResolutionStatus`` but its values have shifted
    over time; the price collapse is the invariant every resolved market shows.
    """
    if not m.get("closed"):
        return None
    prices = _json_list(m.get("outcomePrices"))
    if len(prices) < 2:
        return None
    yes, no = _price(prices[0]), _price(prices[1])
    if yes is None or no is None:
        return None
    if yes >= 0.99 and no <= 0.01:
        return "YES"
    if yes <= 0.01 and no >= 0.99:
        return "NO"
    return None


def parse_market(m: dict, event_slug: str) -> Optional[Market]:
    """Normalise one Gamma market dict; None when it is not a binary yes/no."""
    if not is_binary_yesno(m):
        return None
    prices = _json_list(m.get("outcomePrices"))
    yes_price = _price(prices[0]) if prices else None
    try:
        volume = float(m.get("volumeNum") or m.get("volume") or 0.0)
    except (TypeError, ValueError):
        volume = 0.0
    return Market(
        market_id=str(m.get("id")),
        event_slug=event_slug,
        market_slug=str(m.get("slug") or ""),
        question=str(m.get("question") or "").strip(),
        description=str(m.get("description") or "").strip(),
        yes_price=yes_price,
        closed=bool(m.get("closed")),
        end_date=m.get("endDate"),
        volume_usd=volume,
        resolved_outcome=resolved_outcome(m),
    )


class GammaClient:
    def __init__(self, base_url: str = GAMMA_BASE, timeout: float = 20.0,
                 transport: httpx.BaseTransport | None = None):
        self._client = httpx.Client(base_url=base_url, headers=_HEADERS, timeout=timeout, transport=transport)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "GammaClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def event_markets(self, event_slug: str) -> list[Market]:
        """All binary markets of one event, in Gamma's order. [] when unknown."""
        resp = self._client.get("/events", params={"slug": event_slug})
        resp.raise_for_status()
        events = resp.json() or []
        if not events:
            return []
        out: list[Market] = []
        for raw in events[0].get("markets", []) or []:
            m = parse_market(raw, event_slug)
            if m is not None:
                out.append(m)
        return out

    def universe(self, event_slugs: tuple[str, ...] | list[str]) -> list[Market]:
        markets: list[Market] = []
        for slug in event_slugs:
            markets.extend(self.event_markets(slug))
        return markets
