"""Live Polymarket market lookup for the MCP tools.

Thin adapter over tm.polymarket (Gamma API) that resolves a user-supplied market
reference — a polymarket.com URL, an event/market slug, or a numeric Gamma market
id — to a live Gamma market dict, and pulls the current YES-outcome price from it.

Keeps Gamma/CLOB specifics out of the tool layer. NB: the local dev network is
firewalled off polymarket.com/gamma, so these run only on the Oracle host / prod
(see the "Polymarket search quirks" runbook).
"""

import json
from typing import Optional

import httpx

from tm.polymarket import GAMMA_BASE, _lookup_by_keywords, _lookup_by_url

_GAMMA_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; TruthMachine/1.0)"}


async def _lookup_by_id(market_id: str) -> Optional[dict]:
    """Fetch a Gamma market by numeric id."""
    async with httpx.AsyncClient(timeout=15) as client:
        try:
            r = await client.get(
                f"{GAMMA_BASE}/markets", params={"id": market_id}, headers=_GAMMA_HEADERS
            )
            if r.status_code == 200 and r.json():
                data = r.json()
                return data[0] if isinstance(data, list) else data
        except Exception:
            return None
    return None


async def resolve_market(market: str) -> Optional[dict]:
    """Resolve a market reference to a live Gamma market dict, or None.

    Accepts, in order:
      - a full polymarket.com URL  → event/market-slug lookup
      - a bare numeric Gamma id    → direct id lookup
      - anything else (a slug/text)→ keyword search
    """
    market = (market or "").strip()
    if not market:
        return None
    if "polymarket.com" in market:
        return await _lookup_by_url(market)
    if market.isdigit():
        return await _lookup_by_id(market)
    # Treat as a slug: try the event lookup with a synthesized URL first (precise
    # when it's really a slug), then fall back to keyword search.
    by_slug = await _lookup_by_url(f"https://polymarket.com/event/{market}")
    if by_slug:
        return by_slug
    return await _lookup_by_keywords([market], market)


def _json_list(raw) -> list:
    """Gamma serialises list fields (outcomes, outcomePrices) as JSON strings."""
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            v = json.loads(raw)
            return v if isinstance(v, list) else []
        except Exception:
            return []
    return []


def current_yes_price(market: dict) -> Optional[float]:
    """Current YES-outcome implied probability [0,1], or None.

    Reads outcomePrices[0] (index 0 = Yes) first — already in the market payload,
    no extra API call — then falls back to lastTradePrice. Mirrors
    tm.polymarket_harvest._extract_outcome's price parsing.
    """
    prices = _json_list(market.get("outcomePrices"))
    if prices:
        try:
            p = float(prices[0])
            if 0.0 <= p <= 1.0:
                return p
        except (ValueError, TypeError):
            pass
    last = market.get("lastTradePrice")
    if last is not None:
        try:
            p = float(last)
            if 0.0 <= p <= 1.0:
                return p
        except (ValueError, TypeError):
            pass
    return None


def is_binary_yesno(market: dict) -> bool:
    """True when the market is a plain binary Yes/No (what the edge tool supports)."""
    outcomes = [str(o).strip().lower() for o in _json_list(market.get("outcomes"))]
    return outcomes[:2] == ["yes", "no"]


def summarize_market(market: dict) -> dict:
    """Compact, agent-friendly view of a Gamma market dict."""
    outcomes = _json_list(market.get("outcomes"))
    prices = _json_list(market.get("outcomePrices"))
    slug = market.get("slug") or ""
    return {
        "id": market.get("id"),
        "question": market.get("question"),
        "slug": slug,
        "url": f"https://polymarket.com/event/{slug}" if slug else None,
        "outcomes": outcomes,
        "outcome_prices": [float(p) for p in prices if _is_float(p)],
        "yes_price": current_yes_price(market),
        "binary_yesno": is_binary_yesno(market),
        "volume": market.get("volumeNum") or market.get("volume"),
        "end_date": market.get("endDate"),
        "active": market.get("active"),
        "closed": market.get("closed"),
    }


def _is_float(v) -> bool:
    try:
        float(v)
        return True
    except (ValueError, TypeError):
        return False
