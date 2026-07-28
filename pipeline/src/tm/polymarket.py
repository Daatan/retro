"""
Polymarket historical data fetcher.

Lookup order for each event:
  1. If ev["polymarket"]["url"] is set, extract event-slug + market-slug from the URL
     and query the Gamma events API directly — precise, no ambiguity.
  2. Fall back to keyword search via Gamma's /public-search (NOT /markets?search=,
     which ignores the query entirely — see _lookup_by_keywords).

Price history is fetched from the CLOB API using the YES-outcome token ID.
Timestamps from the CLOB are Unix seconds (not milliseconds).

Cache schema (data/polymarket/{event_id}.json):
{
  "event_id": "C05",
  "condition_id": "0xabc...",
  "clob_token_yes": "12345...",
  "question": "Another Iran strike on Israel in 2024?",
  "market_url": "https://polymarket.com/event/...",
  "invert": false,          # true if PM question is framed opposite to our outcome
  "prices": [               # sorted oldest → newest, YES-token probability
    {"date": "2024-04-07", "probability": 0.12},
    ...
  ]
}

prices: [] means the market was found but has no CLOB history (old/purged market).
Missing file means the market was not found at all.
"""

import json
import re
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
from rich.console import Console

console = Console()

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE  = "https://clob.polymarket.com"


# ── URL parsing ────────────────────────────────────────────────────────────────

def _slugs_from_url(pm_url: str) -> tuple[str, str]:
    """
    Return (event_slug, market_slug) from a Polymarket URL.

    Format 1: polymarket.com/event/{event-slug}
      → event_slug = market_slug = the single slug

    Format 2: polymarket.com/event/{event-slug}/{market-slug}
      → different event and market slugs
    """
    m = re.search(r"polymarket\.com/event/([^?#]+)", pm_url)
    if not m:
        return "", ""
    parts = m.group(1).rstrip("/").split("/")
    return parts[0], parts[-1]


# ── Gamma API lookup ───────────────────────────────────────────────────────────

async def _lookup_by_url(pm_url: str) -> Optional[dict]:
    """
    Look up the Gamma market using the event-slug embedded in the PM URL.
    Returns a market dict (with clobTokenIds) or None.
    """
    event_slug, market_slug = _slugs_from_url(pm_url)
    if not event_slug:
        return None

    async with httpx.AsyncClient(timeout=15) as client:
        try:
            r = await client.get(
                f"{GAMMA_BASE}/events",
                params={"slug": event_slug, "limit": 1},
            )
            if r.status_code != 200 or not r.json():
                return None
            ev_data = r.json()[0]
        except Exception:
            return None

    markets = ev_data.get("markets", [])
    if not markets:
        return None

    # Prefer the market whose slug matches market_slug; fall back to first.
    def slug_score(mk: dict) -> int:
        s = mk.get("slug", "")
        if s == market_slug:
            return 2
        if s.startswith(market_slug[:30]):
            return 1
        return 0

    return max(markets, key=slug_score)


_STOPWORDS = {
    "the", "a", "an", "is", "are", "will", "be", "to", "of", "in", "on", "for",
    "and", "or", "by", "at", "next", "before", "after", "with", "this", "that",
}


def _significant_words(text: str) -> set[str]:
    """Lowercase words of length >= 4, minus common stopwords."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if len(w) >= 4 and w not in _STOPWORDS}


def _relevance_score(query: str, question: str) -> int:
    """Count of shared significant words between a search query and a
    candidate market's question — a cheap guard against Gamma's own search
    returning an unrelated top hit (observed: it can score 0 lexical overlap)."""
    return len(_significant_words(query) & _significant_words(question))


def _relevance_ratio(query: str, question: str) -> float:
    """Share of the query's significant words the candidate covers.

    A raw count was sufficient while `/markets?search=` ignored the query and
    every candidate shared literally nothing (#334). Against `/public-search`,
    which actually works, candidates are topically *adjacent*, so one shared
    entity is routinely enough to score 1 while answering a different question
    — live examples: "Estonia four-day working week" matching "Will Mart Helme
    be the next President of Estonia?", and "Bitcoin reach $200,000" matching
    "Will MicroStrategy buy 200+ Bitcoin in next purchase?". Requiring a
    proportion of the query to be covered rejects both while keeping genuine
    matches, which share most of their salient words.
    """
    q_words = _significant_words(query)
    if not q_words:
        return 0.0
    return _relevance_score(query, question) / len(q_words)


# Half the query's significant words must appear in the candidate's question.
_MIN_RELEVANCE_RATIO = 0.5


# Query shaping for /public-search. Deliberately separate from _STOPWORDS above:
# that set guards *relevance scoring* (content words, length >= 4), this one
# shapes *the query we send*, so it also drops short function words and the
# framing verbs ("officially", "signs", "announces") that never appear in a
# market title.
_QUERY_STOPWORDS = {
    "a", "an", "the", "of", "or", "and", "to", "in", "on", "for", "by", "be", "will", "is",
    "are", "was", "were", "between", "with", "from", "at", "as", "that", "this", "it", "its",
    "they", "their", "than", "then", "officially", "official", "formally", "formal", "sign",
    "signed", "signs", "announce", "announced", "announces", "reach", "reached", "within",
    "before", "after", "during", "over", "about", "into", "per", "whether",
}

# Market titles say "US", never "USA" — the literal token zeroes out results.
_TOKEN_ALIASES = {"usa": "us", "u.s.a": "us", "u.s": "us"}

_MAX_QUERY_TOKENS = 6


def _compact_amounts(text: str) -> str:
    """Rewrite comma-grouped amounts into the compact form market titles use.

    Claims are written "$200,000"; Polymarket titles say "$200k". Without this
    the tokenizer splits the amount into "200" and "000" — two fragments too
    short for the relevance guard to count, which also burn two of the six
    query slots. Verified live: `bitcoin 200 000 2026` finds nothing, while
    `bitcoin 200k` returns the real "Will Bitcoin reach $200k in ...?" markets.
    """
    def repl(match: re.Match) -> str:
        n = int(match.group(0).replace(",", ""))
        if 1000 <= n < 1_000_000 and n % 1000 == 0:
            return f" {n // 1000}k "
        return f" {n} "

    return re.sub(r"\d{1,3}(?:,\d{3})+", repl, text)


def _search_query(text: str) -> str:
    """Reduce a verbose claim or event name to a short, search-friendly query.

    `/public-search` returns nothing for long natural-language strings and
    weights leading tokens heavily, so keep only salient content words and lead
    with proper nouns — the tokens market titles key on. Mirrors daatan's
    `buildMarketSearchQuery()` (`src/lib/services/external-markets.ts`), which
    solved the same quirks against the same API.
    """
    cleaned = _compact_amounts(text)
    cleaned = re.sub(r"\([^)]*\)", " ", cleaned)
    cleaned = re.sub(r"[^A-Za-z0-9\s.]", " ", cleaned)
    words = [w for w in cleaned.split() if w]

    def norm(w: str) -> str:
        lw = w.lower().rstrip(".")
        return _TOKEN_ALIASES.get(lw, lw)

    def keep(w: str) -> bool:
        return len(w) >= 2 and w not in _QUERY_STOPWORDS

    proper = [norm(w) for w in words if w[:1].isupper()]
    ordered = [w for w in proper if keep(w)] + [w for w in map(norm, words) if keep(w)]

    out: list[str] = []
    for w in ordered:
        if w not in out:
            out.append(w)
        if len(out) >= _MAX_QUERY_TOKENS:
            break
    return " ".join(out)


async def _public_search(client: httpx.AsyncClient, q: str) -> list[dict]:
    """Flatten `/public-search` into a list of Gamma market rows.

    The response is `{"events": [{"markets": [...]}, ...]}`; those market rows
    carry the same `question`/`conditionId`/`clobTokenIds` fields the rest of
    this module expects. Closed markets are kept deliberately — this is a
    historical fetcher, so a settled market is usually exactly what we want.
    """
    try:
        r = await client.get(f"{GAMMA_BASE}/public-search", params={"q": q})
        if r.status_code != 200:
            return []
        data = r.json()
    except Exception:
        return []
    if not isinstance(data, dict):
        return []
    return [
        mk
        for ev in (data.get("events") or [])
        for mk in (ev.get("markets") or [])
        if isinstance(mk, dict)
    ]


async def _lookup_by_keywords(
    keywords: list[str], event_name: str, *, prefer_open: bool = False
) -> Optional[dict]:
    """Keyword search fallback via Gamma's `/public-search`.

    Deliberately NOT `/markets?search=`: that endpoint ignores the query
    entirely and returns a volume-ranked default listing, so this fallback could
    never match on keywords at all (#339 — `search=bitcoin` returned no Bitcoin
    market). `/public-search` is Polymarket's real search, but it returns
    nothing for long natural-language strings, so each phrasing is reduced to
    its salient tokens first, with one broader retry.

    The relevance guard from #334 stays as a backstop, tightened from "shares at
    least one significant word" to "covers at least `_MIN_RELEVANCE_RATIO` of
    the query's significant words" — see `_relevance_ratio` for why the count
    alone stops discriminating once the search actually returns topical results.
    A phrasing that yields nothing acceptable falls through to the next one
    rather than confidently returning an unrelated market.

    `prefer_open` breaks ties in relevance ratio toward an open market rather
    than the first one Gamma happened to list. It never overrides a genuinely
    more relevant match — only ties. Off by default: this function is shared
    with the batch historical fetcher (`resolve_market` below), which wants a
    period-matched settled market, not the newest recurring one. The live MCP
    lookup (`api/.../polymarket_live.py`) passes `prefer_open=True`, because a
    trader typing a natural-language question almost always wants the current,
    tradeable market — Gamma runs many near-identical templated markets across
    months (e.g. a fresh "Bitcoin reach $Nk" market every month), and without
    this a same-ratio stale one can win on iteration order alone. Observed
    live: "Will Bitcoin reach $200,000 in 2026?" matched an October-2025,
    already-closed "$200k in October?" market over an equally-relevant, still-
    open "$200K in July?" one purely because it came first in Gamma's list.
    """
    queries = [kw for kw in keywords if kw and not kw.startswith('"')] + [event_name]
    queries += [kw.strip('"') for kw in keywords if kw.startswith('"')]

    async with httpx.AsyncClient(timeout=15) as client:
        for raw in queries[:4]:
            q = _search_query(raw)
            if not q:
                continue
            candidates = await _public_search(client, q)
            # Rare or over-specific token sets come back empty; retry once with
            # just the leading two tokens for broader recall before giving up.
            tokens = q.split()
            if not candidates and len(tokens) > 2:
                candidates = await _public_search(client, " ".join(tokens[:2]))
            if not candidates:
                continue

            def rank(m: dict) -> tuple:
                ratio = _relevance_ratio(raw, m.get("question", ""))
                # `not m.get("closed", False)` is True for an open market, which
                # sorts after False — so ties in ratio favor open. The ratio
                # always compares first, so this never outranks a better match.
                return (ratio, not m.get("closed", False)) if prefer_open else (ratio,)

            best = max(candidates, key=rank)
            if _relevance_ratio(raw, best.get("question", "")) >= _MIN_RELEVANCE_RATIO:
                return best
    return None


def _extract_clob_token(market: dict) -> Optional[str]:
    """Return the YES-outcome CLOB token ID from a Gamma market dict."""
    tokens = market.get("clobTokenIds") or []
    if isinstance(tokens, str):
        try:
            tokens = json.loads(tokens)
        except Exception:
            tokens = []
    return str(tokens[0]) if tokens else None


# ── CLOB price history ─────────────────────────────────────────────────────────

def _fetch_price_history_sync(clob_token_yes: str, outcome_date: str) -> list[dict]:
    """
    Fetch daily prices from the CLOB API (synchronous — CLOB returns empty with AsyncClient).
    Returns [{date, probability}] sorted oldest→newest, up to outcome_date.
    CLOB timestamps are Unix seconds (not milliseconds).
    """
    try:
        r = httpx.get(
            f"{CLOB_BASE}/prices-history",
            params={"market": clob_token_yes, "interval": "max", "fidelity": 1440},
            timeout=20,
        )
        if r.status_code != 200:
            return []
        history = r.json().get("history", [])
        outcome_dt = datetime.strptime(outcome_date, "%Y-%m-%d").date()
        by_date: dict[str, float] = {}
        for point in history:
            ts = point.get("t", 0)
            prob = point.get("p")
            if ts and prob is not None:
                dt = datetime.fromtimestamp(ts).date()  # CLOB = Unix seconds
                if dt <= outcome_dt:
                    by_date[dt.strftime("%Y-%m-%d")] = round(float(prob), 4)
        return [{"date": d, "probability": v} for d, v in sorted(by_date.items())]
    except Exception as e:
        console.print(f"    [dim red]CLOB price error: {e}[/dim red]")
        return []


async def _fetch_price_history(clob_token_yes: str, outcome_date: str) -> list[dict]:
    """Async wrapper — runs the sync CLOB fetch in a thread pool."""
    return await asyncio.to_thread(_fetch_price_history_sync, clob_token_yes, outcome_date)


# ── Main entry point ───────────────────────────────────────────────────────────

async def fetch_event_prices(event: dict, cache_dir: Path) -> list[dict]:
    """
    Fetch and cache PM price history for one event.
    Returns the prices list (may be empty if no data found).
    Reads from cache if the file exists and has a valid condition_id.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{event['id']}.json"

    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
        # Valid cache: has condition_id AND either has prices or explicitly has no token
        if cached.get("condition_id") and (cached.get("prices") or not cached.get("clob_token_yes")):
            return cached.get("prices", [])

    pm_meta = event.get("polymarket") or {}
    pm_url = pm_meta.get("url", "")
    console.print(f"  [dim cyan]Polymarket lookup: {event['id']} — {event['name'][:50]}[/dim cyan]")

    # Try URL-based lookup first
    market = None
    if pm_url:
        market = await _lookup_by_url(pm_url)
        if market:
            console.print(f"  [dim green]URL lookup → {market.get('question','')[:60]}[/dim green]")
        else:
            console.print(f"  [dim yellow]URL lookup failed, trying keyword search[/dim yellow]")

    if not market:
        market = await _lookup_by_keywords(event.get("search_keywords", []), event["name"])
        if market:
            console.print(f"  [dim green]Keyword search → {market.get('question','')[:60]}[/dim green]")

    if not market:
        console.print(f"  [dim]No Polymarket market found for {event['id']}[/dim]")
        cache_path.write_text(json.dumps({
            "event_id": event["id"], "condition_id": None, "prices": [],
        }))
        return []

    condition_id = market.get("conditionId", "")
    clob_token = _extract_clob_token(market)
    question = market.get("question", "")
    slug = market.get("slug", "")
    market_url = pm_url or (f"https://polymarket.com/event/{slug}" if slug else "")

    prices = []
    if clob_token:
        prices = await _fetch_price_history(clob_token, event["outcome_date"])
        console.print(f"  [dim]CLOB: {len(prices)} daily price points[/dim]")
    else:
        console.print(f"  [dim yellow]No CLOB token found — no price history[/dim yellow]")

    result = {
        "event_id": event["id"],
        "condition_id": condition_id,
        "clob_token_yes": clob_token,
        "question": question,
        "market_url": market_url,
        "invert": pm_meta.get("invert", False),
        "prices": prices,
    }
    cache_path.write_text(json.dumps(result, indent=2))
    return prices


async def prefetch_all(events_dir: Path, cache_dir: Path, event_ids: list[str]):
    """Fetch Polymarket price history for all given event IDs (max 3 concurrent)."""
    events = []
    for eid in event_ids:
        p = events_dir / f"{eid}.json"
        if p.exists():
            events.append(json.loads(p.read_text()))

    sem = asyncio.Semaphore(3)

    async def _fetch_one(ev: dict) -> list:
        async with sem:
            await asyncio.sleep(0.5)  # small stagger to avoid CLOB burst
            return await fetch_event_prices(ev, cache_dir)

    results = await asyncio.gather(*[_fetch_one(ev) for ev in events], return_exceptions=True)
    found = sum(1 for r in results if isinstance(r, list) and r)
    console.print(f"[bold]Polymarket:[/bold] price data for {found}/{len(events)} events")


if __name__ == "__main__":
    import os, sys
    data_dir = Path(os.environ.get("DATA_DIR", "data"))
    event_ids = sys.argv[1:] if len(sys.argv) > 1 else []
    if not event_ids:
        event_ids = [p.stem for p in sorted((data_dir / "events").glob("*.json"))]
    asyncio.run(prefetch_all(data_dir / "events", data_dir / "polymarket", event_ids))
