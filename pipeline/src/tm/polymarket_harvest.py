"""
Polymarket Harvest — bulk download of resolved political/geopolitical markets.

Fetches all resolved binary markets from the Gamma API (2023-01-01 to today),
filters to political/geopolitical categories, validates outcome clarity, and
writes data/poc/pm_harvest/events.jsonl for poc_event_gen.py to consume.

Usage:
    python -m tm.polymarket_harvest
    python -m tm.polymarket_harvest --data-dir /path/to/data/poc --start 2024-01-01

Output:
    data/poc/pm_harvest/events.jsonl   — one JSON object per line
    data/poc/pm_harvest/raw_page_N.json — cached raw API pages (resumable)
"""

import argparse
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

logger = logging.getLogger(__name__)
console = Console()

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"
PAGE_SIZE = 100
REQUEST_DELAY = 0.5  # seconds between pages (be polite)

# Polymarket category values to include (exact match, case-insensitive)
# NOTE: only old (<2022) markets have categories populated; newer ones are empty.
POLITICAL_CATEGORIES = {
    "global politics",
    "us-current-affairs",
    "politics",
    "elections",
    "geopolitics",
    "world",
}

# Keyword fragments — question must contain at least one (case-insensitive)
POLITICAL_KEYWORDS = [
    # Electoral
    "election", "elect", "vote", "ballot", "candidate", "primary", "runoff",
    "inaugur", "senate", "congress", "parliament", "referendum", "poll ",
    # Leadership
    "president", "prime minister", "chancellor", "premier", "minister",
    "trump", "biden", "harris", "zelensky", "putin", "macron", "scholz",
    "xi jinping", "netanyahu", "modi", "erdogan", "meloni",
    # Geopolitical events
    "war", "ceasefire", "sanction", "invasion", "nato", "un security",
    "treaty", "diplomacy", "annexat", "occupation", "missile", "nuclear",
    "ukraine", "russia", "israel", "gaza", "iran", "north korea", "taiwan",
    # Policy
    "legislation", "bill pass", "impeach", "indict", "convict", "resign",
    "tariff", "trade deal", "g7", "g20",
]

# Resolution clarity thresholds
OUTCOME_YES_THRESHOLD = 0.95   # final prob ≥ this → outcome = True
OUTCOME_NO_THRESHOLD = 0.05    # final prob ≤ this → outcome = False
MIN_PRICE_POINTS = 5           # need at least this many daily price points


# Patterns that disqualify a question even if it contains political keywords
_NOISE_PATTERNS = re.compile(
    r"nhl:|nba:|nfl:|mlb:|premier league|bundesliga|la liga|serie a|"
    r"ufc \d|will .{0,30} (score|win|lose|beat) .{0,20} (in|at) (game|match|quarter)|"
    r"will .{0,20} say ['\"]|will .{0,20} tweet|"
    r"\bsports?\b|\bchess\b|horse racing|formula [e1]|"
    r"cryptocurrency|bitcoin|ethereum|crypto|nft|"
    r"box office|oscars?|grammy|emmy|super bowl halftime",
    re.I,
)


def _is_political(market: dict) -> bool:
    """Check if market belongs to political/geopolitical category."""
    question = (market.get("question") or "").lower()

    # Hard exclusions — disqualify noise even if political keyword present
    if _NOISE_PATTERNS.search(question):
        return False

    # Old markets have explicit categories
    category = (market.get("category") or "").strip().lower()
    if category in POLITICAL_CATEGORIES:
        return True

    # New markets (~2022+) have empty category — use question keyword matching
    return any(kw in question for kw in POLITICAL_KEYWORDS)


def _parse_outcome_date(market: dict) -> str | None:
    """Extract resolution date as YYYY-MM-DD string."""
    for field in ("endDate", "resolutionTime", "startDate"):
        val = market.get(field)
        if not val:
            continue
        try:
            # ISO format: "2024-11-05T00:00:00Z" or "2024-11-05"
            dt = datetime.fromisoformat(val.replace("Z", "+00:00"))
            return dt.strftime("%Y-%m-%d")
        except Exception:
            pass
    return None


def _extract_outcome(market: dict) -> bool | None:
    """
    Determine binary outcome from the market's outcomePrices field.
    outcomePrices is a JSON array string like '["0", "1"]' where index 0 = Yes, index 1 = No.
    Returns True (Yes resolved), False (No resolved), or None if ambiguous.
    """
    # Try outcomePrices first (fastest — already in market data, no API call)
    raw = market.get("outcomePrices")
    if raw:
        try:
            prices_arr = json.loads(raw) if isinstance(raw, str) else raw
            if len(prices_arr) >= 2:
                yes_p = float(prices_arr[0])
                no_p = float(prices_arr[1])
                if yes_p >= OUTCOME_YES_THRESHOLD:
                    return True
                if no_p >= OUTCOME_YES_THRESHOLD or yes_p <= OUTCOME_NO_THRESHOLD:
                    return False
        except (ValueError, TypeError, json.JSONDecodeError):
            pass

    # Fall back to lastTradePrice
    last = market.get("lastTradePrice")
    if last is not None:
        try:
            p = float(last)
            if p >= OUTCOME_YES_THRESHOLD:
                return True
            if p <= OUTCOME_NO_THRESHOLD:
                return False
        except (ValueError, TypeError):
            pass

    # Explicit resolution string
    res_val = (market.get("resolutionValue") or "").strip().lower()
    if res_val in ("yes", "1", "true"):
        return True
    if res_val in ("no", "0", "false"):
        return False

    return None  # ambiguous


def _fetch_price_history(clob_token_yes: str, outcome_date: str, client: httpx.Client) -> list[dict]:
    """
    Fetch and normalise daily prices for a market using the CLOB API.
    clob_token_yes is the first element of clobTokenIds (the "Yes" outcome token).
    Timestamps are Unix seconds (not milliseconds) in the CLOB API.
    """
    try:
        r = client.get(
            f"{CLOB_BASE}/prices-history",
            params={"market": clob_token_yes, "interval": "max", "fidelity": 1440},
            timeout=30,
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
                # CLOB API returns Unix seconds (not milliseconds)
                dt = datetime.fromtimestamp(ts, tz=timezone.utc).date()
                if dt <= outcome_dt:
                    by_date[dt.strftime("%Y-%m-%d")] = round(float(prob), 4)
        return [{"date": d, "probability": v} for d, v in sorted(by_date.items())]
    except Exception as exc:
        logger.debug("Price history error for token %s: %s", clob_token_yes, exc)
        return []


def harvest(
    data_dir: Path,
    start_date: str = "2023-01-01",
    end_date: str | None = None,
) -> list[dict]:
    """
    Main harvest function. Returns list of harvested event dicts and
    writes data_dir/pm_harvest/events.jsonl.
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    harvest_dir = data_dir / "pm_harvest"
    harvest_dir.mkdir(parents=True, exist_ok=True)
    output_path = harvest_dir / "events.jsonl"

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    console.print(f"[bold cyan]Polymarket Harvest[/bold cyan] — {start_date} → {end_date}")
    console.print(f"Output: {output_path}")

    # Load already-harvested IDs to allow resumption
    seen_ids: set[str] = set()
    existing_events: list[dict] = []
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    ev = json.loads(line)
                    seen_ids.add(ev["pm_id"])
                    existing_events.append(ev)
        console.print(f"  Resuming — {len(seen_ids)} events already harvested")

    new_events: list[dict] = []
    page = 0
    total_checked = 0
    total_skipped_category = 0
    total_skipped_ambiguous = 0

    with httpx.Client(timeout=30) as client:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Fetching pages...", total=None)

            while True:
                raw_cache = harvest_dir / f"raw_page_{page}.json"

                # Load from cache, or fetch and cache
                markets = None
                if raw_cache.exists():
                    try:
                        with open(raw_cache) as f:
                            markets = json.load(f)
                    except (json.JSONDecodeError, ValueError):
                        logger.warning("Corrupted cache page %d — re-fetching", page)
                        raw_cache.unlink()

                if markets is None:
                    try:
                        r = client.get(
                            f"{GAMMA_BASE}/markets",
                            params={
                                "closed": "true",
                                "limit": PAGE_SIZE,
                                "offset": page * PAGE_SIZE,
                            },
                            timeout=20,
                        )
                        if r.status_code != 200:
                            logger.warning("Page %d: HTTP %d", page, r.status_code)
                            break
                        markets = r.json()
                        with open(raw_cache, "w") as f:
                            json.dump(markets, f)
                        time.sleep(REQUEST_DELAY)
                    except Exception as exc:
                        logger.error("Page %d fetch error: %s", page, exc)
                        break

                if not markets:
                    break  # End of pagination

                for market in markets:
                    total_checked += 1
                    market_id = market.get("id") or market.get("conditionId", "")
                    if not market_id or market_id in seen_ids:
                        continue

                    # Date filter
                    outcome_date = _parse_outcome_date(market)
                    if not outcome_date:
                        continue
                    outcome_dt = datetime.strptime(outcome_date, "%Y-%m-%d")
                    if not (start_dt <= outcome_dt <= end_dt):
                        continue

                    # Category filter
                    if not _is_political(market):
                        total_skipped_category += 1
                        continue

                    # Outcome clarity — use outcomePrices from market (no API call needed)
                    outcome = _extract_outcome(market)
                    if outcome is None:
                        total_skipped_ambiguous += 1
                        continue

                    slug = market.get("slug", "")
                    question = market.get("question") or market.get("title") or market.get("name") or ""
                    if not question:
                        continue

                    # Infer category label
                    category = (market.get("category") or "Politics").strip()

                    # clobTokenIds[0] = Yes-outcome token (needed for CLOB price history API)
                    # The Gamma API sometimes returns clobTokenIds as a JSON-encoded string
                    clob_tokens = market.get("clobTokenIds") or []
                    if isinstance(clob_tokens, str):
                        try:
                            clob_tokens = json.loads(clob_tokens)
                        except Exception:
                            clob_tokens = []
                    clob_token_yes = str(clob_tokens[0]) if clob_tokens else None

                    event = {
                        "pm_id": market_id,
                        "question": question.strip(),
                        "outcome": outcome,
                        "outcome_date": outcome_date,
                        "category": category,
                        "pm_url": f"https://polymarket.com/event/{slug}" if slug else "",
                        "clob_token_yes": clob_token_yes,
                        # prices fetched separately via fetch_prices() to keep harvest fast
                        "prices": [],
                    }

                    seen_ids.add(market_id)
                    new_events.append(event)

                    # Append to output file immediately (crash-safe)
                    with open(output_path, "a") as f:
                        f.write(json.dumps(event) + "\n")

                progress.update(
                    task,
                    description=(
                        f"Page {page} — {len(existing_events) + len(new_events)} events "
                        f"(checked {total_checked}, skipped cat={total_skipped_category} "
                        f"ambig={total_skipped_ambiguous})"
                    ),
                )
                page += 1

    all_events = existing_events + new_events
    console.print(f"\n[bold green]Done.[/bold green] Total events: {len(all_events)}")
    console.print(f"  New this run: {len(new_events)}")
    console.print(f"  Skipped — not political: {total_skipped_category}")
    console.print(f"  Skipped — ambiguous outcome: {total_skipped_ambiguous}")
    return all_events


def backfill_clob_tokens(data_dir: Path) -> int:
    """
    Add clob_token_yes to events that lack it.
    First tries cached raw_page_N.json files; falls back to paging the Gamma API.
    Also resets prices_fetched=False for updated events so fetch_prices() re-runs them.
    Returns count of events updated.
    """
    harvest_dir = data_dir / "pm_harvest"
    output_path = harvest_dir / "events.jsonl"
    if not output_path.exists():
        console.print("[red]events.jsonl not found[/red]")
        return 0

    events: list[dict] = []
    with open(output_path) as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))

    # Build set of pm_ids that still need a clob token
    # Also treat broken tokens (len<=2, e.g. "[") as missing — they were mis-parsed
    def _token_valid(ev: dict) -> bool:
        t = ev.get("clob_token_yes")
        return bool(t and len(str(t)) > 10)

    need_token: set[str] = {str(ev["pm_id"]) for ev in events if not _token_valid(ev)}
    if not need_token:
        console.print("[green]All events already have clob_token_yes — nothing to do.[/green]")
        return 0

    console.print(f"[bold cyan]Backfill CLOB tokens[/bold cyan] — {len(need_token)} events need token")

    # Pass 1: try cached raw pages (fast, no API calls)
    clob_lookup: dict[str, str] = {}
    page = 0
    cached_pages = 0
    while True:
        raw_cache = harvest_dir / f"raw_page_{page}.json"
        if not raw_cache.exists():
            break
        try:
            with open(raw_cache) as f:
                markets = json.load(f)
            for m in markets:
                mid = str(m.get("id") or m.get("conditionId", ""))
                tokens = m.get("clobTokenIds") or []
                if isinstance(tokens, str):
                    try:
                        tokens = json.loads(tokens)
                    except Exception:
                        tokens = []
                if mid and tokens and mid in need_token:
                    clob_lookup[mid] = str(tokens[0])
            cached_pages += 1
        except Exception:
            pass
        page += 1

    if cached_pages:
        console.print(f"  From cache ({cached_pages} pages): found {len(clob_lookup)} tokens")

    # Pass 2: fetch remaining from Gamma API (paging through all closed markets)
    still_need = need_token - set(clob_lookup.keys())
    if still_need:
        console.print(f"  Fetching {len(still_need)} remaining tokens from Gamma API...")
        fetched_from_api = 0
        api_page = 0
        with httpx.Client(timeout=30) as client:
            while still_need:
                raw_cache = harvest_dir / f"raw_page_{api_page}.json"
                markets = None
                if raw_cache.exists():
                    try:
                        with open(raw_cache) as f:
                            markets = json.load(f)
                    except Exception:
                        pass

                if markets is None:
                    try:
                        r = client.get(
                            f"{GAMMA_BASE}/markets",
                            params={"closed": "true", "limit": PAGE_SIZE, "offset": api_page * PAGE_SIZE},
                            timeout=20,
                        )
                        if r.status_code != 200:
                            logger.warning("Backfill page %d: HTTP %d", api_page, r.status_code)
                            break
                        markets = r.json()
                        # Cache for future use
                        with open(raw_cache, "w") as f:
                            json.dump(markets, f)
                        time.sleep(REQUEST_DELAY)
                    except Exception as exc:
                        logger.error("Backfill page %d error: %s", api_page, exc)
                        break

                if not markets:
                    break

                for m in markets:
                    mid = str(m.get("id") or m.get("conditionId", ""))
                    tokens = m.get("clobTokenIds") or []
                    if isinstance(tokens, str):
                        try:
                            tokens = json.loads(tokens)
                        except Exception:
                            tokens = []
                    if mid and tokens and mid in still_need:
                        clob_lookup[mid] = str(tokens[0])
                        still_need.discard(mid)
                        fetched_from_api += 1

                api_page += 1
                if api_page % 20 == 0:
                    console.print(f"    Page {api_page}, {len(still_need)} still needed...")

        console.print(f"  From API: found {fetched_from_api} additional tokens")

    # Apply updates
    updated = 0
    for ev in events:
        mid = str(ev.get("pm_id", ""))
        if ev.get("clob_token_yes"):
            continue
        token = clob_lookup.get(mid)
        if token:
            ev["clob_token_yes"] = token
            ev["prices_fetched"] = False  # force re-fetch with correct token
            updated += 1

    with open(output_path, "w") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")

    console.print(f"[bold green]Backfill done.[/bold green] Updated {updated}/{len(events)} events.")
    return updated


def _atomic_rewrite(output_path: Path, events: list[dict]) -> None:
    """Rewrite ``output_path`` with ``events`` via a temp-file + atomic rename.

    ``os.replace`` is atomic on POSIX and Windows, so a crash mid-write leaves
    either the old file intact or the new one fully written — never a
    truncated/partial file.
    """
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")
    os.replace(tmp_path, output_path)


def fetch_prices(data_dir: Path) -> None:
    """
    Second pass: fetch price history for all events in events.jsonl that have empty prices.
    Uses the CLOB API with clob_token_yes (stored during harvest).

    Crash-safe: like harvest(), progress is persisted incrementally (via an atomic
    temp-file + rename after each fetched event) rather than accumulated in memory
    and written once at the end — a crash partway through only loses the
    in-flight event, and the already-fetched ones' `prices_fetched=True` flag
    lets a re-run skip them and resume from where it stopped.
    Run after harvest() completes.
    """
    output_path = data_dir / "pm_harvest" / "events.jsonl"
    if not output_path.exists():
        console.print("[red]events.jsonl not found — run harvest first[/red]")
        return

    events: list[dict] = []
    with open(output_path) as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))

    # "prices_fetched" flag distinguishes "not yet fetched" from "fetched but no history"
    # Events without clob_token_yes can't use the CLOB API — backfill first
    need_prices = [e for e in events if not e.get("prices_fetched", False)]
    no_token = [e for e in need_prices if not e.get("clob_token_yes")]
    if no_token:
        console.print(
            f"[yellow]WARNING: {len(no_token)} events have no clob_token_yes — "
            f"run --backfill-clob-tokens first to recover from cached pages[/yellow]"
        )
    need_prices = [e for e in need_prices if e.get("clob_token_yes")]
    console.print(f"[bold cyan]Fetch prices[/bold cyan] — {len(need_prices)}/{len(events)} events need price history")

    fetched = 0
    with httpx.Client(timeout=30) as client:
        for i, ev in enumerate(need_prices):
            try:
                prices = _fetch_price_history(ev["clob_token_yes"], ev["outcome_date"], client)
                ev["prices"] = prices
                ev["prices_fetched"] = True
                fetched += 1
            finally:
                # Persist after every event (success or failure) so a crash never
                # loses more than the single in-flight fetch.
                _atomic_rewrite(output_path, events)
            if (i + 1) % 50 == 0:
                console.print(f"  {i+1}/{len(need_prices)} fetched — last had {len(prices)} points...")
                time.sleep(0.5)

    console.print(f"[bold green]Done.[/bold green] Price history written for {fetched} events.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)

    parser = argparse.ArgumentParser(description="Harvest Polymarket political events")
    parser.add_argument("--data-dir", default="data/poc", help="Base data directory (default: data/poc)")
    parser.add_argument("--start", default="2023-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--fetch-prices", action="store_true",
                        help="Second pass: fetch price history for harvested events via CLOB API")
    parser.add_argument("--backfill-clob-tokens", action="store_true",
                        help="Backfill clob_token_yes from cached raw pages (run once on existing data)")
    args = parser.parse_args()

    base = Path(os.environ.get("DATA_DIR", args.data_dir))
    if args.backfill_clob_tokens:
        backfill_clob_tokens(base)
    elif args.fetch_prices:
        fetch_prices(base)
    else:
        harvest(base, start_date=args.start, end_date=args.end)
