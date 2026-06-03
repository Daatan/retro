"""
GDELT batch ingestor — unfiltered keyword search across all ~65k GDELT sources.

Runs one query per event (no hardcoded domain list). Results land in
data/raw_ingest/gdelt/{event_id}/, same schema as site_search.py and
web_search_ingest.py, so the orchestrator picks them up unchanged.

Rate limit: GDELT enforces ~1 req / 10s. We use a 12s gap to be safe.
Retries on 429/5xx with exponential backoff.

Usage:
    DATA_DIR=/path/to/data uv run python -m tm.gdelt_ingest
    DATA_DIR=/path/to/data uv run python -m tm.gdelt_ingest --events C05 C07 E07
    DATA_DIR=/path/to/data uv run python -m tm.gdelt_ingest --limit 15 --force
"""

import asyncio
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional

import httpx
from bs4 import BeautifulSoup
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn
from rich.table import Table

console = Console()

GDELT_BASE = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_MIN_INTERVAL = 12.0  # seconds between calls
_last_call: float = 0.0

MVP_EVENTS = [
    "C05", "C06", "C07", "C08", "C09",
    "B04", "B08", "B09", "B10", "B11", "B13",
    "A04", "A12", "A13", "A14", "A15", "A19",
    "D02", "D03",
    "E07", "E08",
    "G02", "G05", "G06",
    "F05",
]


from .utils import _is_ascii, existing_articles
from .article_text import extract_article_body


async def _gdelt_query(
    keywords: List[str],
    start_date: datetime,
    end_date: datetime,
    max_records: int = 10,
    retries: int = 4,
) -> List[Dict]:
    """
    Query GDELT Doc API without a domain filter.
    Returns list of {title, url, date, domain}.
    """
    global _last_call
    english_kws = [k.strip('"') for k in keywords if _is_ascii(k) and k.strip('"')]
    if not english_kws:
        return []

    # OR-join up to 3 keywords — no domain: filter
    kw_query = " OR ".join(f'"{kw}"' for kw in english_kws[:3])

    params = {
        "query": kw_query,
        "mode": "artlist",
        "format": "json",
        "startdatetime": start_date.strftime("%Y%m%d000000"),
        "enddatetime": end_date.strftime("%Y%m%d235959"),
        "maxrecords": max_records,
        "sort": "DateDesc",
    }

    for attempt in range(retries):
        # Enforce rate limit
        elapsed = time.time() - _last_call
        wait = GDELT_MIN_INTERVAL - elapsed
        if wait > 0:
            await asyncio.sleep(wait)
        if attempt > 0:
            backoff = GDELT_MIN_INTERVAL * (2 ** attempt)
            console.print(f"    [dim]GDELT retry {attempt}/{retries-1} — backoff {backoff:.0f}s[/dim]")
            await asyncio.sleep(backoff)

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.get(GDELT_BASE, params=params)
            _last_call = time.time()

            if r.status_code == 200:
                try:
                    data = r.json()
                except Exception:
                    # GDELT sometimes returns 200 with empty/whitespace body under load
                    console.print(f"    [dim yellow]GDELT 200 but empty body — treating as rate-limit[/dim yellow]")
                    _last_call = time.time()
                    continue
                articles = []
                for art in data.get("articles") or []:
                    pub = art.get("seendate", "")
                    try:
                        pub_str = datetime.strptime(pub[:8], "%Y%m%d").strftime("%Y-%m-%d")
                    except ValueError:
                        pub_str = ""
                    articles.append({
                        "title": art.get("title", ""),
                        "url": art.get("url", ""),
                        "date": pub_str,
                        "domain": art.get("domain", ""),
                    })
                return articles

            elif r.status_code in (429, 500, 502, 503):
                _last_call = time.time()
                console.print(f"    [dim yellow]GDELT {r.status_code} — will retry[/dim yellow]")
                continue
            else:
                console.print(f"    [dim red]GDELT {r.status_code}[/dim red]")
                return []

        except Exception as e:
            _last_call = time.time()
            console.print(f"    [dim red]GDELT error: {e}[/dim red]")
            if attempt == retries - 1:
                return []

    return []


async def _fetch_text(url: str) -> str:
    try:
        async with httpx.AsyncClient(
            timeout=20,
            headers={"User-Agent": "Mozilla/5.0 (compatible; TruthMachine/1.0)"},
            follow_redirects=True,
        ) as client:
            r = await client.get(url)
            if r.status_code != 200:
                return ""
        return extract_article_body(BeautifulSoup(r.text, "html.parser"))
    except Exception:
        return ""


async def ingest_event(
    event: dict,
    raw_ingest_dir: Path,
    limit: int = 10,
    force: bool = False,
) -> int:
    """Fetch and save GDELT articles for one event. Returns number saved."""
    eid = event["id"]
    cell_dir = raw_ingest_dir / "gdelt" / eid
    existing = existing_articles(cell_dir)
    if existing and not force:
        return len(existing)

    outcome_dt = datetime.strptime(event["outcome_date"], "%Y-%m-%d")
    window = int(event.get("predictive_window_days", 30))
    start_dt = outcome_dt - timedelta(days=window)

    keywords = event.get("duel_keywords") or event.get("search_keywords", [])

    # Run up to 2 keyword queries (different keyword sets) and merge results
    seen_urls: set[str] = set()
    candidates: list[dict] = []

    kw_groups = [keywords, keywords[1:]] if len(keywords) > 1 else [keywords]
    for kws in kw_groups[:2]:
        results = await _gdelt_query(kws, start_dt, outcome_dt, max_records=limit)
        if not results:
            break  # GDELT is rate-limited or dry; skip second query group
        for art in results:
            if art["url"] and art["url"] not in seen_urls:
                seen_urls.add(art["url"])
                candidates.append(art)

    if not candidates:
        return 0

    cell_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    article_idx = len(existing)

    for art in candidates[:limit]:
        await asyncio.sleep(1.0)
        url = art["url"]
        text = await _fetch_text(url)
        if len(text) < 200:
            continue

        article = {
            "headline": art["title"],
            "text": text,
            "published_at": art["date"] or start_dt.strftime("%Y-%m-%d"),
            "estimated_date": not bool(art["date"]),
            "author": "Unknown",
            "url": url,
            "source_domain": art["domain"],
        }
        article_idx += 1
        out = cell_dir / f"article_{article_idx:02d}.json"
        out.write_text(json.dumps(article, indent=2, ensure_ascii=False))
        saved += 1

    return saved


async def run_batch(
    data_dir: Path,
    event_ids: List[str],
    limit: int,
    force: bool,
):
    events_dir = data_dir / "events"
    raw_ingest_dir = data_dir / "raw_ingest"

    events: dict[str, dict] = {}
    for eid in event_ids:
        p = events_dir / f"{eid}.json"
        if p.exists():
            events[eid] = json.loads(p.read_text())
        else:
            console.print(f"[yellow]Event {eid} not found, skipping[/yellow]")

    console.print(
        f"\n[bold]GDELT Batch Ingest[/bold]  "
        f"{len(events)} events · up to {limit} articles each\n"
    )

    results: dict[str, int] = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(), MofNCompleteColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Fetching…", total=len(events))

        for eid, event in events.items():
            progress.update(task, description=f"[cyan]{eid}[/cyan] — {event['name'][:40]}")
            count = await ingest_event(event, raw_ingest_dir, limit, force)
            results[eid] = count
            label = f"[green]{count} saved[/green]" if count else "[dim]0[/dim]"
            console.print(f"  {eid}: {label}")
            progress.advance(task)

    table = Table(title="GDELT Ingest Summary")
    table.add_column("Event", style="cyan")
    table.add_column("Articles saved", justify="right")
    grand = 0
    for eid in event_ids:
        if eid not in results:
            continue
        n = results[eid]
        table.add_row(eid, str(n) if n else "·")
        grand += n

    console.print(table)
    console.print(f"\n[bold green]Total articles saved: {grand}[/bold green]")
    console.print(
        f"Next: [bold]DATA_DIR={data_dir} uv run python -m tm.orchestrator local_file[/bold]"
    )


def main():
    import argparse
    p = argparse.ArgumentParser(description="GDELT batch ingestor (unfiltered keyword search)")
    p.add_argument("--events", nargs="+", default=MVP_EVENTS, metavar="EID")
    p.add_argument("--limit", type=int, default=10, help="Max articles per event (default 10)")
    p.add_argument("--force", action="store_true", help="Re-fetch already populated cells")
    args = p.parse_args()

    data_dir = Path(os.environ.get("DATA_DIR", "/app/data"))
    asyncio.run(run_batch(data_dir, args.events, args.limit, args.force))


if __name__ == "__main__":
    main()
