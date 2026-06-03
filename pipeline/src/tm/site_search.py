"""
Direct site-search scraper — no API keys, no rate limits.

Searches each news site's own search engine and scrapes article URLs,
then fetches full text. Much more reliable than GDELT for historical articles.

Supported sources:
  toi       → timesofisrael.com/?s=QUERY
  jpost     → jpost.com/?s=QUERY
  reuters   → reuters.com/site-search/?query=QUERY  (JS-rendered, uses snippets)
  haaretz   → haaretz.com/misc/search-results  (English edition)
  globes    → en.globes.co.il/en/search/  (English)

Usage:
    DATA_DIR=/path/to/data uv run python -m tm.site_search
    DATA_DIR=/path/to/data uv run python -m tm.site_search --events C05 E07 --sources toi jpost
"""

import asyncio
import json
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
from urllib.parse import urlencode, quote_plus

import httpx
from bs4 import BeautifulSoup
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn
from rich.table import Table

console = Console()

MVP_EVENTS = [
    "C05", "C06", "C07", "C08", "C09",
    "B04", "B08", "B09", "B10", "B11", "B13",
    "A04", "A12", "A13", "A14", "A15", "A19",
    "D02", "D03",
    "E07", "E08",
    "G02", "G05", "G06",
    "F05",
]

from .article_text import BROWSER_HEADERS, extract_article_body


async def _get(url: str, timeout: int = 20) -> Optional[BeautifulSoup]:
    try:
        async with httpx.AsyncClient(
            headers=BROWSER_HEADERS, timeout=timeout, follow_redirects=True
        ) as client:
            r = await client.get(url)
            if r.status_code == 200:
                return BeautifulSoup(r.text, "html.parser")
    except Exception as e:
        console.print(f"    [dim red]fetch error: {e}[/dim red]")
    return None


async def search_toi(keywords: List[str], start_date: datetime, end_date: datetime) -> List[str]:
    """Times of Israel search — returns article URLs."""
    q = " ".join(k.strip('"') for k in keywords if _is_ascii(k))[:60]
    url = f"https://www.timesofisrael.com/?s={quote_plus(q)}"
    soup = await _get(url)
    if not soup:
        return []
    urls = []
    for a in soup.select("a.item-title, h2.post-title a, h3.post-title a, .article-title a"):
        href = a.get("href", "")
        if "timesofisrael.com" in href and "/20" in href:
            urls.append(href)
    return list(dict.fromkeys(urls))[:8]  # deduplicate, max 8


async def search_jpost(keywords: List[str], start_date: datetime, end_date: datetime) -> List[str]:
    """Jerusalem Post search."""
    q = " ".join(k.strip('"') for k in keywords if _is_ascii(k))[:60]
    url = f"https://www.jpost.com/?s={quote_plus(q)}"
    soup = await _get(url)
    if not soup:
        return []
    urls = []
    for a in soup.select("article a, .article-title a, h2 a, h3 a"):
        href = a.get("href", "")
        if "jpost.com" in href and href.startswith("https://"):
            urls.append(href)
    return list(dict.fromkeys(urls))[:8]


async def search_haaretz(keywords: List[str], start_date: datetime, end_date: datetime) -> List[str]:
    """Haaretz English — often paywalled but titles are useful."""
    q = " ".join(k.strip('"') for k in keywords if _is_ascii(k))[:60]
    url = f"https://www.haaretz.com/search/?q={quote_plus(q)}"
    soup = await _get(url)
    if not soup:
        return []
    urls = []
    for a in soup.select("a[href*='/news/'], a[href*='/israel-news/'], a[href*='/opinion/']"):
        href = a.get("href", "")
        if href.startswith("https://www.haaretz.com") and len(href) > 40:
            urls.append(href)
    return list(dict.fromkeys(urls))[:8]


async def search_reuters(keywords: List[str], start_date: datetime, end_date: datetime) -> List[str]:
    """Reuters site search — returns snippets (JS-rendered, fallback to URL scrape)."""
    q = " ".join(k.strip('"') for k in keywords if _is_ascii(k))[:80]
    # Reuters search API (often JS, try direct)
    url = f"https://www.reuters.com/site-search/?query={quote_plus(q)}&section=world"
    soup = await _get(url)
    if not soup:
        return []
    urls = []
    for a in soup.select("a[href*='/world/'], a[href*='/business/'], a[href*='/markets/']"):
        href = a.get("href", "")
        if href.startswith("/") and len(href) > 20:
            href = "https://www.reuters.com" + href
        if "reuters.com" in href and "-202" in href:
            urls.append(href)
    return list(dict.fromkeys(urls))[:8]


async def search_globes(keywords: List[str], start_date: datetime, end_date: datetime) -> List[str]:
    """Globes English edition search."""
    q = " ".join(k.strip('"') for k in keywords if _is_ascii(k))[:60]
    url = f"https://en.globes.co.il/en/search/{quote_plus(q)}"
    soup = await _get(url)
    if not soup:
        return []
    urls = []
    for a in soup.select("a[href*='article']"):
        href = a.get("href", "")
        if "globes.co.il" in href and "article" in href:
            urls.append(href)
    return list(dict.fromkeys(urls))[:8]


SEARCH_FNS = {
    "toi":     (search_toi,     "timesofisrael.com"),
    "jpost":   (search_jpost,   "jpost.com"),
    "haaretz": (search_haaretz, "haaretz.com"),
    "reuters": (search_reuters, "reuters.com"),
    "globes":  (search_globes,  "en.globes.co.il"),
}


async def fetch_article_text(url: str) -> str:
    soup = await _get(url, timeout=25)
    if not soup:
        return ""
    return extract_article_body(soup)


def _parse_date_from_url(url: str, fallback: str) -> str:
    """Extract YYYY-MM-DD from URL if present."""
    m = re.search(r"/(20\d\d)[/-](\d\d)[/-](\d\d)", url)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return fallback


from .utils import _is_ascii, existing_articles, save_article


def _in_window(date_str: str, start: datetime, end: datetime) -> bool:
    try:
        dt = datetime.strptime(date_str[:10], "%Y-%m-%d")
        return start <= dt <= end
    except ValueError:
        return True  # keep if we can't determine


async def ingest_cell(
    event: dict,
    source_id: str,
    raw_ingest_dir: Path,
    force: bool = False,
) -> int:
    cell_dir = raw_ingest_dir / source_id / event["id"]
    existing = existing_articles(cell_dir)
    if existing and not force:
        return len(existing)

    if source_id not in SEARCH_FNS:
        return 0

    search_fn, domain = SEARCH_FNS[source_id]
    outcome_dt = datetime.strptime(event["outcome_date"], "%Y-%m-%d")
    window = int(event.get("predictive_window_days", 14))
    start_dt = outcome_dt - timedelta(days=window)
    kws = event.get("duel_keywords") or event.get("search_keywords", [])

    urls = await search_fn(kws, start_dt, outcome_dt)
    if not urls:
        return 0

    saved = 0

    for i, url in enumerate(urls[:5]):
        await asyncio.sleep(1.5)  # polite delay between article fetches

        text = await fetch_article_text(url)
        if len(text) < 250:
            continue

        art_date = _parse_date_from_url(url, start_dt.strftime("%Y-%m-%d"))

        article = {
            "headline": url.split("/")[-1].replace("-", " ").strip("/"),
            "text": text,
            "published_at": art_date,
            "estimated_date": not bool(_parse_date_from_url(url, "")),
            "author": "Unknown",
            "url": url,
        }
        save_article(cell_dir, i + 1, article)
        saved += 1

    return saved


async def run_batch(
    data_dir: Path,
    event_ids: List[str],
    source_filter: Optional[List[str]],
    force: bool,
):
    events_dir = data_dir / "events"
    raw_ingest_dir = data_dir / "raw_ingest"

    events = {}
    for eid in event_ids:
        p = events_dir / f"{eid}.json"
        if p.exists():
            events[eid] = json.loads(p.read_text())
        else:
            console.print(f"[yellow]Event {eid} not found, skipping[/yellow]")

    sources = [s for s in SEARCH_FNS if source_filter is None or s in source_filter]
    total = len(events) * len(sources)

    console.print(f"\n[bold]Site-Search Batch Ingest[/bold]  "
                  f"{len(events)} events × {len(sources)} sources = {total} cells\n")

    results: dict[str, dict] = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(), MofNCompleteColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Fetching…", total=total)

        for eid, event in events.items():
            results[eid] = {}
            for sid in sources:
                progress.update(task, description=f"[cyan]{eid}[/cyan]/[blue]{sid}[/blue]")
                count = await ingest_cell(event, sid, raw_ingest_dir, force)
                results[eid][sid] = count
                label = f"[green]{count} art[/green]" if count else "[dim]0[/dim]"
                console.print(f"  {eid}/{sid}: {label}")
                progress.advance(task)
                await asyncio.sleep(2)  # polite gap between site searches

    # Summary
    table = Table(title="Ingest Summary")
    table.add_column("Event", style="cyan")
    for sid in sources:
        table.add_column(sid, justify="center")
    table.add_column("Total", justify="right", style="bold")

    grand = 0
    for eid in event_ids:
        if eid not in results:
            continue
        row = [eid]
        t = 0
        for sid in sources:
            n = results[eid].get(sid, 0)
            row.append(str(n) if n else "·")
            t += n
        row.append(str(t))
        table.add_row(*row)
        grand += t

    console.print(table)
    console.print(f"\n[bold green]Total articles saved: {grand}[/bold green]")
    console.print(f"Next: [bold]DATA_DIR={data_dir} uv run python -m tm.orchestrator local_file[/bold]")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--events", nargs="+", default=MVP_EVENTS)
    p.add_argument("--sources", nargs="+", default=None)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    data_dir = Path(os.environ.get("DATA_DIR", "/app/data"))
    asyncio.run(run_batch(data_dir, args.events, args.sources, args.force))


if __name__ == "__main__":
    main()
