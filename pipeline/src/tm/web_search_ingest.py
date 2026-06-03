"""
Web Search batch ingestor — uses the same search stack as daatan.

Provider chain: DataForSEO → SerpAPI → Serper → Brave (first available).
All providers support date-filtered historical queries via Google's CDR parameter,
giving much better pre-event article coverage than GDELT's domain filter.

Results land in data/raw_ingest/web_search/{event_id}/, same schema as
gdelt_ingest.py and site_search.py, so the orchestrator picks them up unchanged.

Usage:
    DATA_DIR=/path/to/data uv run python -m tm.web_search_ingest
    DATA_DIR=/path/to/data uv run python -m tm.web_search_ingest --events C07 B10 A19
    DATA_DIR=/path/to/data uv run python -m tm.web_search_ingest --events C07 --limit 15 --force
"""

import asyncio
import json
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import httpx
from bs4 import BeautifulSoup
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn
from rich.table import Table

from tm import web_search as _ws

console = Console()

_MONTH_ABBR = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}

_URL_DATE_RE = re.compile(
    r"/(?P<y>20\d{2})/"
    r"(?P<m>0?[1-9]|1[0-2]|jan|feb|mar|apr|may|jun|jul|aug|sept?|oct|nov|dec)/"
    r"(?P<d>0?[1-9]|[12]\d|3[01])(?:/|$)",
    re.IGNORECASE,
)


def _date_from_url(url: str) -> Optional[str]:
    """Extract YYYY-MM-DD from URL paths like /2025/dec/11/ or /2024/03/15/."""
    m = _URL_DATE_RE.search(url)
    if not m:
        return None
    y = int(m.group("y"))
    raw_m = m.group("m").lower()
    mo = _MONTH_ABBR.get(raw_m, int(raw_m) if raw_m.isdigit() else 0)
    d = int(m.group("d"))
    if not (1 <= mo <= 12 and 1 <= d <= 31):
        return None
    try:
        return datetime(y, mo, d).strftime("%Y-%m-%d")
    except ValueError:
        return None


def _date_from_html(soup: BeautifulSoup) -> Optional[str]:
    """Extract publish date from common HTML metadata locations."""
    # OpenGraph / article meta
    for sel in [
        ('meta[property="article:published_time"]', "content"),
        ('meta[name="article:published_time"]', "content"),
        ('meta[property="og:article:published_time"]', "content"),
        ('meta[name="datePublished"]', "content"),
        ('meta[itemprop="datePublished"]', "content"),
        ('meta[name="pubdate"]', "content"),
        ('meta[name="publishdate"]', "content"),
        ('meta[name="DC.date.issued"]', "content"),
        ('meta[property="article:modified_time"]', "content"),
    ]:
        el = soup.select_one(sel[0])
        if el and el.get(sel[1]):
            d = _parse_iso_date(el[sel[1]])
            if d:
                return d
    # <time datetime="...">
    el = soup.find("time", attrs={"datetime": True})
    if el and el.get("datetime"):
        d = _parse_iso_date(el["datetime"])
        if d:
            return d
    # JSON-LD blocks with datePublished
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
        except Exception:
            continue
        for entry in (data if isinstance(data, list) else [data]):
            if not isinstance(entry, dict):
                continue
            for key in ("datePublished", "dateCreated", "uploadDate"):
                if key in entry:
                    d = _parse_iso_date(entry[key])
                    if d:
                        return d
    return None


def _parse_iso_date(raw: str) -> Optional[str]:
    """Best-effort ISO-8601 → YYYY-MM-DD."""
    if not raw or not isinstance(raw, str):
        return None
    s = raw.strip()[:10]
    try:
        datetime.strptime(s, "%Y-%m-%d")
        return s
    except ValueError:
        return None


# Domains whose pages stay live and get content-updated indefinitely. The
# stored published_at reflects original creation, not the version we read,
# so they leak post-cutoff outcomes regardless of date metadata.
_EVERGREEN_DOMAIN_SUFFIXES = (
    "wikipedia.org",
    "britannica.com",
    "cfr.org",
    "encyclopedia.com",
    "history.com",
    "investopedia.com",
)


def _is_evergreen_domain(url: str) -> bool:
    from urllib.parse import urlparse
    netloc = urlparse(url).netloc.lower()
    netloc = re.sub(r"^www\.", "", netloc)
    return any(netloc.endswith(s) for s in _EVERGREEN_DOMAIN_SUFFIXES)

DUEL_EVENTS = [
    "A19", "B04", "B05", "B08", "B09", "B10",
    "C07", "C08", "C09", "E07", "E08",
]

from .utils import _is_ascii, existing_articles, save_article
from .article_text import BROWSER_HEADERS, extract_article_body



async def _fetch_text(url: str) -> Tuple[str, Optional[str]]:
    """Return (visible_text, html_publish_date_or_None) for the page."""
    try:
        async with httpx.AsyncClient(headers=BROWSER_HEADERS, timeout=25, follow_redirects=True) as c:
            r = await c.get(url)
            if r.status_code != 200:
                return "", None
        full_soup = BeautifulSoup(r.text, "html.parser")
        # Read the publish date before extract_article_body() strips boilerplate.
        html_date = _date_from_html(full_soup)
        return extract_article_body(full_soup), html_date
    except Exception:
        return "", None


async def ingest_event(
    event: dict,
    raw_ingest_dir: Path,
    limit: int,
    force: bool,
    t_days: int = 0,
) -> int:
    eid = event["id"]
    cell_dir = raw_ingest_dir / "web_search" / eid
    existing = existing_articles(cell_dir)
    if existing and not force:
        return len(existing)

    outcome_dt = datetime.strptime(event["outcome_date"], "%Y-%m-%d")
    # When t_days > 0, search ends at outcome_date - t_days (not the outcome itself)
    end_dt = outcome_dt - timedelta(days=t_days) if t_days > 0 else outcome_dt
    window = int(event.get("predictive_window_days", 30))
    start_dt = end_dt - timedelta(days=window)

    keywords = event.get("duel_keywords") or event.get("search_keywords", [])
    ascii_kws = [k for k in keywords if _is_ascii(k)]
    if not ascii_kws:
        console.print(f"  [dim yellow]{eid}: no ASCII keywords, skipping[/dim yellow]")
        return 0

    if t_days > 0:
        console.print(f"  [dim]{eid}: T-{t_days} window [{start_dt.date()} → {end_dt.date()}][/dim]")

    # Collect unique URLs across up to 3 keyword queries
    seen_urls: set[str] = set()
    candidates: list[_ws.SearchResult] = []

    for kw in ascii_kws[:3]:
        try:
            results = await asyncio.to_thread(
                _ws.search_articles, kw, limit, start_dt, end_dt
            )
            for r in results:
                if r.url and r.url not in seen_urls:
                    seen_urls.add(r.url)
                    candidates.append(r)
            provider = getattr(_ws._provider_local, "name", "?")
            console.print(
                f"  [dim]{eid} «{kw[:40]}»: {len(results)} hits via {provider}[/dim]"
            )
        except Exception as e:
            console.print(f"  [dim red]{eid} search error: {e}[/dim red]")
        await asyncio.sleep(1.0)

    if not candidates:
        return 0

    saved = 0
    article_idx = len(existing)  # continue numbering if partially populated

    skipped_no_date = 0
    skipped_evergreen = 0
    for result in candidates[:limit]:
        # Skip encyclopedic / evergreen sources: their stored published_at is
        # the page creation date, but the content is continuously updated, so
        # reading the page now reveals outcomes that postdate any cutoff.
        if _is_evergreen_domain(result.url):
            skipped_evergreen += 1
            continue

        await asyncio.sleep(1.0)
        text, html_date = await _fetch_text(result.url)
        if len(text) < 250:
            continue

        # Resolve publish date: provider → URL path → page metadata. No fallback.
        provider_date = (result.published_date or "")[:10]
        url_date = _date_from_url(result.url)
        pub_date = provider_date or url_date or html_date
        if not pub_date:
            skipped_no_date += 1
            continue
        if pub_date > end_dt.strftime("%Y-%m-%d"):
            # Article published after the cutoff — drop, can't be predictive.
            continue

        article = {
            "headline": result.title,
            "text": text,
            "published_at": pub_date,
            "estimated_date": False,
            "date_source": "provider" if provider_date else ("url" if url_date else "html"),
            "author": "Unknown",
            "url": result.url,
            "snippet": result.snippet,
        }
        article_idx += 1
        save_article(cell_dir, article_idx, article)
        saved += 1

    if skipped_no_date:
        console.print(f"  [dim yellow]{eid}: dropped {skipped_no_date} articles with no resolvable date[/dim yellow]")
    if skipped_evergreen:
        console.print(f"  [dim yellow]{eid}: dropped {skipped_evergreen} evergreen-domain articles[/dim yellow]")
    return saved


async def run_batch(
    data_dir: Path,
    event_ids: List[str],
    limit: int,
    force: bool,
    t_days: int = 0,
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

    t_label = f" · T-{t_days}" if t_days else ""
    console.print(
        f"\n[bold]Web Search Ingest[/bold]  "
        f"{len(events)} events · up to {limit} articles each{t_label}\n"
    )

    # Show which provider is active
    _ws._refresh_keys_if_stale()
    configured = [
        name for name, key in [
            ("dataforseo", _ws.DATAFORSEO_API_KEY),
            ("serpapi",    _ws.SERPAPI_API_KEY),
            ("serper",     _ws.SERPER_API_KEY),
            ("brave",      _ws.BRAVE_API_KEY),
        ] if key
    ]
    console.print(f"Configured providers: [cyan]{', '.join(configured) or 'none'}[/cyan]\n")

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
            count = await ingest_event(event, raw_ingest_dir, limit, force, t_days)
            results[eid] = count
            label = f"[green]{count} saved[/green]" if count else "[dim]0[/dim]"
            console.print(f"  {eid}: {label}")
            progress.advance(task)

    table = Table(title="Web Search Ingest Summary")
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
    p = argparse.ArgumentParser(description="Web Search batch ingestor for duel events")
    p.add_argument("--events", nargs="+", default=DUEL_EVENTS, metavar="EID")
    p.add_argument("--limit", type=int, default=10, help="Max articles per event (default 10)")
    p.add_argument("--force", action="store_true", help="Re-fetch already populated cells")
    p.add_argument("--t-days", type=int, default=0, metavar="N",
                   help="Search window ends at outcome_date - N days (default 0 = full window)")
    args = p.parse_args()

    data_dir = Path(os.environ.get("DATA_DIR", "/app/data"))
    asyncio.run(run_batch(data_dir, args.events, args.limit, args.force, args.t_days))


if __name__ == "__main__":
    main()
