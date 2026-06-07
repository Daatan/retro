"""
Sitemap-based news crawler for Atlas pipeline coverage.

Replaces expensive SERP calls for cells where Oracle /search returns 0
domain-matched results. Discovers article URLs via sitemaps (free, dated,
no API key), extracts full text with trafilatura + BeautifulSoup fallback,
and saves to data/raw_ingest/{source_id}/{event_id}/ — the exact format that
orchestrator's local_file_search already consumes.

Usage:
    uv run python -m tm.crawler --events A13 G02 [--sources jpost ynet] [--force]
    uv run python -m tm.crawler --all-empty [--sources jpost ynet] [--force]
"""

import hashlib
import json
import logging
import os
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import httpx
import trafilatura
from bs4 import BeautifulSoup
from rich.console import Console
from rich.table import Table

from .article_text import BROWSER_HEADERS, extract_article_body
from .models import CellStatus
from .progress import load_state
from .utils import existing_articles, save_article, KNOWN_SOURCE_IDS

console = Console()
logger = logging.getLogger(__name__)

_NS_SM = "http://www.sitemaps.org/schemas/sitemap/0.9"
_NS_NEWS = "http://www.google.com/schemas/sitemap-news/0.9"

_SITEMAP_FALLBACKS = [
    "/sitemap.xml",
    "/sitemap_index.xml",
    "/news-sitemap.xml",
    "/sitemap_news.xml",
    "/sitemap-news.xml",
    "/sitemaps/news.xml",
]

_MIN_ARTICLE_CHARS = 200
_MAX_ARTICLES_PER_CELL = 10
_MAX_CANDIDATES = 20


def _source_domain(source: dict) -> str:
    url = source.get("url", "")
    return (
        url.replace("https://www.", "")
        .replace("http://www.", "")
        .replace("https://", "")
        .replace("http://", "")
        .split("/")[0]
    )


def _find_sitemaps(domain: str, client: httpx.Client) -> list[str]:
    """Return sitemap URLs via robots.txt, falling back to common patterns."""
    sitemaps: list[str] = []
    try:
        r = client.get(f"https://{domain}/robots.txt", timeout=10)
        if r.status_code == 200:
            for line in r.text.splitlines():
                if line.lower().startswith("sitemap:"):
                    url = line.split(":", 1)[1].strip()
                    if url not in sitemaps:
                        sitemaps.append(url)
    except Exception:
        pass
    if not sitemaps:
        sitemaps = [f"https://{domain}{path}" for path in _SITEMAP_FALLBACKS]
    return sitemaps


def _parse_date_str(s: str) -> Optional[datetime]:
    """Parse date strings from sitemaps (ISO-8601 variants)."""
    if not s:
        return None
    s = s.strip()
    for n in (19, 10):
        try:
            return datetime.fromisoformat(s[:n])
        except (ValueError, TypeError):
            pass
    return None


def _urls_from_urlset(
    root: ET.Element, date_from: datetime, date_to: datetime
) -> list[tuple[str, datetime]]:
    results = []
    for url_el in root.iter(f"{{{_NS_SM}}}url"):
        loc = url_el.findtext(f"{{{_NS_SM}}}loc", "").strip()
        if not loc:
            continue

        pub_date: Optional[datetime] = None

        # Google News sitemap: <news:publication_date>
        pub_el = url_el.find(f".//{{{_NS_NEWS}}}publication_date")
        if pub_el is not None and pub_el.text:
            pub_date = _parse_date_str(pub_el.text)

        # Standard sitemap: <lastmod>
        if pub_date is None:
            lastmod = url_el.findtext(f"{{{_NS_SM}}}lastmod", "")
            if lastmod:
                pub_date = _parse_date_str(lastmod)

        if pub_date is None:
            continue

        if date_from <= pub_date <= date_to:
            results.append((loc, pub_date))

    return results


def _parse_sitemap(
    url: str,
    date_from: datetime,
    date_to: datetime,
    client: httpx.Client,
    depth: int = 0,
) -> list[tuple[str, datetime]]:
    """Fetch and parse a sitemap; recurse into sitemapindex once."""
    if depth > 1:
        return []
    try:
        r = client.get(url, timeout=15, follow_redirects=True)
        if r.status_code != 200:
            return []
        root = ET.fromstring(r.text)
    except Exception as exc:
        logger.debug("Sitemap %s: %s", url, exc)
        return []

    local_tag = root.tag.split("}")[-1] if "}" in root.tag else root.tag

    if local_tag == "sitemapindex":
        results: list[tuple[str, datetime]] = []
        for sm_el in root.iter(f"{{{_NS_SM}}}sitemap"):
            loc = sm_el.findtext(f"{{{_NS_SM}}}loc", "").strip()
            if not loc:
                continue
            # Skip sub-sitemaps clearly outside our date window
            lastmod = sm_el.findtext(f"{{{_NS_SM}}}lastmod", "")
            if lastmod:
                sm_date = _parse_date_str(lastmod)
                if sm_date and sm_date < date_from - timedelta(days=1):
                    continue
            results.extend(_parse_sitemap(loc, date_from, date_to, client, depth + 1))
            if len(results) >= _MAX_CANDIDATES * 2:
                break
        return results

    if local_tag == "urlset":
        return _urls_from_urlset(root, date_from, date_to)

    return []


def _fetch_article(url: str, pub_date: datetime, client: httpx.Client) -> Optional[dict]:
    """Fetch URL and extract full text. Returns None if text is too short (paywall/stub)."""
    try:
        r = client.get(url, timeout=20, follow_redirects=True)
        if r.status_code != 200:
            return None
        html = r.text
    except Exception as exc:
        logger.debug("Article fetch %s: %s", url, exc)
        return None

    text = trafilatura.extract(html, include_comments=False, include_tables=False) or ""
    title = ""
    author = "Unknown"
    if text:
        meta = trafilatura.extract_metadata(html)
        if meta:
            title = meta.title or ""
            author = meta.author or "Unknown"

    if len(text) < _MIN_ARTICLE_CHARS:
        soup = BeautifulSoup(html, "html.parser")
        title_tag = soup.find("title")
        if not title and title_tag:
            title = title_tag.get_text(strip=True)
        text = extract_article_body(soup)

    if len(text) < _MIN_ARTICLE_CHARS:
        return None

    return {
        "headline": title,
        "text": text,
        "published_at": pub_date.strftime("%Y-%m-%d"),
        "author": author,
        "url": url,
    }


class SitemapCrawler:
    def __init__(self, data_dir: Path, rate_limit_s: float = 1.5):
        self.data_dir = data_dir
        self.raw_ingest_dir = data_dir / "raw_ingest"
        self.rate_limit_s = rate_limit_s

    def crawl_cell(self, event: dict, source: dict, force: bool = False) -> int:
        """Crawl one (event, source) cell. Returns count of articles saved."""
        event_id = event["id"]
        source_id = source["id"]
        cell_dir = self.raw_ingest_dir / source_id / event_id

        if not force and existing_articles(cell_dir):
            return 0

        outcome_date = datetime.strptime(event["outcome_date"], "%Y-%m-%d")
        window_days = event.get("predictive_window_days", 14)
        date_from = outcome_date - timedelta(days=window_days)
        date_to = outcome_date
        domain = _source_domain(source)
        if not domain:
            return 0

        console.print(
            f"  [bold]{source_id}[/bold] / {event_id} — {domain}"
            f" [{date_from.date()} → {date_to.date()}]"
        )

        with httpx.Client(headers=BROWSER_HEADERS, follow_redirects=True) as client:
            sitemap_urls = _find_sitemaps(domain, client)
            candidates: list[tuple[str, datetime]] = []
            seen_urls: set[str] = set()
            for sm_url in sitemap_urls:
                for url, dt in _parse_sitemap(sm_url, date_from, date_to, client):
                    if domain in url and url not in seen_urls:
                        candidates.append((url, dt))
                        seen_urls.add(url)
                if len(candidates) >= _MAX_CANDIDATES:
                    break

        if not candidates:
            console.print(f"    [dim]No sitemap candidates for {domain}[/dim]")
            return 0

        candidates.sort(key=lambda x: x[1], reverse=True)
        candidates = candidates[:_MAX_CANDIDATES]
        console.print(f"    [dim]{len(candidates)} sitemap candidates[/dim]")

        seen_hashes: set[str] = set()
        saved = 0
        start_idx = len(existing_articles(cell_dir))

        with httpx.Client(headers=BROWSER_HEADERS, follow_redirects=True) as client:
            for url, pub_date in candidates:
                if saved >= _MAX_ARTICLES_PER_CELL:
                    break
                time.sleep(self.rate_limit_s)
                art = _fetch_article(url, pub_date, client)
                if art is None:
                    console.print(f"    [dim]stub/paywall: {url[:70]}[/dim]")
                    continue
                content_sig = hashlib.md5(art["text"][:500].encode()).hexdigest()
                if content_sig in seen_hashes:
                    continue
                seen_hashes.add(content_sig)
                save_article(cell_dir, start_idx + saved, art)
                saved += 1
                console.print(f"    [green]+[/green] {art['headline'][:65]} ({pub_date.date()})")

        console.print(f"    → {saved} articles saved")
        return saved

    def crawl_event(
        self,
        event_id: str,
        source_ids: Optional[list[str]] = None,
        force: bool = False,
    ) -> dict[str, int]:
        """Crawl all sources for one event. Returns {source_id: count}."""
        event_path = self.data_dir / "events" / f"{event_id}.json"
        if not event_path.exists():
            console.print(f"[red]Event not found: {event_path}[/red]")
            return {}
        event = json.loads(event_path.read_text())

        all_sources = {
            p.stem: json.loads(p.read_text())
            for p in (self.data_dir / "sources").glob("*.json")
        }
        wanted = source_ids if source_ids else KNOWN_SOURCE_IDS
        results: dict[str, int] = {}
        for sid in wanted:
            if sid in all_sources:
                results[sid] = self.crawl_cell(event, all_sources[sid], force=force)
        return results


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Sitemap crawler for Atlas cells")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--events", nargs="+", metavar="EVENT_ID")
    group.add_argument(
        "--all-empty",
        action="store_true",
        help="Crawl all cells with status no_predictions or pending",
    )
    parser.add_argument("--sources", nargs="+", metavar="SOURCE_ID")
    parser.add_argument("--force", action="store_true",
                        help="Re-crawl cells that already have articles")
    parser.add_argument("--rate-limit", type=float, default=1.5, metavar="SECONDS",
                        help="Delay between article GETs per domain (default: 1.5)")
    args = parser.parse_args()

    data_dir = Path(os.environ.get("DATA_DIR", "/app/data"))
    crawler = SitemapCrawler(data_dir, rate_limit_s=args.rate_limit)
    totals: dict[str, int] = {}

    if args.all_empty:
        state = load_state()
        target: list[tuple[str, str]] = []
        for key, cell in state.cells.items():
            if cell.status in (CellStatus.no_predictions, CellStatus.pending):
                eid, sid = key.split(":", 1)
                if args.sources and sid not in args.sources:
                    continue
                target.append((eid, sid))
        console.print(f"[bold]{len(target)} empty cells to crawl[/bold]")

        all_sources = {
            p.stem: json.loads(p.read_text())
            for p in (data_dir / "sources").glob("*.json")
        }
        all_events = {
            p.stem: json.loads(p.read_text())
            for p in (data_dir / "events").glob("*.json")
        }
        for eid, sid in target:
            if eid not in all_events or sid not in all_sources:
                continue
            n = crawler.crawl_cell(all_events[eid], all_sources[sid], force=args.force)
            totals[f"{eid}/{sid}"] = n
    else:
        for eid in args.events:
            result = crawler.crawl_event(eid, source_ids=args.sources, force=args.force)
            totals.update({f"{eid}/{sid}": n for sid, n in result.items()})

    table = Table(title="Crawler Summary", show_lines=False)
    table.add_column("Cell", style="cyan")
    table.add_column("Saved", style="green", justify="right")
    for cell, n in sorted(totals.items()):
        if n > 0:
            table.add_row(cell, str(n))
    console.print(table)
    filled = sum(1 for n in totals.values() if n > 0)
    console.print(f"[bold]Filled {filled} / {len(totals)} cells[/bold]")


if __name__ == "__main__":
    main()
