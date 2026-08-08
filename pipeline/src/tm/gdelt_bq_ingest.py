"""GDELT-BigQuery batch ingestor — free, deterministic historical article discovery.

Why this exists
---------------
The retro engines (Bediavad source-leaderboard + the Duel) need articles published
*before* an event, often years back. The paid SERP chain is expensive and drifts
run-to-run; GDELT's free DOC API only covers a ~3-month rolling window and is
429-throttled from our server IP. GDELT's **GKG table in BigQuery** has the full
history, real date filters, and — over a partition-pruned window — costs ~$0
(a 27-day window scans ~10 GB, inside BigQuery's free 1 TB/month). It is also
*reproducible*: the same query over the same fixed table returns the same URLs,
which is exactly what a scientific backtest needs.

What it does
------------
For each event, ONE BigQuery scan (cheap) pulls URLs across all tracked outlets,
buckets them by ``source_id``, fetches each article's text **Wayback-first**
(recovers dead 3-year-old URLs *and* reads the article as it appeared pre-outcome,
hardening anti-lookahead), and writes them to
``data/raw_ingest/{source_id}/{event_id}/article_NN.json`` — the exact layout the
orchestrator's ``local_file`` mode consumes, so extraction → Atlas is unchanged.

Free-only by design: this ingestor never calls a paid provider. When an outlet has
no free GKG coverage for an event, the cell is left empty and logged (a "miss") —
honest, and cost stays provably bounded.

Cost rule baked in: **one query per event, bucket sources locally.** A per-source
query would re-scan the same partitions 20× (≈14 TB ≈ $80). One per-event scan with
a ``SourceCommonName IN (...)`` filter is free (the predicate adds no scanned bytes).

Usage
-----
    DATA_DIR=/path/to/data uv run python -m tm.gdelt_bq_ingest --events A19 C07 B10
    DATA_DIR=/path/to/data uv run python -m tm.gdelt_bq_ingest --all --limit 8
    DATA_DIR=/path/to/data uv run python -m tm.gdelt_bq_ingest --discover "Rafah offensive" \
        --date-from 2024-01-01 --date-to 2024-05-01
Then:
    DATA_DIR=/path/to/data uv run python -m tm.orchestrator local_file
"""

import asyncio
import json
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote

import httpx
from bs4 import BeautifulSoup
from rich.console import Console
from rich.table import Table

from tm import web_search as _ws
from .utils import KNOWN_SOURCE_IDS, existing_articles, save_article, predates_outcome
from .article_text import BROWSER_HEADERS, extract_article_body
from .net_guard import safe_get_async, UnsafeURLError
# Reuse the date/evergreen helpers rather than re-implement them.
from .web_search_ingest import _is_evergreen_domain, _date_from_url, _date_from_html
# Keep the ingest window identical to the window the backtest actually scores.
from .backtest import MIN_DAYS_BEFORE_EVENT, MAX_DAYS_BEFORE_EVENT

console = Console()

# One BigQuery scan per event returns rows for ALL tracked outlets; this caps how
# many rows the SQL returns. LIMIT does not change bytes scanned in BigQuery, so a
# generous cap is free and ensures low-volume outlets aren't crowded out by a
# high-volume one in the same scan.
_BQ_ROW_CAP = 2000

# Politeness delays (seconds). GKG is one query/event; the fetch loop hits
# archive.org and live sites, which we pace.
_FETCH_DELAY = 0.7
_MIN_TEXT_CHARS = 300


def _norm_host(url_or_host: str) -> str:
    """Bare lowercase hostname: strip scheme, ``www.``, and any path."""
    h = re.sub(r"^https?://", "", (url_or_host or "").strip().lower())
    h = re.sub(r"^www\.", "", h).split("/")[0]
    return h


def build_domain_map(sources_dir: Path) -> Dict[str, str]:
    """Map ``SourceCommonName`` host → ``source_id`` for scoreable outlets.

    Built from ``data/sources/*.json`` restricted to ``KNOWN_SOURCE_IDS`` (the only
    sources the backtest scores). GDELT's ``SourceCommonName`` is a bare host
    (``ynet.co.il``), so we key on the normalised host of each source's ``url``.
    On a host collision the first source_id (sorted) wins — deterministic.
    """
    mapping: Dict[str, str] = {}
    for sf in sorted(sources_dir.glob("*.json")):
        try:
            src = json.loads(sf.read_text())
        except Exception:
            continue
        sid = src.get("id")
        if sid not in KNOWN_SOURCE_IDS:
            continue
        host = _norm_host(src.get("url", ""))
        if host and host not in mapping:
            mapping[host] = sid
    return mapping


def _spread_sample(items: list, k: int) -> list:
    """Pick up to ``k`` items spread evenly across ``items`` (order preserved).

    Chooses which of a source's in-window articles to fetch. Taking the first
    ``k`` clusters on one end of the date range (GKG rows arrive DATE-sorted),
    over-sampling the reactive tail; spreading gives coverage across the whole
    predictive window, where forward-looking predictions actually live.
    """
    n = len(items)
    if k <= 0:
        return []
    if n <= k:
        return list(items)
    if k == 1:
        return [items[n // 2]]
    step = (n - 1) / (k - 1)
    idxs = sorted({round(i * step) for i in range(k)})
    return [items[i] for i in idxs]


def _raw_snapshot_url(snapshot_url: str) -> str:
    """Turn a Wayback snapshot URL into its raw (``id_``) variant.

    ``…/web/20231004093000/https://…`` → ``…/web/20231004093000id_/https://…``
    The ``id_`` modifier serves the original archived page without Wayback's
    injected navigation toolbar, so ``extract_article_body`` sees clean HTML.
    """
    return re.sub(r"(/web/\d{14})/", r"\1id_/", snapshot_url, count=1)


async def _wayback_raw_snapshot(
    client: httpx.AsyncClient, url: str, prefer_ts: str
) -> Tuple[Optional[str], Optional[str]]:
    """Return (raw_snapshot_url, snapshot_date YYYY-MM-DD) closest to ``prefer_ts``.

    ``prefer_ts`` is YYYYMMDD (the article's publish date) so the Availability API
    returns the snapshot nearest to publication. Uses the ``id_`` raw variant so we
    get the original page without Wayback's injected toolbar. Returns (None, None)
    when the Archive has no snapshot.
    """
    api = f"https://archive.org/wayback/available?url={quote(url, safe='')}&timestamp={prefer_ts}"
    try:
        r = await safe_get_async(client, api, timeout=20)
        if r.status_code != 200:
            return None, None
        snap = (r.json().get("archived_snapshots") or {}).get("closest") or {}
    except (httpx.HTTPError, UnsafeURLError, ValueError):
        return None, None
    if not snap.get("available") or str(snap.get("status")) not in ("200", "None", ""):
        # status may be absent; treat only explicit non-200 as unusable.
        if snap.get("status") and str(snap["status"]) != "200":
            return None, None
    snap_url = snap.get("url") or ""
    snap_ts = snap.get("timestamp") or ""
    if not snap_url or len(snap_ts) < 8:
        return None, None
    # Insert `id_` after the 14-digit timestamp to fetch the raw archived page.
    raw_url = _raw_snapshot_url(snap_url)
    try:
        snap_date = datetime.strptime(snap_ts[:8], "%Y%m%d").strftime("%Y-%m-%d")
    except ValueError:
        return None, None
    return raw_url, snap_date


async def _fetch_page_text(client: httpx.AsyncClient, url: str) -> str:
    """GET a page (SSRF-guarded) and extract its article body, or '' on failure."""
    try:
        r = await safe_get_async(client, url, timeout=25)
        if r.status_code != 200:
            return ""
        return extract_article_body(BeautifulSoup(r.text, "html.parser"))
    except (httpx.HTTPError, UnsafeURLError):
        return ""


async def fetch_article(
    client: httpx.AsyncClient,
    url: str,
    publish_date: str,
    outcome_date: str,
    allow_live: bool,
) -> Tuple[str, str]:
    """Return (text, text_source) for ``url``. ``text_source`` ∈ wayback|live|"".

    Wayback-first: prefer the snapshot nearest the publish date, and accept it only
    when the snapshot itself predates the outcome (so the archived *text* cannot
    contain the outcome). Falls back to a live fetch only when ``allow_live`` — a
    live page may have been edited post-outcome, so this is opt-in; the GKG publish
    date (pre-outcome by query construction) is still what we store.
    """
    prefer_ts = re.sub(r"[^0-9]", "", publish_date)[:8] or "20000101"
    raw_url, snap_date = await _wayback_raw_snapshot(client, url, prefer_ts)
    if raw_url and snap_date and predates_outcome(snap_date, outcome_date):
        text = await _fetch_page_text(client, raw_url)
        if len(text) >= _MIN_TEXT_CHARS:
            return text, "wayback"
    if allow_live:
        text = await _fetch_page_text(client, url)
        if len(text) >= _MIN_TEXT_CHARS:
            return text, "live"
    return "", ""


async def ingest_event(
    event: dict,
    sources_map: Dict[str, str],
    raw_ingest_dir: Path,
    per_source_limit: int,
    force: bool,
    t_days: int,
    allow_live: bool,
) -> Dict[str, int]:
    """Backfill one event across all tracked outlets from a single GKG scan.

    Returns {source_id: articles_saved}. Sources absent from the return (or with 0)
    are misses — no free GKG coverage in the window.
    """
    eid = event["id"]
    outcome_date = event["outcome_date"]
    outcome_dt = datetime.strptime(outcome_date, "%Y-%m-%d")
    # Align the ingest window with the window the backtest actually scores:
    # [outcome − MAX_DAYS, outcome − MIN_DAYS]. The last MIN_DAYS_BEFORE_EVENT days
    # are reactive news the backtest discards, so fetching them just wastes
    # requests and clogs cells with non-predictive articles. t_days (the duel's
    # T-day protocol) can push the end earlier, never later.
    min_gap = max(t_days, MIN_DAYS_BEFORE_EVENT)
    end_dt = outcome_dt - timedelta(days=min_gap)
    window = int(event.get("predictive_window_days", MAX_DAYS_BEFORE_EVENT))
    start_dt = outcome_dt - timedelta(days=max(window, min_gap + 1))

    keywords = event.get("duel_keywords") or event.get("search_keywords") or []
    query = " ".join(keywords) if keywords else event.get("name", "")
    if not query.strip():
        console.print(f"  [dim yellow]{eid}: no keywords/name — skipping[/dim yellow]")
        return {}

    # ONE BigQuery scan for the whole event, restricted to tracked outlets.
    try:
        results = await asyncio.to_thread(
            _ws._search_gdelt_bq,
            query, _BQ_ROW_CAP, start_dt, end_dt,
            list(sources_map.keys()),   # domains
            _BQ_ROW_CAP,                # max_rows (LIMIT is free)
        )
    except RuntimeError as e:
        # e.g. no extractable entity terms (proper-noun heuristic over-filtered).
        console.print(f"  [dim yellow]{eid}: GKG query yielded nothing ({e})[/dim yellow]")
        return {}
    except Exception as e:
        console.print(f"  [dim red]{eid}: GKG query error: {e}[/dim red]")
        return {}

    if not results:
        console.print(f"  [dim]{eid}: 0 GKG rows in window[/dim]")
        return {}

    # Bucket URLs by source_id (GKG SourceCommonName → known source).
    buckets: Dict[str, List[_ws.SearchResult]] = {}
    for r in results:
        sid = sources_map.get(_norm_host(r.source or r.url))
        if not sid:
            continue
        buckets.setdefault(sid, []).append(r)

    saved_counts: Dict[str, int] = {}
    async with httpx.AsyncClient(headers=BROWSER_HEADERS, follow_redirects=False) as client:
        for sid, items in buckets.items():
            cell_dir = raw_ingest_dir / sid / eid
            existing = existing_articles(cell_dir)
            if existing and not force:
                saved_counts[sid] = len(existing)
                continue
            idx = len(existing)
            saved = 0
            # Sort by publish date, then fetch a candidate pool spread across the
            # window (3× the target, capped) so a run of fetch failures can't
            # collapse the cell onto one end of the date range.
            items_by_date = sorted(items, key=lambda r: (r.published_date or ""))
            candidates = _spread_sample(items_by_date, min(per_source_limit * 3, len(items_by_date)))
            for res in candidates:
                if saved >= per_source_limit:
                    break
                if _is_evergreen_domain(res.url):
                    continue
                # Publish date: GKG date is authoritative + pre-outcome by construction.
                pub_date = (res.published_date or "")[:10] or _date_from_url(res.url) or ""
                if not pub_date or not predates_outcome(pub_date, end_dt.strftime("%Y-%m-%d")):
                    continue
                await asyncio.sleep(_FETCH_DELAY)
                text, text_source = await fetch_article(
                    client, res.url, pub_date, outcome_date, allow_live
                )
                if not text:
                    continue
                idx += 1
                save_article(cell_dir, idx, {
                    "headline": res.title,
                    "text": text,
                    "published_at": pub_date,
                    "estimated_date": False,
                    "date_source": "gdelt",
                    "author": "Unknown",
                    "url": res.url,
                    "snippet": "",
                    "text_source": text_source,
                })
                saved += 1
            saved_counts[sid] = saved
    return saved_counts


async def run_batch(
    data_dir: Path,
    event_ids: List[str],
    per_source_limit: int,
    force: bool,
    t_days: int,
    allow_live: bool,
):
    events_dir = data_dir / "events"
    sources_map = build_domain_map(data_dir / "sources")
    if not sources_map:
        console.print("[red]No tracked-source domains resolved — check data/sources/[/red]")
        return

    events: Dict[str, dict] = {}
    for eid in event_ids:
        p = events_dir / f"{eid}.json"
        if p.exists():
            events[eid] = json.loads(p.read_text())
        else:
            console.print(f"[yellow]Event {eid} not found, skipping[/yellow]")

    console.print(
        f"\n[bold]GDELT-BigQuery Ingest[/bold]  {len(events)} events · "
        f"{len(sources_map)} tracked outlets · ≤{per_source_limit}/source · "
        f"{'live-fallback ON' if allow_live else 'Wayback-only'}\n"
    )

    all_counts: Dict[str, Dict[str, int]] = {}
    for eid, event in events.items():
        console.print(f"[cyan]{eid}[/cyan] — {event.get('name','')[:50]}")
        all_counts[eid] = await ingest_event(
            event, sources_map, data_dir / "raw_ingest",
            per_source_limit, force, t_days, allow_live,
        )

    # Summary + miss-log.
    tracked = sorted(set(sources_map.values()))
    table = Table(title="GDELT-BQ Ingest — articles saved per (event × source)")
    table.add_column("Event", style="cyan")
    table.add_column("Sources with coverage", justify="right")
    table.add_column("Articles", justify="right")
    table.add_column("Misses (no free coverage)", style="dim")
    grand = 0
    for eid in event_ids:
        counts = all_counts.get(eid, {})
        hit = {s: n for s, n in counts.items() if n}
        n_articles = sum(hit.values())
        grand += n_articles
        misses = [s for s in tracked if not counts.get(s)]
        table.add_row(
            eid, str(len(hit)), str(n_articles) if n_articles else "·",
            ", ".join(misses) if misses else "—",
        )
    console.print(table)
    console.print(f"\n[bold green]Total articles saved: {grand}[/bold green]")
    console.print(
        f"Next: [bold]DATA_DIR={data_dir} uv run python -m tm.orchestrator local_file[/bold]"
    )


def _entity_regex_patterns(terms: List[str]) -> List[str]:
    """Build case-insensitive REGEXP_CONTAINS patterns for BigQuery entity terms.

    Returns one ``(?i)<escaped term>`` pattern per term, meant to be bound as a
    BigQuery ``ArrayQueryParameter`` rather than interpolated into SQL text.
    ``re.escape()`` only neutralises regex metacharacters (so the pattern matches
    the term literally) — it says nothing about SQL syntax, so the caller must
    still bind these as parameters, never splice them into the query string.
    """
    return [f"(?i){re.escape(t)}" for t in terms]


async def discover(keywords: str, date_from: str, date_to: str, data_dir: Path):
    """Surface NO-event candidates: which tracked outlets covered a topic, how heavily.

    Runs ONE partition-pruned aggregate over the tracked outlets and prints coverage
    volume per source for the window. Heavy pre-window coverage of a thing that then
    *didn't* happen is a NO-event candidate — but confirming the outcome is human work
    this tool can't do, so it reports coverage, not outcomes.
    """
    from google.cloud import bigquery as _bq
    client = _ws._get_bq_client()
    sources_map = build_domain_map(data_dir / "sources")
    terms = _ws._extract_bq_terms(keywords)
    if not terms:
        console.print("[yellow]No proper-noun entity terms extracted from keywords[/yellow]")
        return
    # Entity terms and date bounds are bound as query parameters — not spliced
    # into the SQL string — so nothing the caller supplies can alter the query's
    # structure, regardless of what characters ever end up in a term.
    patterns = _entity_regex_patterns(terms)
    dom = _ws._sql_domain_filter(list(sources_map.keys()))
    sql = f"""
        SELECT SourceCommonName AS source, COUNT(*) AS n,
               MIN(DATE) AS first_seen, MAX(DATE) AS last_seen
        FROM `gdelt-bq.gdeltv2.gkg_partitioned`
        WHERE _PARTITIONTIME BETWEEN TIMESTAMP(@date_from) AND TIMESTAMP(@date_to)
          AND EXISTS (
              SELECT 1 FROM UNNEST(@patterns) AS pattern
              WHERE REGEXP_CONTAINS(COALESCE(V2Persons,''), pattern)
                 OR REGEXP_CONTAINS(COALESCE(AllNames,''), pattern)
          ) {dom}
        GROUP BY source ORDER BY n DESC LIMIT 50
    """
    job = client.query(
        sql,
        job_config=_bq.QueryJobConfig(
            maximum_bytes_billed=_ws._GDELT_BQ_MAX_BYTES_BILLED,
            query_parameters=[
                _bq.ScalarQueryParameter("date_from", "STRING", date_from),
                _bq.ScalarQueryParameter("date_to", "STRING", date_to),
                _bq.ArrayQueryParameter("patterns", "STRING", patterns),
            ],
        ),
    )
    rows = list(await asyncio.to_thread(job.result))
    table = Table(title=f"GKG coverage for {terms} · {date_from}→{date_to}")
    table.add_column("Outlet", style="cyan")
    table.add_column("source_id", style="green")
    table.add_column("Articles", justify="right")
    table.add_column("First", style="dim")
    table.add_column("Last", style="dim")
    for r in rows:
        table.add_row(
            r.source, sources_map.get(_norm_host(r.source), "—"), str(r.n),
            str(r.first_seen)[:8], str(r.last_seen)[:8],
        )
    console.print(table)
    console.print(
        f"[dim]Scanned ~{job.total_bytes_processed/1e9:.1f} GB "
        f"(${job.total_bytes_processed/1e12*6.25:.3f}).[/dim]"
    )


def main():
    import argparse
    p = argparse.ArgumentParser(description="GDELT-BigQuery free historical ingestor for retro")
    p.add_argument("--events", nargs="+", metavar="EID", help="Event IDs to backfill")
    p.add_argument("--all", action="store_true", help="Backfill all events in data/events/")
    p.add_argument("--limit", type=int, default=8, help="Max articles per source cell (default 8)")
    p.add_argument("--force", action="store_true", help="Re-fetch already-populated cells")
    p.add_argument("--t-days", type=int, default=0, metavar="N",
                   help="Window ends at outcome_date - N days (default 0 = full window)")
    p.add_argument("--allow-live", action="store_true",
                   help="Fall back to a live fetch when no pre-outcome Wayback snapshot exists "
                        "(live pages may be post-outcome-edited; off by default)")
    p.add_argument("--discover", metavar="KEYWORDS",
                   help="Discovery mode: report tracked-outlet coverage volume for a topic")
    p.add_argument("--date-from", metavar="YYYY-MM-DD", help="Discovery window start")
    p.add_argument("--date-to", metavar="YYYY-MM-DD", help="Discovery window end")
    args = p.parse_args()

    data_dir = Path(os.environ.get("DATA_DIR", "/app/data"))

    if args.discover:
        if not (args.date_from and args.date_to):
            p.error("--discover requires --date-from and --date-to")
        asyncio.run(discover(args.discover, args.date_from, args.date_to, data_dir))
        return

    if args.all:
        event_ids = sorted(pp.stem for pp in (data_dir / "events").glob("*.json"))
    elif args.events:
        event_ids = args.events
    else:
        p.error("provide --events EID... or --all (or --discover)")
    asyncio.run(run_batch(data_dir, event_ids, args.limit, args.force, args.t_days, args.allow_live))


if __name__ == "__main__":
    main()
