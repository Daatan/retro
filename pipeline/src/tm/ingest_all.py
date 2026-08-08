"""
Master ingestor — runs GDELT + site_search + web_search for duel events, then
feeds results through the orchestrator into vault2.

Each source writes to its own raw_ingest/{source}/{event}/ directory.
The orchestrator SHA256-deduplicates across sources before running gatekeeper+extractor.
Safe to re-run: already-populated cells are skipped unless --force is passed.

Usage:
    DATA_DIR=/path/to/data uv run python -m tm.ingest_all
    DATA_DIR=/path/to/data uv run python -m tm.ingest_all --events C07 B10 A19
    DATA_DIR=/path/to/data uv run python -m tm.ingest_all --skip gdelt site_search
    DATA_DIR=/path/to/data uv run python -m tm.ingest_all --force
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from rich.console import Console
from rich.rule import Rule

from tm import gdelt_ingest, site_search, web_search_ingest

console = Console()

DUEL_EVENTS = [
    "A19", "B04", "B05", "B08", "B09", "B10",
    "C07", "C08", "C09", "E07", "E08",
]

ALL_SOURCES = ["gdelt", "site_search", "web_search"]


async def run_all(
    data_dir: Path,
    event_ids: list[str],
    skip: list[str],
    limit: int,
    force: bool,
):
    events_dir = data_dir / "events"
    raw_ingest_dir = data_dir / "raw_ingest"

    # Load event dicts for sources that need them
    events: dict[str, dict] = {}
    for eid in event_ids:
        p = events_dir / f"{eid}.json"
        if p.exists():
            events[eid] = json.loads(p.read_text())
        else:
            console.print(f"[yellow]Event {eid} not found, skipping[/yellow]")

    totals: dict[str, int] = {}

    # ── GDELT ──────────────────────────────────────────────────────────────────
    if "gdelt" not in skip:
        console.print(Rule("[bold cyan]GDELT[/bold cyan]"))
        total = 0
        for eid, event in events.items():
            try:
                count = await gdelt_ingest.ingest_event(event, raw_ingest_dir, limit, force)
            except Exception as e:
                console.print(f"    [dim red]GDELT {eid}: error, skipping — {e}[/dim red]")
                continue
            total += count
        totals["gdelt"] = total
        console.print(f"GDELT total: [green]{total}[/green] articles\n")
    else:
        console.print("[dim]Skipping GDELT[/dim]")

    # ── site_search ────────────────────────────────────────────────────────────
    if "site_search" not in skip:
        console.print(Rule("[bold cyan]site_search[/bold cyan]"))
        total = 0
        for eid, event in events.items():
            for source_id in site_search.SEARCH_FNS:
                try:
                    count = await site_search.ingest_cell(event, source_id, raw_ingest_dir, force)
                except Exception as e:
                    console.print(
                        f"    [dim red]site_search {eid}/{source_id}: error, skipping — {e}[/dim red]"
                    )
                    continue
                total += count
        totals["site_search"] = total
        console.print(f"site_search total: [green]{total}[/green] articles\n")
    else:
        console.print("[dim]Skipping site_search[/dim]")

    # ── web_search ─────────────────────────────────────────────────────────────
    if "web_search" not in skip:
        console.print(Rule("[bold cyan]web_search[/bold cyan]"))
        total = 0
        for eid, event in events.items():
            try:
                count = await web_search_ingest.ingest_event(event, raw_ingest_dir, limit, force)
            except Exception as e:
                console.print(f"    [dim red]web_search {eid}: error, skipping — {e}[/dim red]")
                continue
            total += count
        totals["web_search"] = total
        console.print(f"web_search total: [green]{total}[/green] articles\n")
    else:
        console.print("[dim]Skipping web_search[/dim]")

    # ── Summary ────────────────────────────────────────────────────────────────
    console.print(Rule("[bold]Summary[/bold]"))
    grand = sum(totals.values())
    for src, n in totals.items():
        console.print(f"  {src}: {n}")
    console.print(f"\n[bold green]Grand total new articles: {grand}[/bold green]")
    if grand == 0:
        console.print("[bold yellow]Warning: no new articles ingested. Use --force to re-fetch existing cells.[/bold yellow]")
    console.print(
        f"\nNext: [bold]DATA_DIR={data_dir} uv run python -m tm.orchestrator local_file "
        f"--events {' '.join(event_ids)}[/bold]"
    )


def main():
    p = argparse.ArgumentParser(description="Run all ingestors for duel events")
    p.add_argument("--events", nargs="+", default=DUEL_EVENTS, metavar="EID")
    p.add_argument(
        "--skip", nargs="+", default=[], choices=ALL_SOURCES, metavar="SOURCE",
        help="Sources to skip (gdelt, site_search, web_search)"
    )
    p.add_argument("--limit", type=int, default=10, help="Max articles per event for web_search (default 10)")
    p.add_argument("--force", action="store_true", help="Re-fetch already populated cells")
    args = p.parse_args()

    data_dir = Path(os.environ.get("DATA_DIR", "/app/data"))
    asyncio.run(run_all(data_dir, args.events, args.skip, args.limit, args.force))


if __name__ == "__main__":
    main()
