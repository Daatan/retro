"""
Post-processing: find all atlas entries with stance spread > threshold and
run article-level LLM aggregation to collapse them to a single prediction.

Usage:
    python -m tm.reaggregate [--threshold 0.4] [--dry-run]
"""

import asyncio
import json
import argparse
from pathlib import Path

from rich.console import Console

from .aggregator import aggregate_article_predictions, STANCE_SPREAD_THRESHOLD
from .models import PredictionExtraction

console = Console()


def find_problematic_entries(atlas_dir: Path, threshold: float) -> list[dict]:
    entries = []
    for entry_file in sorted(atlas_dir.rglob("entry_*.json")):
        try:
            data = json.loads(entry_file.read_text())
            preds = data.get("predictions", [])
            if len(preds) <= 1:
                continue
            stances = [p["stance"] for p in preds]
            spread = max(stances) - min(stances)
            if spread > threshold:
                parts = entry_file.parts
                # atlas/{event_id}/{source_id}/entry_*.json
                event_id = parts[-3]
                source_id = parts[-2]
                entries.append({
                    "path": entry_file,
                    "event_id": event_id,
                    "source_id": source_id,
                    "article_date": data.get("article_date", ""),
                    "headline": data.get("headline", ""),
                    "predictions": preds,
                    "spread": spread,
                    "n": len(preds),
                    "data": data,
                })
        except Exception:
            continue
    entries.sort(key=lambda x: x["spread"], reverse=True)
    return entries


def load_event_name(events_dir: Path, event_id: str) -> str:
    event_file = events_dir / f"{event_id}.json"
    if event_file.exists():
        return json.loads(event_file.read_text()).get("name", event_id)
    return event_id


def load_source_name(sources_dir: Path, source_id: str) -> str:
    source_file = sources_dir / f"{source_id}.json"
    if source_file.exists():
        return json.loads(source_file.read_text()).get("name", source_id)
    return source_id


async def reaggregate(data_dir: Path, threshold: float, dry_run: bool):
    atlas_dir = data_dir / "atlas"
    events_dir = data_dir / "events"
    sources_dir = data_dir / "sources"

    entries = find_problematic_entries(atlas_dir, threshold)
    console.print(f"[bold]Found {len(entries)} problematic entries (spread > {threshold})[/bold]")

    for i, entry in enumerate(entries, 1):
        eid = entry["event_id"]
        sid = entry["source_id"]
        spread = entry["spread"]
        n = entry["n"]
        headline = entry["headline"][:60]
        console.print(f"\n[{i}/{len(entries)}] {eid}/{sid} spread={spread:.2f} n={n} — {headline}")

        if dry_run:
            console.print("  [dim](dry-run, skipping)[/dim]")
            continue

        event_name = load_event_name(events_dir, eid)
        source_name = load_source_name(sources_dir, sid)
        preds = [PredictionExtraction(**p) for p in entry["predictions"]]

        try:
            agg = await aggregate_article_predictions(
                preds,
                event_name=event_name,
                source_name=source_name,
                article_date=entry["article_date"],
            )
            # Overwrite entry with single aggregated prediction
            new_data = dict(entry["data"])
            new_data["predictions"] = [agg.model_dump()]
            new_data["aggregated"] = True
            new_data["original_prediction_count"] = n
            entry["path"].write_text(json.dumps(new_data, indent=2, ensure_ascii=False))
            console.print(f"  [green]✓ aggregated to stance={agg.stance:.2f}[/green]")
        except Exception as e:
            console.print(f"  [bold red]✗ failed: {e}[/bold red]")

        await asyncio.sleep(2)

    console.print(f"\n[bold green]Done. Processed {len(entries)} entries.[/bold green]")


async def main():
    parser = argparse.ArgumentParser(description="Re-aggregate problematic atlas entries")
    parser.add_argument("--threshold", type=float, default=STANCE_SPREAD_THRESHOLD,
                        help=f"Stance spread threshold (default: {STANCE_SPREAD_THRESHOLD})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only report — do not modify any files")
    import os
    args = parser.parse_args()

    data_dir = Path(os.environ.get("DATA_DIR", "/app/data"))
    await reaggregate(data_dir, args.threshold, args.dry_run)


if __name__ == "__main__":
    asyncio.run(main())
