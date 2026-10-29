"""
One-time migration: compute cell_signal.json for all existing atlas entries.
No LLM calls — pure aggregation from vault data already on disk.

Usage:
    DATA_DIR=/path/to/data uv run python -m tm.migrate_cell_signals
"""

import json
import os
from pathlib import Path

from .models import PredictionExtraction
from .aggregator import aggregate_predictions


def migrate(data_dir: Path):
    atlas_dir = data_dir / "atlas"
    ok, skipped, failed = 0, 0, 0

    for event_dir in sorted(atlas_dir.iterdir()):
        if not event_dir.is_dir():
            continue
        for source_dir in sorted(event_dir.iterdir()):
            if not source_dir.is_dir():
                continue
            eid, sid = event_dir.name, source_dir.name

            entries = list(source_dir.glob("entry_*.json"))
            if not entries:
                skipped += 1
                continue

            all_predictions: list[PredictionExtraction] = []
            for entry_file in entries:
                try:
                    data = json.loads(entry_file.read_text())
                    for p in data.get("predictions", []):
                        all_predictions.append(PredictionExtraction(**p))
                except Exception as e:  # noqa: BLE001 - logged, fail-open
                    print(f"  WARN {eid}/{sid} {entry_file.name}: {e}")

            if not all_predictions:
                skipped += 1
                continue

            try:
                signal = aggregate_predictions(all_predictions)
                out = source_dir / "cell_signal.json"
                out.write_text(json.dumps(signal.model_dump(), indent=2, ensure_ascii=False))
                print(f"  {eid}/{sid}: {signal.claim_count} claims → stance={signal.stance:.2f}")
                ok += 1
            except Exception as e:  # noqa: BLE001 - logged, fail-open
                print(f"  FAIL {eid}/{sid}: {e}")
                failed += 1

    print(f"\nDone: {ok} written, {skipped} skipped (no predictions), {failed} failed")


if __name__ == "__main__":
    data_dir = Path(os.environ.get("DATA_DIR", "/app/data"))
    migrate(data_dir)
