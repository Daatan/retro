"""Report resolution-shadow-credibility gate status (retro#604).

The gate (``settings.resolution_shadow_credibility_enabled``) stays off until
``count_resolutions()`` clears ``settings.resolution_shadow_min_global_predictions``
scoreable resolutions. Before this script, checking that meant SSM-ing onto the
Oracle box and re-deriving the count by hand each time. This just prints where
things stand.

    uv run python scripts/check_resolution_shadow_gate.py [--path FILE]

Exit 0 when the gate is met (actionable — worth a PR to flip the flag), exit 1
when it is not (per retro#395: a measurement script must never read as a pass
when it measured "not yet").
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# forecast_api.config instantiates a module-level ApiSettings() on import,
# which requires oracle_api_key — this script never serves requests, so a
# dummy value unblocks the import (same convention as scan_outlier_estimates.py).
os.environ.setdefault("ORACLE_API_KEY", "dummy")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from forecast_api.config import ApiSettings  # noqa: E402
from forecast_api.resolution_scorer import count_resolutions  # noqa: E402


def gate_status(ingest_path: Path, threshold: int) -> dict:
    n = count_resolutions(ingest_path)
    return {
        "n": n,
        "threshold": threshold,
        "gate_met": n >= threshold,
        "remaining": max(0, threshold - n),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--path", type=Path, default=None,
                     help="override resolution_feedback.jsonl path (default: settings.resolved_resolution_feedback_path)")
    args = ap.parse_args()

    settings = ApiSettings()
    ingest_path = args.path or settings.resolved_resolution_feedback_path
    status = gate_status(ingest_path, settings.resolution_shadow_min_global_predictions)
    print(json.dumps(status))
    return 0 if status["gate_met"] else 1


if __name__ == "__main__":
    sys.exit(main())
