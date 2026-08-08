"""Sanity-check the daily PM-analysis refresh output before it's committed.

Run after fetch_node_history.py / compute_edge_probs.py / apply_html_data.py,
before pm_analysis_refresh.yml commits anything. Exits non-zero (failing the
workflow, so nothing gets committed or opened as a PR) on a NaN/inf value, an
out-of-range probability, or a diff far larger than a day's worth of drift.
"""

import json
import math
import subprocess
import sys
from pathlib import Path

BAYESORACLE_DIR = Path(__file__).parent
WEIGHTS_FILE = BAYESORACLE_DIR / "edge_weights.json"
ALL_JSON = BAYESORACLE_DIR / "node_history" / "all.json"

PROB_FIELDS = ("pY", "pN", "implied_p", "pm_p", "pY_bin", "pN_bin", "pY_blend", "pN_blend")

# A day's real drift is a handful of edges nudging by a few hundredths.
# This just needs to catch "the run silently truncated or duplicated data" —
# generous on purpose so it doesn't flake on a legitimately busy news day.
MAX_CHANGED_LINES = 5000


def _check_finite_probability(value, label: str) -> None:
    if value is None:
        return
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{label} is not numeric: {value!r}")
    if math.isnan(value) or math.isinf(value):
        raise ValueError(f"{label} is NaN/inf: {value!r}")
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{label} out of [0,1] range: {value!r}")


def validate_edge_weights() -> None:
    edges = json.loads(WEIGHTS_FILE.read_text())
    if not isinstance(edges, list) or not edges:
        raise ValueError(f"{WEIGHTS_FILE} is empty or not a list")
    for i, edge in enumerate(edges):
        for field in PROB_FIELDS:
            if field in edge:
                _check_finite_probability(edge[field], f"edge[{i}].{field} ({edge.get('source')}->{edge.get('target')})")


def validate_all_json() -> None:
    data = json.loads(ALL_JSON.read_text())
    if not isinstance(data, dict) or not data:
        raise ValueError(f"{ALL_JSON} is empty or not an object")


def validate_diff_size() -> None:
    result = subprocess.run(
        ["git", "diff", "--stat", "--", "bayesoracle/"],
        capture_output=True, text=True, check=True,
    )
    summary = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
    changed = sum(int(n) for n in __import__("re").findall(r"(\d+) (?:insertion|deletion)", summary))
    if changed > MAX_CHANGED_LINES:
        raise ValueError(f"diff touches {changed} lines under bayesoracle/, over the {MAX_CHANGED_LINES} sanity cap: {summary!r}")


def main() -> None:
    checks = [validate_edge_weights, validate_all_json, validate_diff_size]
    for check in checks:
        try:
            check()
        except Exception as exc:
            print(f"VALIDATION FAILED ({check.__name__}): {exc}", file=sys.stderr)
            sys.exit(1)
    print("All refresh sanity checks passed.")


if __name__ == "__main__":
    main()
