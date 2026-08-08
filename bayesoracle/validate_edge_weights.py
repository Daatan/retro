"""Sanity-check edge_weights.json before apply_html_data.py patches its values
into the public index.html (retro#430).

apply_html_data.py bakes pY_blend/pN_blend straight into the page's embedded
JS via str(value) — a NaN or inf from compute_edge_probs.py's regression
fallback would write the literal token `pY:nan` (not a valid JS number),
breaking bayes.daatan.com's script for every visitor with no automated
warning. Run this after compute_edge_probs.py and before apply_html_data.py.
"""

import json
import math
import sys
from pathlib import Path

WEIGHTS_FILE = Path(__file__).parent / "edge_weights.json"

PROB_FIELDS = ("pY_blend", "pN_blend")
CI_FIELDS = ("pY_ci", "pN_ci")


def _is_finite_number(v: object) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def validate(edges: list[dict]) -> list[str]:
    errors = []
    if not edges:
        return ["edge_weights.json is empty"]

    for edge in edges:
        label = f"{edge.get('source', '?')}→{edge.get('target', '?')}"

        for field in PROB_FIELDS:
            v = edge.get(field)
            if not _is_finite_number(v):
                errors.append(f"{label}: {field}={v!r} is not a finite number")
            elif not (0.0 <= v <= 1.0):
                errors.append(f"{label}: {field}={v!r} is out of [0,1]")

        for field in CI_FIELDS:
            ci = edge.get(field)
            if not (isinstance(ci, list) and len(ci) == 2):
                errors.append(f"{label}: {field}={ci!r} is not a 2-element interval")
                continue
            lo, hi = ci
            if not _is_finite_number(lo):
                errors.append(f"{label}: {field}[0]={lo!r} is not a finite number")
            if not _is_finite_number(hi):
                errors.append(f"{label}: {field}[1]={hi!r} is not a finite number")
            if _is_finite_number(lo) and _is_finite_number(hi) and lo > hi:
                errors.append(f"{label}: {field} lower bound {lo} exceeds upper bound {hi}")

    return errors


def main() -> None:
    edges = json.loads(WEIGHTS_FILE.read_text())
    errors = validate(edges)
    if errors:
        print(f"edge_weights.json failed validation ({len(errors)} issue(s)):", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        sys.exit(1)
    print(f"edge_weights.json validated OK ({len(edges)} edges)")


if __name__ == "__main__":
    main()
