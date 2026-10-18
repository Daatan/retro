#!/usr/bin/env python3
"""
Audit data/events/*.json for weak llm_referee_criteria (gatekeeper / extractor anchor).

Run from repo root:
  python pipeline/scripts/audit_event_criteria.py
  DATA_DIR=./data python pipeline/scripts/audit_event_criteria.py

After tightening criteria for an event, re-run extraction for affected cells, e.g.:
  DATA_DIR=./data uv run python -m tm.orchestrator local_file --events A18 --force-reextract
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def find_events_dir() -> Path:
    root = Path(__file__).resolve().parents[2]
    for candidate in (root / "data" / "events", Path("data/events")):
        if candidate.is_dir():
            return candidate
    return root / "data" / "events"


def audit(events_dir: Path) -> list[tuple[str, list[str], str]]:
    """Return list of (event_id, issues, criteria_snippet)."""
    flagged: list[tuple[str, list[str], str]] = []
    for path in sorted(events_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        eid = data.get("id", path.stem)
        criteria = (data.get("llm_referee_criteria") or "").strip()
        issues: list[str] = []
        if not criteria:
            issues.append("empty_criteria")
        elif len(criteria) < 35:
            issues.append("short_criteria")
        low = criteria.lower()
        if criteria and "must" not in low and "will " not in low and "mark " not in low:
            issues.append("no_anchor_verb")
        if issues:
            flagged.append((eid, issues, criteria[:100]))
    return flagged


def main() -> int:
    events_dir = find_events_dir()
    if not events_dir.is_dir():
        print(f"No events directory at {events_dir}", file=sys.stderr)
        return 2

    flagged = audit(events_dir)
    if not flagged:
        print(f"OK — all {len(list(events_dir.glob('*.json')))} events passed heuristic criteria checks.")
        return 0

    print(f"Flagged {len(flagged)} event(s) under {events_dir}:\n")
    for eid, issues, snippet in flagged:
        print(f"  {eid}: {', '.join(issues)}")
        if snippet:
            print(f"    → {snippet!r}{'…' if len(snippet) >= 100 else ''}")
    print(
        "\nNext: edit llm_referee_criteria to a single crisp binary question, then "
        "for each id run orchestrator with --force-reextract on that event's cells."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
