#!/usr/bin/env python3
"""
Validate the schema of data/events/*.json — the Factum Atlas case studies.

Complements audit_event_criteria.py, which judges the *quality* of one field
(llm_referee_criteria). This one checks that the file is structurally sound:
required fields present and correctly typed, ids unique and matching their
filename, and outcome_date a real past date.

Run from repo root:
  python pipeline/scripts/validate_events.py
  DATA_DIR=./data python pipeline/scripts/validate_events.py

Exit codes: 0 = clean, 1 = findings, 2 = no events directory.

Deliberately NOT covered (retro#348 item 4): whether search_keywords actually
surface articles. That needs live search-provider calls, so it belongs in a
scheduled job rather than a static validator anyone can run offline.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import sys
from pathlib import Path
from typing import Any

# Required on every event. `outcome` is nullable by the documented contract
# (data/README.md: null == unresolved), and `polymarket` is required but
# nullable — all 91 files carry the key, 65 with an explicit null, so a missing
# polymarket key is a real omission rather than "no market".
REQUIRED_TYPES: dict[str, type | tuple[type, ...]] = {
    "id": str,
    "name": str,
    "outcome": (bool, type(None)),
    "search_keywords": list,
    "llm_referee_criteria": str,
    "category": list,
    "tags": list,
    "polymarket": (dict, type(None)),
}

# Optional per data/README.md. `outcome_date` is conditionally required — see
# the resolved/unresolved rule in validate_event — and `predictive_window_days`
# is documented as optional with a default, so its absence is not a defect
# even though all committed events currently set it.
OPTIONAL_TYPES: dict[str, type | tuple[type, ...]] = {
    "outcome_date": str,
    "predictive_window_days": int,
    "oracle_question": str,
    "duel_keywords": list,
    "scope": str,
}

# Letter prefix + zero-padded number, e.g. A01. The letter is the category
# bucket (A=Israeli politics, B=Gaza, C=Iran/regional, ...); new letters are
# expected as the atlas grows, so the letter itself is not enumerated.
ID_PATTERN = re.compile(r"^[A-Z]\d{2,}$")

# Present values as of 2026-08-02: exact, close, indirect.
POLYMARKET_MATCH_QUALITIES = {"exact", "close", "indirect"}

# String-valued lists whose entries must be non-blank.
STRING_LISTS = ("search_keywords", "category", "tags", "duel_keywords")


def find_events_dir() -> Path:
    root = Path(__file__).resolve().parents[2]
    for candidate in (root / "data" / "events", Path("data/events")):
        if candidate.is_dir():
            return candidate
    return root / "data" / "events"


def _type_ok(value: Any, expected: type | tuple[type, ...]) -> bool:
    """isinstance, but bool is never accepted as int and vice versa.

    `bool` subclasses `int`, so a plain isinstance would let `outcome: 1` pass
    as a boolean and `predictive_window_days: true` pass as an integer — both
    are exactly the hand-edit typos this validator exists to catch.
    """
    allowed = expected if isinstance(expected, tuple) else (expected,)
    # Only one direction needs care: `isinstance(1, bool)` is already False, so
    # `outcome: 1` fails naturally, but `predictive_window_days: true` would pass.
    if isinstance(value, bool) and bool not in allowed:
        return False
    return isinstance(value, allowed)


def validate_event(
    data: Any,
    stem: str,
    today: dt.date | None = None,
) -> list[str]:
    """Check one event dict. Returns a list of finding codes (empty == clean)."""
    findings: list[str] = []

    if not isinstance(data, dict):
        return ["not_an_object"]

    for field, expected in REQUIRED_TYPES.items():
        if field not in data:
            findings.append(f"missing_field:{field}")
        elif not _type_ok(data[field], expected):
            findings.append(f"wrong_type:{field}")

    for field, expected in OPTIONAL_TYPES.items():
        if field in data and not _type_ok(data[field], expected):
            findings.append(f"wrong_type:{field}")

    known = set(REQUIRED_TYPES) | set(OPTIONAL_TYPES)
    for field in sorted(set(data) - known):
        findings.append(f"unknown_field:{field}")

    eid = data.get("id")
    if isinstance(eid, str):
        if not ID_PATTERN.match(eid):
            findings.append("malformed_id")
        if eid != stem:
            findings.append("id_filename_mismatch")

    if not str(data.get("name") or "").strip():
        findings.append("empty_name")

    # outcome/outcome_date consistency. A *resolved* event (outcome true/false)
    # must carry a real, already-past resolution date; an unresolved one
    # (outcome null) legitimately has neither, and any date it does carry is a
    # target rather than a resolution, so a future value is fine there.
    resolved = isinstance(data.get("outcome"), bool)
    raw_date = data.get("outcome_date")
    if resolved and not str(raw_date or "").strip():
        findings.append("resolved_without_outcome_date")
    if isinstance(raw_date, str) and raw_date.strip():
        try:
            outcome_date = dt.date.fromisoformat(raw_date)
        except ValueError:
            findings.append("unparseable_outcome_date")
        else:
            if resolved and outcome_date > (today or dt.date.today()):
                findings.append("future_outcome_date")

    for field in STRING_LISTS:
        value = data.get(field)
        if not isinstance(value, list):
            continue
        if field != "duel_keywords" and not value:
            findings.append(f"empty_{field}")
        if any(not isinstance(item, str) for item in value):
            findings.append(f"non_string_in:{field}")
        elif any(not item.strip() for item in value):
            findings.append(f"blank_entry_in:{field}")

    window = data.get("predictive_window_days")
    if isinstance(window, int) and not isinstance(window, bool) and window <= 0:
        findings.append("non_positive_window")

    market = data.get("polymarket")
    if isinstance(market, dict):
        for field in ("question", "url"):
            if not str(market.get(field) or "").strip():
                findings.append(f"polymarket_missing:{field}")
        quality = market.get("match_quality")
        if quality not in POLYMARKET_MATCH_QUALITIES:
            findings.append("polymarket_bad_match_quality")
        if "invert" in market and not isinstance(market["invert"], bool):
            findings.append("polymarket_bad_invert")

    return findings


def validate_events_dir(
    events_dir: Path,
    today: dt.date | None = None,
) -> tuple[list[tuple[str, list[str]]], int]:
    """Validate every event file. Returns (findings_by_event, files_checked)."""
    flagged: list[tuple[str, list[str]]] = []
    seen_ids: dict[str, str] = {}
    paths = sorted(events_dir.glob("*.json"))

    for path in paths:
        stem = path.stem
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            flagged.append((stem, [f"invalid_json:{exc.lineno}:{exc.colno}"]))
            continue

        findings = validate_event(data, stem, today=today)

        eid = data.get("id") if isinstance(data, dict) else None
        if isinstance(eid, str):
            if eid in seen_ids:
                findings.append(f"duplicate_id:also_in_{seen_ids[eid]}")
            else:
                seen_ids[eid] = stem

        if findings:
            flagged.append((stem, findings))

    return flagged, len(paths)


def main() -> int:
    events_dir = find_events_dir()
    if not events_dir.is_dir():
        print(f"No events directory at {events_dir}", file=sys.stderr)
        return 2

    flagged, checked = validate_events_dir(events_dir)
    if not flagged:
        print(f"OK — all {checked} events passed schema validation.")
        return 0

    print(f"Flagged {len(flagged)} of {checked} event(s) under {events_dir}:\n")
    for stem, findings in flagged:
        print(f"  {stem}: {', '.join(findings)}")
    print("\nSee pipeline/scripts/validate_events.py for what each code means.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
