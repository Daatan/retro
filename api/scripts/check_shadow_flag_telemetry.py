"""Detect shadow-only flags producing zero log volume (retro#806).

Several config flags in ``forecast_api.config`` gate shadow-only behavior —
logged, never enforced — pending accumulated evidence before promotion.
retro#601 shipped ``premise_verifier_enabled`` believing it was live in
shadow; ten days passed with **zero** ``event=premise_verifier`` lines before
anyone noticed, because nothing distinguished "shadow flag producing
telemetry" from "shadow flag silently inert" except a human happening to
look and check.

This generalizes that one-off check: for every flag in ``FLAG_REGISTRY`` that
is currently enabled in ``ApiSettings()``, scan a log file for its expected
``event=<name>`` line(s) within a recent window and report SILENT when none
showed up.

    uv run python scripts/check_shadow_flag_telemetry.py \\
        [--log-file /home/ubuntu/truthmachine/oracle_log.txt] \\
        [--since-hours 48] [--json]

Exit 0 only when every *enabled* flag has matching telemetry in the window;
exit 1 if any enabled flag is silent (per retro#395: a measurement script
must never read as a pass when it measured "not yet"). That makes this
composable as a CI/cron check, not just a human-read report — see the
`/audit` skill, which runs it against the Oracle box's log via SSM.

Adding a new shadow flag later needs one more ``ShadowFlag`` entry in
``FLAG_REGISTRY``, not a new script.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional

# forecast_api.config instantiates a module-level ApiSettings() on import,
# which requires oracle_api_key — this script never serves requests, so a
# dummy value unblocks the import (same convention as
# check_resolution_shadow_gate.py / scan_outlier_estimates.py).
os.environ.setdefault("ORACLE_API_KEY", "dummy")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from forecast_api.config import ApiSettings  # noqa: E402


# main.py's logging.basicConfig sets format
#   "%(asctime)s %(levelname)s %(name)s — %(message)s"
# and asctime defaults to "YYYY-MM-DD HH:MM:SS,mmm" in the server's local
# time — the same clock this script runs against when invoked on-box via SSM,
# which is how naive (tz-less) comparison below stays correct.
_TIMESTAMP_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d{3}")


@dataclass(frozen=True)
class ShadowFlag:
    """One shadow-only flag: its ``ApiSettings`` attribute and the
    ``event=<name>`` log line(s) it is expected to emit whenever it fires."""

    settings_attr: str
    events: tuple[str, ...]
    issue: str = ""


# Start here. A new shadow flag needs one more entry, not a new script.
# resolution_shadow_credibility_enabled and hazard_shadow_enabled are
# deliberately NOT here — those are promotion gates with their own dedicated
# scripts (check_resolution_shadow_gate.py); this registry is specifically
# for the "believed to be shadow-logging but silently isn't" failure mode.
FLAG_REGISTRY: tuple[ShadowFlag, ...] = (
    ShadowFlag(
        settings_attr="premise_verifier_enabled",
        events=("event=premise_verifier", "event=premise_verifier_error"),
        issue="retro#575 / retro#601",
    ),
    ShadowFlag(
        settings_attr="precursor_match_enabled",
        events=("event=precursor_match", "event=precursor_match_crash"),
        issue="retro#608",
    ),
    ShadowFlag(
        settings_attr="settled_grounding_enabled",
        events=("event=settled_grounding", "event=settled_grounding_crash"),
        issue="retro#609",
    ),
    ShadowFlag(
        settings_attr="retry_relaxed_search_enabled",
        events=("event=retry_relaxed_search",),
        issue="retro#621",
    ),
)


def _parse_timestamp(line: str) -> Optional[datetime]:
    m = _TIMESTAMP_RE.match(line)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def count_events(log_lines: Iterable[str], events: tuple[str, ...], since: datetime) -> int:
    """Count lines timestamped at/after ``since`` that carry one of ``events``.

    Requires whitespace or end-of-line right after the token, so
    ``event=premise_verifier`` does not also match an unrelated
    ``event=premise_verifier_something_else`` line that happens to share the
    prefix.
    """
    patterns = [re.compile(re.escape(e) + r"(?:\s|$)") for e in events]
    count = 0
    for line in log_lines:
        ts = _parse_timestamp(line)
        if ts is None or ts < since:
            continue
        if any(p.search(line) for p in patterns):
            count += 1
    return count


def check_flags(settings, log_lines: Iterable[str], since: datetime,
                 registry: tuple[ShadowFlag, ...] = FLAG_REGISTRY) -> list[dict]:
    """Per-flag verdicts: OFF (not enabled, nothing to check), SILENT (enabled
    but zero matching events in the window — the retro#601 failure mode), or
    PASS (enabled and telemetry showed up)."""
    log_lines = list(log_lines)
    results = []
    for flag in registry:
        enabled = bool(getattr(settings, flag.settings_attr, False))
        n = count_events(log_lines, flag.events, since) if enabled else 0
        if not enabled:
            verdict = "OFF"
        elif n > 0:
            verdict = "PASS"
        else:
            verdict = "SILENT"
        results.append({
            "flag": flag.settings_attr,
            "enabled": enabled,
            "event_count": n,
            "verdict": verdict,
            "issue": flag.issue,
        })
    return results


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--log-file", type=Path,
                     default=Path("/home/ubuntu/truthmachine/oracle_log.txt"),
                     help="Oracle log file to scan (not journald — see retro/CLAUDE.md)")
    ap.add_argument("--since-hours", type=float, default=48.0,
                     help="lookback window in hours (default 48 — catch a dead flag fast)")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args()

    settings = ApiSettings()
    since = datetime.now() - timedelta(hours=args.since_hours)

    try:
        log_lines = args.log_file.read_text(errors="replace").splitlines()
    except OSError as exc:
        print(f"error: could not read {args.log_file}: {exc}", file=sys.stderr)
        return 1

    results = check_flags(settings, log_lines, since)
    any_silent = any(r["verdict"] == "SILENT" for r in results)

    if args.json:
        print(json.dumps({
            "since_hours": args.since_hours,
            "log_file": str(args.log_file),
            "flags": results,
        }, indent=2))
    else:
        print(f"Shadow flag telemetry — since {since.isoformat()} (local), log={args.log_file}")
        for r in results:
            marker = {"OFF": "--", "PASS": "OK", "SILENT": "!!"}[r["verdict"]]
            issue = f"  ({r['issue']})" if r["issue"] else ""
            print(f"  [{marker}] {r['flag']:<32} enabled={r['enabled']!s:<5} "
                  f"events={r['event_count']:<4} verdict={r['verdict']}{issue}")
        if any_silent:
            print("\nSILENT: an enabled shadow flag produced zero matching log lines in the window.")
        else:
            print("\nAll enabled shadow flags have telemetry in the window.")

    return 1 if any_silent else 0


if __name__ == "__main__":
    sys.exit(main())
