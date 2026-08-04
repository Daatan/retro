"""Replay the article reduction over pool rows exported from daatan (retro#398).

Answers the question nothing could answer before: **do stored rows still reduce
to the numbers they shipped with?** ``test_claims_detail.py`` checks that for
claims built moments earlier in-process; this checks it for rows written weeks
ago by an older build, which is the case retroactive backtesting depends on.

Input: a JSON array of pool rows, each carrying ``claims_detail`` plus whichever
stored scalars you want checked (missing ones are simply not compared)::

    [{"url": "...", "claims_detail": [ ... ],
      "stance": 0.62, "certainty": 0.71, "evidence_weight": 0.6,
      "evidence_class": "reported_fact", "settled": true,
      "settlement_event_date": "2026-06-15", "quantitative_estimate": null,
      "fact_signal": 0.3, "event_actors": "...", "event_target": "...",
      "is_occurrence": false, "verified": true}]

Produce it from daatan's ``evidence_pool_articles`` (the rows live there; the
reduction lives here). Rows with a null ``claims_detail`` are **skipped, not
failed** — the column is forward-only from 2026-08-02 with no backfill, so their
absence is expected and counting them as mismatches would bury the signal.

    uv run python scripts/replay_claims_detail.py rows.json [--json-out out.json] [--limit N]

Exit code is 1 when the replay could not measure anything (no row was
replayable, or every replayable row errored) or when any row disagrees. A clean
run that measured nothing must never read as a pass — that is the trap the
settlement-verifier replay walked into (retro#395).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from forecast_api.claims_replay import replay_rows  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("rows", type=Path)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--json-out", type=Path)
    ap.add_argument("--show", type=int, default=15, help="how many disagreements to print")
    args = ap.parse_args()

    rows = json.loads(args.rows.read_text(encoding="utf-8"))
    if args.limit:
        rows = rows[: args.limit]
    report = replay_rows(rows)

    print(f"\n{report.total} stored rows")
    print(f"  replayable (carry claims_detail): {report.replayed}"
          f"  ({report.coverage:.1%} coverage)")
    print(f"  skipped (column predates the row): {report.skipped}")
    print(f"  errored (unparseable):             {report.errored}")
    if report.replayed:
        print(f"\n  AGREED: {report.agreed}/{report.replayed} ({report.agreement:.1%})")

    if report.by_field:
        print("\ndisagreements by field:")
        for field, n in sorted(report.by_field.items(), key=lambda kv: -kv[1]):
            print(f"  {field:<26}{n:>6}")

    shown = 0
    for r in report.results:
        if not r.diffs or shown >= args.show:
            continue
        shown += 1
        print(f"\n  {r.key}")
        for d in r.diffs:
            print(f"    {d.field}: stored={d.stored!r} replayed={d.replayed!r}")

    if args.json_out:
        args.json_out.write_text(json.dumps({
            "total": report.total, "replayed": report.replayed, "agreed": report.agreed,
            "skipped": report.skipped, "errored": report.errored,
            "by_field": report.by_field,
            "disagreements": [
                {"key": r.key, "diffs": [
                    {"field": d.field, "stored": d.stored, "replayed": d.replayed}
                    for d in r.diffs
                ]}
                for r in report.results if r.diffs
            ],
        }, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    if report.replayed == 0:
        print("\nNOTHING WAS REPLAYABLE — this is not a result. Every row lacked "
              "claims_detail, or the export omitted the column.", file=sys.stderr)
        return 1
    if report.errored == report.total:
        print(f"\nEVERY row errored ({report.errored}/{report.total}) — this is not a "
              "result. Check the export shape.", file=sys.stderr)
        return 1
    if report.by_field:
        print(f"\n{sum(report.by_field.values())} field disagreements across "
              f"{sum(1 for r in report.results if r.diffs)} rows — stored rows no longer "
              "reduce to what they carry.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
