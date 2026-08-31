#!/usr/bin/env python3
"""Stage A driver for retro#526 — score stored Oracul estimates for outlier-ness.

Two modes:

    uv run python scripts/scan_outlier_estimates.py sql
        Print the ready-to-run prod dump command. The evidence lives in daatan's
        Postgres on the daatan prod box, not on the Oracul box.

    uv run python scripts/scan_outlier_estimates.py score dump.json [--out scan_a.json]
        Recompute every dumped snapshot against its own frozen roster and print
        the per-signal distributions.

The scoring pass runs the real aggregator **in-process** — no HTTP, so no
60/min rate limit and no dependence on the network. That is only legitimate
while local code equals deployed code, which is what ``--deployed-commit``
enforces: pass the sha from ``GET https://oracle.daatan.com/version`` and the
run aborts rather than warns if it differs from local HEAD. A Stage A run under
code that is not in production measures a system nobody is using.

**No thresholds.** Nothing here decides what counts as an outlier; it prints
distributions and the top rows per signal, and a human picks the cuts in the
issue. Exit 1 when nothing was scorable — a run that measured nothing must never
read as a pass (retro#395).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

# ApiSettings requires `oracle_api_key`, and the settlement verifier is the one
# path in an otherwise offline recompute that would reach Bedrock. Both are set
# before `forecast_api` is imported, because settings are read at import time —
# which is also why `--settlement-verifier` is read off `sys.argv` here rather
# than from the parsed arguments below.
#
# Leaving the verifier off is NOT free, and the flag exists because of it: with
# it off, `_apply_settlement_match_gate` returns the aggregate untouched, so a
# pin that prod's gate vetoed at publication time is re-applied by the recompute.
# That shows up as an S1 gap and a reproduction disagreement on a forecast where
# the estimator did the right thing. The default run is offline and fully
# deterministic; pass the flag to measure the same corpus the way prod sees it
# (one Bedrock call per pinned pool, fail-open) and compare the two.
os.environ.setdefault("ORACLE_API_KEY", "dummy")
if "--settlement-verifier" not in sys.argv:
    os.environ.setdefault("SETTLEMENT_VERIFIER_ENABLED", "false")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from forecast_api.outlier_scan import (  # noqa: E402
    SIGNAL_FIELDS,
    Distribution,
    ScanRow,
    distributions,
    scan,
    split_by_corpus,
    split_by_settled,
    top_rows,
)

# Runs inside `docker exec -i daatan-postgres psql -U daatan -d daatan -X -A -t`
# on the daatan PROD box (i-04ea44d4243d35624) — the DB is there, not on the
# Oracul box. Emits one JSON object per line (not one giant json_agg): a
# truncated dump then fails to parse at the truncation point instead of
# silently yielding a shorter array, which matters because SSM caps output at
# ~24KB and this dump is megabytes — write it to a file on the box and pull it
# back gzipped and base64-split.
#
# The universe is the LATEST non-clock snapshot per prediction that carries a
# source roster. Clock rows are excluded because they are arithmetic glide with
# no evidence behind them, and `insufficient_data` rows because an abstention
# published no number to be an outlier about.
#
# `published_date` / `settlement_event_date` are TEXT columns on
# evidence_pool_articles and are carried verbatim inside the snapshot JSON, so
# nothing here formats a date.
DUMP_SQL = """\
WITH latest AS (
  SELECT DISTINCT ON (cs."predictionId")
         cs.id                AS snapshot_id,
         cs."predictionId"    AS pid,
         cs."createdAt"       AS snapshot_created_at,
         cs.kind, cs.origin, cs.oracle_snapshot
  FROM context_snapshots cs
  WHERE cs.oracle_snapshot IS NOT NULL
    AND cs.kind <> 'clock'
    AND cs.insufficient_data IS NOT TRUE
    AND jsonb_typeof(cs.oracle_snapshot -> 'sources') = 'array'
    AND jsonb_array_length(cs.oracle_snapshot -> 'sources') > 0
  ORDER BY cs."predictionId", cs."createdAt" DESC
)
SELECT json_build_object(
  'pid',                  p.id,
  'claim',                p."claimText",
  'status',               p.status,
  'outcome_type',         p."outcomeType",
  'resolved_at',          p."resolvedAt",
  'claim_created_at',     p."createdAt",
  'claim_direction',      p.claim_direction,
  'claim_deadline',       p.claim_deadline,
  'claim_archetype',      p.claim_archetype,
  'confidence',           p.confidence,
  'ai_ci_low',            p.ai_ci_low,
  'ai_ci_high',           p.ai_ci_high,
  'snapshot_id',          l.snapshot_id,
  'snapshot_created_at',  l.snapshot_created_at,
  'kind',                 l.kind,
  'origin',               l.origin,
  'oracle_snapshot',      l.oracle_snapshot
)::text
FROM latest l JOIN predictions p ON p.id = l.pid;
"""

DUMP_HELP = """\
# 1. On the daatan PROD box (i-04ea44d4243d35624), write the dump to a file.
#    Base64 the SQL through SSM rather than quoting it inline.
docker exec -i daatan-postgres psql -U daatan -d daatan -X -A -t > /tmp/stage_a.jsonl <<'SQL'
{sql}SQL
wc -l /tmp/stage_a.jsonl; du -h /tmp/stage_a.jsonl

# 2. Pull it back. SSM truncates output at ~24KB and errors hide mid-output, so
#    compress, base64 and split — then reassemble locally in the listed order.
gzip -c /tmp/stage_a.jsonl | base64 -w0 | split -b 20000 - /tmp/stage_a.b64.
ls -1 /tmp/stage_a.b64.*
#    locally:  cat parts... | base64 -d | gunzip > stage_a.jsonl

# 3. Score it (V4 gate: pass the sha from https://oracle.daatan.com/version):
uv run python scripts/scan_outlier_estimates.py score stage_a.jsonl \\
    --out scan_a.json --deployed-commit <sha>
"""


def load_records(path: Path) -> list[dict]:
    """Accept either a JSON array or the JSONL the dump emits."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


#: What "today's rules" is made of. `outlier_scan.py` is excluded on purpose: it
#: is read by nothing in the estimator, so a branch that only adds the scanner
#: still computes exactly the numbers prod computes.
ESTIMATOR_PATHS = (
    "api/src/forecast_api",
    "pipeline/src/tm",
    ":(exclude)api/src/forecast_api/outlier_scan.py",
)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True, cwd=str(REPO_ROOT)
    ).stdout.strip()


def git_head() -> str:
    try:
        return _git("rev-parse", "HEAD")
    except Exception:  # noqa: BLE001 — provenance is best-effort, not a gate
        return "unknown"


def estimator_drift_vs(sha: str) -> list[str]:
    """Estimator files that differ between the deployed commit and local HEAD.

    Deliberately NOT a HEAD-vs-deployed sha comparison. Those shas never match
    on a feature branch — including the branch that adds this script — so a
    literal equality check would make the tool unrunnable everywhere and get
    switched off, which is the worst outcome for a gate. What actually has to
    hold is narrower and checkable: the code that decides the numbers must be
    byte-identical to what is serving. Raises if git cannot answer, because an
    unanswerable gate must fail closed.
    """
    return [f for f in _git("diff", "--name-only", sha, "HEAD", "--", *ESTIMATOR_PATHS).splitlines() if f]


def _fmt(v: float) -> str:
    if v != v:  # NaN
        return "     -"
    if abs(v) >= 1000 or (v and abs(v) < 0.001):
        return f"{v:>6.2e}"
    return f"{v:>6.3f}"


def print_table(title: str, dists: list[Distribution]) -> None:
    print(f"\n### {title}")
    print(f"{'signal':<42}{'n':>5}{'mean':>8}{'p10':>8}{'p50':>8}{'p90':>8}{'max':>8}")
    for d in dists:
        if d.n == 0:
            print(f"{d.label:<42}{0:>5}{'     -':>8}{'     -':>8}{'     -':>8}{'     -':>8}{'     -':>8}")
            continue
        print(f"{d.label:<42}{d.n:>5}{_fmt(d.mean):>8}{_fmt(d.p10):>8}"
              f"{_fmt(d.p50):>8}{_fmt(d.p90):>8}{_fmt(d.max):>8}")


def print_top(rows: list[ScanRow], k: int) -> None:
    print(f"\n## Top {k} per signal")
    for fieldname, label in SIGNAL_FIELDS:
        top = top_rows(rows, fieldname, k)
        if not top:
            continue
        print(f"\n{label}")
        for r in top:
            val = float(getattr(r, fieldname))
            stored = "  -" if r.stored_mean_pct is None else f"{r.stored_mean_pct:>3.0f}"
            recomp = "  -" if r.recomputed_mean_pct is None else f"{r.recomputed_mean_pct:>3d}"
            print(f"  {_fmt(val)}  stored={stored}% recomp={recomp}% "
                  f"n={r.n_scored:<3d} settled={str(r.settled_now)[:5]:<5} "
                  f"{r.claim[:64]}")


async def run_score(args: argparse.Namespace) -> int:
    local_head = git_head()
    if args.deployed_commit:
        try:
            drift = estimator_drift_vs(args.deployed_commit)
        except Exception as exc:  # noqa: BLE001 — an unanswerable gate fails closed
            print(f"ABORT: could not compare against {args.deployed_commit[:12]}: {exc}", file=sys.stderr)
            return 2
        if drift:
            print(
                f"ABORT: estimator code differs from deployed {args.deployed_commit[:12]}:\n  "
                + "\n  ".join(drift)
                + "\nAn in-process recompute is only 'today's rules' while these match.",
                file=sys.stderr,
            )
            return 2

    from forecast_api.forecaster import run_pool_aggregate

    records = load_records(args.dump)
    if args.limit:
        records = records[: args.limit]
    if not records:
        print(f"NOTHING TO SCORE: {args.dump} held no records.", file=sys.stderr)
        return 1

    total = len(records)

    def progress(done: int, _total: int) -> None:
        if done % 20 == 0:
            print(f"  ... {done}/{total}", file=sys.stderr)

    report = await scan(records, run_pool_aggregate, loo=not args.no_loo, on_progress=progress)

    print(f"\n# Stage A outlier scan — retro#526")
    print(f"records in dump : {report.total_input}")
    print(f"settlement match gate: "
          f"{'ENFORCING (as prod)' if args.settlement_verifier else 'OFF — pins prod vetoed may be re-applied'}")
    print(f"scored          : {report.scored}")
    print(f"skipped         : {sum(report.skipped.values())}  {report.skipped or ''}")
    if report.repro_agreement is not None:
        print(f"as-published reproduction agreement: {report.repro_agreement:.1%} "
              "(reported, not asserted — recency decay and rules changes both live here)")
    if report.loo_weight_agreement is not None:
        print(f"V8 max-weight == max-LOO row       : {report.loo_weight_agreement:.1%}")

    if report.scored == 0:
        print(
            "\nNOTHING WAS SCORABLE — this is not a result. Every record lacked a "
            "source roster, abstained, or errored. Check the dump's shape before "
            "reading anything above as a finding.",
            file=sys.stderr,
        )
        return 1

    print_table("all scored snapshots", distributions(report.rows))
    for name, subset in split_by_settled(report.rows).items():
        if subset:
            print_table(f"{name} ({len(subset)})", distributions(subset))
    for name, subset in split_by_corpus(report.rows).items():
        if subset:
            print_table(f"{name} ({len(subset)})", distributions(subset))

    print_top(report.rows, args.top)

    if args.out:
        args.out.write_text(
            json.dumps(
                report.to_artifact(
                    label=args.label, git_commit=local_head, deployed_commit=args.deployed_commit
                ),
                indent=2, ensure_ascii=False, default=str,
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {args.out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    sub.add_parser("sql", help="print the ready-to-run prod dump command")

    sc = sub.add_parser("score", help="score a dump and print the distributions")
    sc.add_argument("dump", type=Path)
    sc.add_argument("--out", type=Path, help="write the self-describing JSON artifact here")
    sc.add_argument("--label", default="stage-a")
    sc.add_argument("--deployed-commit", help="sha from GET https://oracle.daatan.com/version; aborts on mismatch")
    sc.add_argument("--limit", type=int, default=None)
    sc.add_argument("--top", type=int, default=10)
    sc.add_argument("--no-loo", action="store_true", help="skip leave-one-out attribution (N+ aggregates per pool)")
    sc.add_argument("--settlement-verifier", action="store_true",
                    help="run the settlement match gate as prod does (one Bedrock call per pinned pool)")

    args = ap.parse_args()
    if args.mode == "sql":
        print(DUMP_HELP.format(sql=DUMP_SQL))
        return 0
    return asyncio.run(run_score(args))


if __name__ == "__main__":
    raise SystemExit(main())
