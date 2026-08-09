"""Eyeball what a lower `cluster_jaccard_threshold` would newly cluster (retro#414).

retro#414 measured that the live threshold (0.5) has never once been reached in
prod (max observed `max_jaccard` across 24 pools: 0.457) -- the correlated-evidence
discount cannot fire, even with `cluster_downweight_exponent` turned on. Two
candidate fixes were proposed; this script supports the cheaper one (lower the
threshold) by showing the actual TEXT behind each pairwise score, not just the
number -- `event=evidence_clusters` never logs pair-level text, only aggregate
stats, so there was no way to tell real echo from coincidence before this.

Input: a JSON dump of per-prediction pool rows from the daatan prod DB (the DB
lives on the daatan prod box, not the Oracle box). Print the ready-to-run dump
command with:

    python api/scripts/eyeball_cluster_pairs.py --sql

run it there via SSM (`~/.claude/skills/ssm-exec/ssm-run.sh prod ...`), then:

    uv run python api/scripts/eyeball_cluster_pairs.py pairs.json

Uses the real `cluster_text_for_claims`/`shingles`/`jaccard` from
`forecast_api.clustering` -- not a reimplementation -- so scores exactly match
what `/forecast` and `/pool/aggregate` would compute.

Stdlib-only; runs anywhere with the `forecast_api` package importable.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from forecast_api.clustering import cluster_text_for_claims, jaccard, shingles  # noqa: E402

DUMP_SQL = """\
WITH elig AS (
  SELECT "predictionId" pid FROM evidence_pool_articles
  WHERE claims_detail IS NOT NULL AND jsonb_array_length(claims_detail) > 0
    AND excluded IS NOT TRUE
  GROUP BY "predictionId" HAVING count(*) >= 2)
SELECT json_agg(json_build_object(
  'pid', elig.pid,
  'rows', (SELECT json_agg(json_build_object(
       'id', e.id, 'url', e.url, 'title', e.title, 'claims_detail', e.claims_detail))
     FROM evidence_pool_articles e
     WHERE e."predictionId" = elig.pid AND e.claims_detail IS NOT NULL
       AND jsonb_array_length(e.claims_detail) > 0 AND e.excluded IS NOT TRUE)
))
FROM elig;
"""


def iter_pairs(pools, shingle_size=3):
    """Yield (pid, row_a, row_b, text_a, text_b, score) for every textful pair."""
    for pool in pools:
        rows = pool.get("rows") or []
        texts = [cluster_text_for_claims(r.get("claims_detail")) for r in rows]
        shingle_sets = [shingles(t, shingle_size) if t else None for t in texts]
        for i in range(len(rows)):
            if not shingle_sets[i]:
                continue
            for j in range(i + 1, len(rows)):
                if not shingle_sets[j]:
                    continue
                score = jaccard(shingle_sets[i], shingle_sets[j])
                yield pool["pid"], rows[i], rows[j], texts[i], texts[j], score


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dump", type=Path, nargs="?")
    ap.add_argument("--sql", action="store_true", help="print the dump SQL and exit")
    ap.add_argument("--low", type=float, default=0.20, help="lower bound of the range to print")
    ap.add_argument("--high", type=float, default=0.50, help="upper bound of the range to print")
    ap.add_argument("--shingle-size", type=int, default=3)
    ap.add_argument("--snippet", type=int, default=280, help="chars of each text to print")
    args = ap.parse_args()

    if args.sql:
        print(DUMP_SQL)
        return 0
    if not args.dump:
        ap.error("dump is required unless --sql")

    pools = json.loads(args.dump.read_text(encoding="utf-8")) or []
    all_pairs = list(iter_pairs(pools, args.shingle_size))
    in_range = [p for p in all_pairs if args.low <= p[-1] <= args.high]

    print(f"{len(pools)} pools, {len(all_pairs)} textful pairs total")
    print(f"{len(in_range)} pairs score in [{args.low}, {args.high}]\n")

    for pid, row_a, row_b, text_a, text_b, score in sorted(in_range, key=lambda p: -p[-1]):
        print(f"--- pid={pid}  score={score:.3f} ---")
        print(f"  A: {row_a.get('title') or row_a.get('url')}")
        print(f"     {text_a[: args.snippet]}")
        print(f"  B: {row_b.get('title') or row_b.get('url')}")
        print(f"     {text_b[: args.snippet]}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
