"""Step 0 offline measurement for retro#782 (Rule 3 of the source-dependence
plan, umbrella retro#779) — proposes `min_stratum_share`, the one tunable
number the issue's own text calls out ("propose it from the offline
measurement, not from a backtest").

**No code in aggregation.** This script only measures, on a prod dump, what
`source_lineage_stratum_factors` (`aggregation.py`) would do to real pools at
a range of candidate thresholds, using the real function — not a
reimplementation — so the numbers reported are exactly what enabling the
flag would produce.

Per pool this computes, for each candidate `min_stratum_share` in
`CANDIDATE_THRESHOLDS`:
  - how many distinct strata qualify vs. get folded into "other"
  - the needle delta (pp) between today's per-row pooling and the pool with
    `source_lineage_stratum_factors` applied (grand-total-preserving, per its
    docstring — this is NOT a comparison against the issue's literal
    normalize-to-1.0 spec, which `aggregation.py` deliberately departs from)

`source_id` is reconstructed the same way the live pipeline does it
(`forecaster._source_id_from_url(row.url)`), not read from a separately
stored column — this is what keeps the measurement's grouping identical to
what `aggregate_pool()` will actually group on in prod.

Input: a JSON dump from prod (the DB lives on the daatan prod box, not the
Oracul box). Print the ready-to-run dump command with:

    python api/scripts/measure_source_strata_782.py --sql

run it there via SSM, then:

    uv run python api/scripts/measure_source_strata_782.py pools.json

Stdlib-only beyond the `forecast_api` imports; runs anywhere with
`forecast_api` importable. Requires `data/source_strata.json` to exist
(generate it first with `generate_source_strata_782.py`).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from forecast_api.aggregation import (  # noqa: E402
    pool_sources,
    recency_weight,
    relevance_weight,
    source_lineage_stratum_factors,
    stance_to_prob,
)
from forecast_api.forecaster import _source_id_from_url  # noqa: E402

DUMP_SQL = """\
WITH elig AS (
  SELECT e."predictionId" pid FROM evidence_pool_articles e
  JOIN predictions p ON p.id = e."predictionId"
  WHERE e.claims_detail IS NOT NULL AND jsonb_array_length(e.claims_detail) > 0
    AND e.excluded IS NOT TRUE
    AND p."resolveByDatetime" > now()
  GROUP BY e."predictionId" HAVING count(*) >= 2)
SELECT json_agg(json_build_object(
  'pid', elig.pid,
  'rows', (SELECT json_agg(json_build_object(
       'id', r.id, 'url', r.url,
       'publishedDate', r.published_date,
       'stance', r.stance,
       'credibilityWeight', r.credibility_weight,
       'evidenceWeight', r.evidence_weight,
       'relevanceScore', r.relevance_score))
     FROM evidence_pool_articles r
     WHERE r."predictionId" = elig.pid AND r.claims_detail IS NOT NULL
       AND jsonb_array_length(r.claims_detail) > 0 AND r.excluded IS NOT TRUE)
))
FROM elig;
"""

RECENCY_HALF_LIFE_DAYS = 7.0  # config.py OracleConfig.recency_half_life_days default
STRATA_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "source_strata.json"
CANDIDATE_THRESHOLDS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]


def _load_strata_map() -> dict[str, str]:
    if not STRATA_PATH.exists():
        raise SystemExit(f"{STRATA_PATH} not found — run generate_source_strata_782.py first")
    return json.loads(STRATA_PATH.read_text()).get("strata", {})


def _row_weight(r: dict, ref_date) -> float:
    cred = r.get("credibilityWeight")
    cred = cred if cred is not None else 1.0
    ev = r.get("evidenceWeight")
    ev = ev if ev is not None else 0.6
    rw = recency_weight(r.get("publishedDate"), ref_date, RECENCY_HALF_LIFE_DAYS)
    rel = r.get("relevanceScore")
    rel_w = relevance_weight(rel) if rel is not None else 1.0
    return cred * ev * rw * rel_w


def measure_pool(pool: dict, strata_map: dict[str, str]) -> dict:
    rows = pool.get("rows") or []
    n = len(rows)
    source_ids = [_source_id_from_url(r["url"]) if r.get("url") else None for r in rows]
    ref_date = max((r.get("publishedDate") for r in rows if r.get("publishedDate")), default=None)
    weights = [_row_weight(r, ref_date) for r in rows]
    stances = [r.get("stance") for r in rows]

    mapped = sum(1 for sid in source_ids if sid is not None and sid in strata_map)
    mapped_fill_rate = mapped / n if n else 0.0

    usable = [(s, w) for s, w in zip(stances, weights) if s is not None]
    baseline_mean = pool_sources(list(s for s, _ in usable), list(w for _, w in usable))[0] if len(usable) >= 2 else None

    per_threshold = {}
    for thr in CANDIDATE_THRESHOLDS:
        factors = source_lineage_stratum_factors(source_ids, weights, strata_map, thr, True)
        if factors is None or baseline_mean is None:
            per_threshold[thr] = None
            continue
        adj_weights = [w * f for w, f in zip(weights, factors)]
        usable_adj = [(s, w) for s, w in zip(stances, adj_weights) if s is not None]
        if len(usable_adj) < 2:
            per_threshold[thr] = None
            continue
        adj_mean = pool_sources(list(s for s, _ in usable_adj), list(w for _, w in usable_adj))[0]
        delta_pp = (stance_to_prob(adj_mean) - stance_to_prob(baseline_mean)) * 100.0

        groups: dict[object, float] = {}
        for sid, w in zip(source_ids, weights):
            key = strata_map.get(sid) if sid is not None else None
            key = key if key is not None else object()
            groups[key] = groups.get(key, 0.0) + w
        total = sum(weights)
        n_qualifying = sum(1 for g in groups.values() if total > 0 and g / total >= thr)
        per_threshold[thr] = {"delta_pp": delta_pp, "n_strata_qualifying": n_qualifying}

    return {
        "pid": pool.get("pid"),
        "rows": n,
        "mapped_fill_rate": mapped_fill_rate,
        "n_distinct_source_ids": len(set(sid for sid in source_ids if sid is not None)),
        "per_threshold": per_threshold,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dump", type=Path, nargs="?")
    ap.add_argument("--sql", action="store_true", help="print the dump SQL and exit")
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    if args.sql:
        print(DUMP_SQL)
        return 0
    if not args.dump:
        ap.error("dump is required unless --sql")

    strata_map = _load_strata_map()
    pools = json.loads(args.dump.read_text(encoding="utf-8")) or []
    results = [measure_pool(p, strata_map) for p in pools]

    total_rows = sum(r["rows"] for r in results)
    print(f"{len(results)} pools, {total_rows} eligible rows, {len(strata_map)} outlets in strata map\n")

    fill_rates = [r["mapped_fill_rate"] for r in results if r["rows"]]
    if fill_rates:
        print(f"source_id mapped-into-a-stratum fill rate: mean={statistics.mean(fill_rates):.1%} "
              f"median={statistics.median(fill_rates):.1%}\n")

    print(f"{'threshold':<10} {'n_pools':<8} {'mean_abs_delta_pp':<18} {'median_abs_delta_pp':<20} {'pools|delta|>=5pp':<18} {'mean_n_strata_qualifying'}")
    for thr in CANDIDATE_THRESHOLDS:
        rows = [r["per_threshold"][thr] for r in results if r["per_threshold"].get(thr) is not None]
        if not rows:
            print(f"{thr:<10} 0")
            continue
        deltas = [abs(row["delta_pp"]) for row in rows]
        n_strata = [row["n_strata_qualifying"] for row in rows]
        big = sum(1 for d in deltas if d >= 5.0)
        print(f"{thr:<10} {len(rows):<8} {statistics.mean(deltas):<18.2f} "
              f"{statistics.median(deltas):<20.2f} {f'{big}/{len(rows)}':<18} "
              f"{statistics.mean(n_strata):.2f}")

    if args.json_out:
        args.json_out.write_text(
            json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
