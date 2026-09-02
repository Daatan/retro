"""Step 0 offline measurement for retro#780 (Rule 1 of the source-dependence
plan, umbrella retro#779).

**No code in aggregation.** This script only measures, on a prod dump, how many
pool rows collapse under three candidate dependence keys, and what the pooled
needle would do if they did. It reads only fields already on the wire
(``voice.attributed_to``, the retro#682 event key, and shingle-Jaccard text
similarity via ``forecast_api.clustering``) — see retro#780 for the rule this
measurement is meant to justify or refute.

Candidate keys:
  (a) attribution-only  — dominant claim's ``voice.attributed_to``, lower/stripped,
      NO alias table yet (the issue calls for one; this measures the raw string
      first to see whether it is worth building).
  (b) event-key-only     — ``clustering.event_key_for_row`` (retro#682), unchanged.
  (c) combined            — union of rows where (attribution matches AND event_key
      matches) OR (event_key matches AND claim-text Jaccard clears
      ``config.cluster_jaccard_threshold``).

For each pool this also estimates a **needle delta**: the pooled mean today
(each row its own vote) vs. the pooled mean if rule (c)'s groups each collapsed
to one row (stance = weighted mean of the group using the SAME per-row weight
described below — not credibility alone, despite retro#780's rule text saying
"credibility-weighted" — weight = MAX member weight). Per-row weight is reconstructed from
the stored ``credibilityWeight`` / ``evidenceWeight`` / ``relevanceScore`` /
``publishedDate`` columns using the real ``aggregation.recency_weight`` /
``relevance_weight`` / ``pool_sources`` — not a reimplementation — so the delta
is the same shape the live pipeline would produce, just computed offline on a
static dump.

Input: a JSON dump from prod (the DB lives on the daatan prod box, not the
Oracul box). Print the ready-to-run dump command with:

    python api/scripts/measure_dependence_key_780.py --sql

run it there via SSM, then:

    uv run python api/scripts/measure_dependence_key_780.py pools.json

Stdlib-only beyond the two `forecast_api` imports; runs anywhere with
`forecast_api` importable.
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
    stance_to_prob,
    weighted_mean,
)
from forecast_api.clustering import (  # noqa: E402
    cluster_by_event_key,
    cluster_text_for_claims,
    event_key_for_row,
    jaccard,
    shingles,
)

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
       'id', r.id, 'url', r.url, 'source', r.source,
       'publishedDate', r.published_date,
       'stance', r.stance,
       'credibilityWeight', r.credibility_weight,
       'evidenceWeight', r.evidence_weight,
       'relevanceScore', r.relevance_score,
       'claims_detail', r.claims_detail))
     FROM evidence_pool_articles r
     WHERE r."predictionId" = elig.pid AND r.claims_detail IS NOT NULL
       AND jsonb_array_length(r.claims_detail) > 0 AND r.excluded IS NOT TRUE)
))
FROM elig;
"""

RECENCY_HALF_LIFE_DAYS = 7.0  # config.py OracleConfig.recency_half_life_days default
JACCARD_THRESHOLD = 0.40  # config.py OracleConfig.cluster_jaccard_threshold default


def _get(obj, name):
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _dominant_claim(claims_detail):
    """Highest claim_strength wins, ties broken by array position. NOT the same
    ranking as clustering._pick_claim_facets, which first drops claims lacking
    event_actors/event_target before ranking — this function ranks every claim,
    so the attribution key and the event key can come from different claims on
    the same row (deliberate: a byline claim with no event facets can still
    carry voice)."""
    best = None
    best_rank = None
    for i, c in enumerate(claims_detail or []):
        try:
            strength = float(_get(c, "claim_strength") or 0.0)
        except (TypeError, ValueError):
            strength = 0.0
        rank = (-strength, i)
        if best_rank is None or rank < best_rank:
            best_rank, best = rank, c
    return best


def _attribution_key(claims_detail):
    """Raw ``voice.attributed_to`` of the dominant claim, lower/stripped. No
    alias table — this measurement decides whether one is worth building."""
    c = _dominant_claim(claims_detail)
    if c is None:
        return None
    voice = _get(c, "voice")
    if voice is None:
        return None
    attributed_to = _get(voice, "attributed_to")
    if not attributed_to:
        return None
    key = " ".join(str(attributed_to).lower().split())
    return key or None


class _UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.parent[max(rx, ry)] = min(rx, ry)

    def cluster_ids(self):
        ids: dict[int, int] = {}
        out = []
        for i in range(len(self.parent)):
            root = self.find(i)
            if root not in ids:
                ids[root] = len(ids)
            out.append(ids[root])
        return out


def _cluster_stats(cluster_ids):
    counts: dict[int, int] = {}
    for cid in cluster_ids:
        counts[cid] = counts.get(cid, 0) + 1
    echoed = sum(n for n in counts.values() if n > 1)
    return {
        "rows": len(cluster_ids),
        "distinct_keys": len(counts),
        "echoed_rows": echoed,
        "largest": max(counts.values()) if counts else 0,
    }


def measure_pool(pool: dict) -> dict:
    rows = pool.get("rows") or []
    n = len(rows)
    claims = [r.get("claims_detail") for r in rows]

    attribution_keys = [_attribution_key(c) for c in claims]
    event_keys = [event_key_for_row(c) for c in claims]
    texts = [cluster_text_for_claims(c) for c in claims]
    shingle_sets = [shingles(t, 3) if t else frozenset() for t in texts]

    # (a) attribution-only
    attr_uf = _UnionFind(n)
    attr_groups: dict[str, list[int]] = {}
    for i, k in enumerate(attribution_keys):
        if k is not None:
            attr_groups.setdefault(k, []).append(i)
    for members in attr_groups.values():
        for j in members[1:]:
            attr_uf.union(members[0], j)
    attr_stats = _cluster_stats(attr_uf.cluster_ids())

    # (b) event-key-only (retro#682, unchanged)
    event_ids, event_stats_raw = cluster_by_event_key(event_keys)

    # (c) combined: attribution+event_key match, OR event_key match + near-identical text
    combined_uf = _UnionFind(n)
    event_groups: dict[str, list[int]] = {}
    for i, k in enumerate(event_keys):
        if k is not None:
            event_groups.setdefault(k, []).append(i)
    for members in event_groups.values():
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                i, j = members[a], members[b]
                same_attr = (
                    attribution_keys[i] is not None
                    and attribution_keys[i] == attribution_keys[j]
                )
                near_identical = (
                    bool(shingle_sets[i])
                    and bool(shingle_sets[j])
                    and jaccard(shingle_sets[i], shingle_sets[j]) >= JACCARD_THRESHOLD
                )
                if same_attr or near_identical:
                    combined_uf.union(i, j)
    combined_ids = combined_uf.cluster_ids()
    combined_stats = _cluster_stats(combined_ids)

    attribution_fill_rate = (
        sum(1 for k in attribution_keys if k is not None) / n if n else 0.0
    )

    # Needle delta: today's per-row pooling vs. rule (c)'s collapsed groups.
    stances = [r.get("stance") for r in rows]
    ref_date = max((r.get("publishedDate") for r in rows if r.get("publishedDate")), default=None)
    weights = []
    for r in rows:
        cred = r.get("credibilityWeight")
        cred = cred if cred is not None else 1.0
        ev = r.get("evidenceWeight")
        ev = ev if ev is not None else 0.6
        rw = recency_weight(r.get("publishedDate"), ref_date, RECENCY_HALF_LIFE_DAYS)
        rel = r.get("relevanceScore")
        rel_w = relevance_weight(rel) if rel is not None else 1.0
        weights.append(cred * ev * rw * rel_w)

    def _collapsed_delta(cluster_ids):
        usable = [(s, w) for s, w in zip(stances, weights) if s is not None]
        if len(usable) < 2:
            return None
        u_stances, u_weights = zip(*usable)
        baseline_mean, *_ = pool_sources(list(u_stances), list(u_weights))

        groups: dict[int, list[int]] = {}
        for idx, cid in enumerate(cluster_ids):
            if stances[idx] is None:
                continue
            groups.setdefault(cid, []).append(idx)
        collapsed_stances, collapsed_weights = [], []
        for members in groups.values():
            m_stances = [stances[i] for i in members]
            m_weights = [weights[i] for i in members]
            collapsed_stances.append(weighted_mean(m_stances, m_weights))
            collapsed_weights.append(max(m_weights))
        if len(collapsed_stances) < 2:
            return None
        collapsed_mean, *_ = pool_sources(collapsed_stances, collapsed_weights)
        return (stance_to_prob(collapsed_mean) - stance_to_prob(baseline_mean)) * 100.0

    needle_delta_pp = _collapsed_delta(combined_ids)
    # Contrast: dropping the attribution/text corroboration requirement and
    # collapsing on event_key match alone (rule (b)) — event_key rarely disagrees
    # with a same-story judgement even without the extra check; this shows how
    # much of the rule's power the corroboration requirement is currently giving up.
    event_key_only_delta_pp = _collapsed_delta(list(event_ids))

    return {
        "pid": pool.get("pid"),
        "rows": n,
        "attribution": attr_stats,
        "event_key": {**event_stats_raw.__dict__, "distinct_keys": event_stats_raw.clusters},
        "combined": combined_stats,
        "attribution_fill_rate": attribution_fill_rate,
        "needle_delta_pp": needle_delta_pp,
        "event_key_only_delta_pp": event_key_only_delta_pp,
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

    pools = json.loads(args.dump.read_text(encoding="utf-8")) or []
    results = [measure_pool(p) for p in pools]

    total_rows = sum(r["rows"] for r in results)
    print(f"{len(results)} pools, {total_rows} eligible rows\n")

    def _agg(key):
        rows = sum(r["rows"] for r in results)
        distinct = sum(r[key]["distinct_keys"] for r in results)
        echoed = sum(r[key]["echoed_rows"] for r in results)
        return rows, distinct, echoed

    for label, key in (("attribution-only", "attribution"), ("event-key-only", "event_key"), ("combined", "combined")):
        rows, distinct, echoed = _agg(key)
        collapse_pct = 100.0 * (1 - distinct / rows) if rows else 0.0
        echo_pct = 100.0 * echoed / rows if rows else 0.0
        print(f"{label:<18} rows={rows:<6} distinct_keys={distinct:<6} "
              f"collapse={collapse_pct:5.1f}%  echoed_rows%={echo_pct:5.1f}%")

    fill_rates = [r["attribution_fill_rate"] for r in results if r["rows"]]
    if fill_rates:
        print(f"\nvoice.attributed_to fill rate: mean={statistics.mean(fill_rates):.1%} "
              f"median={statistics.median(fill_rates):.1%}")

    deltas = [r["needle_delta_pp"] for r in results if r["needle_delta_pp"] is not None]
    if deltas:
        abs_deltas = sorted((abs(d), d, r["pid"]) for d, r in zip(deltas, [r for r in results if r["needle_delta_pp"] is not None]))
        print(f"\nneedle delta (pp) under rule (c) collapse: n_pools={len(deltas)} "
              f"mean_abs={statistics.mean(abs(d) for d in deltas):.2f} "
              f"median_abs={statistics.median(abs(d) for d in deltas):.2f} "
              f"max_abs={max(abs(d) for d in deltas):.2f}")
        print(f"pools with |delta| >= 5pp: {sum(1 for d in deltas if abs(d) >= 5.0)}/{len(deltas)}")
        print("\ntop 10 by |needle delta|:")
        for abs_d, d, pid in sorted(abs_deltas, reverse=True)[:10]:
            print(f"  pid={pid}  delta={d:+.2f}pp")

    ek_deltas = [r["event_key_only_delta_pp"] for r in results if r["event_key_only_delta_pp"] is not None]
    if ek_deltas:
        print(f"\ncontrast — needle delta (pp) under event_key-ONLY collapse (rule (b), no "
              f"attribution/text corroboration): n_pools={len(ek_deltas)} "
              f"mean_abs={statistics.mean(abs(d) for d in ek_deltas):.2f} "
              f"median_abs={statistics.median(abs(d) for d in ek_deltas):.2f} "
              f"max_abs={max(abs(d) for d in ek_deltas):.2f}")
        print(f"pools with |delta| >= 5pp: {sum(1 for d in ek_deltas if abs(d) >= 5.0)}/{len(ek_deltas)}")

    if args.json_out:
        args.json_out.write_text(
            json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
