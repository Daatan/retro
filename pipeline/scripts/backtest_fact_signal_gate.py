#!/usr/bin/env python3
"""
Gate harness for the fact_signal -> estimator cutover: paired stance-vs-fact_signal
comparison over the fact_signal-covered pool subset, through the REAL /pool/aggregate
(no re-extraction -- the shadow columns ARE the snapshot). `fact_signal` is currently
read by nothing in aggregation (docs/ORACLE_VARIABLES.md, SourceSignal table); any
future cutover is gated on this harness showing real outcome-based evidence.

Two modes, automatic per prediction:
  * RESOLVED with coverage  -> Brier(stance) vs Brier(fact_signal), paired dBrier == THE GATE
  * ACTIVE/PENDING coverage -> P_stance vs P_fact divergence (impact preview of a cutover)

"0 resolved predictions yet" is a correct, honest report -- the gate fills in as
fact_signal-era predictions resolve; re-run then.

Input is a JSON dump of per-prediction pool rows from the daatan prod DB. The DB lives
on the daatan prod box (not the Oracul box); print the ready-to-run dump command with:

  python pipeline/scripts/backtest_fact_signal_gate.py --sql

run it there via SSM (`~/.claude/skills/ssm-exec/ssm-run.sh prod ...` -- mind the 24KB
SSM output cap when pulling the dump back), then:

  ORACLE_URL=https://oracle.daatan.com ORACLE_API_KEY=... \\
      python pipeline/scripts/backtest_fact_signal_gate.py gate_data.json [gate_report.json]

Stdlib-only; runs anywhere that can reach the Oracul -- including the prod box itself,
where the first run happened (/home/ubuntu/factsig_gate/, 2026-07-24; this file is that
box script committed, unchanged in behavior). Throttled to /pool/aggregate's 60/min
rate limit (THROTTLE env, seconds between calls).
"""

import json
import os
import statistics as st
import sys
import time
import urllib.error
import urllib.request

# Runs inside `docker exec -i daatan-postgres psql -U daatan -d daatan -X -A -t` on the
# daatan prod box; emits the JSON array this harness takes as input.
DUMP_SQL = """\
WITH elig AS (
  SELECT "predictionId" pid FROM evidence_pool_articles
  WHERE fact_signal IS NOT NULL AND stance IS NOT NULL AND excluded IS NOT TRUE
  GROUP BY "predictionId" HAVING count(*)>=3)
SELECT json_agg(json_build_object(
  'pid', p.id, 'claim', left(p."claimText",160), 'status', p.status,
  'resolvedAt', p."resolvedAt", 'claim_direction', p.claim_direction, 'claim_deadline', p.claim_deadline,
  'rows', (SELECT json_agg(json_build_object(
       'stance',e.stance,'fact_signal',e.fact_signal,'certainty',e.certainty,
       'credibility_weight',e.credibility_weight,'relevance_score',e.relevance_score,
       'evidence_weight',e.evidence_weight,'evidence_class',e.evidence_class,
       'settled',e.settled,'published_date',e.published_date))
     FROM evidence_pool_articles e
     WHERE e."predictionId"=p.id AND e.fact_signal IS NOT NULL AND e.stance IS NOT NULL AND e.excluded IS NOT TRUE)))
FROM predictions p JOIN elig ON elig.pid=p.id;
"""

THROTTLE = float(os.environ.get("THROTTLE", "1.1"))  # /pool/aggregate is 60/min


def clamp(v):
    return max(-1.0, min(1.0, v))


def post_aggregate(oracle, key, body):
    req = urllib.request.Request(
        oracle + "/pool/aggregate",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json", "x-api-key": key},
        method="POST",
    )
    for attempt in range(4):
        time.sleep(THROTTLE)
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 3:
                time.sleep(5 * (attempt + 1))
                continue
            raise RuntimeError(f"HTTP {e.code}: {e.read().decode()[:300]}") from e


def aggregate(oracle, key, rows, signal, cdir, cdl, drop_opinion=False):
    """One /pool/aggregate call with `signal` (stance or fact_signal) in the stance slot.

    drop_opinion was an exploratory knob in the July-2026 analysis; kept so those
    variants stay reproducible.
    """
    src = []
    for r in rows:
        v = r.get(signal)
        if v is None:
            continue
        if drop_opinion and (r.get("evidence_class") == "opinion"):
            continue
        src.append(
            dict(
                stance=clamp(v),
                certainty=r["certainty"] if r.get("certainty") is not None else 0.5,
                credibility_weight=r["credibility_weight"] if r.get("credibility_weight") is not None else 1.0,
                relevance_score=r["relevance_score"] if r.get("relevance_score") is not None else 0.5,
                evidence_weight=r.get("evidence_weight"),
                published_date=r.get("published_date"),
                settled=bool(r.get("settled")),
            )
        )
    body = dict(sources=src, claim_direction=cdir, claim_deadline=(cdl or None))
    return post_aggregate(oracle, key, body), len(src)


def main(argv):
    if "--sql" in argv:
        print("# run on the daatan prod box (the DB is there, not on the Oracul box):")
        print("docker exec -i daatan-postgres psql -U daatan -d daatan -X -A -t <<'SQL'")
        print(DUMP_SQL + "SQL")
        return 0
    if not 1 <= len(argv) <= 2:
        print(__doc__)
        return 2
    oracle = os.environ["ORACLE_URL"].rstrip("/")
    key = os.environ["ORACLE_API_KEY"]
    data = json.load(open(argv[0]))
    report_path = argv[1] if len(argv) == 2 else "gate_report.json"

    out = []
    for d in data:
        rr = d.get("rows") or []
        if len(rr) < 3:
            continue
        cdir = (d.get("claim_direction") or "").lower() or None  # daatan stores ARRIVAL/SURVIVAL; Oracul wants lowercase
        if cdir not in ("arrival", "survival"):
            cdir = None
        cdl = (d.get("claim_deadline") or "")[:10] or None  # date only
        try:
            a, _ns = aggregate(oracle, key, rr, "stance", cdir, cdl)
            b, _nf = aggregate(oracle, key, rr, "fact_signal", cdir, cdl)
        except Exception as e:
            print("ERR", d["pid"], repr(e)[:140])
            continue
        p_stance, p_fact = (a["mean"] + 1) / 2, (b["mean"] + 1) / 2
        rec = dict(
            pid=d["pid"],
            status=d["status"],
            n=len(rr),
            claim=d["claim"],
            Ps=round(p_stance, 4),
            Pf=round(p_fact, 4),
            dP=round(p_fact - p_stance, 4),
            settled_s=a.get("settled"),
            settled_f=b.get("settled"),
            resolvedAt=d.get("resolvedAt"),
            mean_stance=round(st.mean([r["stance"] for r in rr]), 3),
            mean_fact=round(st.mean([r["fact_signal"] for r in rr]), 3),
        )
        if d["status"] in ("RESOLVED_CORRECT", "RESOLVED_WRONG"):
            y = 1.0 if d["status"] == "RESOLVED_CORRECT" else 0.0  # positively-phrased claim assumption
            rec["outcome"] = y
            rec["brier_stance"] = round((p_stance - y) ** 2, 4)
            rec["brier_fact"] = round((p_fact - y) ** 2, 4)
        out.append(rec)

    # ---- preview (active/pending): impact of a cutover ----
    prev = [x for x in out if "outcome" not in x]
    prev.sort(key=lambda x: -abs(x["dP"]))
    print(f"=== PREVIEW: P(stance) vs P(fact_signal) on {len(prev)} un-resolved predictions ===")
    print(f"{'status':8s} {'n':>3s} {'P_st':>5s} {'P_fs':>5s} {'dP':>7s}  claim")
    for x in prev:
        print(f"{x['status'][:8]:8s} {x['n']:3d} {x['Ps']:.3f} {x['Pf']:.3f} {x['dP']:+.4f}  {x['claim'][:56]}")
    if prev:
        dps = [x["dP"] for x in prev]
        print(
            f"\nN={len(prev)}  mean_dP={st.mean(dps):+.4f}  mean|dP|={st.mean([abs(v) for v in dps]):.4f}  "
            f"max|dP|={max(abs(v) for v in dps):.3f}"
        )
        print(
            f"fact_signal LOWERS P: {sum(1 for v in dps if v < -0.02)}  RAISES: {sum(1 for v in dps if v > 0.02)}  "
            f"~same(|dP|<=.02): {sum(1 for v in dps if abs(v) <= 0.02)}"
        )

    # ---- gate (resolved): Brier ----
    res = [x for x in out if "outcome" in x]
    print(f"\n=== GATE: resolved predictions with fact_signal coverage = {len(res)} ===")
    if res:
        bs = [x["brier_stance"] for x in res]
        bf = [x["brier_fact"] for x in res]
        print(
            f"mean Brier  stance={st.mean(bs):.4f}  fact_signal={st.mean(bf):.4f}  "
            f"dBrier(fact-stance)={st.mean(bf) - st.mean(bs):+.4f}"
        )
        wins = sum(1 for x in res if x["brier_fact"] < x["brier_stance"])
        print(f"fact_signal better on {wins}/{len(res)} predictions")
        for x in res:
            print(
                f"  {x['status'][:3]} out={x['outcome']:.0f} P_st={x['Ps']:.3f} P_fs={x['Pf']:.3f} "
                f"B_st={x['brier_stance']:.3f} B_fs={x['brier_fact']:.3f}  {x['claim'][:50]}"
            )
    else:
        print("(none yet -- the gate fills in as fact_signal-era predictions resolve; re-run then.)")

    json.dump(out, open(report_path, "w"), default=str)
    print(f"\nwrote {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
