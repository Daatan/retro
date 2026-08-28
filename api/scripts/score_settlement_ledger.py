"""Score the settlement gates against pins the OUTCOME later contradicted (retro#691).

**Read-only. No LLM calls.** Every other measurement in this lane scores the
gates against another fallible judgement — the settlement verifier's verdicts
(``backtest_settlement_semantic.py``) or a Sonnet labeller's
(``score_settlement_labels.py``). This one scores them against what actually
happened, which is the only label that is not itself a model output.

## The catch, up front

That label barely exists. It needs a question that was **pinned and then
resolved**: 12 rows in the pin ledger, 5 of them contradicted. And the pin's
vote-set has to be reconstructable, which needs ``claims_detail`` — forward-only
from 2026-08-02, never backfilled. Run this and it prints the coverage before it
prints anything else, because the interesting number here is almost always the
sample size.

This script therefore exists to be **re-run as pins accrue** (retro#500's
recurring shape), not to settle the question today.

## Inputs

``--ledger`` — ``settlement_pin_ledger.jsonl`` off the Oracle box
(``data/settlement_pin_ledger.jsonl``; ``GET /leaderboard/settlement-pin-report``
serves the same rows).

``--data`` — a JSON array mixing ``{"kind":"pred",...}`` and ``{"kind":"row",...}``
objects exported from daatan prod; see ``settlement_backtest_export.sql`` for the
row shape and the three filters that make it faithful.

## What "would_block" means here

The gates are run over the vote-set **as of the resolution timestamp**, so a row
added after the question resolved cannot influence a pin that preceded it. The
pin is counted as blocked when the surviving rows no longer clear
``settlement_min_sources`` distinct outlets — the same arithmetic production uses.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from forecast_api.aggregation import settlement_vote_validity  # noqa: E402
from forecast_api.settlement_semantic import (  # noqa: E402
    ALL_GATES,
    SettlementCandidate,
    apply_gates,
    claim_subject_from_question,
    pin_survives,
)

DEFAULT_GATES = ("point_in_time", "occurrence_consistency", "facet_missing")


def candidates_for(rows: list[dict], pid: str, as_of: str, pred: dict) -> list[SettlementCandidate]:
    """The settled claims that were in the pool when the question resolved.

    Same reconstruction as ``backtest_settlement_semantic.candidates_for`` — row-level
    ``settlement_vote_validity`` then claim expansion, deliberately NOT re-applying
    ``settlement_grade`` (it ran at claim level upstream, so the stored ``settled``
    column is already its output; re-applying it to a row's mean stance halved
    reconstruction fidelity when measured).
    """
    out: list[SettlementCandidate] = []
    for row in rows:
        if row.get("pid") != pid or (row.get("added_at") or "") > as_of:
            continue
        if settlement_vote_validity(
            float(row.get("stance") or 0.0), row.get("sed"), row.get("published"),
            (pred.get("direction") or "").lower() or None,
            pred.get("deadline"), pred.get("created"), pred.get("archetype"),
            today=as_of[:10],
        ) is not None:
            continue
        for c in row.get("claims") or []:
            out.append(SettlementCandidate(
                claim=c.get("claim") or "",
                stance=float(c.get("st") or 0.0),
                certainty=float(c.get("ct") or 0.0),
                outlet=row.get("source"),
                event_actors=c.get("ac") or row.get("actors"),
                event_target=c.get("tg") or row.get("target"),
                event_date=c.get("ed") or row.get("sed"),
                is_occurrence=(c.get("occ") == "true") if c.get("occ") is not None else row.get("occ"),
                facet=c.get("fc") or row.get("facet"),
                evidence_class=c.get("cls") or row.get("cls"),
            ))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--gates", default=",".join(DEFAULT_GATES))
    ap.add_argument("--min-sources", type=int, default=2)
    args = ap.parse_args()

    ledger = [json.loads(l) for l in args.ledger.read_text().splitlines() if l.strip()]
    blob = json.loads(args.data.read_text())
    preds = {o["id"]: o for o in blob if o.get("kind") == "pred"}
    rows = [o for o in blob if o.get("kind") == "row"]
    gates = tuple(g.strip() for g in args.gates.split(",") if g.strip() in ALL_GATES)
    if not gates:
        print("no recognised gates", file=sys.stderr)
        return 1

    print("=" * 78)
    print("SETTLEMENT GATES vs OUTCOME — retro#691, ledger from retro#361/#455")
    print(f"gates: {'+'.join(gates)}   min_sources={args.min_sources}")
    print("=" * 78)

    contradicted = [e for e in ledger if e.get("contradicted")]
    print(f"\nledger entries        {len(ledger)}")
    print(f"  pins CONTRADICTED   {len(contradicted)}   <-- the only true positives that exist")
    print(f"  pins upheld         {len(ledger) - len(contradicted)}")

    scored, skipped = [], []
    for e in ledger:
        pid = e["prediction_id"]
        pred = preds.get(pid)
        if pred is None:
            skipped.append((pid, e, "no prediction row"))
            continue
        as_of = (e.get("resolved_at") or "")[:19].replace("T", " ")
        cands = candidates_for(rows, pid, as_of, pred)
        if not cands:
            # Two very different failures, and collapsing them into one message
            # hides which road is actually blocked: no rows at all means the
            # backfill gap, rows-but-no-candidates means the pin's real vote-set
            # is elsewhere and what survives today could not have voted.
            have = [r for r in rows if r.get("pid") == pid]
            why = ("no claims_detail (forward-only from 2026-08-02)" if not have else
                   f"{len(have)} rows carry claims_detail, but none is a valid vote as of "
                   f"{as_of[:10]} — the pin's own vote-set is not among them")
            skipped.append((pid, e, why))
            continue
        subject = claim_subject_from_question(pred.get("claim") or "")
        outcome = apply_gates(subject, cands, gates=gates, deadline=pred.get("deadline"))
        scored.append({
            "pid": pid, "contradicted": bool(e.get("contradicted")),
            "claim": (pred.get("claim") or "")[:88],
            "cands": len(cands), "demoted": len(outcome.demoted),
            "outlets": outcome.distinct_outlets,
            "blocked": not pin_survives(outcome, min_sources=args.min_sources),
            "reasons": sorted({r for _, r in outcome.demoted}),
        })

    print(f"\nreconstructable       {len(scored)} of {len(ledger)}")
    print(f"  of the contradicted {sum(1 for s in scored if s['contradicted'])} of {len(contradicted)}")
    if skipped:
        print("\nNOT SCOREABLE")
        for pid, e, why in skipped:
            mark = "CONTRADICTED" if e.get("contradicted") else "upheld      "
            print(f"  {pid[:12]}  {mark}  {why}")
        if not any(s["contradicted"] for s in scored):
            print("\n  Every contradicted pin is unscoreable, so nothing below speaks to")
            print("  whether the gates CATCH a bad pin — only to what they would cost.")

    if not scored:
        print("\nNOTHING SCOREABLE — refusing to report a pass (retro#395).")
        return 1

    print("\nper pin")
    print(f"  {'pin':<14}{'outcome':<14}{'gates':<10}{'votes':>6}{'demoted':>8}{'outlets':>8}  question")
    for s in sorted(scored, key=lambda x: not x["contradicted"]):
        print(f"  {s['pid'][:12]:<14}"
              f"{'CONTRADICTED' if s['contradicted'] else 'upheld':<14}"
              f"{'BLOCK' if s['blocked'] else 'allow':<10}"
              f"{s['cands']:>6}{s['demoted']:>8}{s['outlets']:>8}  {s['claim']}")
        if s["reasons"]:
            print(f"  {'':<14}{'':<14}{', '.join(s['reasons'])}")

    tp = sum(1 for s in scored if s["contradicted"] and s["blocked"])
    fn = sum(1 for s in scored if s["contradicted"] and not s["blocked"])
    fp = sum(1 for s in scored if not s["contradicted"] and s["blocked"])
    tn = sum(1 for s in scored if not s["contradicted"] and not s["blocked"])
    print(f"\n  caught {tp}/{tp + fn} contradicted pins;"
          f" would have cost {fp}/{fp + tn} upheld ones")

    # The whole point of printing this is that it is too small to act on. Say so
    # in the output rather than trusting whoever pastes it into an issue.
    if tp + fn < 5 or fp + tn < 5:
        print("\n  !! NOT A MEASUREMENT. Both classes are single digits — this is a")
        print("     worked example, not precision and recall. Re-run as pins accrue")
        print("     (retro#500); do not put these ratios in an enforcement argument.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
