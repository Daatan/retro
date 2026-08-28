#!/usr/bin/env python3
"""retro#691 step 2 — score the candidate gates against the widened label set.

Companion to `label_settlement_candidates.py`. Where
`backtest_settlement_semantic.py` scores gates against the settlement
verifier's own 9 recorded blocks, this scores them against independently
labelled (question, settled claim) pairs — the same judgement the gates make,
at the level they make it, across every settled row in the prod pool rather
than the handful the verifier happened to be called on.

Label semantics (from the labeller, blind to every gate input):
    ADJACENT — a real but different event; the gates SHOULD demote it.
    SETTLES  — the question's own event; the gates MUST NOT demote it.
    UNCLEAR  — undecidable from the text; scored separately, never counted as
               either a hit or a miss. Folding it into one of the other two
               would let a gate look good by firing on ambiguity.

Both the claim level and the pin level are reported. The claim level is where a
gate acts; the pin level is what a user sees, and the two can disagree — a gate
that demotes half the claims changes nothing if two independent outlets still
survive, because `settlement_min_sources` counts outlets, not claims.

The claim subject is still a regex proxy off the question text (see
`settlement_semantic.claim_subject_from_question`), so gate recall here remains
a LOWER bound on what a classifier-supplied dyad would achieve.

Usage:
    uv run python scripts/score_settlement_labels.py \
        --candidates label_candidates.jsonl --labels labels.jsonl
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forecast_api.settlement_semantic import (  # noqa: E402
    ALL_GATES,
    SettlementCandidate,
    apply_gates,
    claim_subject_from_question,
    pin_survives,
)

ADJACENT, SETTLES, UNCLEAR = "ADJACENT", "SETTLES", "UNCLEAR"


def _f(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _b(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in ("true", "false"):
        return value.lower() == "true"
    return None


def to_candidate(row: dict) -> SettlementCandidate:
    return SettlementCandidate(
        claim=row.get("claim") or "",
        stance=_f(row.get("stance")),
        certainty=_f(row.get("certainty")),
        outlet=row.get("outlet") or row.get("outlet_row"),
        event_actors=row.get("actors"),
        event_target=row.get("target"),
        event_date=row.get("event_date"),
        is_occurrence=_b(row.get("occ")),
        facet=row.get("facet"),
        evidence_class=row.get("cls"),
    )


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip().startswith("{")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--min-sources", type=int, default=2)
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    # Import here so the scorer and the labeller can never disagree on the key.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from label_settlement_candidates import candidate_key  # noqa: E402

    cands = load_jsonl(args.candidates)
    labels = {rec["key"]: rec for rec in load_jsonl(args.labels)}

    pairs = []
    for row in cands:
        rec = labels.get(candidate_key(row))
        if rec is None or rec["verdict"] == "ERROR":
            continue
        pairs.append((row, rec["verdict"]))

    print("=" * 78)
    print("SETTLEMENT GATES vs LABELLED PAIRS — retro#691 step 2")
    print("Claim subject is still a regex PROXY; recall is a LOWER bound.")
    print("=" * 78)

    dist = Counter(v for _, v in pairs)
    errors = sum(1 for rec in labels.values() if rec["verdict"] == "ERROR")
    unlabelled = len(cands) - len(pairs) - errors
    print(f"\ncandidate pairs        {len(cands)}  across {len({r['pid'] for r in cands})} questions")
    print(f"  labelled + scoreable {len(pairs)}   {dict(dist)}")
    print(f"  labeller ERRORs      {errors}   (excluded — a failed call is not a label)")
    print(f"  never labelled       {unlabelled}")

    scored = [(r, v) for r, v in pairs if v in (ADJACENT, SETTLES)]
    if not scored:
        print("\nNOTHING SCOREABLE — refusing to report a pass (retro#395).")
        return 1
    n_adj = sum(1 for _, v in scored if v == ADJACENT)
    n_set = sum(1 for _, v in scored if v == SETTLES)
    print(f"\nscoring on {len(scored)} decided pairs: {n_adj} ADJACENT (should demote), "
          f"{n_set} SETTLES (must keep)")
    if n_adj == 0 or n_set == 0:
        print("ONE-SIDED LABEL SET — precision or recall would be undefined; refusing to score.")
        return 1

    # ── claim level ──────────────────────────────────────────────────────────
    # Every non-empty subset, singles first. Reading the best row off this table
    # and shipping it would be selecting on the evaluation set; the table is here
    # to show which gate carries which cost, and the combination that ships has
    # to be defensible from WHY a gate misfires, not from its rank here.
    names = list(ALL_GATES)
    combos = sorted(
        (c for r in range(1, len(names) + 1) for c in combinations(names, r)),
        key=lambda c: (len(c), c),
    )
    table, json_rows = [], []
    for combo in combos:
        tp = fp = fn = tn = 0
        reasons = Counter()
        for row, verdict in scored:
            subject = claim_subject_from_question(row["question"])
            outcome = apply_gates(subject, [to_candidate(row)], gates=combo,
                                  deadline=row.get("deadline"))
            fired = bool(outcome.demoted)
            if fired:
                reasons.update(r for _, r in outcome.demoted)
            if verdict == ADJACENT:
                tp, fn = (tp + 1, fn) if fired else (tp, fn + 1)
            else:
                fp, tn = (fp + 1, tn) if fired else (fp, tn + 1)
        prec = tp / (tp + fp) if (tp + fp) else float("nan")
        rec = tp / (tp + fn) if (tp + fn) else float("nan")
        name = combo[0] if len(combo) == 1 else "+".join(g[:4] for g in combo)
        table.append((name, tp, fn, fp, tn, prec, rec))
        json_rows.append(dict(gate=name, caught=tp, missed=fn, false_pos=fp, clean=tn,
                              precision=prec, recall=rec, reasons=dict(reasons)))

    print(f"\n{'gate':<26}{'caught':>7}{'missed':>7}{'FALSE+':>8}{'clean':>7}{'prec':>7}{'recall':>8}")
    print("-" * 78)
    for name, tp, fn, fp, tn, prec, rec in table:
        print(f"{name:<26}{tp:>7}{fn:>7}{fp:>8}{tn:>7}{prec:>7.2f}{rec:>8.2f}")

    # ── pin level ────────────────────────────────────────────────────────────
    # A gate only matters if it changes whether the pin fires. Group by question
    # and ask that directly.
    by_q = defaultdict(list)
    for row, verdict in scored:
        by_q[row["pid"]].append((row, verdict))
    print(f"\npin level ({len(by_q)} questions, min_sources={args.min_sources}):")
    print(f"  {'gate':<26}{'pins before':>12}{'pins after':>12}{'lost a TRUE pin':>17}")
    print("  " + "-" * 66)
    for combo in combos:
        before = after = lost_true = 0
        for pid, group in by_q.items():
            subject = claim_subject_from_question(group[0][0]["question"])
            all_c = [to_candidate(r) for r, _ in group]
            base = apply_gates(subject, all_c, gates=(), deadline=group[0][0].get("deadline"))
            gated = apply_gates(subject, all_c, gates=combo, deadline=group[0][0].get("deadline"))
            b = pin_survives(base, min_sources=args.min_sources)
            a = pin_survives(gated, min_sources=args.min_sources)
            before += b
            after += a
            # "true pin" = the question had at least one SETTLES-labelled claim,
            # so a pin here is defensible and losing it is a real cost.
            if b and not a and any(v == SETTLES for _, v in group):
                lost_true += 1
        name = combo[0] if len(combo) == 1 else "+".join(g[:4] for g in combo)
        print(f"  {name:<26}{before:>12}{after:>12}{lost_true:>17}")

    if args.json_out:
        args.json_out.write_text(json.dumps(
            {"pairs": len(pairs), "distribution": dict(dist), "gates": json_rows}, indent=2))
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
