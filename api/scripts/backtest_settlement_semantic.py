"""Score candidate deterministic settlement gates against the verifier's own verdicts (retro#691).

**Read-only. Makes no LLM calls and touches no pipeline state.** The sibling
``replay_settlement_verifier.py`` re-runs the *model* over pins; this runs
*candidate code gates* over decisions the model has already made, so a gate can
be measured for free before anyone proposes enforcing it.

Why these labels: ``settlement_verifier`` is currently the only check on whether
a settled fact IS the claim's event, and the production log records every verdict
it ever reached. Those verdicts are an imperfect but pre-existing label set — the
gates' job is to reproduce the blocks without touching the allows.

## Building the dataset

Labels, from the oracle box (``/home/ubuntu/truthmachine/oracle_log.txt``)::

    grep 'event=settlement_verifier ' oracle_log.txt | grep -v 'outcome=skipped'

parsed by ``--verdicts``' loader below; both the pre- and post-2026-08-14 line
formats are accepted (``cached=``/``samples=``/``agree=`` were added later).

Questions, from daatan prod — every ``predictions`` row, keyed by
``sha256(claimText.strip().casefold())[:12]``, which is ``forecaster._question_hash``.

Votes, from daatan prod — see ``settlement_backtest_export.sql``.

    uv run python scripts/backtest_settlement_semantic.py \
        --verdicts verdicts.json --predictions pred_by_hash.json --rows settled_rows.json

## What it will NOT tell you

``claims_detail`` is forward-only from 2026-08-02 and was never backfilled, so a
verdict whose votes predate it has no reconstructable vote set. Those cases are
reported as UNRECONSTRUCTED and excluded from scoring — never silently counted as
agreement. A run that scores nothing exits 1: retro#395 shipped a settlement
replay that measured nothing and read as a pass, and that must not happen twice.

The claim side of every dyad is a PROXY derived from question text
(``claim_subject_from_question``), not the classifier field the gate actually
wants. Every score here is a lower bound, and the header says so on every run.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
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

_LINE = re.compile(
    r"^(?P<ts>\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+ \w+ [\w.]+ . "
    r"event=settlement_verifier settles=(?P<settles>True|False) "
    r"errored=(?P<errored>True|False) enforced=(?P<enforced>True|False) "
    r"votes=(?P<votes>\d+) "
    r"(?:cached=(?:True|False) samples=\d+ agree=\d+ )?"
    r"question=(?P<qhash>[0-9a-f]+) reason=(?P<reason>.*)$"
)


def load_verdicts(path: Path) -> list[dict]:
    """Accept either the raw log lines or an already-parsed JSON array."""
    text = path.read_text()
    if text.lstrip().startswith("["):
        return json.loads(text)
    out = []
    for line in text.splitlines():
        m = _LINE.match(line.strip())
        if not m:
            continue
        g = m.groupdict()
        out.append({
            "ts": g["ts"], "qhash": g["qhash"], "votes": int(g["votes"]),
            "settles": g["settles"] == "True", "errored": g["errored"] == "True",
            "enforced": g["enforced"] == "True", "reason": g["reason"].strip().strip("'\""),
        })
    return out


def candidates_for(rows: list[dict], pid: str, as_of: str, pred: dict) -> list[SettlementCandidate]:
    """The settled claims production would actually have put to the verifier.

    Mirrors the real path: rows are filtered ROW-level by ``settlement_grade`` and
    ``settlement_vote_validity`` (that is what ``agg.settlement_vote_indices``
    holds), then each surviving row expands into its settled claims
    (``votes_for_index``). Reconstructing from every settled row instead — the
    obvious shortcut — reproduced production's vote count only 44% of the time.

    ``today`` is pinned to the verdict's own timestamp so a window that has since
    closed is evaluated as it was then, not as it is now.
    """
    out: list[SettlementCandidate] = []
    for row in rows:
        if row.get("pid") != pid or (row.get("added_at") or "") > as_of:
            continue
        # No grade filter here: settlement_grade runs at CLAIM level upstream
        # (forecaster.py:329) and the stored `settled` column is already its
        # output, so re-applying it to the row's claim-weighted mean stance
        # wrongly drops rows whose settling claim is +1.0 inside a mixed
        # article. Measured: doing so cut vote-count fidelity from 44% to 22%.
        stance = float(row.get("stance") or 0.0)
        if settlement_vote_validity(
            stance, row.get("sed"), row.get("published"),
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
    ap.add_argument("--verdicts", type=Path, required=True)
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--rows", type=Path, required=True)
    ap.add_argument("--min-sources", type=int, default=2)
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    verdicts = load_verdicts(args.verdicts)
    preds = json.loads(args.predictions.read_text())
    rows = json.loads(args.rows.read_text())

    print("=" * 78)
    print("SETTLEMENT SEMANTIC GATE BACKTEST — retro#691")
    print("Claim-side dyad is a PROXY from question text; scores are a LOWER BOUND.")
    print("=" * 78)

    # ── dataset honesty ──────────────────────────────────────────────────────
    per_q = Counter(v["qhash"] for v in verdicts)
    joined = [v for v in verdicts if v["qhash"] in preds]
    print(f"\nverdicts                 {len(verdicts)}")
    print(f"  joined to a prediction {len(joined)}  ({len(verdicts) - len(joined)} unmatched question hashes)")
    print(f"DISTINCT QUESTIONS       {len(per_q)}   <-- the real sample size, not the verdict count")
    top = ", ".join(f"{n}" for _, n in per_q.most_common(5))
    print(f"  verdicts per question, top 5: {top}")
    labels = Counter(("errored" if v["errored"] else "block" if not v["settles"] else "allow")
                     for v in joined)
    print(f"  labels: {dict(labels)}  (errored excluded from scoring — the verifier failed open)")

    # ── reconstruction ───────────────────────────────────────────────────────
    cases, unreconstructed = [], 0
    for v in joined:
        if v["errored"]:
            continue
        p = preds[v["qhash"]]
        cands = candidates_for(rows, p["id"], v["ts"], p)
        if not cands:
            unreconstructed += 1
            continue
        cases.append((v, p, cands))
    exact = sum(1 for v, _, c in cases if len(c) == v["votes"])
    print(f"\nreconstruction: {len(cases)} scoreable, {unreconstructed} UNRECONSTRUCTED "
          f"(no claims_detail — forward-only from 2026-08-02, never backfilled)")
    if cases:
        print(f"  vote-count matches the log exactly on {exact}/{len(cases)} "
              f"({100 * exact / len(cases):.0f}%) — a proxy for whether the pool was "
              f"reconstructed as production saw it")

    if not cases:
        print("\nNOTHING SCOREABLE — refusing to report a pass (retro#395).")
        return 1

    # ── scoring ──────────────────────────────────────────────────────────────
    # Deduped unit: one decision per (question, vote-set), so a question re-priced
    # 144 times cannot dominate the score.
    def signature(cands):
        return (len(cands), tuple(sorted((c.outlet or "", c.claim[:60]) for c in cands)))

    combos = [(name,) for name in ALL_GATES] + [tuple(ALL_GATES)]
    results = {}
    for combo in combos:
        seen, tp = {}, None
        per_dec = {}
        for v, p, cands in cases:
            key = (v["qhash"], signature(cands))
            if key in per_dec:
                continue
            subject = claim_subject_from_question(p["claim"])
            outcome = apply_gates(subject, cands, gates=combo, deadline=p.get("deadline"))
            blocked = not pin_survives(outcome, min_sources=args.min_sources)
            per_dec[key] = (not v["settles"], blocked, Counter(r for _, r in outcome.demoted))
        tp = sum(1 for want, got, _ in per_dec.values() if want and got)
        fn = sum(1 for want, got, _ in per_dec.values() if want and not got)
        fp = sum(1 for want, got, _ in per_dec.values() if not want and got)
        tn = sum(1 for want, got, _ in per_dec.values() if not want and not got)
        reasons = Counter()
        for _, _, rs in per_dec.values():
            reasons.update(rs)
        results["+".join(combo)] = dict(tp=tp, fp=fp, fn=fn, tn=tn,
                                        decisions=len(per_dec), reasons=dict(reasons))

    print(f"\nunique decisions scored: {next(iter(results.values()))['decisions']}"
          f"  (deduped by question + vote-set)")
    print(f"\n{'gate':<40} {'caught':>7} {'missed':>7} {'FALSE+':>7} {'clean':>6}")
    print("-" * 72)
    for name, r in results.items():
        print(f"{name:<40} {r['tp']:>7} {r['fn']:>7} {r['fp']:>7} {r['tn']:>6}")
    print("\ncaught = verifier blocked and the gate would too (the win)")
    print("FALSE+ = verifier allowed but the gate would block (the cost — pins we would lose)")

    print("\ndemotion reasons fired:")
    for reason, n in Counter(results["+".join(tuple(ALL_GATES))]["reasons"]).most_common():
        print(f"  {reason:<38} {n}")

    if args.json_out:
        args.json_out.write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
