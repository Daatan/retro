"""Replay the settlement match gate over pins production has already published.

This is the measurement that justifies letting the gate act (retro#388/#360).
The gate ships enabled-but-shadow; before ``settlement_verifier_enforce`` is
turned on, someone has to be able to say what it would have done to the pins we
already have — including the three that resolved WRONG.

Input: a JSON array, one object per pin::

    [{"q": "<the claim, verbatim>",
      "status": "ACTIVE | RESOLVED_CORRECT | RESOLVED_WRONG",
      "m": <published mean, 0-100>,
      "votes": [{"o": "<outlet>", "d": "<settlement event date>",
                 "c": [{"claim": "...", "quote": "..."}]}]}]

Produce it from the daatan snapshot store — the settling rows of the latest
pinning snapshot per prediction. ``quote`` is only present on rows extracted
after claims_detail shipped (F1/retro#364), and the gate is measurably weaker
without it, so the script reports quote coverage alongside the verdicts: a
replay over summary-only rows is a LOWER BOUND on live behaviour.

    uv run python scripts/replay_settlement_verifier.py pins.json [--model M] [--concurrency N]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from forecast_api.settlement_verifier import SettlementVote, verify_settlement  # noqa: E402


async def _one(case: dict, model: str, timeout_s: int, sem: asyncio.Semaphore) -> dict:
    votes = [
        SettlementVote(outlet=v.get("o"), claim=c.get("claim") or "",
                       quote=c.get("quote"), event_date=v.get("d"))
        for v in case.get("votes") or []
        for c in v.get("c") or []
    ]
    # `m` is the published mean on the daatan 0-100 scale; >= 50 is a YES pin.
    answer = "YES" if float(case.get("m") or 0) >= 50 else "NO"
    async with sem:
        verdict = await verify_settlement(
            case["q"], votes, model=model, timeout_s=timeout_s, answer=answer,
        )
    return {
        "q": case["q"],
        "status": case.get("status"),
        "answer": answer,
        "n_votes": len(votes),
        "has_quotes": any(v.quote for v in votes),
        "settles": verdict.settles,
        "errored": verdict.errored,
        "reason": verdict.reason,
    }


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", type=Path)
    ap.add_argument("--model", default="bedrock/eu.anthropic.claude-haiku-4-5-20251001-v1:0")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=30)
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    sem = asyncio.Semaphore(args.concurrency)
    results = await asyncio.gather(*[_one(c, args.model, args.timeout, sem) for c in cases])

    by_status: dict[str, Counter] = {}
    for r in results:
        key = "ERRORED" if r["errored"] else ("KEEP" if r["settles"] else "VETO")
        by_status.setdefault(r["status"] or "UNKNOWN", Counter())[key] += 1

    print(f"\n{len(results)} pins replayed · model {args.model}")
    print(f"quote coverage: {sum(1 for r in results if r['has_quotes'])}/{len(results)} "
          f"(summary-only pins understate the gate — see module docstring)\n")
    print(f"{'status':<18}{'keep':>6}{'veto':>6}{'error':>7}")
    for status, counts in sorted(by_status.items()):
        print(f"{status:<18}{counts['KEEP']:>6}{counts['VETO']:>6}{counts['ERRORED']:>7}")

    print("\nvetoed:")
    for r in results:
        if not r["settles"] and not r["errored"]:
            print(f"  [{r['status']}/{r['answer']}] {r['q'][:70]}\n      → {r['reason'][:150]}")

    if args.json_out:
        args.json_out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
