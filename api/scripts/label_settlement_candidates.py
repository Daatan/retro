#!/usr/bin/env python3
"""retro#691 step 2 — build the labelled set the gate backtest needs.

`backtest_settlement_semantic.py` scores the candidate gates against the
settlement verifier's own recorded verdicts. That label set is too small to
ship on: 622 verdicts collapse to 23 distinct questions and 9 blocks. This
script widens it.

Three widening paths were considered; two are dead:

  * **Pin-vs-outcome ground truth.** The honest label — "did the pin agree with
    the eventual resolution" — tops out at SEVEN examples. Prod has 59 resolved
    predictions and 12 that were ever pinned, and the intersection is 7 (4 of
    which still have their pool rows). `settlement_pin_ledger.jsonl` on the
    Oracul box holds 12 entries, 5 contradicted. Not a label set.
  * **Replaying the verifier over more questions.** Generates more verdicts, but
    the verifier is the thing under test — its own output cannot grade it.

  * **What this script does instead:** label each (question, settled claim) pair
    directly, on the question the gates actually decide — *does this settled
    fact establish the question's own event, or a different one?* That needs no
    resolution, so it is answerable for all 388 pairs across 52 questions in the
    prod pool, pinned or not.

The labeller is Claude Sonnet 5 on Bedrock. Prod's verifier runs Claude Haiku
4.5 (`settlement_verifier_model` falls through to `extractor_model`, which the
oracle-api drop-in pins to Haiku), so the labeller is a strictly stronger and
different model — not the graded system grading itself.

It is also deliberately BLIND to every field the gates consume: no stance, no
claim_strength/certainty, no `settled` flag, no event_actors/event_target, no
is_occurrence, no facet, no evidence_class. It sees the question, the claim,
the quote and the dates. A labeller shown the gate's inputs is marking the
gate's own worksheet.

Usage:
    uv run python scripts/label_settlement_candidates.py \
        --candidates label_candidates.jsonl --out labels.jsonl [--limit N]

Resumable: every verdict is appended to --out and re-runs skip what is already
labelled, so an interrupted run costs nothing to restart.
"""
import argparse
import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Literal, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pipeline" / "src"))

from pydantic import BaseModel, Field  # noqa: E402

from tm.llm import complete_structured  # noqa: E402

DEFAULT_MODEL = "bedrock/eu.anthropic.claude-sonnet-5"

SETTLES = "SETTLES"
ADJACENT = "ADJACENT"
UNCLEAR = "UNCLEAR"

# Kept out of the per-call prompt body so Bedrock can cache it: it is
# byte-identical on all 388 calls.
INSTRUCTIONS = """\
You are auditing a forecasting system's evidence pool.

The system reads news articles about a forecasting QUESTION and extracts CLAIMS
from them. Some claims are marked as *settling* the question — meaning the
system believes the claim reports that the question's outcome is now a matter of
record, not a prediction. When enough settling claims accumulate, the system
pins the forecast at ~97% and stops updating it. A wrong pin is expensive and
hard to undo.

Your job is to judge ONE claim against ONE question and answer exactly this:

  Does this claim report the question's OWN event having occurred (or having
  been definitively foreclosed), or does it report a DIFFERENT event?

Verdicts:

  SETTLES  — the claim reports the question's own event: the same actor, the
             same act, within the question's own timeframe, and reported as
             something that has happened rather than something expected,
             planned, demanded, feared, or announced-as-intent. Choosing this
             means you would be comfortable freezing the forecast at 97% on the
             strength of this claim.

  ADJACENT — the claim is about something real and topically related, but it is
             not the question's own event. This is the common failure and it has
             many shapes: a different actor; a related but distinct act; a
             precursor, an announcement, a plan, a threat, a demand, a proposal,
             an indictment, a scheduled date; the right event in the wrong
             timeframe; a partial or a reversible step; commentary or analysis
             about the question rather than a report of its outcome. When the
             question asks about a state on a specific date, an event before
             that date does not settle it — the state can still change.

  UNCLEAR  — you genuinely cannot tell from the text given, or the quote does
             not support the claim.

Judge only what the text in front of you says. Do not use outside knowledge of
how the situation actually turned out — a claim that happens to guess right is
still ADJACENT if it is not a report of the question's own event. When SETTLES
and ADJACENT feel equally defensible, answer ADJACENT: the asymmetry is that a
missed pin merely leaves a live forecast slightly less confident, while a false
pin publishes a wrong answer at 97% and latches.

`reason` must be one sentence naming the specific mismatch (or the specific
match), not a restatement of the verdict.
"""


class Label(BaseModel):
    verdict: Literal["SETTLES", "ADJACENT", "UNCLEAR"]
    reason: str = Field(description="One sentence naming the specific mismatch or match.")


def candidate_key(row: dict) -> str:
    raw = "\x1f".join([
        row.get("pid") or "", row.get("claim") or "", row.get("quote") or "", row.get("url") or "",
    ])
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def build_prompt(row: dict) -> str:
    parts = [
        f"QUESTION: {row['question']}",
        f"Question deadline: {row.get('deadline') or 'not stated'}",
        "",
        f"Article outlet: {row.get('outlet') or 'unknown'}",
        f"Article published: {row.get('published') or 'unknown'}",
        "",
        f"CLAIM: {row.get('claim') or ''}",
        f"QUOTE FROM THE ARTICLE: {row.get('quote') or '(none given)'}",
        "",
        "Verdict?",
    ]
    return "\n".join(parts)


async def label_one(row: dict, model: str, sem: asyncio.Semaphore) -> dict:
    async with sem:
        try:
            out, usage = await complete_structured(
                model=model,
                response_model=Label,
                prompt=build_prompt(row),
                cached_prefix=INSTRUCTIONS,
                # 300 truncated some replies mid-JSON ("output is incomplete due to a
                # max_tokens length limit") and lost them as ERROR rows.
                max_tokens=1000,
                timeout=60,
                # Sonnet 5 rejects the parameter outright on Bedrock; omitting it
                # is the only way to call this model family at all.
                temperature=None,
            )
            return {
                "key": candidate_key(row), "pid": row["pid"], "url": row.get("url"),
                "claim": row.get("claim"), "quote": row.get("quote"),
                "verdict": out.verdict, "reason": out.reason,
                "model": model, "usage": usage,
            }
        except Exception as exc:  # noqa: BLE001 — one bad pair must not sink the run
            # Recorded, not silently dropped: an ERROR row is visible in the
            # tally instead of quietly shrinking the label set, which is how a
            # fail-open harness reports a broken run as a clean one.
            return {
                "key": candidate_key(row), "pid": row["pid"], "url": row.get("url"),
                "claim": row.get("claim"), "quote": row.get("quote"),
                "verdict": "ERROR", "reason": f"{type(exc).__name__}: {exc}"[:300],
                "model": model, "usage": {},
            }


async def main_async(args: argparse.Namespace) -> int:
    # psql prints its own "Output format is unaligned." banner ahead of the rows;
    # skip anything that isn't a JSON object rather than making the caller sed it out.
    candidates = [
        json.loads(line) for line in Path(args.candidates).read_text().splitlines()
        if line.strip().startswith("{")
    ]
    out_path = Path(args.out)
    done: set[str] = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                if rec.get("verdict") != "ERROR":
                    done.add(rec["key"])

    todo = [r for r in candidates if candidate_key(r) not in done]
    if args.limit:
        todo = todo[: args.limit]

    print(f"candidates={len(candidates)} questions={len({r['pid'] for r in candidates})} "
          f"already_labelled={len(done)} to_label={len(todo)} model={args.model}", flush=True)
    if not todo:
        print("nothing to do")
        return 0

    sem = asyncio.Semaphore(args.concurrency)
    results = []
    with out_path.open("a") as fh:
        for i in range(0, len(todo), 25):
            batch = todo[i:i + 25]
            for rec in await asyncio.gather(*(label_one(r, args.model, sem) for r in batch)):
                fh.write(json.dumps(rec) + "\n")
                results.append(rec)
            fh.flush()
            print(f"  {min(i + 25, len(todo))}/{len(todo)}", flush=True)

    tally: dict[str, int] = {}
    tok_in = tok_out = 0
    for rec in results:
        tally[rec["verdict"]] = tally.get(rec["verdict"], 0) + 1
        tok_in += (rec.get("usage") or {}).get("prompt_tokens", 0) or 0
        tok_out += (rec.get("usage") or {}).get("completion_tokens", 0) or 0
    print("\nverdicts:", json.dumps(tally, sort_keys=True))
    print(f"tokens: in={tok_in} out={tok_out}")

    errors = tally.get("ERROR", 0)
    if errors and errors == len(results):
        print("ERROR: every call failed — not a clean run", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", required=True, help="JSONL from export_settlement_labels.sql")
    ap.add_argument("--out", required=True, help="JSONL to append verdicts to (resumable)")
    ap.add_argument("--model", default=os.environ.get("LABEL_MODEL", DEFAULT_MODEL))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=6)
    return asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
