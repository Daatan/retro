#!/usr/bin/env python3
"""retro#697 acceptance test — does emitting WHO/WHAT/SCOPE stop adjacent settlements?

retro#691 established that **58% of `settled=true` claims are about a different
event than the question** (215 ADJACENT of 371 decided pairs), and that no
downstream gate fixes it: the shipped trio stops 7 of 9 zero-support pins but
misses the founding case, `predicate_echo` adds zero pins over it, and requiring
the surviving outlets to agree on one event stops all 9 only by killing 9 of 20
real pins. The decision has to move upstream, to where `settled` is assigned.

This measures that change **before anyone edits `PROMPT_PREFIX`**, which is the
only way to propose an edit to a byte-identical cacheable prefix shared by every
call system-wide.

## What is actually varied

Arm A is the live rules, **sliced out of `PROMPT_PREFIX` at runtime** rather than
pasted here — a control that can drift from the thing it controls for is not a
control. Arm B is retro#697: the same rules, plus (a) the WHO/WHAT/SCOPE
decomposition the prompt already *mandates* is made an emitted output rather
than an unobservable internal step, and (b) the "Buried facts" paragraph is
subordinated to MATCH THE EVENT instead of standing beside it — today it reads
as a licence to settle on an incidental clause, and the founding case IS a
buried-fact extraction.

Arm B therefore changes two things at once, wording and output shape. That is
deliberate — it is what retro#697 proposes — but it means a positive result
does not attribute between them.

## What this is NOT

It is not the full extractor call. Prod stores no article text (only `snippet`
and per-claim `quote`), so this re-decides `settled` from the question, claim and
quote — the same blind inputs the Sonnet labeller saw. The real extractor sees
the whole article and emits ~20 fields at once, and "Buried facts" is precisely
about material elsewhere in the article. So:

  * a NEGATIVE result here is decisive — if the rewrite cannot separate adjacent
    from settling claims even on the isolated judgement, it will not do better
    inside a longer call;
  * a POSITIVE result is necessary, not sufficient, and must still go through
    `docs/AB_HARNESS.md` on whole articles before any prompt PR.

Model is the LIVE extractor (Haiku 4.5 via the oracle-api drop-in), not a
stronger one — the point is what production would do.

Usage:
    uv run python scripts/ab_settled_decision.py \
        --candidates cand.jsonl --labels labels.jsonl --out decisions.jsonl
    uv run python scripts/ab_settled_decision.py --score --candidates ... --labels ... --out ...

Resumable: every decision is appended to --out, keyed by (arm, candidate).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pipeline" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tm.extractor import PROMPT_PREFIX  # noqa: E402
from tm.llm import complete_structured  # noqa: E402

from label_settlement_candidates import candidate_key  # noqa: E402

#: The live extractor, per infra/oracle-api.service.d/extractor-model.conf. Not a
#: stronger model on purpose: the question is what production does, not what a
#: better model could do.
DEFAULT_MODEL = "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"

ARM_A, ARM_B = "baseline", "decomposed"


def slice_section(heading: str, *, level: int) -> str:
    """Pull one section out of the live PROMPT_PREFIX by its heading.

    Copy-pasting the baseline rules into this file would make the control drift
    silently the first time someone edits the real prompt — and this script
    exists precisely to be run across such an edit.
    """
    start = PROMPT_PREFIX.find(heading)
    if start < 0:
        raise SystemExit(
            f"heading {heading!r} is no longer in PROMPT_PREFIX — the baseline arm "
            "cannot be reconstructed, so this run would compare against nothing."
        )
    stop = re.compile(rf"^#{{1,{level}}} ", re.M)
    m = stop.search(PROMPT_PREFIX, start + len(heading))
    return PROMPT_PREFIX[start:m.start() if m else len(PROMPT_PREFIX)].rstrip()


def build_arms() -> dict[str, str]:
    """Both arms, assembled from the live prompt.

    ``### Buried facts`` is a SUBSECTION of ``## SETTLED``, not a sibling — so the
    baseline is the SETTLED section as-is, and arm B is that same section with the
    one paragraph substituted in place. Appending a replacement instead of
    substituting would leave both wordings in the prompt, the permissive one first,
    and the comparison would measure nothing. (It did, in the first run of this
    script; hence this docstring and TestArms.)
    """
    settled = slice_section("## SETTLED — the event already happened", level=2)
    match_event = slice_section("## MATCH THE EVENT — do not credit a near-miss", level=2)
    buried = slice_section("### Buried facts — extract settlement even when", level=3)
    if buried not in settled:
        raise SystemExit(
            "'### Buried facts' is no longer inside '## SETTLED' — the arms would "
            "differ in structure as well as wording, so this comparison is void."
        )

    preamble = (
        "You are the settlement stage of a news-extraction pipeline. You are given a "
        "forecasting QUESTION (the RELATED EVENT) and one factual CLAIM drawn from a "
        "news article, with the verbatim QUOTE it came from.\n\n"
        "Decide only this: does that claim report the QUESTION's own outcome as an "
        "accomplished fact? Apply the rules below exactly as written.\n\n"
    )
    baseline = preamble + settled + "\n\n" + match_event

    # retro#697. Two changes, deliberately together because that is what #697
    # proposes — so a positive result does not attribute between them:
    #   1. "Buried facts" is subordinated to MATCH THE EVENT rather than standing
    #      beside it. Today it reads as a licence to settle on an incidental
    #      clause ("mark settled true regardless of how minor its role"), and the
    #      founding retro#691 case IS a buried-fact extraction.
    #   2. The WHO/WHAT/SCOPE decomposition MATCH THE EVENT already mandates is
    #      made an emitted output instead of an unobservable internal step, so
    #      something can actually tell whether the model did it.
    buried_subordinated = (
        "### Buried facts — a settlement may be incidental to the article's main topic\n"
        "A clear past-tense statement of the RELATED EVENT can appear as a single "
        "supporting clause inside an article whose main subject is something else "
        "entirely. Scan the WHOLE article, not just the headline or opening paragraph. "
        "A fact does not have to be the article's primary subject to be settled.\n"
        "This does NOT relax MATCH THE EVENT. An incidental clause must still match "
        "WHO, WHAT and SCOPE in full; being buried is not evidence that it matches, "
        "and a passing mention of a DIFFERENT event — an earlier episode, a related "
        "contest, a family member, a neighbouring arena — is never settled, however "
        "plainly it is stated. A statement of capability, intent, or a similar event "
        "elsewhere is not a past-tense report of THIS event."
    )
    decomposed = (
        preamble
        + settled.replace(buried, buried_subordinated)
        + "\n\n" + match_event
        + "\n\n## REQUIRED OUTPUT — decompose before you decide\n"
        "MATCH THE EVENT already requires the WHO/WHAT/SCOPE decomposition. Write it "
        "down instead of doing it silently: fill question_who/what/scope from the "
        "RELATED EVENT, and fact_who/what/scope from the CLAIM as reported. Then set "
        "matches_all_three, and only then settled. If any one of the three does not "
        "match, matches_all_three is false and settled MUST be false — a near-miss is "
        "evidence, never the event."
    )
    return {ARM_A: baseline, ARM_B: decomposed}


class BaselineDecision(BaseModel):
    settled: bool = Field(description="true iff the claim reports the question's own outcome as accomplished fact")
    reason: str = Field(description="one sentence")


class DecomposedDecision(BaseModel):
    question_who: str
    question_what: str
    question_scope: str
    fact_who: str
    fact_what: str
    fact_scope: str
    matches_all_three: bool
    settled: bool
    reason: str = Field(description="one sentence")


def build_body(row: dict) -> str:
    return "\n".join([
        f"RELATED EVENT (the question): {row['question']}",
        f"Question deadline: {row.get('deadline') or 'not stated'}",
        "",
        f"Article published: {row.get('published') or 'unknown'}",
        f"CLAIM: {row.get('claim') or ''}",
        f"QUOTE FROM THE ARTICLE: {row.get('quote') or '(none given)'}",
        "",
        "Is this claim settled?",
    ])


async def decide(row: dict, arm: str, prefix: str, model: str, sem: asyncio.Semaphore) -> dict:
    model_cls = BaselineDecision if arm == ARM_A else DecomposedDecision
    base = {"key": candidate_key(row), "arm": arm, "model": model, "pid": row["pid"],
            "claim": row.get("claim"), "url": row.get("url"), "quote": row.get("quote")}
    async with sem:
        try:
            out, usage = await complete_structured(
                model=model, response_model=model_cls, prompt=build_body(row),
                cached_prefix=prefix, max_tokens=1000, timeout=90, temperature=0,
            )
            return {**base, **out.model_dump(),
                    "usage": usage if isinstance(usage, dict) else None}
        except Exception as exc:  # noqa: BLE001 — one bad pair must not sink the run
            # Recorded, never silently dropped: a run where every call failed must
            # look like a failure, not like a clean sweep of "not settled".
            return {**base, "settled": None, "error": f"{type(exc).__name__}: {exc}"}



# ── pin level ────────────────────────────────────────────────────────────────
# A claim-level win is not the deliverable. The pin is what reaches production
# (settlement_min_sources=2 DISTINCT outlets at stance 0.94), so a rewrite that
# drops 40% of adjacent claims and still leaves two outlets standing in every bad
# pool has changed nothing that matters. These are the same 29/9/20 pools the
# gate numbers on retro#691 are quoted against — do not redefine them here.

def _outlet(row: dict) -> str:
    """The same outlet identity the gate scoring used — the stored ``outlet``
    column, not a domain re-derived from the URL. Two different derivations of
    "distinct source" would silently produce two different pin counts."""
    stored = (row.get("outlet") or row.get("outlet_row") or "").strip().lower()
    if stored:
        return stored
    from urllib.parse import urlparse
    try:
        return (urlparse(row.get("url") or "").netloc or "").lower().removeprefix("www.")
    except ValueError:
        return ""


def classify_pools(cands: list[dict], labels: dict[str, str]) -> dict[str, dict]:
    """Group candidates into pools and mark which pins are known-bad.

    ``zero_support`` = every settled claim in the pool is ADJACENT, with no
    UNCLEAR to hide behind: the pin fired and nothing in it speaks to the
    question. The strict reading (9 pools) is the one the shipped gates are
    scored against; the loose one (11, counting pools whose only non-ADJACENT
    claims are UNCLEAR) is printed beside it so the choice stays visible.
    """
    from collections import defaultdict
    pools: dict[str, dict] = {}
    grouped = defaultdict(list)
    for r in cands:
        grouped[r["pid"]].append(r)
    for pid, rows in grouped.items():
        outlets = {_outlet(r) for r in rows} - {""}
        labs = [labels.get(candidate_key(r)) for r in rows]
        pools[pid] = {
            "rows": rows,
            "pins_today": len(outlets) >= 2,
            "zero_support": not any(l == "SETTLES" for l in labs)
                            and not any(l == "UNCLEAR" for l in labs),
            "zero_support_loose": not any(l == "SETTLES" for l in labs),
            # >=2 outlets carrying a claim labelled SETTLES: this pin is defensible
            # on the labels alone, so losing it is unambiguously a regression. The
            # 20 "not known bad" pools include 6 that pin on a single SETTLES outlet
            # plus adjacent filler — those are not clean wins to protect.
            "well_supported": len({_outlet(r) for r, l in zip(rows, labs)
                                   if l == "SETTLES"} - {""}) >= 2,
            "question": rows[0].get("question", ""),
        }
    return pools


def pin_table(pools: dict[str, dict], got: dict[str, Optional[bool]]) -> dict[str, int]:
    """Would each pin still fire if only the claims this arm settled counted?

    A pool where the arm errored on some claim is scored on what it did decide —
    an error suppresses a claim, which flatters the arm, so the error count is
    printed alongside and a run with many errors should not be read.
    """
    out = {"bad_stopped": 0, "bad_total": 0, "good_kept": 0, "good_total": 0,
           "strict_kept": 0, "strict_total": 0, "undecided": 0}
    for pool in pools.values():
        if not pool["pins_today"]:
            continue
        kept = {_outlet(r) for r in pool["rows"]
                if got.get(candidate_key(r)) is True} - {""}
        out["undecided"] += sum(1 for r in pool["rows"]
                                if got.get(candidate_key(r), "missing") in (None, "missing"))
        if pool["zero_support"]:
            out["bad_total"] += 1
            out["bad_stopped"] += len(kept) < 2
        else:
            out["good_total"] += 1
            out["good_kept"] += len(kept) >= 2
        if pool["well_supported"]:
            out["strict_total"] += 1
            out["strict_kept"] += len(kept) >= 2
    return out


def short_model(model: str) -> str:
    """Last path segment of a Bedrock id, minus the version suffix — nova-lite,
    claude-haiku-4-5. Used as the run label, because an arm re-run on a different
    model is a different arm."""
    tail = model.rsplit("/", 1)[-1].removeprefix("us.").removeprefix("anthropic.")
    tail = tail.removeprefix("amazon.")
    return tail.split("-v1:")[0].rsplit("-2025", 1)[0]


def run_label(decision: dict) -> str:
    return f"{decision['arm']}@{short_model(decision.get('model') or DEFAULT_MODEL)}"


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


async def run(args) -> int:
    cands = load_jsonl(args.candidates)
    arms = build_arms()
    if args.only_arm:
        arms = {k: v for k, v in arms.items() if k == args.only_arm}
        if not arms:
            raise SystemExit(f"--only-arm {args.only_arm!r} is not one of {ARM_A}, {ARM_B}")
    done = {(run_label(d), d["key"]) for d in load_jsonl(args.out) if d.get("settled") is not None}
    label = short_model(args.model)
    todo = [(r, arm) for r in cands for arm in arms
            if (f"{arm}@{label}", candidate_key(r)) not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(cands)} candidates x {len(arms)} arms; {len(done)} already done; running {len(todo)}")
    if not todo:
        return 0
    sem = asyncio.Semaphore(args.concurrency)
    # Append as each call lands, not once at the end: a 770-call run that dies at
    # call 700 must not throw away 700 answers, and --out is the resume state.
    results = []
    with args.out.open("a", buffering=1) as fh:
        for fut in asyncio.as_completed([decide(r, arm, arms[arm], args.model, sem)
                                         for r, arm in todo]):
            row = await fut
            fh.write(json.dumps(row) + "\n")
            results.append(row)
            if len(results) % 50 == 0:
                print(f"  {len(results)}/{len(todo)}", flush=True)
    errors = sum(1 for r in results if r.get("settled") is None)
    print(f"wrote {len(results)} decisions, {errors} errors")
    if errors == len(results):
        print("EVERY call failed — refusing to report this as a run.", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--only-arm", choices=[ARM_A, ARM_B],
                    help="run one arm (e.g. the baseline on a second model)")
    ap.add_argument("--score", action="store_true", help="score an existing --out, make no calls")
    ap.add_argument("--dump-arms", action="store_true")
    args = ap.parse_args()

    if args.dump_arms:
        for name, text in build_arms().items():
            print(f"\n{'=' * 70}\nARM: {name}  ({len(text)} chars)\n{'=' * 70}\n{text}")
        return 0
    if args.score:
        return score(args)
    return asyncio.run(run(args))


def score(args) -> int:
    from collections import Counter
    labels = {d["key"]: d["verdict"] for d in load_jsonl(args.labels)}
    decisions = load_jsonl(args.out)
    cands = {candidate_key(r): r for r in load_jsonl(args.candidates)}

    print("=" * 74)
    print("SETTLED-DECISION A/B — retro#697 acceptance test")
    print("Isolated judgement, NOT the full extractor call. A negative result is")
    print("decisive; a positive one still needs the whole-article harness.")
    print("=" * 74)

    by_arm: dict[str, dict[str, Optional[bool]]] = {}
    for d in decisions:
        by_arm.setdefault(run_label(d), {})[d["key"]] = d.get("settled")

    print("\nPRODUCTION is the row this table is measured against: every one of these")
    print("claims was emitted with settled=true by the live extractor, so production")
    print("scores 215/215 ADJACENT kept and 156/156 SETTLES kept. An arm that does not")
    print("come close to 215/215 is not reproducing the failure, and cannot be read as")
    print("a fix for it.\n")
    print(f"{'run':<30}{'n':>6}{'errors':>8}{'ADJACENT kept settled':>24}{'SETTLES kept settled':>22}")
    for arm, got in sorted(by_arm.items()):
        keys = [k for k in got if labels.get(k) in ("ADJACENT", "SETTLES")]
        errs = sum(1 for k in got if got[k] is None)
        adj = [k for k in keys if labels[k] == "ADJACENT" and got[k] is not None]
        sett = [k for k in keys if labels[k] == "SETTLES" and got[k] is not None]
        a_kept = sum(1 for k in adj if got[k])
        s_kept = sum(1 for k in sett if got[k])
        print(f"{arm:<30}{len(keys):>6}{errs:>8}"
              f"{f'{a_kept}/{len(adj)} ({100*a_kept/max(1,len(adj)):.0f}%)':>24}"
              f"{f'{s_kept}/{len(sett)} ({100*s_kept/max(1,len(sett)):.0f}%)':>22}")
    print("\n  ADJACENT kept settled = the failure being fixed (lower is better)")
    print("  SETTLES kept settled  = the capability being risked (higher is better)")

    pairs = [(a, b) for a in by_arm for b in by_arm
             if a.startswith(ARM_A + "@") and b.startswith(ARM_B + "@")
             and a.split("@")[1] == b.split("@")[1]]
    for ARM_A_L, ARM_B_L in pairs:
        both = [k for k in by_arm[ARM_A_L]
                if by_arm[ARM_A_L][k] is not None and by_arm[ARM_B_L].get(k) is not None]
        flip = Counter((labels.get(k), by_arm[ARM_A_L][k], by_arm[ARM_B_L][k]) for k in both)
        print(f"\n{ARM_A_L} -> {ARM_B_L}: per-claim movement on {len(both)} pairs both decided")
        for (lab, a, b), n in sorted(flip.items(), key=lambda x: -x[1]):
            if a == b:
                continue
            arrow = "settled -> not" if a else "not -> settled"
            print(f"  {lab or 'UNLABELLED':<11} {arrow:<18} {n}")

    pools = classify_pools(load_jsonl(args.candidates), labels)
    pinning = sum(1 for p in pools.values() if p["pins_today"])
    bad = sum(1 for p in pools.values() if p["pins_today"] and p["zero_support"])
    loose = sum(1 for p in pools.values() if p["pins_today"] and p["zero_support_loose"])
    print(f"\nPIN LEVEL — {len(pools)} pools; {pinning} pin today (>=2 distinct outlets), "
          f"of which {bad} are zero-support (every settled claim ADJACENT, none UNCLEAR).")
    print(f"  {loose} on the looser reading that also counts UNCLEAR-only pools.")
    print("  Reference points already on retro#691: no gate stops 0 of 9; the shipped")
    print("  trio stops 7 of 9 and keeps 17 of 20; predicate_echo adds nothing to it.")
    strict = sum(1 for p in pools.values() if p["pins_today"] and p["well_supported"])
    print(f"  {strict} of the {pinning} are well-supported (>=2 outlets carrying a")
    print("  SETTLES claim) — losing one of those is unambiguously a regression.")
    print(f"\n{'run':<30}{'bad pins stopped':>20}{'good kept':>13}{'well-supported kept':>21}{'undecided':>11}")
    print(f"{'PRODUCTION (no gate)':<30}{'0/9':>20}{'20/20':>13}{f'{strict}/{strict}':>21}{0:>11}")
    for arm, got in sorted(by_arm.items()):
        t = pin_table(pools, got)
        stopped = f"{t['bad_stopped']}/{t['bad_total']}"
        keptp = f"{t['good_kept']}/{t['good_total']}"
        strictp = f"{t['strict_kept']}/{t['strict_total']}"
        print(f"{arm:<30}{stopped:>20}{keptp:>13}{strictp:>21}{t['undecided']:>11}")
    print("\n  An arm with undecided claims is scored on what it did decide, which")
    print("  flatters it — a suppressed claim cannot hold up a pin. Read the pin")
    print("  numbers only when that column is 0.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
