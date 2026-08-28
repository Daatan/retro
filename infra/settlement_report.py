#!/usr/bin/env python3
"""Runs ON EC2 — called by ``logs.sh settlement`` via SSM.

Pairs each ``event=settlement_semantic_gates`` shadow line (retro#691, PR #698)
with the ``event=settlement_verifier`` verdict emitted on the same vote-set, and
reports where the deterministic gates and the LLM disagree.

Why this exists rather than a grep: the shadow gates are a candidate stand-in for
a verifier call that **fails open by design**, and the only thing that settles
whether they can stand in for it is agreement measured on live traffic. This is
also the tool that answers "is it safe to enforce yet" after a week of data.

Two counting traps it exists to avoid:

* The raw line count is not a decision count. Both events re-fire on every
  recompute; the verifier's own log once looked like 622 decisions and was 23
  distinct questions, one of them re-priced 144 times. Everything below is
  deduped before it is counted, and the raw totals are printed separately so the
  gap is visible rather than flattening silently.
* An errored verifier is not an allowed pin — it is an *unchecked* one. Those are
  broken out on their own line, because they are the population the gates were
  proposed for.

Read-only. No LLM calls, no writes, no network beyond reading the log file.
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
from collections import Counter
from pathlib import Path

DEFAULT_LOG = "/home/ubuntu/truthmachine/oracle_log.txt"

#: Both lines start with the stdlib ``asctime`` format, so a plain string
#: comparison against a ``YYYY-MM-DD`` prefix is a valid "since" filter.
_TS = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)")

_SHADOW = re.compile(
    r"event=settlement_semantic_gates would_block=(?P<block>True|False) "
    r"votes=(?P<votes>\d+) demoted=(?P<demoted>\d+) outlets_left=(?P<outlets>\d+) "
    r"gates=(?P<gates>[\w,]*) reasons=(?P<reasons>\{.*?\}) question=(?P<qhash>[0-9a-f]+)"
)
_VERIFIER = re.compile(
    r"event=settlement_verifier settles=(?P<settles>True|False) "
    r"errored=(?P<errored>True|False) enforced=(?P<enforced>True|False) "
    r"votes=(?P<votes>\d+) "
    r"(?:cached=(?:True|False) samples=\d+ agree=\d+ )?"
    r"question=(?P<qhash>[0-9a-f]+)"
)

#: A shadow line and the verifier verdict for the same vote-set are emitted from
#: the same call, microseconds apart. Anything further apart is a different
#: pricing of the same question, and pairing across it would invent agreement.
PAIR_WINDOW_SECONDS = 120


def _seconds(ts: str) -> int:
    """Coarse clock for the pairing window. Same-day arithmetic is enough: the
    two lines are emitted from one call, so a midnight straddle just declines to
    pair a handful of rows rather than pairing the wrong ones."""
    h, m, s = ts[11:].split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def parse(lines, since: str = "") -> tuple[list[dict], list[dict], int]:
    shadows: list[dict] = []
    verdicts: list[dict] = []
    errors = 0
    for line in lines:
        if "event=settlement_" not in line:
            continue
        m = _TS.match(line)
        if not m or (since and m.group(1) < since):
            continue
        ts = m.group(1)
        if "event=settlement_semantic_gates outcome=error" in line:
            errors += 1
            continue
        s = _SHADOW.search(line)
        if s:
            try:
                reasons = ast.literal_eval(s.group("reasons"))
            except (ValueError, SyntaxError):
                reasons = {}
            shadows.append({
                "ts": ts, "qhash": s.group("qhash"),
                "block": s.group("block") == "True", "votes": int(s.group("votes")),
                "demoted": int(s.group("demoted")), "outlets": int(s.group("outlets")),
                "gates": s.group("gates"), "reasons": reasons,
            })
            continue
        v = _VERIFIER.search(line)
        if v:
            verdicts.append({
                "ts": ts, "qhash": v.group("qhash"),
                "settles": v.group("settles") == "True",
                "errored": v.group("errored") == "True",
                "enforced": v.group("enforced") == "True",
                "votes": int(v.group("votes")),
            })
    return shadows, verdicts, errors


def pair(shadows: list[dict], verdicts: list[dict]) -> list[dict]:
    """Attach each verdict to the shadow line from the same call.

    Matched on question hash within :data:`PAIR_WINDOW_SECONDS`, nearest first.
    A shadow line with no verdict is kept unpaired — that is the verifier being
    disabled, skipped or short-circuited, and it is exactly the population the
    gates are meant to cover, so it must not be dropped.
    """
    by_q: dict[str, list[dict]] = {}
    for v in verdicts:
        by_q.setdefault(v["qhash"], []).append(v)
    used: set[int] = set()
    out = []
    for sh in shadows:
        best, best_gap = None, None
        for v in by_q.get(sh["qhash"], []):
            if id(v) in used:
                continue
            gap = abs(_seconds(v["ts"]) - _seconds(sh["ts"]))
            if gap <= PAIR_WINDOW_SECONDS and (best_gap is None or gap < best_gap):
                best, best_gap = v, gap
        if best is not None:
            used.add(id(best))
        out.append({**sh, "verdict": best})
    for v in verdicts:
        if id(v) not in used:
            out.append({"ts": v["ts"], "qhash": v["qhash"], "verdict": v, "block": None})
    return out


def dedupe(rows: list[dict]) -> list[dict]:
    """One row per distinct decision, not per re-pricing."""
    seen: dict[tuple, dict] = {}
    for r in rows:
        v = r["verdict"]
        key = (r["qhash"], r.get("votes"), r.get("block"), r.get("demoted"),
               None if v is None else (v["settles"], v["errored"], v["enforced"]))
        seen.setdefault(key, r)
    return list(seen.values())


def report(rows: list[dict], raw_shadow: int, raw_verdict: int, shadow_errors: int,
           since: str, out=sys.stdout) -> None:
    p = lambda *a: print(*a, file=out)  # noqa: E731
    p("=" * 74)
    p(f"SETTLEMENT SHADOW REPORT — retro#691" + (f", since {since}" if since else ""))
    p("=" * 74)
    p(f"\nraw lines          shadow={raw_shadow}  verifier={raw_verdict}"
      f"  shadow_errors={shadow_errors}")
    if shadow_errors:
        p("  !! the shadow path raised — it swallows and logs, so forecasts are")
        p("     unaffected, but the gates measured nothing on those calls.")
    p(f"distinct questions {len({r['qhash'] for r in rows})}")
    p(f"DECISIONS          {len(rows)}   <-- the real sample size; the raw counts")
    p("                       above are re-pricings of the same vote-sets")

    if not rows:
        p("\nNothing to report yet — no settled pool has been priced in this window.")
        return

    paired = [r for r in rows if r["verdict"] is not None and r["block"] is not None]
    errored = [r for r in paired if r["verdict"]["errored"]]
    live = [r for r in paired if not r["verdict"]["errored"]]
    shadow_only = [r for r in rows if r["verdict"] is None]
    verdict_only = [r for r in rows if r["block"] is None]

    p(f"\nverifier ran, gates ran     {len(paired)}")
    p(f"  of which it ERRORED       {len(errored)}   <-- fail-open: pin published unchecked")
    p(f"gates ran, no verdict line  {len(shadow_only)}   <-- verifier off/skipped")
    p(f"verdict line, no gates      {len(verdict_only)}   <-- pre-#698, or gates disabled")

    if live:
        tp = sum(1 for r in live if r["block"] and not r["verdict"]["settles"])
        fp = sum(1 for r in live if r["block"] and r["verdict"]["settles"])
        fn = sum(1 for r in live if not r["block"] and not r["verdict"]["settles"])
        tn = sum(1 for r in live if not r["block"] and r["verdict"]["settles"])
        p(f"\nAGREEMENT on the {len(live)} decisions where both actually decided")
        p(f"{'':<22}{'verifier BLOCKED':>18}{'verifier allowed':>18}")
        p(f"{'gates would block':<22}{tp:>18}{fp:>18}")
        p(f"{'gates would allow':<22}{fn:>18}{tn:>18}")
        agree = 100 * (tp + tn) / len(live)
        p(f"\n  agreement            {agree:.0f}%")
        if tp + fn:
            p(f"  reproduces           {100 * tp / (tp + fn):.0f}% of the verifier's blocks")
        p(f"  cost                 {fp} pins the verifier allowed would be blocked")
        p("\n  Neither side is ground truth: the verifier is one fail-open LLM call,")
        p("  and the gates carry a regex proxy for the claim side. Disagreement is a")
        p("  question to go read, not a defect on either side.")

    if errored:
        caught = sum(1 for r in errored if r["block"])
        p(f"\nFAIL-OPEN EXPOSURE: of {len(errored)} unchecked pins, the gates would")
        p(f"  have blocked {caught}. That number is the case for enforcing them.")

    reasons = Counter()
    for r in rows:
        reasons.update(r.get("reasons") or {})
    if reasons:
        p("\ndemotion reasons fired")
        for reason, n in reasons.most_common():
            p(f"  {reason:<38} {n}")

    gates = {r.get("gates") for r in rows if r.get("gates")}
    if len(gates) > 1:
        p(f"\n!! gate set changed mid-window ({', '.join(sorted(gates))}) — the")
        p("   numbers above mix configurations and are not comparable.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", default=DEFAULT_LOG)
    ap.add_argument("--since", default="", help="YYYY-MM-DD, compared against the line's timestamp")
    args = ap.parse_args()

    path = Path(args.log)
    if not path.exists():
        print(f"no log at {path}", file=sys.stderr)
        return 1
    with path.open(errors="replace") as fh:
        shadows, verdicts, errors = parse(fh, since=args.since)
    rows = dedupe(pair(shadows, verdicts))
    rows.sort(key=lambda r: r["ts"])
    report(rows, len(shadows), len(verdicts), errors, args.since)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
