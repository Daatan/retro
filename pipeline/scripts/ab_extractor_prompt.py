"""Extractor prompt A/B harness (retro#470) — baseline vs patched prompt over a
fixed case sample, on the live model, diffed on the facets that matter.

NOT a CI test — it calls Bedrock. This formalizes the ad hoc methodology
already run twice by hand (PR#309, PR#314) and once as a standalone script
(eval_extractor_adjacent_events.py) into something reusable for every future
extractor prompt edit — starting with retro#352 and retro#353, both blocked
on this. See docs/AB_HARNESS.md for the full walkthrough.

## Usage — two phases, because "baseline" and "patched" are two different
## checkouts and this script only ever runs in one of them at a time.

1. On the BASELINE checkout (e.g. `git checkout origin/main`):

    cd pipeline && AWS_REGION=us-east-1 .venv/bin/python scripts/ab_extractor_prompt.py \\
        run scripts/ab_cases/CASES.json --out /tmp/baseline.json --label baseline

2. On the PATCHED checkout (your branch):

    cd pipeline && AWS_REGION=us-east-1 .venv/bin/python scripts/ab_extractor_prompt.py \\
        run scripts/ab_cases/CASES.json --out /tmp/patched.json --label patched

3. Compare (pure, no network, either checkout):

    .venv/bin/python scripts/ab_extractor_prompt.py compare /tmp/baseline.json /tmp/patched.json

   Exit code IS the zero-regression gate: 0 = no in-scope case regressed. Prints
   a per-case report (regressions / improvements / no-change) plus a summary.

## The same-prompt/same-model control arm

For a case that carries `control_event_description` (retro#353's shape: the
resolution rules the batch path already passes correctly, vs the bare
question the live path passes today), add `--use-control-description` to a
`run` invocation to substitute it in — on the SAME checkout, SAME model,
SAME prompt, isolating the variable to just the input content:

    .venv/bin/python scripts/ab_extractor_prompt.py \\
        run scripts/ab_cases/CASES.json --out /tmp/control.json --label control \\
        --use-control-description

Then `compare /tmp/baseline.json /tmp/control.json` shows what "correct"
already looks like on this exact model/prompt combo — the natural control
the issue asks for wherever a second correct path exists.

## Temporal-leakage cases

Any case whose `article_date` postdates its `claim_deadline` is flagged
`[LEAKAGE]` in the report and excluded from the gate by default (AVeriTeC
prior art: hindsight evidence can make a patched prompt look better than it
really is). Pass `--allow-leakage` to `compare` to include it anyway.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
from pathlib import Path

from tm import llm
from tm.ab_harness import Case, build_case_results, gate_exit_code, load_cases
from tm.config import settings
from tm.extractor import PROMPT_PREFIX, PROMPT_SUFFIX
from tm.models import ExtractionOutput, PredictionExtraction

RUNS_PER_CASE = 5


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True,
        ).strip()
    except Exception:
        return "unknown"


async def _run_case(case: Case, model: str, *, use_control: bool, runs: int) -> list[list[dict]]:
    description = case.control_event_description if use_control else case.event_description
    if use_control and description is None:
        print(f"    [{case.id}] no control_event_description on this case — skipping control run")
        return []
    prompt = PROMPT_SUFFIX.format(
        article_text=case.article_text,
        source_name=case.source_name,
        journalist=case.journalist,
        article_date=case.article_date,
        event_name=case.event_name,
        event_description=description,
        claim_deadline=case.claim_deadline,
    )
    runs_out: list[list[dict]] = []
    for _ in range(runs):
        try:
            out, _usage = await llm.complete_structured(
                model, ExtractionOutput, prompt, max_tokens=1200, timeout=180,
                cached_prefix=PROMPT_PREFIX,
            )
        except Exception as exc:  # noqa: BLE001 — report, keep going
            print(f"    [{case.id}] EXCEPTION: {type(exc).__name__}: {exc}")
            continue
        runs_out.append([p.model_dump() for p in out.predictions])
    return runs_out


async def _cmd_run(args: argparse.Namespace) -> None:
    cases = load_cases(Path(args.cases))
    model = args.model or settings.extractor_model
    print(f"Running {len(cases)} cases x{args.runs_per_case} against {model}"
          f"{' (control description)' if args.use_control_description else ''}")
    results: dict[str, list[list[dict]]] = {}
    for case in cases:
        runs = await _run_case(
            case, model, use_control=args.use_control_description, runs=args.runs_per_case,
        )
        results[case.id] = runs
        print(f"  [{case.id}] {len(runs)} usable runs")
    payload = {
        "label": args.label,
        "git_commit": _git_commit(),
        "model": model,
        "use_control_description": args.use_control_description,
        "cases": {cid: {"case": _case_asdict(c), "runs": results[c.id]} for c in cases for cid in [c.id]},
    }
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"Wrote {args.out}")

    # Fail closed on an empty arm. Every per-case exception is caught in _run_case so one
    # bad case can't abort the sweep -- but that also means a run where EVERY call failed
    # (expired API key, revoked model access, wrong region) still wrote a well-formed
    # results file and exited 0. `compare` then reads two arms, finds no predictions to
    # disagree about, and reports no regression: a total outage renders as a clean pass.
    # The results file is still written -- the exceptions in it are the diagnosis -- but
    # the exit code now says the arm is unusable.
    empty = [cid for cid, runs in results.items() if not runs]
    if empty:
        print(f"\nFAIL: {len(empty)}/{len(cases)} cases produced 0 usable runs "
              f"({', '.join(empty[:5])}{', ...' if len(empty) > 5 else ''}) -- "
              f"this arm is not comparable. See the errors above.")
        sys.exit(1)


def _case_asdict(case: Case) -> dict:
    return {
        "id": case.id, "event_name": case.event_name,
        "event_description": case.event_description, "claim_deadline": case.claim_deadline,
        "article_date": case.article_date, "article_text": case.article_text,
        "expect": case.expect, "source_name": case.source_name, "journalist": case.journalist,
        "control_event_description": case.control_event_description, "tags": list(case.tags),
    }


def _load_results(path: str) -> tuple[dict, dict[str, list[list[PredictionExtraction]]]]:
    payload = json.loads(Path(path).read_text())
    predictions = {
        cid: [[PredictionExtraction(**p) for p in run] for run in entry["runs"]]
        for cid, entry in payload["cases"].items()
    }
    return payload, predictions


def _cmd_compare(args: argparse.Namespace) -> None:
    baseline_payload, baseline_preds = _load_results(args.baseline)
    patched_payload, patched_preds = _load_results(args.patched)
    cases = load_cases(Path(args.cases)) if args.cases else [
        Case(**{k: v for k, v in c["case"].items() if k != "tags"}, tags=tuple(c["case"]["tags"]))
        for c in baseline_payload["cases"].values()
    ]

    missing = [c.id for c in cases if c.id not in baseline_preds or c.id not in patched_preds]
    if missing:
        print(f"WARNING: cases missing from one side, skipped: {missing}")
        cases = [c for c in cases if c.id not in missing]

    # Refuse a dead arm rather than scoring it. A case present but with 0 usable runs meets no
    # facets, so an all-failed BASELINE cannot be regressed against -- the gate passes and every
    # case prints as `improved ... fixed`, reading an outage as a win. (An all-failed `patched`
    # is caught by the gate honestly, and both-empty prints `no change`.) `run` exits 1 on this
    # now, but results files get reused and re-compared long after that exit code is gone.
    dead = {
        "baseline": [c.id for c in cases if not baseline_preds.get(c.id)],
        "patched": [c.id for c in cases if not patched_preds.get(c.id)],
    }
    if any(dead.values()):
        for arm, ids in dead.items():
            if ids:
                print(f"UNUSABLE: {arm} has 0 usable runs for {len(ids)}/{len(cases)} cases: "
                      f"{', '.join(ids[:5])}{', ...' if len(ids) > 5 else ''}")
        print("\nRefusing to compare -- re-run the affected arm. A dead baseline scores as a win.")
        sys.exit(2)

    results = build_case_results(cases, baseline_preds, patched_preds)

    print(f"\nbaseline: {baseline_payload['label']} @ {baseline_payload['git_commit']} "
          f"({baseline_payload['model']})")
    print(f"patched:  {patched_payload['label']} @ {patched_payload['git_commit']} "
          f"({patched_payload['model']})\n")

    any_regression = False
    for r in results:
        leak = " [LEAKAGE]" if r.case.is_temporal_leakage else ""
        if r.regressions:
            any_regression = True
            print(f"  REGRESSION {r.case.id}{leak}: lost {sorted(r.regressions)}")
        elif r.improvements:
            print(f"  improved   {r.case.id}{leak}: fixed {sorted(r.improvements)}")
        elif r.patched_unmet:
            print(f"  no change  {r.case.id}{leak}: still failing {sorted(r.patched_unmet)}")
        else:
            print(f"  pass       {r.case.id}{leak}")

    code = gate_exit_code(results, allow_leakage=args.allow_leakage)
    gated_regressions = [
        r for r in results
        if r.regressions and (args.allow_leakage or not r.case.is_temporal_leakage)
    ]
    print(f"\nGate: {'FAIL' if code else 'PASS'} "
          f"({len(gated_regressions)} in-scope regression(s)"
          f"{', leakage cases excluded' if not args.allow_leakage and any_regression else ''})")
    sys.exit(code)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run a case sample against the live model, write results JSON")
    p_run.add_argument("cases", help="Path to a case-sample JSON file")
    p_run.add_argument("--out", required=True, help="Where to write the results JSON")
    p_run.add_argument("--label", required=True, help="Human label for this run (e.g. baseline, patched)")
    p_run.add_argument("--model", default=None, help="Override settings.extractor_model")
    p_run.add_argument("--runs-per-case", type=int, default=RUNS_PER_CASE)
    p_run.add_argument("--use-control-description", action="store_true",
                        help="Use each case's control_event_description instead of event_description")
    p_run.set_defaults(func=lambda a: asyncio.run(_cmd_run(a)))

    p_cmp = sub.add_parser("compare", help="Diff two results JSON files, exit non-zero on regression")
    p_cmp.add_argument("baseline")
    p_cmp.add_argument("patched")
    p_cmp.add_argument("--cases", default=None,
                        help="Case file to re-load (defaults to the cases embedded in --baseline)")
    p_cmp.add_argument("--allow-leakage", action="store_true",
                        help="Include temporal-leakage cases in the regression gate")
    p_cmp.set_defaults(func=_cmd_compare)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
