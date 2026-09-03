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
from collections import Counter
from pathlib import Path

from tm import llm
from tm.ab_harness import (
    CONFIDENT_MIN_RUNS, Case, QuantityDiagnostic, build_case_results, gate_exit_code,
    load_cases, quantity_diagnostics,
)
from tm.config import settings
from tm.extractor import PROMPT_PREFIX, PROMPT_SUFFIX
from tm.models import ExtractionOutput, PredictionExtraction

RUNS_PER_CASE = 5

# retro#757: a case tagged `volatile` in its corpus JSON is run at this many times
# regardless of `--runs-per-case`, since the whole point of the tag is a measured
# history of a false regression at the default count (see `Case.volatile`).
VOLATILE_MIN_RUNS = 15


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True,
        ).strip()
    except Exception:
        return "unknown"


# Everything on ExtractionOutput that is NOT the predictions list. The gate scores
# per-prediction facets only, so until retro#686 an arm's results file recorded nothing
# at all about the article-level elicited fields — you could add one, ship it, and have
# no way to ask this harness whether the model ever filled it. `author_lean` had been
# invisible here since it landed. Recorded per run, beside the predictions.
_ARTICLE_FIELDS = (
    "author_lean", "author_lean_certainty", "consensus_view",
    # retro#697. QUESTION-level rather than article-level, which is why they get the
    # extra consistency line below: `author_lean` may legitimately differ between two
    # articles, this decomposition may not — it is of the event, and the event is the
    # same one every run was handed.
    "claim_actor", "claim_predicate", "claim_scope",
)


def _article_value(out, field):
    """One article-level field, JSON-safe. `claim_actor` is a nested model, and the
    results file has to survive `json.dumps` — everything else here is a scalar."""
    value = getattr(out, field, None)
    return value.model_dump() if hasattr(value, "model_dump") else value


async def _run_case(
    case: Case, model: str, *, use_control: bool, runs: int,
) -> tuple[list[list[dict]], list[dict], list[dict]]:
    """Returns (per-run predictions, per-run article-level fields, per-run token usage)."""
    description = case.control_event_description if use_control else case.event_description
    if use_control and description is None:
        print(f"    [{case.id}] no control_event_description on this case — skipping control run")
        return [], [], []
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
    article_out: list[dict] = []
    usage_out: list[dict] = []
    for _ in range(runs):
        try:
            out, usage = await llm.complete_structured(
                model, ExtractionOutput, prompt, max_tokens=1200, timeout=180,
                cached_prefix=PROMPT_PREFIX,
            )
        except Exception as exc:  # noqa: BLE001 — report, keep going
            print(f"    [{case.id}] EXCEPTION: {type(exc).__name__}: {exc}")
            continue
        runs_out.append([p.model_dump() for p in out.predictions])
        article_out.append({f: _article_value(out, f) for f in _ARTICLE_FIELDS})
        usage_out.append(dict(usage or {}))
    return runs_out, article_out, usage_out


async def _cmd_run(args: argparse.Namespace) -> None:
    cases = load_cases(Path(args.cases))
    model = args.model or settings.extractor_model
    print(f"Running {len(cases)} cases x{args.runs_per_case} against {model}"
          f"{' (control description)' if args.use_control_description else ''}")
    results: dict[str, list[list[dict]]] = {}
    article: dict[str, list[dict]] = {}
    usage: dict[str, list[dict]] = {}
    for case in cases:
        case_runs = max(args.runs_per_case, VOLATILE_MIN_RUNS) if case.volatile else args.runs_per_case
        runs, article_runs, usage_runs = await _run_case(
            case, model, use_control=args.use_control_description, runs=case_runs,
        )
        results[case.id] = runs
        article[case.id] = article_runs
        usage[case.id] = usage_runs
        volatile_note = " (volatile, retro#757)" if case.volatile else ""
        print(f"  [{case.id}] {len(runs)} usable runs{volatile_note}")
    payload = {
        "label": args.label,
        "git_commit": _git_commit(),
        "model": model,
        "use_control_description": args.use_control_description,
        "cases": {
            c.id: {
                "case": _case_asdict(c),
                "runs": results[c.id],
                "article_runs": article[c.id],
                "usage_runs": usage[c.id],
            }
            for c in cases
        },
    }
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"Wrote {args.out}")

    # A one-line-per-field fill summary for the article-level elicitations. The gate
    # cannot speak to these -- it scores per-prediction facets -- so without this an
    # arm's only evidence about a new article-level field is a JSON file nobody reads.
    all_article = [a for runs_ in article.values() for a in runs_]
    if all_article:
        print(f"\nArticle-level fill over {len(all_article)} run(s):")
        for f in _ARTICLE_FIELDS:
            vals = [a[f] for a in all_article if a[f] is not None]
            share = f"{100 * len(vals) / len(all_article):.0f}%"
            # A nested field's distribution lives on its discriminating key, not on the
            # whole object: `claim_actor` is {name, type} and it is `type` that carries
            # the kill criterion, exactly as `kind` does for `voice`.
            flat = [v.get("type") or v.get("kind") if isinstance(v, dict) else v for v in vals]
            counts = Counter(v for v in flat if isinstance(v, str))
            # Free text would print one singleton per case and say nothing. Suppress the
            # roll-call and report the property that actually matters for these two.
            noisy = len(counts) > len(all_article) / 3
            detail = "" if noisy or not counts else "  " + ", ".join(
                f"{k}={n}" for k, n in counts.most_common())
            print(f"  {f:<24} {len(vals):>3}/{len(all_article)} ({share:>4}){detail}")
        _print_decomposition_consistency(article)
    _print_quantity_report(cases, results, arm=args.label)

    total_tokens = sum(u.get("total_tokens", 0) for runs_ in usage.values() for u in runs_)
    n_calls = sum(len(runs_) for runs_ in usage.values())
    if n_calls:
        print(f"\nTokens: {total_tokens} over {n_calls} call(s) "
              f"({total_tokens / n_calls:.0f}/call)")

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


# ── the quantity report (retro#683 item 1.6) ─────────────────────────────────
#
# "Compare `quantity` against the question's threshold in code and log the result per
# rater beside `stance`." An arm is one model, so an arm's report IS the per-rater row.
#
# It is a DIAGNOSTIC and not part of the gate, on purpose: `quantity` is a shadow field
# whose validator has not been run yet, and `unmet_facets` is what decides whether a
# prompt PR may land. See the note on `Case.question_threshold`.

def _diag_rows(cases: list[Case], results: dict[str, list[list[dict]]]) -> list[QuantityDiagnostic]:
    parsed = {
        cid: [[PredictionExtraction(**p) for p in run] for run in runs]
        for cid, runs in results.items()
    }
    return quantity_diagnostics(cases, parsed)


def _pct(n: int, d: int) -> str:
    return f"{100 * n / d:.0f}%" if d else "  --"


def _print_quantity_report(
    cases: list[Case], results: dict[str, list[list[dict]]], *, arm: str,
) -> None:
    diags = _diag_rows(cases, results)
    if not diags:
        return
    print(f"\nQuantity vs question threshold — rater: {arm}")
    print(f"  {'case':<45} {'target':>10} {'fill':>10} {'exact':>10} "
          f"{'code=stance':>13} {'code ok':>9} {'stance ok':>10}")
    for d in diags:
        print(f"  {d.case_id:<45} "
              f"{d.target_hit:>3}/{d.runs:<3} {_pct(d.target_hit, d.runs):>3} "
              f"{d.filled:>3}/{d.predictions:<3} {_pct(d.filled, d.predictions):>3} "
              f"{d.exact:>3}/{d.labelled:<3} {_pct(d.exact, d.labelled):>3} "
              f"{d.agree:>3}/{d.agree + d.disagree:<3} {_pct(d.agree, d.agree + d.disagree):>4} "
              f"{d.code_correct:>4} {d.stance_correct:>9}")
    print("  " + _quantity_summary(diags))


def _quantity_summary(diags: list[QuantityDiagnostic]) -> str:
    """One line: the validator's own number first, then fill and the code-vs-stance gap.

    `target` leads because it is what retro#683's validator asks — did the rater get THIS
    case's number, unit and comparator right — while `exact` is per-prediction and so is
    diluted by correct extractions of other figures in the same article.
    """
    runs = sum(d.runs for d in diags)
    hit = sum(d.target_hit for d in diags)
    miscomp = sum(d.target_miscomparated for d in diags)
    preds = sum(d.predictions for d in diags)
    filled = sum(d.filled for d in diags)
    exact = sum(d.exact for d in diags)
    labelled = sum(d.labelled for d in diags)
    agree = sum(d.agree for d in diags)
    decided = agree + sum(d.disagree for d in diags)
    undecidable = sum(d.undecidable for d in diags)
    code_ok = sum(d.code_correct for d in diags)
    stance_ok = sum(d.stance_correct for d in diags)
    return (f"TOTAL target {hit}/{runs} ({_pct(hit, runs).strip()}), "
            f"of which right-number-wrong-comparator {miscomp}; "
            f"fill {filled}/{preds} ({_pct(filled, preds).strip()}), "
            f"exact {exact}/{labelled} ({_pct(exact, labelled).strip()}), "
            f"code agrees with stance {agree}/{decided} ({_pct(agree, decided).strip()}), "
            f"code abstained {undecidable}; "
            f"correct: code {code_ok}, stance {stance_ok}, of {preds} prediction(s)")


def _case_asdict(case: Case) -> dict:
    return {
        "id": case.id, "event_name": case.event_name,
        "event_description": case.event_description, "claim_deadline": case.claim_deadline,
        "article_date": case.article_date, "article_text": case.article_text,
        "expect": case.expect, "source_name": case.source_name, "journalist": case.journalist,
        "control_event_description": case.control_event_description, "tags": list(case.tags),
        # retro#683 — without these the arm's results file could not answer any quantity
        # question after the fact, which is the same blindness _ARTICLE_FIELDS fixed.
        "question_threshold": case.question_threshold, "expect_quantity": case.expect_quantity,
        "volatile": case.volatile,
    }


def _print_decomposition_consistency(article: dict) -> None:
    """How often the SAME question got the SAME decomposition across its runs.

    The fill rate cannot answer this and it is the field's whole premise: WHO/WHAT/
    SCOPE describe the RELATED EVENT, so every run of one case was handed the same
    event and must answer the same way. A field that fills at 100% while answering
    differently on every run is not a decomposition, it is a paraphrase generator —
    and its consumer (`settlement_semantic.ClaimSubject`) would be comparing the
    article against a different subject each time.

    Reported as distinct answers per case, lower is better and 1.0 is perfect. Free
    text is normalised on case and surrounding whitespace only: two genuinely
    different phrasings SHOULD count as two, since a downstream echo-match would
    treat them as two.
    """
    fields = ("claim_actor", "claim_predicate", "claim_scope")
    rows = []
    for f in fields:
        per_case = []
        for runs_ in article.values():
            vals = [a[f] for a in runs_ if a[f] is not None]
            if not vals:
                continue
            norm = {
                (v.get("name", ""), v.get("type", "")) if isinstance(v, dict)
                else str(v).strip().lower()
                for v in vals
            }
            per_case.append(len(norm))
        if per_case:
            rows.append((f, sum(per_case) / len(per_case), max(per_case), len(per_case)))
    if not rows:
        return
    print("\nDecomposition consistency (distinct answers per case, 1.0 = identical every run):")
    for f, mean, worst, n in rows:
        print(f"  {f:<24} mean {mean:.2f}   worst {worst}   over {n} case(s)")


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

    # The gate below scores facets; `quantity` is not one of them (retro#683). Print both
    # arms' quantity lines here anyway — a results file outlives the `run` that printed
    # them, and an arm-vs-arm read of fill and validator match is the whole reason two
    # arms exist.
    for label, payload in (("baseline", baseline_payload), ("patched", patched_payload)):
        diags = quantity_diagnostics(
            cases, {cid: [[PredictionExtraction(**p) for p in run] for run in e["runs"]]
                    for cid, e in payload["cases"].items()},
        )
        if diags:
            print(f"  quantity [{label}: {payload['model']}] {_quantity_summary(diags)}")
    print()

    print(f"\nbaseline: {baseline_payload['label']} @ {baseline_payload['git_commit']} "
          f"({baseline_payload['model']})")
    print(f"patched:  {patched_payload['label']} @ {patched_payload['git_commit']} "
          f"({patched_payload['model']})\n")

    any_regression = False
    for r in results:
        leak = " [LEAKAGE]" if r.case.is_temporal_leakage else ""
        if r.regressions:
            any_regression = True
            confidence_note = (
                f" -- LOW CONFIDENCE at {r.n_patched_runs} run(s), re-measure at "
                f"--runs-per-case {CONFIDENT_MIN_RUNS}+ before trusting this (retro#757)"
                if r.low_confidence_regression else ""
            )
            print(f"  REGRESSION {r.case.id}{leak}: lost {sorted(r.regressions)}{confidence_note}")
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
    low_confidence = [r for r in gated_regressions if r.low_confidence_regression]
    print(f"\nGate: {'FAIL' if code else 'PASS'} "
          f"({len(gated_regressions)} in-scope regression(s)"
          f"{', leakage cases excluded' if not args.allow_leakage and any_regression else ''})")
    if low_confidence:
        print(f"  {len(low_confidence)} of those measured at fewer than {CONFIDENT_MIN_RUNS} "
              f"runs -- retro#757: re-measure before treating as a real regression")
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
