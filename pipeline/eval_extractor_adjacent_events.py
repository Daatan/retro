"""Extractor false-settlement / adjacent-event calibration eval.

NOT a CI test — it calls Bedrock. Run manually before changing the extractor prompt
OR the extractor model, and whenever re-evaluating whether a cheaper model than the
live Claude Haiku 4.5 override can be trusted:

    cd pipeline && AWS_REGION=us-east-1 .venv/bin/python eval_extractor_adjacent_events.py

## Background

The 2026-07-11/12 Illouz/Likud incident (a member leaving a party scored settled=true
for "a party withdraws from the race") is the single measured case behind the
extractor's Haiku override — see docs/ORACLE_VARIABLES.md and
tests/test_extractor_prompt.py. That A/B only ever tested ONE incident article; this
script widens the same methodology (temp default, n runs per case, count false
settlements) across the categories the prompt itself enumerates
(extractor.py's "## THE EVENT ITSELF vs. ADJACENT EVENTS", "## Single-winner
contests", "## Negated events", "## Capability and intent are not occurrence") with
synthetic, de-named cases in the house style (Company X / Candidate A / Force F —
see tests/test_extractor_prompt.py's de-naming note).

## What this measures

For each (article, event, expected_settled) case, run N times against each model in
MODELS and count how many runs produce `settled=True` when `expected_settled` is
False — a false settlement is the single highest-impact failure mode this prompt
guards against (it feeds directly into forecast resolution). This does NOT replace a
full quality eval — it's a fast, targeted regression/comparison check on exactly the
failure class that motivated the current model choice.

## Go/no-go (see the cost-reduction plan this script was built for)

Baseline from the one measured incident: Nova Lite 8/10 (80%) false-settlement rate
with the current prompt hardening; Haiku 4.5 0/10 (0%). Applied to this wider case
set:
  - Hard NO-GO: any candidate model averaging >= 30% false settlements across CASES.
  - GO: any candidate averaging <= 10% (matching Haiku's demonstrated result within
    normal model noise).
  - Otherwise: escalate to a human call — this script does not auto-decide.
"""
import asyncio
from collections import defaultdict

from tm.config import settings
from tm.extractor import PROMPT_PREFIX, PROMPT_SUFFIX
from tm.models import ExtractionOutput
from tm import llm

# Models to compare. settings.extractor_model is whatever's actually configured in
# THIS environment — the Haiku override only applies where oracle-api's systemd
# drop-in is present (i.e. on the live host), so running this locally will usually
# just resolve to the Nova Lite default. The explicit Haiku ID is listed separately
# so the comparison is meaningful even when run off the live host; deduped so the
# same model is never tested (and paid for) twice.
_LIVE_HAIKU_MODEL = "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"
MODELS = list(dict.fromkeys([
    "bedrock/amazon.nova-lite-v1:0",
    _LIVE_HAIKU_MODEL,
    settings.extractor_model,
]))

RUNS_PER_CASE = 10

# (event_name, event_description, claim_deadline, article_text, expected_settled)
# expected_settled=False means: this article must NOT cause a positive settlement of
# THIS event — it's evidence about an adjacent/related-but-distinct event instead.
CASES = [
    # --- Adjacent events: organization vs. member ---
    (
        "Party P withdraws from the 2026 election",
        "Party P, a mid-sized coalition partner, withdraws its candidate list entirely",
        "2026-08-01",
        "Two senior lawmakers announced they are quitting Party P and will sit as "
        "independents for the remainder of the term, citing policy disagreements with "
        "the leadership.",
        False,  # a member leaving is not the party leaving the race
    ),
    # --- Adjacent events: parent vs. subsidiary ---
    (
        "Company X exits the European market by 2026",
        "Company X, the parent conglomerate, ceases all European operations",
        "2026-12-31",
        "Company X's Belgian logistics subsidiary announced it will shut down its "
        "warehouse in Antwerp next quarter, citing rising rents.",
        False,  # one subsidiary's warehouse closure is not the parent exiting Europe
    ),
    # --- Adjacent events: primary vs. general election ---
    (
        "Candidate A wins the 2026 general election",
        "Candidate A wins the nationwide general election outright",
        "2026-11-03",
        "Candidate A won the party primary decisively last night, securing the "
        "nomination with 68% of the vote against two rivals.",
        False,  # winning a primary is not winning the general election
    ),
    # --- Single-winner contests: a rival's win settles the claim NO ---
    (
        "Team F wins the 2026 tournament",
        "Team F wins the tournament outright",
        "2026-09-15",
        "Team G stunned Team F 3-1 in Tuesday's final, lifting the trophy in front of "
        "a home crowd.",
        False,  # a rival's decisive win settles Team F's claim negatively, not positively
    ),
    # --- Negated events: score the claim AS WRITTEN ---
    (
        "A ceasefire will NOT be implemented in Region R by 2026",
        "The negated claim: no ceasefire takes effect in Region R this year",
        "2026-12-31",
        "Both sides signed a framework agreement Thursday and fighting has paused "
        "along the entire front line, officials on both sides confirmed.",
        False,  # this is strong evidence a ceasefire IS happening -> the NEGATED claim is not settled true
    ),
    # --- Capability and intent are not occurrence ---
    (
        "Force F will strike Target K by 2026",
        "Force F carries out a strike specifically against Target K",
        "2026-12-31",
        "Force F has demonstrated the capability to strike distant infrastructure, "
        "successfully hitting a power station in a different region last month.",
        False,  # a demonstrated capability against a DIFFERENT target is not an occurrence against Target K
    ),
    # --- Adjacent events: qualifying stage vs. the event itself ---
    (
        "Athlete B wins the 2026 championship",
        "Athlete B wins the championship final outright",
        "2026-08-20",
        "Athlete B cruised through the qualifying round, posting the fastest time of "
        "the morning session by a wide margin.",
        False,  # qualifying is not the final
    ),
    # --- Adjacent events: regional vs. national scope ---
    (
        "Party Q wins a majority nationally in 2026",
        "Party Q secures a national parliamentary majority",
        "2026-10-10",
        "Party Q swept the regional council elections in the capital district, "
        "winning every seat on offer.",
        False,  # a regional sweep is not a national majority
    ),
    # --- Stated intent is not an occurrence ---
    (
        "Company X launches Product Z by the end of 2026",
        "Company X ships Product Z to customers",
        "2026-12-31",
        "Company X's CEO said in an interview that the firm intends to bring Product Z "
        "to market as soon as possible, calling it a top priority for the coming year.",
        False,  # a stated intent/vow to ship is not the shipment itself
    ),
    # --- Buried past-tense fact about a DIFFERENT, adjacent matter ---
    (
        "Candidate A is indicted by the end of 2026",
        "Candidate A is formally charged with a crime",
        "2026-12-31",
        "Candidate A's former campaign treasurer was indicted last week on unrelated "
        "fraud charges stemming from a prior business venture.",
        False,  # an associate's indictment is not the candidate's own indictment
    ),
]


def _settlement_rate(results: list[bool]) -> float:
    return sum(results) / len(results) if results else 0.0


async def _run_case(model: str, case: tuple) -> list[bool]:
    event_name, event_description, claim_deadline, article_text, expected_settled = case
    prompt = PROMPT_SUFFIX.format(
        article_text=article_text,
        source_name="Test",
        journalist="unknown",
        article_date="2026-07-15",
        event_name=event_name,
        event_description=event_description,
        claim_deadline=claim_deadline,
    )
    false_settlements = []
    for _ in range(RUNS_PER_CASE):
        try:
            out, _ = await llm.complete_structured(
                model, ExtractionOutput, prompt, max_tokens=1200, timeout=180,
                cached_prefix=PROMPT_PREFIX,
            )
        except Exception as exc:  # noqa: BLE001 — report, keep going
            print(f"    EXCEPTION: {type(exc).__name__}: {exc}")
            continue
        settled_true = any(p.settled for p in out.predictions)
        false_settlements.append(settled_true and not expected_settled)
    return false_settlements


async def main() -> None:
    summary = defaultdict(list)
    for model in MODELS:
        print(f"\n=== {model} ===")
        for i, case in enumerate(CASES):
            event_name = case[0]
            results = await _run_case(model, case)
            rate = _settlement_rate(results)
            summary[model].extend(results)
            flag = " <-- FALSE SETTLEMENTS" if rate > 0 else ""
            print(f"  [{i}] {event_name!r}: {sum(results)}/{len(results)} false settlements{flag}")

    print("\n=== Summary (false-settlement rate across all cases) ===")
    for model, results in summary.items():
        rate = _settlement_rate(results)
        verdict = (
            "HARD NO-GO" if rate >= 0.30 else
            "GO" if rate <= 0.10 else
            "ESCALATE TO HUMAN CALL"
        )
        print(f"  {model}: {sum(results)}/{len(results)} = {rate:.0%} — {verdict}")


if __name__ == "__main__":
    asyncio.run(main())
