"""retro#360: does the live extractor still invert stance on the incident's own articles?

NOT a CI test — it calls Bedrock. Same methodology as
``eval_extractor_adjacent_events.py``: N runs per case against the live model,
count the failures.

## What this settles

The 2026-07-15 incident (question cmrmgga9…, "England will win their FIFA World
Cup 2026 semi-final match against Argentina on July 15") published **97,
settled=true** on four articles that plainly reported *Argentina beat England*,
extracted as **+1.0 for England**. Brier 0.94 — one of the two worst misses in
the resolved corpus.

The prompt was hardened afterwards: PR #291 (2026-07-17, "fix rival-win stance
inversion") added the single-winner-contest rule and an England–Argentina worked
example, and ``test_extractor_prompt.py`` asserts that example is present. But
asserting a string is in the prompt is not evidence the model obeys it. Nothing
has ever re-run the incident's own inputs.

## Why most of these cases are deliberately NOT the ESPN headline

The prompt's worked example is *"Argentina stun England with a late rally to
reach the final"* — near-verbatim the ESPN headline from the incident. Testing
on that string measures whether the model can copy an example sitting in its own
context, which is not the question. It is included as a **control** (it should be
easy), and the real evidence is the other rows:

- the **BBC live-blog** headline, which never says England lost — it reports an
  Argentina *equaliser*, and the original extraction read it +0.80. This is the
  hardest and most important case.
- the **Al Jazeera live blog**, a bare fixture title with no outcome at all.
- **aa.com.tr**, which reports the loss in different words than the prompt uses.

Expected stance for every case is NEGATIVE for "England will win".
"""
import asyncio
from collections import defaultdict

from tm.config import settings
from tm.extractor import PROMPT_PREFIX, PROMPT_SUFFIX
from tm.models import ExtractionOutput
from tm import llm

_LIVE_HAIKU_MODEL = "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"
MODELS = list(dict.fromkeys([_LIVE_HAIKU_MODEL, settings.extractor_model]))

RUNS_PER_CASE = 5

EVENT_NAME = "England will win their FIFA World Cup 2026 semi-final match against Argentina on July 15"
EVENT_DESCRIPTION = (
    "England win their FIFA World Cup 2026 semi-final match against Argentina, "
    "played on 2026-07-15"
)
CLAIM_DEADLINE = "2026-07-16"
ARTICLE_DATE = "2026-07-15"

# (label, article_text, contaminated_by_prompt)
CASES = [
    (
        "BBC live blog — the hard one (originally +0.80)",
        "Enzo Fernandez equalises. Argentina are level against England in the "
        "World Cup semi-final at the Estadio Azteca.",
        False,
    ),
    (
        "Al Jazeera live blog — bare fixture title (originally +1.0)",
        "World Cup 2026 semi-final: Argentina vs England — live updates from the "
        "Estadio Azteca.",
        False,
    ),
    (
        "aa.com.tr — the loss, in different words (originally -1.0, correct)",
        "Argentina rally late to sink England and book their place in the World "
        "Cup final.",
        False,
    ),
    (
        "ESPN — CONTROL, near-verbatim in the prompt (originally +1.0)",
        "Argentina stun England with late rally to reach World Cup final.",
        True,
    ),
    (
        "ESPN second headline (originally +1.0)",
        "Argentina show why they're World Cup champs with epic comeback against "
        "England.",
        False,
    ),
]


async def _run_case(model: str, article_text: str) -> list[dict]:
    prompt = PROMPT_SUFFIX.format(
        article_text=article_text,
        source_name="Test",
        journalist="unknown",
        article_date=ARTICLE_DATE,
        event_name=EVENT_NAME,
        event_description=EVENT_DESCRIPTION,
        claim_deadline=CLAIM_DEADLINE,
    )
    out_rows = []
    for _ in range(RUNS_PER_CASE):
        try:
            out, _ = await llm.complete_structured(
                model, ExtractionOutput, prompt, max_tokens=1200, timeout=180,
                cached_prefix=PROMPT_PREFIX,
            )
        except Exception as exc:  # noqa: BLE001 — report, keep going
            print(f"    EXCEPTION: {type(exc).__name__}: {exc}")
            continue
        # The article's vote is the claim-weighted picture; for a pass/fail read
        # the dominant signed claim is what matters — a positive stance on ANY
        # settled claim is the incident's exact shape.
        stances = [(p.stance, p.certainty, bool(p.settled)) for p in out.predictions]
        worst = max(stances, key=lambda s: s[0], default=(None, None, False))
        out_rows.append({
            "n_claims": len(stances),
            "max_stance": worst[0],
            "any_positive_settled": any(s > 0 and st for s, _c, st in stances),
            "all_negative": bool(stances) and all(s < 0 for s, _c, _st in stances),
        })
    return out_rows


async def main() -> None:
    summary = defaultdict(list)
    for model in MODELS:
        print(f"\n=== {model} ===")
        for label, text, contaminated in CASES:
            rows = await _run_case(model, text)
            if not rows:
                print(f"  {label}: NO RUNS COMPLETED")
                continue
            inverted = sum(1 for r in rows if (r["max_stance"] or 0) > 0)
            false_settle = sum(1 for r in rows if r["any_positive_settled"])
            clean = sum(1 for r in rows if r["all_negative"])
            tag = "  [CONTROL — in prompt]" if contaminated else ""
            print(f"  {label}{tag}")
            print(f"    all-negative {clean}/{len(rows)} · positive stance {inverted}/{len(rows)} "
                  f"· POSITIVE+SETTLED {false_settle}/{len(rows)}")
            if not contaminated:
                summary[model].append((inverted, false_settle, len(rows)))

    print("\n=== verdict (excluding the contaminated control) ===")
    for model, rows in summary.items():
        inv = sum(r[0] for r in rows)
        fs = sum(r[1] for r in rows)
        n = sum(r[2] for r in rows)
        print(f"  {model}")
        print(f"    stance inversions: {inv}/{n}   false settlements: {fs}/{n}")
        print(f"    → {'PASS — the incident does not reproduce' if inv == 0 else 'FAIL — retro#360 still live'}")


if __name__ == "__main__":
    asyncio.run(main())
