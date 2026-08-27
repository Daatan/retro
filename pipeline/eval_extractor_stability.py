"""Extractor stance/certainty stability eval (retro#519).

NOT a CI test — it calls Bedrock (real cost, real latency). Run manually:

    cd pipeline && AWS_REGION=us-east-1 .venv/bin/python eval_extractor_stability.py

## Background

2026-08-11: a WhatsApp-reported case (a Le Monde source scoring 92% confidence on
the Netanyahu-PM-2026 forecast) showed Nova Lite producing wildly different
stance/confidence across repeated runs on identical input — 10 runs, confidence
75-100, median ~95, even at "most deterministic" sampling params
(temperature=0/top_p=1/top_k=1; `seed` is rejected by Nova outright).

2026-08-12 follow-up (retro#519, comment) re-ran the same *shape* of test through
the real production code path (`extract_predictions` -> `complete_structured`,
`instructor`'s schema-constrained MD_JSON mode) instead of a raw/standalone Bedrock
call, and found temp=0 (retro#516/#517, now shipped) tightened variance
substantially on 2/3 real-article cases — casting doubt on "the model itself is
unstable" as the full story. That re-test used substitute articles because the
literal Le Monde article was unfetchable at the time (paywalled + bot-blocked, not
yet in news-indexer's S3 archive via any reachable path).

2026-08-12, same day: root-caused separately as a **data-quality** bug (retro#520,
shipped) — search/pull-path sources fed the extractor a ~200-char title+snippet
fallback, never the real article body, independent of temperature. Once #520
shipped (news-indexer's archived-S3-text lookup wired into retro's fetch fallback)
the literal Le Monde article became fetchable for the first time — case 1 below.

## What this measures

For each (real article, event, claim) case, run N times against each model in
MODELS and record the primary elicited prediction's `stance`/`certainty`. Reports
range and population stdev per (model, case) — the stability question #519 asks,
not accuracy (there's no "correct" stance being checked against a label here).

Since retro#664 it also reports the **sign-flip rate**: the fraction of runs whose
stance sign differs from the modal sign on identical input. Range and stdev answer
"how much does the number move"; the sign-flip rate answers "how often does the
same article vote the opposite way", which is the failure retro#545 is about and
which stdev can hide — a case sitting at +0.05/-0.05 has a tiny stdev and a
catastrophic flip rate, while one swinging +0.2 to +0.9 has a large stdev and votes
the same way every time. It is also the P0 baseline for retro#664's paired
comparison of `sign(stance)` against a separately elicited `direction` field.

## Go/no-go

No formal threshold yet (unlike eval_extractor_adjacent_events.py's false-settlement
rate, there's no single agreed "acceptable variance" number for stance/certainty).
Read the printed ranges/stdevs across models per case: a candidate model whose stdev
is consistently and substantially lower than Nova Lite's, across all cases, is a
switch candidate — escalate to a human call either way, this script does not
auto-decide (same policy as eval_extractor_adjacent_events.py).
"""
import asyncio
from collections import Counter, defaultdict
from statistics import pstdev

from tm.config import settings
from tm.extractor import extract_predictions

# Models to compare. Explicit Haiku ID listed separately (not just
# settings.extractor_model) so the comparison is meaningful even off the live host,
# where oracle-api's systemd drop-in isn't present and settings.extractor_model
# resolves to the Nova Lite default; deduped so the same model is never tested (and
# paid for) twice.
_LIVE_HAIKU_MODEL = "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"
MODELS = list(dict.fromkeys([
    "bedrock/amazon.nova-lite-v1:0",
    _LIVE_HAIKU_MODEL,
    settings.extractor_model,
]))

# Matches the original WhatsApp report's run count (10) closely enough to be
# comparable while keeping the 3-case x 3-model matrix affordable; the 2026-08-12
# re-test used 8.
RUNS_PER_CASE = 8

# Real articles, pulled live via news-indexer's archived-text endpoint
# (GET /articles/text, retro#520/news-indexer#278) — not synthetic/de-named like
# eval_extractor_adjacent_events.py's cases. Each maps to one primary claim so a
# single (stance, certainty) pair per run is the meaningful unit to compare.
CASES = [
    dict(
        name="lemonde_netanyahu_self_interested",
        # The literal article from the case that started this investigation —
        # unfetchable before retro#520 shipped. Self-interested stated-intent
        # ("je vais me présenter... j'ai l'intention de gagner") mixed with
        # negative/competitor signals (a unified Bennett-Lapid opposition list).
        source_name="Le Monde",
        journalist="unknown",
        article_date="2026-06-15",
        event_name="Benjamin Netanyahu is the Prime Minister of Israel on December 31, 2026",
        event_description=(
            "Benjamin Netanyahu holds the office of Prime Minister of Israel as of "
            "December 31, 2026"
        ),
        claim_deadline="2026-12-31",
        language="French",
        article_text="""Le premier ministre israélien, Benyamin Nétanyahou, a déclaré, lundi 15 juin, qu'il comptait se présenter aux élections législatives prévues d'ici à la fin de l'année, alors qu'il fait face à des critiques concernant sa gestion de la guerre au Moyen-Orient et ses conséquences.
« Je vais me présenter aux élections et j'ai l'intention de gagner », a déclaré le dirigeant lors d'une conférence de presse, sa première prise de parole depuis que Washington et Téhéran ont conclu un accord visant à mettre fin à la guerre dans la région.
Benyamin Nétanyahou sera notamment opposé à l'ancien premier ministre israélien Naftali Bennett (droite) et au chef de l'opposition, Yaïr Lapid (centre), qui se présenteront aux prochaines élections sur une liste commune.
« Cette initiative (…) permet de concentrer tous les efforts pour conduire Israël vers la réparation nécessaire », avait expliqué Yaïr Lapid, ajoutant : « Bennett est un homme de droite, mais de droite honnête, et il y a de la confiance entre nous ». Naftali Bennett a promis que, s'il était élu, il nommerait une commission nationale d'enquête sur les défaillances ayant conduit au massacre du 7 octobre 2023, ce que refuse le gouvernement actuel.
Plus d'informations à venir.""",
    ),
    dict(
        name="cbk_kenya_rate_hold_factual",
        # Plain factual/procedural report of a decision that already happened —
        # low ambiguity, no self-interested statement, no adjacent-event confusion.
        # Expected to be the MOST stable case; a useful floor/control.
        source_name="allAfrica",
        journalist="unknown",
        article_date="2026-08-12",
        event_name="Kenya's Central Bank Rate is at or below 9% at the end of 2026",
        event_description="The Central Bank of Kenya's benchmark policy rate is 9% or lower on December 31, 2026",
        claim_deadline="2026-12-31",
        language=None,
        article_text="""Nairobi — The Central Bank of Kenya (CBK) has kept its benchmark interest rate at 8.75 percent as inflation remains within the target range.
The Monetary Policy Committee (MPC) retained the Central Bank Rate (CBR) at 8.75 percent during its meeting on Tuesday, saying the current rate was appropriate despite rising global economic risks.
Kenya's inflation rose slightly to 6.5 percent in July from 6.4 percent in June, driven mainly by higher food prices.
Prices of potatoes, tomatoes, kales, cabbages and onions remained elevated, keeping pressure on household budgets.""",
    ),
    dict(
        name="rba_cash_rate_hold_factual",
        # Second factual/procedural control, different outlet/region — checks
        # whether the "low ambiguity -> low variance" pattern from case 2 holds
        # generally or was a one-off.
        source_name="The Guardian",
        journalist="unknown",
        article_date="2026-08-11",
        event_name="Australia's Reserve Bank cuts its cash rate before the end of 2026",
        event_description="The Reserve Bank of Australia lowers its official cash rate below its current level at least once before December 31, 2026",
        claim_deadline="2026-12-31",
        language=None,
        article_text="""The Reserve Bank of Australia has held the official cash rate steady, defying expectations from some economists for a cut as it weighs persistent inflation pressures against a softening jobs market.
The RBA board kept the cash rate unchanged at its meeting this week, saying it wanted more evidence that inflation was sustainably within its target band before considering any further easing.
Governor comments suggested the central bank was in no rush to move, with house prices and household spending both proving resilient in recent months despite higher borrowing costs.
Money markets had priced in a small chance of a rate cut this month, but the majority of economists surveyed had expected a hold.""",
    ),
]


def _primary(out) -> tuple:
    """(stance, certainty) of the first extracted prediction, or (None, None) if none."""
    if not out.predictions:
        return None, None
    p = out.predictions[0]
    return p.stance, p.certainty


async def _run_case(model: str, case: dict) -> list[tuple]:
    settings.extractor_model = model
    results = []
    for _ in range(RUNS_PER_CASE):
        try:
            out, _ = await extract_predictions(
                case["article_text"],
                case["source_name"],
                case["article_date"],
                case["event_name"],
                case["event_description"],
                journalist=case["journalist"],
                claim_deadline=case["claim_deadline"],
                language=case["language"],
            )
        except Exception as exc:  # noqa: BLE001 — report, keep going
            print(f"    EXCEPTION: {type(exc).__name__}: {exc}")
            continue
        results.append(_primary(out))
    return results


def _spread(values: list[float]) -> tuple[float, float]:
    return (max(values) - min(values), pstdev(values)) if len(values) >= 2 else (0.0, 0.0)


def _sign(value: float) -> int:
    """-1 / 0 / +1. Exact 0.0 is its own category, not folded into either side."""
    return (value > 0) - (value < 0)


def _sign_flip_rate(values: list[float]) -> float:
    """Fraction of runs whose stance sign differs from the modal sign, on identical input.

    This is the statistic retro#545 is actually about, and it is NOT what stdev measures.
    Magnitude moving is survivable — the pool is a weighted mean, so a stance wobbling
    between +0.4 and +0.8 still votes the same way. A sign crossing zero inverts the vote:
    `probability = (stance + 1) / 2` puts +0.4 at 70% and -0.4 at 30%, so two runs over the
    same article can put a source on opposite sides of the question.

    0.0 = every run agreed on direction. 0.5 = an even split, the worst reachable value for
    a two-sign case. Ties resolve to whichever sign Counter sees first, which only affects
    the reported rate at an exact 50/50 — already a maximal-instability result either way.
    """
    if len(values) < 2:
        return 0.0
    signs = [_sign(v) for v in values]
    modal_count = Counter(signs).most_common(1)[0][1]
    return (len(signs) - modal_count) / len(signs)


def _sign_histogram(values: list[float]) -> str:
    """`+5/-3/0x0` — the raw split behind the rate, so a reader can see WHICH way it split."""
    c = Counter(_sign(v) for v in values)
    return f"+{c.get(1, 0)}/-{c.get(-1, 0)}/0x{c.get(0, 0)}"


async def main() -> None:
    summary = defaultdict(dict)
    for model in MODELS:
        print(f"\n=== {model} ===")
        for case in CASES:
            results = await _run_case(model, case)
            stances = [s for s, c in results if s is not None]
            certainties = [c for s, c in results if c is not None]
            n_predictions = len(results)
            no_prediction = RUNS_PER_CASE - n_predictions
            stance_range, stance_stdev = _spread(stances) if stances else (0.0, 0.0)
            cert_range, cert_stdev = _spread(certainties) if certainties else (0.0, 0.0)
            sign_flip_rate = _sign_flip_rate(stances) if stances else 0.0
            summary[case["name"]][model] = {
                "stance_range": stance_range, "stance_stdev": stance_stdev,
                "cert_range": cert_range, "cert_stdev": cert_stdev,
                "no_prediction": no_prediction,
                "sign_flip_rate": sign_flip_rate,
                "sign_histogram": _sign_histogram(stances) if stances else "-",
            }
            print(
                f"  {case['name']}: n={n_predictions}/{RUNS_PER_CASE} "
                f"stance range={stance_range:.2f} stdev={stance_stdev:.3f} | "
                f"certainty range={cert_range:.2f} stdev={cert_stdev:.3f} | "
                f"sign {_sign_histogram(stances) if stances else '-'} "
                f"flip_rate={sign_flip_rate:.3f}"
                + (f" | {no_prediction} run(s) extracted NO prediction" if no_prediction else "")
            )

    print("\n=== Summary (stance SIGN-FLIP RATE by case x model — 0.000 = direction unanimous) ===")
    print("  The retro#545 statistic: how often the same article votes the opposite way.")
    header = "  case".ljust(38) + "".join(m.split("/")[-1][:24].ljust(28) for m in MODELS)
    print(header)
    for case_name, by_model in summary.items():
        row = f"  {case_name}".ljust(38)
        for model in MODELS:
            stats = by_model.get(model, {})
            rate = stats.get("sign_flip_rate", float("nan"))
            row += f"{rate:.3f}  ({stats.get('sign_histogram', '-')})".ljust(28)
        print(row)

    print("\n=== Summary (stance stdev by case x model — lower is more stable) ===")
    header = "  case".ljust(38) + "".join(m.split("/")[-1][:24].ljust(28) for m in MODELS)
    print(header)
    for case_name, by_model in summary.items():
        row = f"  {case_name}".ljust(38)
        for model in MODELS:
            stats = by_model.get(model, {})
            row += f"{stats.get('stance_stdev', float('nan')):.3f}".ljust(28)
        print(row)


if __name__ == "__main__":
    asyncio.run(main())
