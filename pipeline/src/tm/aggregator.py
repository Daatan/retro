"""
Two aggregation functions:

1. aggregate_predictions() — cell-level: collapses all predictions across all articles
   for one (event, source) cell into a single CellSignal.

2. aggregate_article_predictions() — article-level: collapses N predictions extracted
   from a single article into one unified PredictionExtraction using an LLM.
   Call needs_aggregation() first to check whether the LLM step is warranted.
"""

import json
from statistics import median
from collections import Counter
from typing import Optional

from .models import PredictionExtraction, CellSignal, ReaderConfidence
from .config import settings
from .llm import complete_structured

STANCE_SPREAD_THRESHOLD = 0.4

AGGREGATOR_PROMPT = """\
You are a forensic intelligence analyst. You have been given a list of predictions \
extracted from a **single news article** about a specific event. Different parts of the \
article may quote different people or express different views, but the article as a whole \
has an editorial angle and a dominant stance.

Your job is to synthesize all these predictions into **one unified article signal** that \
best represents:
- The article's **dominant directional outlook** on whether the event will occur / succeed
- The **most representative quote** that captures the article's overall stance
- A **single neutral claim** summarising what the article as a whole predicts

### Rules
- Do NOT simply average the numbers — use editorial judgment
- If the article quotes one strong voice and several weaker counterpoints, weight toward the dominant voice
- The `quote` must be an actual excerpt from one of the input predictions (do not fabricate)
- The `claim` must be in English regardless of source language
- Output a **single JSON object** (not a list)

### Input

**Event:** {event_name}
**Source:** {source_name}
**Article date:** {article_date}

**Extracted predictions (N={n_predictions}):**
{predictions_json}

### Output

Return ONLY a raw JSON object — no code block, no explanation:

{{
  "quote": "the single most representative quote from the article",
  "claim": "one-sentence English summary of the article's overall prediction",
  "stance": <float -1.0 to 1.0>,
  "sentiment": <float 0.0 to 1.0>,
  "certainty": <float 0.0 to 1.0>,
  "specificity": <float 0.0 to 1.0>,
  "hedge_ratio": <float 0.0 to 1.0>,
  "conditionality": <float 0.0 to 1.0>,
  "magnitude": <float 0.0 to 1.0>,
  "time_horizon": "<days|weeks|months|years|unspecified>",
  "time_horizon_days": <integer or null>,
  "prediction_type": "<binary|continuous|range|trend>",
  "source_authority": <float 0.0 to 1.0>
}}
"""


def needs_aggregation(predictions: list[PredictionExtraction]) -> bool:
    """Return True if article-level LLM aggregation is warranted."""
    if len(predictions) <= 1:
        return False
    stances = [p.stance for p in predictions]
    return (max(stances) - min(stances)) > STANCE_SPREAD_THRESHOLD


async def aggregate_article_predictions(
    predictions: list[PredictionExtraction],
    event_name: str,
    source_name: str,
    article_date: str,
) -> PredictionExtraction:
    """Call Nova Lite to collapse N predictions from one article into one."""
    predictions_json = json.dumps(
        [p.model_dump() for p in predictions], indent=2, ensure_ascii=False
    )
    prompt = AGGREGATOR_PROMPT.format(
        event_name=event_name,
        source_name=source_name,
        article_date=article_date,
        n_predictions=len(predictions),
        predictions_json=predictions_json,
    )

    output, _usage = await complete_structured(
        settings.extractor_model, PredictionExtraction, prompt, max_tokens=1000, timeout=120,
    )
    # Direct assignment, deliberately not `model_copy(update=...)`: that API is
    # UNVALIDATED in Pydantic v2, so a mistyped key becomes a stray attribute and
    # the value silently never lands (retro#680 shipped that bug once already).
    # Assignment raises on an unknown field name.
    output.reader_confidence = _worst_reader_confidence(predictions)
    _carry_uncollapsible_fields(output, predictions)
    return output


def _worst_reader_confidence(
    predictions: list[PredictionExtraction],
) -> Optional[ReaderConfidence]:
    """The `reader_confidence` of the input claim its reader was least sure of.

    Carried across the LLM collapse in code rather than elicited again (retro#681).
    AGGREGATOR_PROMPT does not ask for the field, so without this the batch lane
    would silently null it on exactly the articles that tripped aggregation —
    articles whose claims disagree by more than STANCE_SPREAD_THRESHOLD, i.e. the
    ones a reader is most likely to have struggled with. A shadow field that
    vanishes where the interesting cases are is worse than no field.

    Whole object, not just the level: the trap that came with the least-confident
    claim is the one worth keeping, and re-pairing a level from one claim with a
    trap from another would invent a reading nothing produced.
    """
    answered = [p.reader_confidence for p in predictions if p.reader_confidence is not None]
    if not answered:
        return None
    order = {"high": 0, "medium": 1, "low": 2}
    return max(answered, key=lambda rc: order[rc.level])


# Fields the LLM collapse drops, and how each is carried across it (retro#721).
#
# `_worst_reader_confidence` above fixed this for ONE field; measurement showed
# eleven more go the same way. AGGREGATOR_PROMPT ends with an output template
# naming twelve fields while the schema instructor serialises into the call names
# ~35, and the template wins: a probe with all fifteen populated came back with
# twelve nulled, identically on 5/5 runs. So this is not sampling noise to be
# prompted away — it is the retro#700 mechanism (a worked example is read as the
# definitive enumeration of its block) applied to the strongest kind of example
# there is, a complete-looking JSON object.
#
# Fixed in code rather than by widening the template on purpose. Asking Nova Lite
# to copy 35 fields faithfully through a JSON round-trip is unauditable, and the
# prompt route would also make this an extractor-prompt PR — sequential slot, A/B
# gate — for what is really a plumbing bug.

# The fields the collapse legitimately produces — the ones AGGREGATOR_PROMPT's
# output template actually names, so the ones the model is genuinely being asked
# to synthesise across claims. (`certainty` in that template is an accepted
# validation_alias of `claim_strength`, retro#680.)
#
# Everything else on the model is carried from the lead claim. The list is written
# this way round on purpose: a field added to PredictionExtraction later is carried
# by DEFAULT rather than silently dropped, so this bug cannot recur by omission.
# test_every_field_is_either_synthesised_or_carried holds the partition total.
_LLM_SYNTHESISED_FIELDS = frozenset({
    "quote",
    "claim",
    "stance",
    "claim_strength",
    "sentiment",
    "specificity",
    "hedge_ratio",
    "conditionality",
    "magnitude",
    "time_horizon",
    "time_horizon_days",
    "prediction_type",
    "source_authority",
})

# Handled outside the lead-claim rule: `reader_confidence` has its own
# worst-of-inputs rule (retro#681), and `speaker` is deliberately dropped.
_SPECIALLY_HANDLED_FIELDS = frozenset({"reader_confidence", "speaker"})


def _carry_uncollapsible_fields(
    output: PredictionExtraction,
    predictions: list[PredictionExtraction],
) -> None:
    """Restore the fields the LLM collapse drops, in place.

    Mutates `output` for the same reason `reader_confidence` is assigned rather
    than `model_copy(update=...)`-ed: that API is unvalidated in Pydantic v2.

    ONE claim's reading travels intact, rather than each field being resolved by
    its own rule. Splitting them would re-pair `fact_signal` from one claim with
    the facets that qualify it (`facet`, `is_occurrence`, `verified`,
    `event_actors`/`event_target`) from another, or `is_conditional` with a
    different claim's `antecedent_text` — inventing a reading no claim produced.
    That is the same trap `_worst_reader_confidence` avoids by carrying the whole
    ReaderConfidence object instead of level and trap separately.

    `speaker` is deliberately NOT carried and stays None. A collapsed article
    signal has no single speaker, and taking the lead claim's would attribute the
    whole article to one quoted person. That is a decision, not an oversight.
    """
    lead = _lead_claim(output, predictions)
    if lead is None:
        return
    for name in type(output).model_fields:
        if name in _LLM_SYNTHESISED_FIELDS or name in _SPECIALLY_HANDLED_FIELDS:
            continue
        setattr(output, name, getattr(lead, name))


def _lead_claim(
    output: PredictionExtraction,
    predictions: list[PredictionExtraction],
) -> Optional[PredictionExtraction]:
    """The input claim the aggregate's stance actually followed, or None.

    Nearest stance to the aggregate's, and only among claims that agree with it
    in SIGN. The sign guard is what keeps `settled` safe: aggregation fires
    precisely on articles whose claims disagree (spread > STANCE_SPREAD_THRESHOLD),
    so a settling claim and the aggregate can easily point opposite ways — an
    article reporting "the deal was signed" alongside "critics expect it to
    collapse". Inheriting a settlement from a claim the aggregate rejected would
    manufacture a settled outcome the article does not carry, and `settled` gates
    an ENFORCING resolution path.

    Returns None when no input shares the aggregate's sign (including when the
    aggregate lands on exactly 0.0), leaving every lead-claim field null. That is
    the same "no single value" answer the unanimity rule gives, and it is correct:
    if nothing in the article points the way the aggregate does, no claim's
    evidence describes it. `enforce_settlement_event_date` in the extractor stays
    the second guard on any positive settlement that does get through here.
    """
    sign = (output.stance > 0) - (output.stance < 0)
    if sign == 0:
        return None
    same_sign = [
        p for p in predictions if ((p.stance > 0) - (p.stance < 0)) == sign
    ]
    if not same_sign:
        return None
    return min(same_sign, key=lambda p: abs(p.stance - output.stance))


def _weighted_mean(values: list[float], weights: list[float]) -> float:
    total = sum(weights)
    if total == 0:
        return sum(values) / len(values)
    return sum(v * w for v, w in zip(values, weights)) / total


def _majority(values: list[str]) -> str:
    return Counter(values).most_common(1)[0][0]


def _weighted_median(values: list[Optional[int]], weights: list[float]) -> Optional[int]:
    pairs = [(v, w) for v, w in zip(values, weights) if v is not None]
    if not pairs:
        return None
    pairs.sort(key=lambda x: x[0])
    total = sum(w for _, w in pairs)
    if total == 0:
        return int(median(v for v, _ in pairs))
    cumulative = 0.0
    for v, w in pairs:
        cumulative += w
        if cumulative >= total / 2:
            return v
    return pairs[-1][0]


def aggregate_predictions(predictions: list[PredictionExtraction]) -> CellSignal:
    if not predictions:
        raise ValueError("Cannot aggregate empty prediction list")

    weights = [p.claim_strength * (p.specificity if p.specificity is not None else 1.0)
               for p in predictions]

    def wmean(attr: str) -> float:
        return _weighted_mean([getattr(p, attr) for p in predictions], weights)

    def optional_wmean(attr: str) -> Optional[float]:
        pairs = [(getattr(p, attr), w) for p, w in zip(predictions, weights)
                 if getattr(p, attr) is not None]
        if not pairs:
            return None
        vals, wts = zip(*pairs)
        return _weighted_mean(list(vals), list(wts))

    th_vals = [p.time_horizon for p in predictions if p.time_horizon is not None]
    pt_vals = [p.prediction_type for p in predictions if p.prediction_type is not None]

    return CellSignal(
        claim_count=len(predictions),
        stance=wmean("stance"),
        # CellSignal.certainty keeps its name (atlas lane, retro#680 renamed only
        # the elicited PredictionExtraction field), but the value is read off
        # PredictionExtraction by attribute NAME via getattr — so the argument
        # here must be the new name or every aggregation raises AttributeError.
        certainty=wmean("claim_strength"),
        sentiment=optional_wmean("sentiment"),
        specificity=optional_wmean("specificity"),
        hedge_ratio=optional_wmean("hedge_ratio"),
        conditionality=optional_wmean("conditionality"),
        magnitude=optional_wmean("magnitude"),
        source_authority=optional_wmean("source_authority"),
        time_horizon=_majority(th_vals) if th_vals else None,
        time_horizon_days=_weighted_median(
            [p.time_horizon_days for p in predictions], weights
        ),
        prediction_type=_majority(pt_vals) if pt_vals else None,
        quotes=[p.quote for p in predictions],
        claims=[p.claim for p in predictions],
    )
