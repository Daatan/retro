"""
Two aggregation functions:

1. aggregate_predictions() — cell-level: collapses all predictions across all articles
   for one (event, source) cell into a single CellSignal.

2. aggregate_article_predictions() — article-level: collapses N predictions extracted
   from a single article into one unified PredictionExtraction using an LLM.
   Call needs_aggregation() first to check whether the LLM step is warranted.
"""

import asyncio
import json
from statistics import median
from collections import Counter
from typing import Optional

from .models import PredictionExtraction, CellSignal
from .config import settings
from .llm import client as _client, RATE_LIMIT_BACKOFF, apply_routing, is_rate_limit_error

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

    last_exc: Exception = RuntimeError("no attempts")
    for _attempt, wait in enumerate([0] + RATE_LIMIT_BACKOFF):
        if wait:
            await asyncio.sleep(wait)
        try:
            kwargs = apply_routing(dict(
                model=settings.extractor_model,
                response_model=PredictionExtraction,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                timeout=120,
                max_retries=1,
            ))
            return await _client.chat.completions.create(**kwargs)
        except Exception as e:
            if is_rate_limit_error(e):
                last_exc = e
                continue
            raise
    raise last_exc


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

    weights = [p.certainty * (p.specificity if p.specificity is not None else 1.0)
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
        certainty=wmean("certainty"),
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
