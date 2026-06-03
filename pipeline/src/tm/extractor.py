import asyncio
from .models import ExtractionOutput
from .config import settings
from .llm import client as _client, RATE_LIMIT_BACKOFF, apply_routing, extract_usage, is_rate_limit_error

PROMPT = """\
You are a forensic prediction analyst. Extract every distinct forward-looking prediction \
from the article below and quantify each one.

Extract ALL predictions — not just explicit forecasts, but also:
- Implied directional views ("the economy is heading toward...")
- Vague sentiment that implies an outcome ("things are deteriorating")
- Predictions hidden between the lines in analysis or sourced commentary

STANCE DEFINITION — this is the most important field:
  stance measures how strongly the prediction implies the RELATED EVENT will occur.
  +1.0 = the author is certain the related event WILL happen
  -1.0 = the author is certain the related event WILL NOT happen
   0.0 = neutral / genuinely uncertain

  The related event is given below. Ask yourself: does this prediction imply the
  event is more likely (+) or less likely (-) to happen?

  Example: related event = "Assad regime falls in Syria"
    "Rebel forces are closing in on Hama" → stance +0.7  (implies Assad will fall)
    "Assad's army is holding the line"    → stance -0.6  (implies Assad will NOT fall)
    "The outcome remains uncertain"       → stance  0.0

  Do NOT use stance to indicate whether the outcome is good or bad.
  Use stance ONLY to indicate whether the prediction points toward the event happening.

For each prediction extract exactly four fields:
- quote: exact sentence(s) from the article (keep original language)
- claim: one-sentence neutral summary in English
- stance: float from -1.0 (event will NOT happen) to +1.0 (event WILL happen)
- certainty: float from 0.0 (very hedged) to 1.0 (absolute confidence)

Article:
<article>
{article_text}
</article>

Source: {source_name}
Journalist: {journalist}
Date: {article_date}
Related event: {event_name} — {event_description}

Return up to 5 predictions. If more exist, take the most specific and highest-stance ones.
The claim field must be in English.

IMPORTANT: Your response must be a JSON object with a "predictions" key containing a list.
Example structure: {{"predictions": [ {{...}}, {{...}} ]}}
Do NOT return a bare JSON array. Always wrap in {{"predictions": [...]}}

CRITICAL: Each prediction must be a JSON object with exactly these four fields:
  quote (string), claim (string), stance (float -1 to 1), certainty (float 0-1)

Example — related event: "Assad regime falls in Syria":
{{
  "quote": "Syrian rebel forces pushed close on Tuesday to the major city of Hama",
  "claim": "Rebel advances toward Hama make Assad's fall increasingly likely",
  "stance": 0.7,
  "certainty": 0.6
}}
"""


async def extract_predictions(
    article_text: str,
    source_name: str,
    article_date: str,
    event_name: str,
    event_description: str,
    journalist: str = "unknown",
) -> tuple["ExtractionOutput", dict]:
    """Returns (ExtractionOutput, usage) where usage has prompt_tokens/completion_tokens/total_tokens."""
    last_exc: Exception = RuntimeError("no attempts")
    for attempt, wait in enumerate([0] + RATE_LIMIT_BACKOFF):
        if wait:
            await asyncio.sleep(wait)
        try:
            kwargs = apply_routing(dict(
                model=settings.extractor_model,
                response_model=ExtractionOutput,
                messages=[
                    {
                        "role": "user",
                        "content": PROMPT.format(
                            article_text=article_text,
                            source_name=source_name,
                            journalist=journalist,
                            article_date=article_date,
                            event_name=event_name,
                            event_description=event_description,
                        ),
                    }
                ],
                max_tokens=1200,
                timeout=180,
                max_retries=1,
            ))
            output, completion = await _client.chat.completions.create_with_completion(**kwargs)
            return output, extract_usage(completion)
        except Exception as e:
            if is_rate_limit_error(e):
                last_exc = e
                continue
            raise
    raise last_exc
