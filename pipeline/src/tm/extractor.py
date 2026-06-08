from .models import ExtractionOutput
from .config import settings
from .llm import complete_structured

PROMPT = """\
You are a forensic prediction analyst. Your job is to extract EVERY signal — \
explicit or implicit — that bears on whether the RELATED EVENT will occur.

## What counts as a signal

Extract ALL of the following:
- Explicit forecasts: "X will happen", "Y is expected to..."
- Implied directional views: "the economy is heading toward...", "pressure is mounting"
- Factual reports whose content logically implies an outcome — e.g. reporting that \
  troops are advancing implies a battle outcome; reporting that negotiations collapsed \
  implies a deal is less likely. INFER the implication.
- Quotes from officials, analysts, or experts that imply a position
- Vague sentiment that colors likelihood: "things are deteriorating", "a breakthrough \
  looks distant"
- Even near-zero certainty signals (certainty=0.1) are valuable — include them

## What does NOT count
- Pure background with zero bearing on the event (e.g. article only covers geography)
- Statements about a wholly different event with no link to the related event

## STANCE — the most important field
Stance measures how strongly this signal implies the RELATED EVENT will occur.
  +1.0 = certain the event WILL happen
  -1.0 = certain the event WILL NOT happen
   0.0 = neutral / no directional signal

Ask yourself: "If this quote/fact is true, does it make the related event more likely \
(positive stance) or less likely (negative stance)?"

Examples — related event: "Assad regime falls in Syria":
  "Rebel forces are closing in on Hama"        → stance +0.7, certainty 0.6
  "Assad's army is holding the line"           → stance −0.6, certainty 0.5
  "The conflict has dragged on for two years"  → stance +0.2, certainty 0.2
  "International sanctions remain in place"   → stance +0.3, certainty 0.3

Note: even factual/contextual sentences have a stance if they imply a direction.
Do NOT use stance to indicate good/bad — only more/less likely to happen.

## Article language
The article may be in Hebrew, Arabic, or English. Always write the claim in English.
Quote the original language verbatim in the quote field.

## Output
Extract up to 5 signals. Prefer higher-certainty ones but do not omit low-certainty \
signals if they are the only content available.

Article:
<article>
{article_text}
</article>

Source: {source_name}
Journalist: {journalist}
Date: {article_date}
Related event: {event_name} — {event_description}

IMPORTANT: Your response must be a JSON object with a "predictions" key containing a list.
Example: {{"predictions": [ {{...}}, {{...}} ]}}

Each prediction has exactly four fields:
  quote (string — original language), claim (string — English), \
stance (float −1 to 1), certainty (float 0 to 1)

Example — related event: "Assad regime falls in Syria":
{{
  "predictions": [
    {{
      "quote": "Syrian rebel forces pushed close on Tuesday to the major city of Hama",
      "claim": "Rebel advances toward Hama make Assad's fall increasingly likely",
      "stance": 0.7,
      "certainty": 0.6
    }}
  ]
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
    prompt = PROMPT.format(
        article_text=article_text,
        source_name=source_name,
        journalist=journalist,
        article_date=article_date,
        event_name=event_name,
        event_description=event_description,
    )
    return await complete_structured(
        settings.extractor_model, ExtractionOutput, prompt, max_tokens=1200, timeout=180,
    )
