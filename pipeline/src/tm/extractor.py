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
- Statements about the SAME subject but a DIFFERENT timeframe, edition, or deadline \
than the related event — e.g. next season's title odds when the event is this season's \
final, a later election, a different year's target. Check any year/date in the claim \
against the related event's timeframe; if they refer to a different occurrence of the \
event, do NOT extract it.

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
  "Rebels have taken Damascus; Assad has fled the country" \
                                               → stance +1.0, certainty 0.95, settled true
  "Assad crushed the uprising; the rebellion is over" \
                                               → stance −1.0, certainty 0.95, settled true

Note: even factual/contextual sentences have a stance if they imply a direction.
Do NOT use stance to indicate good/bad — only more/less likely to happen.

## Numeric thresholds — compare the numbers, not the sentiment
When the related event states a quantitative threshold ("more than 33 seats", \
"below $50,000", "at least 10 medals", "reaches 2800 rating"), judge each signal \
by COMPARING its number against the threshold — never by momentum or by how \
positive the news sounds for the subject. A reported or projected value on the \
wrong side of the threshold is NEGATIVE stance even when it is good news for \
the subject; general success without a number that clears the bar is at most \
weakly positive.

Examples — related event: "Likud wins more than 33 seats in the election":
  "Poll projects Likud at 31 seats"        → stance −0.6, certainty 0.7  (31 ≤ 33: contradicts)
  "Likud is leading in the polls"          → stance +0.2, certainty 0.3  (leading ≠ >33 seats)
  "Poll gives Likud 36 seats"              → stance +0.7, certainty 0.7  (36 > 33: supports)
  "Likud gained two seats since last poll" → stance +0.2, certainty 0.3  (trend, no level given)

## Multi-stage / bracket events — discount single-stage "favorite" framing
When the related event requires winning a SEQUENCE of separate future contests \
(a tournament bracket, a playoff series, a multi-round election, a series of \
confirmation votes) rather than one determination, an article's "favorite," \
"front-runner," or "strong candidate" framing about ONE upcoming stage is weak \
support for the event as a whole — it says nothing about the stages still to \
come. Advancing past one stage narrows the field but does not itself imply the \
final outcome; only raise stance as stages actually clear, and reserve high \
certainty for articles that address the full remaining path, not just the next \
match or round.

Examples — related event: "France wins the 2026 World Cup" (tournament bracket):
  "France is a strong favorite entering the Round of 16"     → stance +0.3, certainty 0.3  (one stage of several remaining)
  "France beats Paraguay to reach the quarter-finals"        → stance +0.4, certainty 0.5  (one stage cleared, more remain)
  "France reaches the final after a dominant semi-final win" → stance +0.6, certainty 0.6  (one stage left)

Examples — related event: "Judge Alvarez is confirmed to the Supreme Court" (committee vote, then floor vote):
  "Alvarez is seen as the clear favorite to be confirmed"       → stance +0.3, certainty 0.3  (favorite framing, no vote yet)
  "The Judiciary Committee advances Alvarez's nomination 12-10" → stance +0.4, certainty 0.5  (one stage cleared, floor vote remains)

Examples — related event: "Diaz wins the presidential runoff" (first round, then runoff):
  "Diaz leads first-round polling by 8 points"        → stance +0.2, certainty 0.3  (first round ≠ runoff win)
  "Diaz advances to the runoff after finishing first" → stance +0.4, certainty 0.5  (one stage cleared, runoff remains)

## SETTLED — the event already happened (or definitively cannot)
When the article REPORTS THE OUTCOME AS AN ACCOMPLISHED FACT — the event occurred, \
or became permanently impossible (deadline passed, subject died, contest decided) — \
set settled to true and use the full ±1.0 stance with certainty ≥ 0.9. Past-tense \
reporting of the outcome ("X won", "the deal was signed", "Y has died") is settled; \
predictions, odds, and expectations ("X is likely to win") are NOT settled, however \
confident. Do not soften a settled outcome into a likelihood — a report that the \
event happened is stance +1.0, not +0.7.

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

Each prediction has exactly five fields:
  quote (string — original language), claim (string — English), \
stance (float −1 to 1), certainty (float 0 to 1), settled (boolean — true only when \
the source reports the outcome as an accomplished fact)

Example — related event: "Assad regime falls in Syria":
{{
  "predictions": [
    {{
      "quote": "Syrian rebel forces pushed close on Tuesday to the major city of Hama",
      "claim": "Rebel advances toward Hama make Assad's fall increasingly likely",
      "stance": 0.7,
      "certainty": 0.6,
      "settled": false
    }},
    {{
      "quote": "Rebels seized the capital on Sunday as Assad fled to Moscow",
      "claim": "The Assad regime has fallen; rebels control Damascus",
      "stance": 1.0,
      "certainty": 0.95,
      "settled": true
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
