from .models import GatekeeperOutput
from .config import settings
from .llm import complete_structured

PROMPT = """\
You are a topic-relevance screener for a forecasting system.

Decide whether this article contains useful evidence about the RELATED EVENT below.
A downstream extractor will pull out directional signals and assign its own stance and
certainty — even a weak or ambiguous signal is valuable. ERR ON THE SIDE OF INCLUDING
the article. Worst case, the extractor will assign near-zero certainty and it won't
affect the forecast.

**PASS (is_prediction=true) if ANY of these apply:**
- The article discusses the related event directly (including reporting current actions,
  announcements, statements, or developments).
- The article covers the specific actors, institutions, places, or situation underlying
  the event (even in a factual / analytical style).
- The article reports causes, consequences, or recent developments a reasonable reader
  would consider relevant evidence for whether the event will occur.
- The article contains explicit or implicit forecasts, analysis, or commentary about
  the event's likelihood or drivers.
- The article is partisan, opinion, or speculative but on-topic.

**REJECT (is_prediction=false) ONLY if:**
- The article is wholly about a clearly different event or domain (e.g. celebrity
  gossip when the event is about monetary policy).
- The article is empty, a paywall/404 stub, or has no substantive content (under ~200
  meaningful words).
- The article only brushes past the event's keywords in passing while the substance is
  about something wholly unrelated.

**Do NOT reject for:**
- Being "only factual reporting" — recent facts ARE evidence for forecasts.
- Lacking explicit "X will happen" language — implicit signal is extracted downstream.
- Being short but on-topic — the extractor handles low-specificity input.
- Covering an adjacent aspect of the same underlying situation.

Article snippet:
<article>
{article_text}
</article>

Source: {source_name}
Date: {article_date}
Related event: {event_name}

Set `is_prediction` to true for on-topic articles even when they contain no explicit
forecast. Set `reason` with a one-sentence justification. Set `prediction_count_estimate`
to how many distinct predictive signals (explicit or implicit) a careful reader could
extract; use 0 for purely factual on-topic articles (but still pass them).
"""


async def check_is_prediction(
    article_text: str,
    source_name: str,
    article_date: str,
    event_name: str,
) -> tuple["GatekeeperOutput", dict]:
    """Returns (GatekeeperOutput, usage) where usage has prompt_tokens/completion_tokens/total_tokens."""
    prompt = PROMPT.format(
        article_text=article_text[200:2700],
        source_name=source_name,
        article_date=article_date,
        event_name=event_name,
    )
    return await complete_structured(
        settings.gatekeeper_model, GatekeeperOutput, prompt, max_tokens=200, timeout=90,
    )
