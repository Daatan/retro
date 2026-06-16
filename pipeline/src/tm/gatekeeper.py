from .models import GatekeeperOutput
from .config import settings
from .llm import complete_structured

PROMPT = """\
You are a topic-relevance screener for a forecasting system.

You do two things for the RELATED EVENT below: a coarse keep/drop gate, and a graded
relevance score that decides how much this article counts. Off-topic articles are NOT
harmless — a confident off-topic article still pulls the forecast — so score honestly.

**1. Coarse gate — set `is_prediction`.**
Set it to false ONLY for clearly unusable input:
- The article is wholly about a different event or domain (e.g. celebrity gossip when the
  event is about monetary policy), or only brushes the event's keywords in passing while
  its substance is about something unrelated.
- The article is empty, a paywall/404 stub, or has no substantive content (under ~200
  meaningful words).
Otherwise set it to true. Keep this bar LOW — borderline articles pass the gate and are
disciplined by the relevance score below. Do NOT reject for being "only factual
reporting", lacking explicit "X will happen" language, or being short but on-topic.

**2. Graded relevance — set `relevance_score` in [0.0, 1.0].**
How directly does this article bear on whether the specific event happens? Anchored bands:
- **0.8–1.0** — directly about the event: reports its actors/situation, its likelihood,
  or developments that move it.
- **0.4–0.6** — tangential or an adjacent aspect of the same broad situation; related
  domain but not this specific event.
- **0.0–0.2** — unrelated, or merely shares a keyword/name while the substance is about
  something else.
Be discriminating: reserve the top band for articles a forecaster would treat as direct
evidence. When in doubt between two bands, choose the lower one.

Article:
<article>
{article_text}
</article>

Source: {source_name}
Date: {article_date}
Related event: {event_name}

Set `reason` with a one-sentence justification covering both the gate and the score. Set
`prediction_count_estimate` to how many distinct predictive signals (explicit or implicit)
a careful reader could extract; use 0 for purely factual on-topic articles (but still pass
them if relevant).
"""


async def check_is_prediction(
    article_text: str,
    source_name: str,
    article_date: str,
    event_name: str,
) -> tuple["GatekeeperOutput", dict]:
    """Returns (GatekeeperOutput, usage) where usage has prompt_tokens/completion_tokens/total_tokens."""
    prompt = PROMPT.format(
        article_text=article_text,
        source_name=source_name,
        article_date=article_date,
        event_name=event_name,
    )
    return await complete_structured(
        settings.gatekeeper_model, GatekeeperOutput, prompt, max_tokens=200, timeout=90,
    )
