from .models import GatekeeperOutput
from .config import settings
from .llm import complete_structured

PROMPT = """\
You are a relevance screener for a forecasting system. Below is a CLAIM (a specific
predicted outcome) and an article. Judge the article by one question only:

  **Does this article change how likely the CLAIM's specific outcome is?**

Judge *evidence-relevance*, not keyword overlap. Two traps to avoid:
- An article can be strongly relevant WITHOUT mentioning the outcome. A report on a
  leader's failing health is strong evidence for "will they die this year?" even if it
  never says "death." Indirect evidence still counts — do not require the outcome to be
  named.
- An article can be about the claim's main actor/topic yet be IRRELEVANT. A story about
  Elon Musk's views on bitcoin tells you nothing about whether he will tweet about some
  specific company. Being *about the actor* is NOT enough — it must bear on THIS outcome.

**1. Coarse gate — set `is_prediction`.**
Set it to false when a forecaster would get nothing to update on for this claim:
- It is about the claim's actor/topic but its substance does not bear on the specific
  outcome (claim "X will tweet about Y" + article about X's unrelated business dealings).
- It is wholly about a different event or domain, or only brushes the claim's keywords in
  passing.
- It is empty, a paywall/404 stub, or has no substantive content (under ~200 meaningful
  words).
Otherwise set it to true. The bar is *bears on the outcome* — borderline-but-relevant
articles pass and are graded below. Do NOT reject for being "only factual reporting",
lacking explicit "X will happen" language, for being short, or for being INDIRECT evidence.

**2. Graded relevance — set `relevance_score` in [0.0, 1.0].**
How much would a forecaster update their estimate of THIS outcome after reading it?
- **0.8–1.0** — direct or strong evidence: reports the outcome, its drivers, or
  developments (even indirect) that clearly move its likelihood.
- **0.4–0.6** — weak/partial bearing: touches the situation but only loosely moves the
  specific outcome.
- **0.0–0.2** — no bearing on the outcome: about the actor/topic but a different matter,
  or merely shares a name/keyword.
Reserve the top band for genuine evidence about the outcome. When in doubt between two
bands, choose the lower one.

Article:
<article>
{article_text}
</article>

Source: {source_name}
Date: {article_date}
Claim: {event_name}

Set `reason` with a one-sentence justification covering both the gate and the score. Set
`prediction_count_estimate` to how many distinct predictive signals (explicit or implicit)
a careful reader could extract; use 0 for purely factual articles that still bear on the
outcome (but pass them).
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
