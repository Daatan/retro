import logging
import re

from .models import GatekeeperOutput
from .config import settings
from .llm import complete_structured

logger = logging.getLogger(__name__)

# Split into a fixed, cacheable PROMPT_PREFIX (identical on every call, no .format()
# placeholders) and a PROMPT_SUFFIX carrying the article + per-call variable fields —
# same rationale as extractor.py's split. Byte-identical prompt content overall.
PROMPT_PREFIX = """\
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
DECISION RULE: if a thoughtful forecaster's estimate of THIS outcome could move even
slightly after reading the article, set is_prediction=true. This INCLUDES indirect
drivers that never name the outcome — a rival's collapse, the actor's party / coalition /
primary dynamics, shifts in the actor's standing or support, the process leading to the
outcome (talks, injuries, setbacks, polls). When in doubt, PASS — the graded
relevance_score below handles weak or loose signal.

Set `is_prediction` to false ONLY when:
- The article is wholly about a DIFFERENT matter, event, or domain — e.g. a claim about an
  election and an article about a sports result, a stock-market move, or an unrelated
  topic — or it merely shares a name/keyword with the claim.
- It is empty, a paywall/404 stub, or has no substantive content (under ~200 meaningful words).
Do NOT reject for being "only factual reporting", for lacking explicit "X will happen"
language, for being short, or for being INDIRECT evidence.

**2. Graded relevance — set `relevance_score` in [0.0, 1.0].**
How much would a forecaster update their estimate of THIS outcome after reading it?
- **0.7–1.0** — genuine evidence: reports the outcome, its drivers, or developments that
  advance or hinder it — INCLUDING the process leading to it (accession talks for "will
  the EU admit members?"; territorial gains/losses or military setbacks for "will they
  lose territory?"; a leader's failing health for "will they die?"). If a reasonable
  reader would treat it as evidence either way, it belongs here — be generous, not stingy
  (sole exception: the process-only cap below).
- **0.3–0.6** — partial/background bearing: about the situation and somewhat informative,
  but only loosely tied to the specific outcome — including articles that confirm the
  underlying process or situation is on track (schedules, dates, procedures) without
  distinguishing between its possible outcomes.
- **0.0–0.2** — no bearing: about the actor/topic but a DIFFERENT matter (Musk's bitcoin
  views for "will Musk tweet about Daatan?"), or merely shares a name/keyword.
Most articles a search returns for a well-formed claim are genuinely relevant — the
0.0–0.2 band is for the actor-but-different-matter case, not for solid evidence that
merely isn't laser-focused on the outcome.

**Process-only cap.** When the claim names ONE specific outcome of a multi-outcome
process (one winner among contenders, one option among alternatives) and the article
only confirms the process itself — its schedule, timetable, procedure, or rules, which
hold equally under every possible outcome — it is on-topic background, not evidence of
WHICH outcome will occur. Such articles still PASS the gate (is_prediction=true), but
their relevance_score is capped: score them 0.3–0.5, never 0.7 or higher. Reserve
0.7–1.0 for reporting that favors or disfavors a specific outcome.
"""

PROMPT_SUFFIX = """\
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


# Appended ONLY for short_form callers. The base PROMPT above rejects anything "under ~200
# meaningful words" — a rule aimed at paywall stubs and 404 pages, which quietly misfires on an
# entire class of source we ingest ON PURPOSE.
#
# Telegram posts arrive as text over MTProto, and news-indexer's `build_pushed_article` explicitly
# bypasses its own length minimum because (its words) "short breaking-news messages are exactly the
# kind of content this path exists for". Measured on the live index: 460 of 552 indexed Telegram
# posts run under 70 words — including 154 of Ben Caspit's 169. So the pipeline deliberately
# collects terse posts from serious journalists, and the gatekeeper then discards every one of them
# for being terse.
#
# This does not weaken the stub protection for real articles: it is opt-in, off by default, and the
# emptiness and different-matter rejections both stay in force.
_SHORT_FORM_OVERRIDE = """

**Short-form source.** This item is a social-media / messaging post (e.g. a journalist's Telegram
channel), NOT a news article. The length rule above does not apply to it: do NOT set
is_prediction=false merely because it runs under ~200 words. Terse posts from serious journalists
are a deliberate part of this corpus and routinely carry real evidence — "elections are now final
for Oct 27" is one line and decisive. Judge it on what it SAYS about the claim, exactly as you
would judge a long article, and score its relevance on the same bands.

Still reject it if it is genuinely empty (an image or video with no caption), or if it is about a
different matter. Brevity alone is not a reason.

**No content, no judgment.** If the post carries no assertable proposition of its own — it is only
a link, a bare @handle, a hashtag, an emoji, or a headline-free teaser pointing elsewhere — set
is_prediction=false and relevance_score=0.0. A URL is not content: do NOT infer what the linked
page says from its domain, its slug, or its numeric id, and do NOT treat the act of a journalist
sharing a link as evidence about the claim. Judge only what the post itself asserts. If that is
nothing, say so — "the post links out without stating anything" is the correct reason, and an
invented summary of the unseen page is the failure this rule exists to prevent.
"""


# Appended when the caller knows the article's language (retro#417). The gatekeeper and
# extractor prompts are written in English, but most of this corpus is not — Hebrew, Russian
# and Arabic posts were judged with no signal that the text is non-English. Append-only, so
# callers that don't send a language keep a byte-for-byte identical prompt.
_LANGUAGE_HINT = """

**Language.** The article text is in {language}. Judge it in that language, exactly as you \
would an English article — being non-English is never, by itself, a reason to reject or \
down-score it.
"""


# Everything that is a POINTER rather than a proposition. Stripped before asking "is there any
# content here at all?" — deliberately only the forms actually observed in the corpus (scheme-ful
# URLs, t.me paths, @handles, #hashtags). Bare domains ("ynet.co.il") are NOT stripped: the pattern
# that catches them also eats ordinary abbreviations, and over-stripping fails in the dangerous
# direction here — a wrongly-rejected post is a curated journalist's scoop lost silently, which is
# the exact failure news-indexer's rescue path exists to undo.
_POINTER_RE = re.compile(r"https?://\S+|www\.\S+|t\.me/\S+|[@#]\w+", re.IGNORECASE)
# A run of two or more Unicode letters. `[^\W\d_]` is "word character that is neither digit nor
# underscore", i.e. a letter in ANY script — Hebrew, Cyrillic, Arabic and CJK all count, which
# matters because most of this corpus is not English. Emoji and digits are not letters, so "🔴"
# and "97%" carry no proposition by this test, correctly.
_LETTER_RUN_RE = re.compile(r"[^\W\d_]{2,}")


def carries_proposition(article_text: str) -> bool:
    """Does this text assert anything of its own, or is it purely a pointer elsewhere?

    The gatekeeper is a language model, and a model handed a bare URL does not answer "there is
    nothing here" — it CONFABULATES. Measured in prod (2026-07-31): a t.me post whose entire text
    was `https://www.c14.co.il/article/1641278` (37 chars) was judged against 127 open forecasts and
    endorsed for 76 of them at relevance 0.7–1.0, with invented justifications — "provides a direct
    statement about Elon Musk tweeting about Daatan by December 31, 2028" for an article that is one
    URL. Five such articles have cost 644 judgments and 247 downstream Oracle runs.

    So the prompt rule above teaches it, and this enforces it: prompt rules are advisory, and this
    verdict gets PERSISTED by news-indexer and can be reused downstream in place of a fresh judgment
    (`reuse_supplied_relevance`), so a confabulated 1.0 does not stay contained. Deterministic and
    free — it also skips the LLM call entirely, which is the cheapest possible way to be right.

    The bar is deliberately the floor and not a length heuristic: ANY two consecutive letters that
    survive pointer-stripping count as content. "מבזק" (4 chars, "newsflash") passes. Deciding
    whether a short-but-real post is substantive enough is the judge's job, not this function's —
    the only question here is whether there is anything at all for it to judge.
    """
    return bool(_LETTER_RUN_RE.search(_POINTER_RE.sub(" ", article_text)))


# What the gate returns instead of asking the model. relevance 0.0 (not the model's 1.0 default) so
# that a caller squaring it for aggregation weight lands on zero either way.
_NO_CONTENT_VERDICT = GatekeeperOutput(
    is_prediction=False,
    reason=(
        "Content-free input: the text carries no assertable proposition of its own (only "
        "links/handles/emoji), so there is nothing to judge against the claim."
    ),
    prediction_count_estimate=0,
    relevance_score=0.0,
)
_NO_USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


async def check_is_prediction(
    article_text: str,
    source_name: str,
    article_date: str,
    event_name: str,
    short_form: bool = False,
    language: str | None = None,
) -> tuple["GatekeeperOutput", dict]:
    """Returns (GatekeeperOutput, usage) where usage has prompt_tokens/completion_tokens/total_tokens.

    `short_form` opts into judging a social-media post on its content rather than its length. It
    defaults to False and only ever APPENDS to the prompt, so the text every existing caller sends
    is byte-for-byte unchanged. Since retro#417 the /forecast path sets it for t.me-host articles,
    matching the /relevance rescue path.

    `language` (retro#417) is an optional caller-supplied hint ("Hebrew", "ru", …) appended to the
    prompt so the model is told the text is non-English instead of having to notice. Also
    append-only; None keeps the prompt unchanged.

    Input that carries no proposition is rejected here without an LLM call — see
    `carries_proposition`. That guard sits in front of BOTH callers on purpose: /relevance is the
    path that produced the measured incident, but /forecast judges articles from search providers
    with the same model and has the same failure mode.
    """
    if not carries_proposition(article_text):
        logger.info(
            "gatekeeper: content-free input rejected without an LLM call "
            "(source=%.40s len=%d) — %.80s",
            source_name, len(article_text), article_text.replace("\n", " "),
        )
        return _NO_CONTENT_VERDICT.model_copy(), _NO_USAGE.copy()

    prompt = PROMPT_SUFFIX.format(
        article_text=article_text,
        source_name=source_name,
        article_date=article_date,
        event_name=event_name,
    )
    if short_form:
        prompt += _SHORT_FORM_OVERRIDE
    if language:
        prompt += _LANGUAGE_HINT.format(language=language)
    return await complete_structured(
        settings.gatekeeper_model, GatekeeperOutput, prompt, max_tokens=200, timeout=90,
        cached_prefix=PROMPT_PREFIX,
    )
