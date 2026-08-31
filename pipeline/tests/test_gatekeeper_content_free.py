"""Content-free input must fail closed — the gatekeeper may not judge what isn't there.

Handed a bare URL, a language model does not answer "there is nothing here"; it confabulates. In
prod on 2026-07-31 a t.me post whose entire text was `https://www.c14.co.il/article/1641278` was
judged against 127 open forecasts and endorsed for 76 of them at relevance 0.7-1.0, with invented
justifications ("provides a direct statement about Elon Musk tweeting about Daatan by December 31,
2028"). Five articles of that shape have cost 644 judgments and 247 downstream Oracul runs.

Two layers, per the system-model principle that the prompt teaches and the code enforces. These
tests pin the code layer, plus the two things that make it safe: it must not reject terse-but-real
posts (the corpus is mostly non-English, and over-rejection loses a curated journalist's scoop
silently), and it must not cost an LLM call when it fires.

Regression test for Daatan/retro#359.
"""

from unittest.mock import AsyncMock, patch

import pytest

from tm import gatekeeper

# Verbatim from the incident and its neighbours in the live index (t.me/edycohendr, t.me/hnaftali).
CONTENT_FREE = [
    "https://www.c14.co.il/article/1641278",   # THE incident article, 37 chars
    "https://www.c14.co.il/article/1629029",   # same shape, fan-out 92
    "https://t.me/edycohendr/3213",
    "@edycohendr",
    "#breaking",
    "🔴",
    "🇮🇱🇮🇱🇮🇱",
    "",
    "   \n  ",
    "97%",                                      # digits are not a proposition
    "https://www.c14.co.il/article/1641278 @edycohendr #news 🔴",
    "www.jpost.com/breaking-news/article-812345",
]

# Terse but real — every one of these must survive. The rescue path exists FOR these.
CARRIES_CONTENT = [
    "Elections are now final for Oct 27.",
    "מבזק",                                     # 4 chars, Hebrew: "newsflash"
    "שרן השכל התפטרה מתפקידה כסגנית שר החוץ",   # the motivating Ben Caspit post
    "טראמפ: אין ספק שנתקוף את הר המכוש",
    "Кажется «Ансар Аллах» готов вступить в полномасштабную войну",
    "Reading the Signs. It's About To Happen! https://m.example.com/x",  # teaser — has words
    "Bennett: I will recognise Qatar. https://t.me/ben_caspit/18097",
    "🔴 Ceasefire agreed.",                     # emoji plus a real clause
]


@pytest.mark.parametrize("text", CONTENT_FREE)
def test_pointer_only_text_carries_no_proposition(text):
    assert gatekeeper.carries_proposition(text) is False


@pytest.mark.parametrize("text", CARRIES_CONTENT)
def test_terse_but_real_posts_still_carry_a_proposition(text):
    # The floor is "any two consecutive letters survive pointer-stripping", NOT a length rule.
    # Deciding whether a short post is substantive is the judge's job; this only asks whether
    # there is anything at all for it to judge.
    assert gatekeeper.carries_proposition(text) is True


@pytest.mark.asyncio
async def test_content_free_input_is_rejected_without_calling_the_model():
    """The point of enforcing in code: the confabulation cannot happen if the model is never asked.
    Being free is not a side benefit — it is why this can sit in front of every caller."""
    with patch("tm.gatekeeper.complete_structured", new=AsyncMock()) as cs:
        out, usage = await gatekeeper.check_is_prediction(
            article_text="https://www.c14.co.il/article/1641278",
            source_name="Edy Cohen",
            article_date="2026-07-31",
            event_name="Israel will strike Iran before 2026-12-31",
            short_form=True,
        )
    cs.assert_not_awaited()
    assert out.is_prediction is False
    assert out.relevance_score == 0.0
    assert out.prediction_count_estimate == 0
    assert usage == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


@pytest.mark.asyncio
async def test_the_guard_covers_the_forecast_path_too_not_just_short_form():
    """/relevance produced the measured incident, but /forecast judges search-provider articles
    with the same model and the same failure mode. short_form is not a condition of the guard."""
    with patch("tm.gatekeeper.complete_structured", new=AsyncMock()) as cs:
        out, _ = await gatekeeper.check_is_prediction(
            article_text="https://www.c14.co.il/article/1629029",
            source_name="c14",
            article_date="2026-07-31",
            event_name="Israel will strike Iran before 2026-12-31",
        )
    cs.assert_not_awaited()
    assert out.is_prediction is False


@pytest.mark.asyncio
async def test_real_content_still_reaches_the_model():
    """The guard must be inert for everything that isn't content-free — it is a floor, not a filter.
    A regression here would silently mute the whole pipeline."""
    with patch("tm.gatekeeper.complete_structured", new=AsyncMock(return_value=(None, {}))) as cs:
        await gatekeeper.check_is_prediction(
            article_text="שרן השכל התפטרה מתפקידה כסגנית שר החוץ",
            source_name="Ben Caspit",
            article_date="2026-07-12",
            event_name="The coalition will collapse before 2026-12-31",
            short_form=True,
        )
    cs.assert_awaited_once()


@pytest.mark.asyncio
async def test_the_returned_verdict_is_never_the_shared_instance():
    """The rejection is built from a module-level template. Callers (and pydantic-mutating test
    code) must not be able to reach through a returned object and edit what the next caller gets."""
    a, _ = await gatekeeper.check_is_prediction("🔴", "x", "", "claim")
    b, _ = await gatekeeper.check_is_prediction("🔴", "x", "", "claim")
    assert a is not b
    assert a is not gatekeeper._NO_CONTENT_VERDICT


def test_short_form_prompt_teaches_the_rule_the_code_enforces():
    """Both layers or neither: the code stops the calls we make, the prompt is what protects a
    judgment when the model is reached by some path this guard doesn't sit in front of."""
    override = gatekeeper._SHORT_FORM_OVERRIDE
    assert "No content, no judgment" in override
    assert "A URL is not content" in override
    # The specific confabulation mechanism, named so a prompt edit can't drop it by accident.
    assert "slug" in override and "domain" in override
