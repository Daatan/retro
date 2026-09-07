"""retro#801: an ASCII ``"`` used as the Hebrew gershayim inside an abbreviation (יו"ר, ח"כ,
צה"ל) is copied verbatim by Haiku into the JSON ``quote`` field and, when it opens the quote,
left unescaped — the parse fails and the article is dropped. The extractor rewrites that
character to U+05F4 ״ in the article text before it reaches the model.

Pins: (1) the helper only touches a quote flanked by Hebrew letters on both sides — quoted
speech, English, and text that already uses ״ are untouched, and the rewrite is idempotent;
(2) the rewritten text is what goes into the rendered prompt; (3) an English article renders
exactly as before (the byte-for-byte prompt invariant in test_extractor_short_form.py holds
because the helper is a no-op there).
"""

from unittest.mock import AsyncMock, patch

import pytest

from tm import extractor
from tm.extractor import normalize_hebrew_gershayim

GERSHAYIM = "״"


@pytest.mark.parametrize(
    "raw, expected",
    [
        ('יו"ר עוצמה יהודית', f"יו{GERSHAYIM}ר עוצמה יהודית"),  # the W2 opener that broke Haiku
        ('כוח צה"ל בשוגג', f"כוח צה{GERSHAYIM}ל בשוגג"),
        ('ח"כ אחד וח"כ שני', f"ח{GERSHAYIM}כ אחד וח{GERSHAYIM}כ שני"),  # every occurrence
    ],
)
def test_ascii_quote_between_hebrew_letters_becomes_gershayim(raw, expected):
    assert normalize_hebrew_gershayim(raw) == expected


@pytest.mark.parametrize(
    "untouched",
    [
        'תנועת "קומו" ובין השאר',  # quoted speech: quote follows a space / precedes a letter
        'אמר כי "לא אתמודד לכנסת"',  # quote after a space
        'בסביבתו של "עמך ישראל" מעוניינים',
        f"יו{GERSHAYIM}ר המפלגה",  # already uses U+05F4
        'He said "no" and the 5"-wide 2"x4" plank',  # English / digits — not Hebrew letters
        '',  # empty
        'no quotes at all',
    ],
)
def test_other_quotes_are_left_alone(untouched):
    assert normalize_hebrew_gershayim(untouched) == untouched


def test_idempotent():
    once = normalize_hebrew_gershayim('יו"ר, תנועת "קומו"')
    assert normalize_hebrew_gershayim(once) == once
    assert once == f'יו{GERSHAYIM}ר, תנועת "קומו"'


def test_quote_glued_to_a_prefix_letter_is_rewritten_too():
    """Documented, accepted: ו"קומו" (conjunction + opening quote) has the same shape as a
    suffixed abbreviation (ח"כים) and cannot be told apart; ״ is a valid Hebrew quotation
    mark, so the meaning survives and the closing quote is left as it was."""
    assert normalize_hebrew_gershayim('ו"קומו" וגם ח"כים') == f'ו{GERSHAYIM}קומו" וגם ח{GERSHAYIM}כים'


_ARGS = dict(
    source_name="srugim",
    article_date="2026-09-06",
    event_name="Will Ben Gvir's list run alone?",
    event_description="Resolves YES if Otzma Yehudit files an independent list.",
)


async def _rendered_prompt(article_text: str) -> str:
    with patch("tm.extractor.complete_structured", new=AsyncMock(return_value=(None, {}))) as cs:
        await extractor.extract_predictions(article_text=article_text, **_ARGS)
    return cs.await_args.args[2]


@pytest.mark.asyncio
async def test_prompt_carries_the_normalised_article_text():
    raw = 'יו"ר עוצמה יהודית, השר איתמר בן גביר, הודיע כי "נרוץ לבד"'
    prompt = await _rendered_prompt(raw)
    assert f"יו{GERSHAYIM}ר עוצמה יהודית" in prompt
    assert 'יו"ר' not in prompt
    assert 'כי "נרוץ לבד"' in prompt  # quoted speech survives verbatim


@pytest.mark.asyncio
async def test_english_article_renders_unchanged():
    raw = 'The chair said "we will run alone" on Tuesday.'
    prompt = await _rendered_prompt(raw)
    assert raw in prompt
