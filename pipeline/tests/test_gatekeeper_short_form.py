"""The short-form override: judge a terse social post on content, not length.

The base gatekeeper prompt rejects anything "under ~200 meaningful words" as insubstantial — a rule
written for paywall stubs and 404 pages. It misfires on a source we ingest ON PURPOSE: 460 of 552
indexed Telegram posts run under 70 words, including 154 of Ben Caspit's 169. So the pipeline
deliberately collects terse posts from serious journalists and the gatekeeper throws every one away.

These tests pin the fix AND, more importantly, that it cannot leak into /forecast.
"""

from unittest.mock import AsyncMock, patch

import pytest

from tm import gatekeeper


async def _capture_prompt(**kwargs) -> str:
    """Reconstructs the full text the model actually sees: the cached PROMPT_PREFIX
    (now passed as its own kwarg for Bedrock/Anthropic prompt caching, not concatenated
    into ``prompt`` anymore) plus the per-call ``prompt`` (PROMPT_SUFFIX, formatted).
    Keeping this reconstruction is what lets the byte-for-byte invariant below still
    mean what it always meant, across the PROMPT_PREFIX/PROMPT_SUFFIX split."""
    with patch("tm.gatekeeper.complete_structured", new=AsyncMock(return_value=(None, {}))) as cs:
        await gatekeeper.check_is_prediction(
            article_text="Elections are now final for Oct 27.",
            source_name="Ben Caspit",
            article_date="2026-07-12",
            event_name="Netanyahu will form the next government",
            **kwargs,
        )
    prompt = cs.await_args.args[2]
    cached_prefix = cs.await_args.kwargs.get("cached_prefix", "")
    return cached_prefix + prompt


@pytest.mark.asyncio
async def test_default_prompt_is_unchanged_byte_for_byte():
    """THE safety property. /forecast calls this with no short_form, and its prompt must be exactly
    what it was — the override may only ever append, never edit the base text."""
    default = await _capture_prompt()
    expected = (gatekeeper.PROMPT_PREFIX + gatekeeper.PROMPT_SUFFIX).format(
        article_text="Elections are now final for Oct 27.",
        source_name="Ben Caspit",
        article_date="2026-07-12",
        event_name="Netanyahu will form the next government",
    )
    assert default == expected
    assert "Short-form source" not in default


@pytest.mark.asyncio
async def test_short_form_appends_the_override_and_keeps_the_base_intact():
    short = await _capture_prompt(short_form=True)
    default = await _capture_prompt()

    assert short.startswith(default)                    # append-only: base text survives verbatim
    assert "Short-form source" in short
    assert "do NOT set\nis_prediction=false merely because it runs under ~200 words" in short


@pytest.mark.asyncio
async def test_short_form_still_rejects_empty_and_off_topic_posts():
    # The override lifts the LENGTH rule only. A caption-less image or a post about a different
    # matter must still be rejected — brevity alone is what stops being disqualifying.
    short = await _capture_prompt(short_form=True)
    assert "genuinely empty" in short
    assert "different matter" in short
    assert "Brevity alone is not a reason" in short
