"""The extractor short-form override: scope a terse post to its own primary topic.

retro#297: a single Telegram post about the IRGC declaring US bases legitimate targets
produced, from the same text, a claim about Ukrainian long-range attrition strategy AND one
about the S&P 500 CAPE ratio — pushed at cosine 0.03-0.09 via the rescue path. The gatekeeper
grew a short_form override in #264/#266; the extractor never did, and its "### Buried facts"
rule ("Scan the WHOLE article ... however minor its role") is correct for a long article with
an incidental clause and actively wrong for a terse multi-topic post.

These tests pin the fix AND, more importantly, that it cannot leak into the default
long-form path — the same safety property test_gatekeeper_short_form.py pins for the gate.
"""

from unittest.mock import AsyncMock, patch

import pytest

from tm import extractor

_ARGS = dict(
    article_text="IRGC declares US bases in the region legitimate targets.",
    source_name="Ben Caspit",
    article_date="2026-07-12",
    event_name="Ukraine will adopt a long-range attrition strategy",
    event_description="Ukraine will adopt a long-range attrition strategy",
)


async def _capture_prompt(**kwargs) -> str:
    """Reconstructs the full text the model actually sees: the cached PROMPT_PREFIX
    (passed as its own kwarg for Bedrock/Anthropic prompt caching, not concatenated
    into ``prompt``) plus the per-call ``prompt`` (PROMPT_SUFFIX, formatted). The
    reconstruction keeps the byte-for-byte invariant below meaningful across the
    PROMPT_PREFIX/PROMPT_SUFFIX split."""
    with patch("tm.extractor.complete_structured", new=AsyncMock(return_value=(None, {}))) as cs:
        await extractor.extract_predictions(**_ARGS, **kwargs)
    prompt = cs.await_args.args[2]
    cached_prefix = cs.await_args.kwargs.get("cached_prefix", "")
    return cached_prefix + prompt


@pytest.mark.asyncio
async def test_default_prompt_is_unchanged_byte_for_byte():
    """THE safety property. Every existing caller sends no short_form, and its prompt must be
    exactly what it was — the override may only ever append, never edit the base text."""
    default = await _capture_prompt()
    expected = (extractor.PROMPT_PREFIX + extractor.PROMPT_SUFFIX).format(
        **_ARGS, journalist="unknown", claim_deadline="not given"
    )
    assert default == expected
    assert "Short-form source" not in default


@pytest.mark.asyncio
async def test_short_form_appends_the_override_and_keeps_the_base_intact():
    short = await _capture_prompt(short_form=True)
    default = await _capture_prompt()

    assert short.startswith(default)                    # append-only: base text survives verbatim
    assert "Short-form source" in short
    # the override must scope to the post's own topic, not contradict Buried facts for articles
    assert "own primary topic" in short
    assert "extract nothing" in short
