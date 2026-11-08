"""retro#1054: the numeric-bar exception is /relevance-only and must not leak into /forecast.

On threshold claims the base prompt treats a report of today's value of the barred quantity
("Brent at $118" for "Brent will reach $130") as a different matter. RELEVANCE_PROMPT_PREFIX adds
an exception inside the rejection rule. Offline it fixed 7/9 such misses on Gemini but gained
nothing on Nova Micro, so only callers that pass `numeric_bar_exception=True` get it.
"""

from unittest.mock import AsyncMock, patch

import pytest

from tm import gatekeeper

ARTICLE = "Oil is trading at $118.07 per barrel this morning, based on the Brent benchmark. " * 5


async def _cached_prefix(**kwargs) -> str:
    with patch("tm.gatekeeper.complete_structured", new=AsyncMock(return_value=(None, {}))) as cs:
        await gatekeeper.check_is_prediction(
            article_text=ARTICLE,
            source_name="Fortune",
            article_date="2026-11-04",
            event_name="The price of Brent crude oil will reach or exceed $130 per barrel",
            **kwargs,
        )
    return cs.await_args.kwargs["cached_prefix"]


@pytest.mark.asyncio
async def test_default_sends_the_base_prefix_unchanged():
    assert await _cached_prefix() == gatekeeper.PROMPT_PREFIX
    assert "numeric bars" not in gatekeeper.PROMPT_PREFIX


@pytest.mark.asyncio
async def test_opt_in_sends_the_relevance_prefix():
    assert await _cached_prefix(numeric_bar_exception=True) == gatekeeper.RELEVANCE_PROMPT_PREFIX


def test_relevance_prefix_only_inserts_the_exception():
    # Same base text with exactly one block inserted, right after the "different matter" rejection
    # it qualifies — not a rewording of anything else.
    rel = gatekeeper.RELEVANCE_PROMPT_PREFIX
    assert rel.replace(gatekeeper._NUMERIC_BAR_EXCEPTION, "", 1) == gatekeeper.PROMPT_PREFIX
    assert gatekeeper._REJECTION_ANCHOR + gatekeeper._NUMERIC_BAR_EXCEPTION in rel
    assert rel.index(gatekeeper._NUMERIC_BAR_EXCEPTION) < rel.index("It is empty, a paywall/404 stub")
