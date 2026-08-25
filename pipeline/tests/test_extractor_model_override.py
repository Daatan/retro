"""retro#652: the extractor accepts a per-call ``model`` override.

A caller with a wider latency/cost budget than live production (e.g. a benchmark
harness) can point a single call at a different litellm model without touching the
process-wide ``settings.extractor_model`` default every other caller relies on.
"""

from unittest.mock import AsyncMock, patch

import pytest

from tm import extractor
from tm.config import settings

_ARGS = dict(
    article_text="IRGC declares US bases in the region legitimate targets.",
    source_name="Ben Caspit",
    article_date="2026-07-12",
    event_name="Ukraine will adopt a long-range attrition strategy",
    event_description="Ukraine will adopt a long-range attrition strategy",
)


async def _capture_model(**kwargs) -> str:
    with patch("tm.extractor.complete_structured", new=AsyncMock(return_value=(None, {}))) as cs:
        await extractor.extract_predictions(**_ARGS, **kwargs)
    return cs.await_args.args[0]


@pytest.mark.asyncio
async def test_default_uses_the_configured_extractor_model():
    assert await _capture_model() == settings.extractor_model


@pytest.mark.asyncio
async def test_model_override_replaces_the_configured_default():
    override = "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"
    assert await _capture_model(model=override) == override
    # the global default is untouched for the next caller
    assert await _capture_model() == settings.extractor_model
