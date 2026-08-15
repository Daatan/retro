"""retro#542 (F6 residue): the batch pipeline must give t.me posts the same short_form
treatment the live /forecast path has had since retro#417 — without it, the gatekeeper's
~200-word floor rejects (or confabulates on) terse Telegram posts, and the extractor
doesn't scope the post to its own primary topic."""

from unittest.mock import AsyncMock, patch

import pytest

from tm.models import ExtractionOutput, GatekeeperOutput
from tm.runner import ArticleInput, run_article

pytestmark = pytest.mark.asyncio


def _article(**overrides) -> ArticleInput:
    defaults = dict(
        text="some article text",
        source_id="src1",
        source_name="Source One",
        article_date="2026-08-01",
        event_id="E1",
        event_name="Event happens",
        event_description="Will the event happen?",
    )
    defaults.update(overrides)
    return ArticleInput(**defaults)


async def _run(article: ArticleInput) -> tuple[AsyncMock, AsyncMock]:
    gate = GatekeeperOutput(is_prediction=True, reason="looks predictive")
    extraction = ExtractionOutput(predictions=[])
    gate_mock = AsyncMock(return_value=(gate, {}))
    extract_mock = AsyncMock(return_value=(extraction, {}))
    with patch("tm.runner.check_is_prediction", new=gate_mock), \
         patch("tm.runner.extract_predictions", new=extract_mock), \
         patch("tm.runner.update_cell"):
        await run_article(article)
    return gate_mock, extract_mock


@pytest.mark.parametrize("url", [
    "https://t.me/somechannel/123",
    "https://www.t.me/somechannel/123",
])
async def test_a_t_me_article_reaches_both_stages_as_short_form(url):
    gate_mock, extract_mock = await _run(_article(article_url=url))
    assert gate_mock.await_args.kwargs["short_form"] is True
    assert extract_mock.await_args.kwargs["short_form"] is True


@pytest.mark.parametrize("url", [
    "https://example.com/news/t.me-story",  # t.me in the path is not a t.me host
    "https://ynet.co.il/article/1",
    None,  # batch rows without a URL stay long-form
])
async def test_everything_else_stays_long_form(url):
    gate_mock, extract_mock = await _run(_article(article_url=url))
    assert gate_mock.await_args.kwargs["short_form"] is False
    assert extract_mock.await_args.kwargs["short_form"] is False
