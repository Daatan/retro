"""retro#688 — the wiring between the routing decision and what actually runs.

Three things have to hold together, and the last two are what make the acceptance
criteria checkable rather than merely plausible:

1. a routed event reaches ``extract_predictions`` with the override;
2. the row's provenance records the model that RAN, not ``settings.extractor_model``
   — the global stops being the truth the moment any row is routed;
3. the negative-marker staleness check compares a row against ITS model, not the
   global, or every routed marker is stale on every cycle and the pipeline
   re-extracts them forever on the expensive model.
"""

from unittest.mock import AsyncMock, patch

import pytest

from tm.models import ExtractionOutput, GatekeeperOutput
from tm.orchestrator import EXTRACTION_PROMPT_VERSION, Orchestrator
from tm.runner import ArticleInput, run_article

# No module-level asyncio mark: pyproject sets `asyncio_mode = "auto"`, and this file
# deliberately mixes async wiring tests with sync ones for the staleness predicate.

HAIKU = "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"
NOVA = "bedrock/us.amazon.nova-lite-v1:0"


def _article(**overrides) -> ArticleInput:
    defaults = dict(
        text="some article text",
        source_id="src1",
        source_name="Source One",
        article_date="2026-08-01",
        event_id="I01",
        event_name="Brent crude oil exceeds $100/barrel (Russia-Ukraine)",
        event_description="Polymarket market resolved Yes.",
    )
    defaults.update(overrides)
    return ArticleInput(**defaults)


async def _run(article: ArticleInput, extractor_model=None, is_prediction=True):
    gate = GatekeeperOutput(is_prediction=is_prediction, reason="looks predictive")
    gate_mock = AsyncMock(return_value=(gate, {}))
    extract_mock = AsyncMock(return_value=(ExtractionOutput(predictions=[]), {}))
    with patch("tm.runner.check_is_prediction", new=gate_mock), \
         patch("tm.runner.extract_predictions", new=extract_mock), \
         patch("tm.runner.update_cell"):
        result = await run_article(article, extractor_model=extractor_model)
    return result, extract_mock


# ── 1. the override reaches the call ─────────────────────────────────────────
async def test_a_routed_article_carries_the_override_into_extraction():
    _, extract_mock = await _run(_article(), extractor_model=HAIKU)
    assert extract_mock.await_args.kwargs["model"] == HAIKU


async def test_an_unrouted_article_passes_None_and_keeps_the_global():
    _, extract_mock = await _run(_article(), extractor_model=None)
    assert extract_mock.await_args.kwargs["model"] is None


async def test_the_gatekeeper_is_not_routed():
    """retro#664's numeric failures were all in extraction, and the gate is a cheap
    binary call — routing it would be spend with no measured return."""
    gate = GatekeeperOutput(is_prediction=True, reason="ok")
    gate_mock = AsyncMock(return_value=(gate, {}))
    with patch("tm.runner.check_is_prediction", new=gate_mock), \
         patch("tm.runner.extract_predictions",
               new=AsyncMock(return_value=(ExtractionOutput(predictions=[]), {}))), \
         patch("tm.runner.update_cell"):
        await run_article(_article(), extractor_model=HAIKU)
    assert "model" not in gate_mock.await_args.kwargs


# ── 2. provenance records what ran, on every path ────────────────────────────
async def test_a_routed_run_reports_the_routed_model():
    result, _ = await _run(_article(), extractor_model=HAIKU)
    assert result.extractor_model == HAIKU


async def test_an_unrouted_run_reports_the_configured_global():
    """Not the empty string: the caller writes this straight into the vault row, and a
    blank there would make the row unattributable."""
    from tm.config import settings
    result, _ = await _run(_article(), extractor_model=None)
    assert result.extractor_model == settings.extractor_model


async def test_a_gate_rejection_still_reports_a_model():
    """The negative-marker path writes provenance too, and its staleness check reads
    that field back. A blank here re-extracts every gate-rejected article forever."""
    result, _ = await _run(_article(), extractor_model=HAIKU, is_prediction=False)
    assert result.is_prediction is False
    assert result.extractor_model == HAIKU


async def test_a_failed_run_still_reports_a_model():
    gate_mock = AsyncMock(side_effect=RuntimeError("bedrock exploded"))
    with patch("tm.runner.check_is_prediction", new=gate_mock), \
         patch("tm.runner.update_cell"):
        result = await run_article(_article(), extractor_model=HAIKU)
    assert result.error == "bedrock exploded"
    assert result.extractor_model == HAIKU


# ── 3. staleness compares a row against ITS model ────────────────────────────
def _marker(extractor_model: str) -> dict:
    from tm.config import settings
    return {
        "status": "gate_rejected",
        "extraction": None,
        "prompt_version": EXTRACTION_PROMPT_VERSION,
        "extractor_model": extractor_model,
        "gatekeeper_model": settings.gatekeeper_model,
    }


def test_a_routed_marker_is_current_against_its_own_model():
    assert Orchestrator._negative_marker_is_current(_marker(HAIKU), HAIKU) is True


def test_a_routed_marker_would_be_stale_against_the_global():
    """The bug this signature change exists to prevent: judged against
    settings.extractor_model, a Haiku-produced marker is stale on every cycle, so the
    pipeline re-extracts it forever — on the expensive model, and only on the rows the
    routing was added to protect."""
    assert Orchestrator._negative_marker_is_current(_marker(HAIKU), NOVA) is False


def test_an_unrouted_marker_is_current_against_the_global():
    from tm.config import settings
    marker = _marker(settings.extractor_model)
    assert Orchestrator._negative_marker_is_current(
        marker, settings.extractor_model) is True


def test_a_prompt_version_bump_still_invalidates_a_routed_marker():
    """Routing must not weaken docs#57 item 2's other half."""
    stale = _marker(HAIKU) | {"prompt_version": "v0-ancient"}
    assert Orchestrator._negative_marker_is_current(stale, HAIKU) is False


def test_a_gatekeeper_change_still_invalidates_a_routed_marker():
    stale = _marker(HAIKU) | {"gatekeeper_model": "bedrock/some-other-gatekeeper"}
    assert Orchestrator._negative_marker_is_current(stale, HAIKU) is False
