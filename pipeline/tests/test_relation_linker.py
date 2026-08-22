"""Unit tests for the relation-linker prompt construction and output schema (retro#574).

Accuracy against real claim pairs is NOT tested here — that needs a real LLM call and is
covered by the manual eval_relation_linker.py script (not CI). These tests only pin: the
prompt is built correctly from the two claim texts, the cached-prefix/suffix split matches
the gatekeeper.py convention, and PairRelationOutput validates the shapes the classifier
is allowed to return.
"""

from unittest.mock import AsyncMock, patch

import pytest
from instructor.core.exceptions import InstructorRetryException
from pydantic import ValidationError

from tm import relation_linker
from tm.models import PairRelationOutput


def _retry_exc() -> InstructorRetryException:
    return InstructorRetryException("malformed output", n_attempts=1, total_usage=0)


async def _capture_prompt(claim_a: str, claim_b: str) -> tuple[str, str]:
    """Returns (cached_prefix, prompt) exactly as passed to complete_structured."""
    with patch("tm.relation_linker.complete_structured", new=AsyncMock(return_value=(None, {}))) as cs:
        await relation_linker.classify_relation(claim_a, claim_b)
    return cs.await_args.kwargs.get("cached_prefix", ""), cs.await_args.args[2]


@pytest.mark.asyncio
async def test_prompt_carries_both_claims_verbatim():
    prefix, prompt = await _capture_prompt(
        "Israel withdraws from southern Lebanon by 31 December 2026.",
        "The IDF maintains a military presence in southern Lebanon through 31 December 2026.",
    )
    assert "Israel withdraws from southern Lebanon by 31 December 2026." in prompt
    assert "The IDF maintains a military presence in southern Lebanon through 31 December 2026." in prompt
    assert prompt == relation_linker.PROMPT_SUFFIX.format(
        claim_a="Israel withdraws from southern Lebanon by 31 December 2026.",
        claim_b="The IDF maintains a military presence in southern Lebanon through 31 December 2026.",
    )
    assert prefix == relation_linker.PROMPT_PREFIX


@pytest.mark.asyncio
async def test_prefix_is_cached_not_concatenated():
    # Mirrors gatekeeper's PROMPT_PREFIX/PROMPT_SUFFIX split: the fixed instructions must
    # travel as their own cached_prefix kwarg, not get baked into the per-call prompt —
    # otherwise Bedrock/Anthropic prompt caching never engages for this module either.
    _, prompt = await _capture_prompt("Claim A text.", "Claim B text.")
    assert "You are a relation-typing classifier" not in prompt


@pytest.mark.asyncio
async def test_uses_extractor_model_tier_not_gatekeeper():
    with patch("tm.relation_linker.complete_structured", new=AsyncMock(return_value=(None, {}))) as cs:
        await relation_linker.classify_relation("A", "B")
    from tm.config import settings
    assert cs.await_args.args[0] == settings.extractor_model


class TestPairRelationOutputSchema:
    def test_minimal_valid_alias(self):
        out = PairRelationOutput(
            relation_type="alias", polarity="opposite", reason="Same event, negated.",
        )
        assert out.direction is None
        assert out.quote_a is None and out.quote_b is None

    def test_nested_with_direction_and_quotes(self):
        out = PairRelationOutput(
            relation_type="nested", direction="a_to_b", polarity="same",
            quote_a="by 2029", quote_b="by 2030", reason="Earlier deadline is a subset.",
        )
        assert out.relation_type == "nested"
        assert out.direction == "a_to_b"

    def test_independent_leaves_direction_and_polarity_null(self):
        out = PairRelationOutput(relation_type="independent", reason="Merely topical.")
        assert out.direction is None
        assert out.polarity is None

    def test_rejects_invalid_relation_type(self):
        with pytest.raises(ValidationError):
            PairRelationOutput(relation_type="opposite", reason="x")

    def test_rejects_invalid_direction(self):
        with pytest.raises(ValidationError):
            PairRelationOutput(relation_type="nested", direction="sideways", reason="x")

    def test_reason_is_required(self):
        with pytest.raises(ValidationError):
            PairRelationOutput(relation_type="independent")

    def test_unwraps_properties_envelope(self):
        # Nova Lite (MD_JSON mode) intermittently wraps output in {"properties": {...}} —
        # same quirk GatekeeperOutput already guards against (retro#306).
        out = PairRelationOutput.model_validate({
            "properties": {"relation_type": "complement", "polarity": "same", "reason": "x"}
        })
        assert out.relation_type == "complement"


class TestClassifyRelationRetriesOnMalformedOutput:
    """Nova Lite's MD_JSON mode was observed (manual runs, see PR description) failing
    Pydantic validation on this schema — sometimes as the {"properties": ...} envelope
    above, sometimes as a bare schema echo with no instance data at all, which no local
    unwrapping can repair. complete_structured's shared max_retries=1 gives instructor
    zero self-correction attempts, so classify_relation retries the whole call itself.
    """

    @pytest.mark.asyncio
    async def test_succeeds_after_transient_malformed_output(self):
        good = PairRelationOutput(relation_type="complement", polarity="same", reason="x")
        mock = AsyncMock(side_effect=[_retry_exc(), _retry_exc(), (good, {"tokens": 10})])
        with patch("tm.relation_linker.complete_structured", new=mock), \
             patch("tm.relation_linker.asyncio.sleep", new=AsyncMock()) as sleep_mock:
            out, usage = await relation_linker.classify_relation("A", "B")
        assert out is good
        assert usage == {"tokens": 10}
        assert mock.await_count == 3
        assert sleep_mock.await_count == 2  # slept between attempts 1->2 and 2->3, not after success

    @pytest.mark.asyncio
    async def test_gives_up_after_max_attempts_and_raises_the_last_error(self):
        mock = AsyncMock(side_effect=[_retry_exc() for _ in range(relation_linker._MAX_ATTEMPTS)])
        with patch("tm.relation_linker.complete_structured", new=mock), \
             patch("tm.relation_linker.asyncio.sleep", new=AsyncMock()):
            with pytest.raises(InstructorRetryException):
                await relation_linker.classify_relation("A", "B")
        assert mock.await_count == relation_linker._MAX_ATTEMPTS

    @pytest.mark.asyncio
    async def test_first_attempt_success_needs_no_retry(self):
        good = PairRelationOutput(relation_type="independent", reason="x")
        mock = AsyncMock(return_value=(good, {}))
        with patch("tm.relation_linker.complete_structured", new=mock):
            out, _ = await relation_linker.classify_relation("A", "B")
        assert out is good
        assert mock.await_count == 1

    @pytest.mark.asyncio
    async def test_does_not_swallow_unrelated_errors(self):
        # A genuine non-malformed-output error (e.g. auth, a real network failure that
        # complete_structured's own rate-limit wrapper didn't classify as transient)
        # must propagate immediately, not get treated as retriable.
        mock = AsyncMock(side_effect=RuntimeError("boom"))
        with patch("tm.relation_linker.complete_structured", new=mock):
            with pytest.raises(RuntimeError):
                await relation_linker.classify_relation("A", "B")
        assert mock.await_count == 1
