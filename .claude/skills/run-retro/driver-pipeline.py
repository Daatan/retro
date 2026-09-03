#!/usr/bin/env python3
"""
Direct-invocation driver for pipeline/ (package `tm`). See SKILL.md "Run: pipeline".

`tm` has no CLI/server — its real entrypoints (`python -m tm.orchestrator api`,
`smoke_test.py`) call AWS Bedrock/OpenRouter for real and must NOT run in a
sandbox with no legitimate production credentials. This script instead drives
the pipeline stages directly, the way pipeline/tests/ does:

  1. Pure functions — zero network, zero LLM, real code paths:
     - tm.dedup.simhash            (SimHash fingerprint used for near-dup detection)
     - tm.extractor.has_conditional_language  (lexical pre-filter, no LLM)
  2. tm.gatekeeper.check_is_prediction() on CONTENT-FREE input — this is a real,
     unmocked call: carries_proposition() short-circuits before any LLM call for
     input with no assertable proposition (only links/handles/emoji), returning
     a canned GatekeeperOutput. Zero mocking needed, zero network.
  3. tm.gatekeeper.check_is_prediction() / tm.extractor.extract_predictions() on
     a REAL article: the LLM boundary (`tm.gatekeeper.complete_structured` /
     `tm.extractor.complete_structured`) is patched with an AsyncMock, exactly
     the pattern pipeline/tests/test_gatekeeper_content_free.py already uses —
     this proves the full code path (prompt construction, parsing, validation)
     without ever calling Bedrock/OpenRouter.

Run: cd pipeline && uv run python ../.claude/skills/run-retro/driver-pipeline.py
"""
import asyncio
from unittest.mock import AsyncMock, patch

from tm.dedup import simhash
from tm.extractor import has_conditional_language, extract_predictions
from tm.gatekeeper import check_is_prediction, GatekeeperOutput
from tm.models import ExtractionOutput, PredictionExtraction


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> None:
    section("1. Pure functions (no LLM, no network)")
    h = simhash("Netanyahu's coalition faces a budget vote by March 31.")
    print(f"simhash(...) = {h:#018x}")
    assert isinstance(h, int)

    cond = has_conditional_language("If the budget fails to pass, the government will collapse.")
    print(f"has_conditional_language('If the budget fails...') = {cond}")
    assert cond is True

    plain = has_conditional_language("The Knesset passed the budget yesterday.")
    print(f"has_conditional_language('The Knesset passed...') = {plain}")
    assert plain is False

    section("2. tm.gatekeeper.check_is_prediction — content-free input (REAL, unmocked call)")
    result, usage = asyncio.run(check_is_prediction(
        article_text="🔗 https://example.com/status/12345 @someone",
        source_name="test",
        article_date="2026-01-01",
        event_name="Test event",
    ))
    print(f"is_prediction={result.is_prediction} reason={result.reason!r}")
    print(f"usage={usage}")
    assert result.is_prediction is False
    assert usage == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    print("Confirmed zero-token, zero-network path (carries_proposition short-circuit).")

    section("3. tm.gatekeeper.check_is_prediction — real article, LLM boundary MOCKED")
    canned_gate = GatekeeperOutput(
        is_prediction=True,
        reason="Article contains an explicit forward-looking claim about the budget vote.",
        prediction_count_estimate=1,
        relevance_score=0.9,
    )
    with patch("tm.gatekeeper.complete_structured", new=AsyncMock(return_value=(canned_gate, {"prompt_tokens": 120, "completion_tokens": 40, "total_tokens": 160}))):
        gate_result, gate_usage = asyncio.run(check_is_prediction(
            article_text=(
                "Senior Likud officials privately acknowledge that if the budget fails to pass "
                "by the end of March, the government will collapse within weeks."
            ),
            source_name="Jerusalem Post",
            article_date="2023-11-15",
            event_name="Budget 2024 passes Knesset",
        ))
    print(f"is_prediction={gate_result.is_prediction} reason={gate_result.reason!r}")
    assert gate_result.is_prediction is True

    section("4. tm.extractor.extract_predictions — LLM boundary MOCKED")
    canned_extraction = ExtractionOutput(
        predictions=[
            PredictionExtraction(
                quote="the government will collapse within weeks",
                claim="Israeli government predicted to collapse within weeks if budget fails.",
                stance=-0.8,
                claim_strength=0.7,
            )
        ]
    )
    with patch("tm.extractor.complete_structured", new=AsyncMock(return_value=(canned_extraction, {"prompt_tokens": 300, "completion_tokens": 90, "total_tokens": 390}))):
        extraction, ex_usage = asyncio.run(extract_predictions(
            article_text=(
                "Senior Likud officials privately acknowledge that if the budget fails to pass "
                "by the end of March, the government will collapse within weeks."
            ),
            source_name="Jerusalem Post",
            article_date="2023-11-15",
            event_name="Budget 2024 passes Knesset",
            event_description="Did the Israeli government pass the state budget for 2024?",
        ))
    pred = extraction.predictions[0]
    print(f"claim={pred.claim!r} stance={pred.stance} claim_strength={pred.claim_strength}")
    assert pred.stance == -0.8

    print("\nAll driver stages completed. No real network or LLM calls were made.")


if __name__ == "__main__":
    main()
