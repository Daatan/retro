"""The A/B driver must call the extractor model the way the live extractor does (retro#1012).

Before this, the driver hard-coded ``max_tokens=1200`` and no ``thinking``, so after retro#921
(Haiku 5.5 + adaptive thinking live) every arm measured a configuration nobody ships.
"""
import asyncio
import importlib.util
from pathlib import Path

from tm.ab_harness import Case
from tm.config import settings
from tm.models import ExtractionOutput

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ab_extractor_prompt.py"
# scripts/ is not an importable package — load the module by path.
_spec = importlib.util.spec_from_file_location("ab_extractor_prompt", _SCRIPT)
ab = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ab)


def _case() -> Case:
    return Case(
        id="c1", article_text="text", source_name="src", journalist="j",
        article_date="2026-01-01", event_name="e", event_description="d",
        claim_deadline="2026-12-31", expect={},
    )


def _calls(monkeypatch, **run_kwargs) -> list[dict]:
    seen: list[dict] = []

    async def fake(model, response_model, prompt, **kwargs):
        seen.append(kwargs)
        return ExtractionOutput(predictions=[]), {}

    monkeypatch.setattr(ab.llm, "complete_structured", fake)
    asyncio.run(ab._run_case(_case(), "m", use_control=False, runs=1, **run_kwargs))
    return seen


def test_runs_with_the_live_output_cap(monkeypatch):
    [kwargs] = _calls(monkeypatch, thinking=True)
    assert kwargs["max_tokens"] == settings.extractor_max_tokens


def test_passes_thinking_through(monkeypatch):
    assert _calls(monkeypatch, thinking=True)[0]["thinking"] is True
    assert _calls(monkeypatch, thinking=False)[0]["thinking"] is False


def test_an_arm_thinks_exactly_when_the_live_extractor_does(monkeypatch):
    monkeypatch.setattr(settings, "extractor_thinking", True)
    assert ab._arm_thinking(no_thinking=False) is True
    assert ab._arm_thinking(no_thinking=True) is False
    monkeypatch.setattr(settings, "extractor_thinking", False)
    assert ab._arm_thinking(no_thinking=False) is False
