"""WHO/WHAT/SCOPE decomposition call (retro#758) — parsing and fail-open contract.

Mirrors test_settlement_verifier.py's TestParsing: an unreachable model, a
timeout, or an unparseable reply must never raise and must never invent a
decomposition — the caller's cue to leave event_description exactly as it was
before this existed.
"""
from __future__ import annotations

import pytest

from forecast_api import event_decomposition as ed


class TestParsing:
    def test_clean_line(self):
        text = "WHO: Country R (country). WHAT: issues an official mobilisation order. SCOPE: by 2026-12-31."
        assert ed.parse_decomposition(text) == text

    def test_line_wrapped_in_prose(self):
        text = (
            "Here is the decomposition:\n"
            "WHO: NASA. WHAT: officially states the Earth is flat. SCOPE: by 2026-12-31.\n"
            "Let me know if you need anything else."
        )
        parsed = ed.parse_decomposition(text)
        assert parsed is not None
        assert parsed.startswith("WHO: NASA.")
        assert parsed.endswith("SCOPE: by 2026-12-31.")
        assert "Let me know" not in parsed

    def test_embedded_newlines_are_collapsed(self):
        text = "WHO: X.\nWHAT: does Y.\nSCOPE: by Z."
        parsed = ed.parse_decomposition(text)
        assert "\n" not in parsed

    @pytest.mark.parametrize("text", ["", "no idea", "WHO: X. WHAT: Y.", "SCOPE only, no who or what"])
    def test_anything_unparseable_fails_open(self, text):
        assert ed.parse_decomposition(text) is None


class TestDecomposeEvent:
    async def test_empty_question_skips_without_a_call(self, monkeypatch):
        called = []

        async def fake_complete(*_a, **_kw):
            called.append(1)
            return "WHO: X. WHAT: Y. SCOPE: Z."

        monkeypatch.setattr(ed, "complete_text_once", fake_complete)
        result = await ed.decompose_event("", None, model="fixture-model")
        assert result is None
        assert called == []

    async def test_a_clean_reply_round_trips(self, monkeypatch):
        expected = "WHO: Elon Musk. WHAT: tweets about Daatan. SCOPE: by 2028-12-31."

        async def fake_complete(model, prompt, **kwargs):
            assert "Elon Musk will tweet about Daatan" in prompt
            return expected

        monkeypatch.setattr(ed, "complete_text_once", fake_complete)
        result = await ed.decompose_event(
            "Elon Musk will tweet about Daatan by December 31, 2028.", None, model="fixture-model",
        )
        assert result == expected

    async def test_resolution_criteria_reaches_the_prompt(self, monkeypatch):
        captured = {}

        async def fake_complete(model, prompt, **kwargs):
            captured["prompt"] = prompt
            return "WHO: X. WHAT: Y. SCOPE: Z."

        monkeypatch.setattr(ed, "complete_text_once", fake_complete)
        await ed.decompose_event(
            "Will X happen?", "Resolves YES if the official record shows X.", model="fixture-model",
        )
        assert "official record shows X" in captured["prompt"]

    async def test_a_call_failure_fails_open_to_none(self, monkeypatch):
        async def fake_complete(*_a, **_kw):
            raise TimeoutError("fixture timeout")

        monkeypatch.setattr(ed, "complete_text_once", fake_complete)
        result = await ed.decompose_event("Will X happen?", None, model="fixture-model")
        assert result is None

    async def test_an_unparseable_reply_fails_open_to_none(self, monkeypatch):
        async def fake_complete(*_a, **_kw):
            return "I'm not sure how to answer that."

        monkeypatch.setattr(ed, "complete_text_once", fake_complete)
        result = await ed.decompose_event("Will X happen?", None, model="fixture-model")
        assert result is None
