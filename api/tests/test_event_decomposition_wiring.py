"""retro#758 wiring: the once-per-question decomposition reaches every article's
event_description exactly when the flag is on, and leaves it byte-identical when
off — the whole safety contract for an append-only, off-by-default change.

Follows test_settlement_verifier.py's fixture shape: gate/extract stubbed so no
model is called for the extraction itself, only the decomposition call under test.
"""
from __future__ import annotations

from datetime import date, timedelta

from forecast_api import forecaster
from forecast_api.config import settings as api_settings
from forecast_api.models import ArticleInput, ForecastRequest
from tm.models import ExtractionOutput, GatekeeperOutput, PredictionExtraction


_FRESH = (date.today() - timedelta(days=1)).isoformat()
_BODY = (
    "Fixture article body for the decomposition-wiring suite; the gatekeeper and "
    "extractor are stubbed, so no model reads it beyond the decomposition call. "
) * 3


def _claim() -> PredictionExtraction:
    return PredictionExtraction(
        quote="Officials confirmed the step was taken.",
        claim="The step has been taken.",
        stance=0.5, certainty=0.6, settled=False,
    )


def _patch_gate_and_extract(monkeypatch):
    """Stub the gate + extractor, capturing every event_description the
    extractor was actually called with."""
    calls: list[str] = []

    async def fake_gate(**kwargs):
        return (GatekeeperOutput(
            is_prediction=True, reason="fixture gate", relevance_score=1.0,
        ), {"total_tokens": 0})

    async def fake_extract(**kwargs):
        calls.append(kwargs["event_description"])
        return (ExtractionOutput(predictions=[_claim()]), {"total_tokens": 0})

    monkeypatch.setattr(forecaster, "check_is_prediction", fake_gate)
    monkeypatch.setattr(forecaster, "extract_predictions", fake_extract)
    return calls


async def _run(monkeypatch, question: str, *, resolution_criteria: str | None = None):
    return await forecaster.run_forecast(ForecastRequest(
        question=question,
        resolution_criteria=resolution_criteria,
        articles=[
            ArticleInput(
                url="https://source.example.test/story",
                title="Fixture headline",
                snippet="Fixture snippet long enough to be usable.",
                source="source", published_date=_FRESH, text=_BODY,
            ),
        ],
    ))


class TestFlagOff:
    async def test_default_is_off_and_event_description_is_unchanged(self, monkeypatch):
        assert api_settings.inject_event_decomposition is False, (
            "must ship off — the sentinel case alone does not establish the "
            "retro#691 387-pair measurement retro#758 itself calls for"
        )
        calls = _patch_gate_and_extract(monkeypatch)
        decompose_calls = []

        async def fake_decompose(*_a, **_kw):
            decompose_calls.append(1)
            return "WHO: X. WHAT: Y. SCOPE: Z."

        monkeypatch.setattr(forecaster, "decompose_event", fake_decompose)
        await _run(monkeypatch, "[decomp-off] Will X happen?", resolution_criteria="Resolves YES if X.")

        assert decompose_calls == [], "off means no call at all, not a call that gets discarded"
        assert calls == ["Resolves YES if X."], "event_description must be byte-identical to pre-#758"


class TestFlagOn:
    async def test_the_decomposition_is_appended_to_every_article(self, monkeypatch):
        monkeypatch.setattr(api_settings, "inject_event_decomposition", True)
        calls = _patch_gate_and_extract(monkeypatch)

        async def fake_decompose(question, resolution_criteria, **kwargs):
            return "WHO: NASA. WHAT: officially states the Earth is flat. SCOPE: by 2026-12-31."

        monkeypatch.setattr(forecaster, "decompose_event", fake_decompose)
        await _run(
            monkeypatch, "[decomp-on] NASA will officially state the Earth is flat by 2026-12-31.",
            resolution_criteria="Resolves YES if NASA issues an official statement.",
        )

        assert len(calls) == 1
        assert calls[0].startswith("Resolves YES if NASA issues an official statement.")
        assert "WHO: NASA. WHAT: officially states the Earth is flat. SCOPE: by 2026-12-31." in calls[0]

    async def test_a_failed_decomposition_leaves_event_description_unchanged(self, monkeypatch):
        """decompose_event fails open to None on any error — this proves the
        None propagates all the way through rather than injecting an empty
        or malformed line."""
        monkeypatch.setattr(api_settings, "inject_event_decomposition", True)
        calls = _patch_gate_and_extract(monkeypatch)

        async def fake_decompose(*_a, **_kw):
            return None

        monkeypatch.setattr(forecaster, "decompose_event", fake_decompose)
        await _run(monkeypatch, "[decomp-fail] Will X happen?", resolution_criteria="Resolves YES if X.")

        assert calls == ["Resolves YES if X."]

    async def test_the_decomposition_call_is_made_once_not_per_article(self, monkeypatch):
        monkeypatch.setattr(api_settings, "inject_event_decomposition", True)
        _patch_gate_and_extract(monkeypatch)
        decompose_calls = []

        async def fake_decompose(*_a, **_kw):
            decompose_calls.append(1)
            return "WHO: X. WHAT: Y. SCOPE: Z."

        monkeypatch.setattr(forecaster, "decompose_event", fake_decompose)
        await forecaster.run_forecast(ForecastRequest(
            question="[decomp-once] Will X happen?",
            articles=[
                ArticleInput(
                    url=f"https://source-{i}.example.test/story",
                    title=f"Fixture headline {i}",
                    snippet=f"Fixture snippet long enough to be usable, variant {i}.",
                    source=f"source-{i}", published_date=_FRESH, text=_BODY,
                )
                for i in (1, 2, 3)
            ],
        ))

        assert len(decompose_calls) == 1, "one call per question, reused across every article in the batch"
