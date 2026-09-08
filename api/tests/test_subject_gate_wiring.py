"""retro#805 wiring: the subject card is derived once per request, the gate is evaluated
per article, shadow by default (logged, nothing dropped), and a fired gate drops the article
with outcome `subject_absent` only behind `subject_gate_enforce`. Every fail-open branch
leaves the article exactly as it was.

Follows test_event_decomposition_wiring.py's fixture shape: gate/extract stubbed so no
model is called; the subject-card derivation is stubbed too.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta

import pytest

from forecast_api import forecaster
from forecast_api.config import settings as api_settings
from forecast_api.models import ArticleInput, ForecastRequest
from forecast_api.subject_card import SubjectActor, SubjectCard
from tm.models import ArticleCard, ArticleCardActor, ExtractionOutput, GatekeeperOutput, PredictionExtraction

_FRESH = (date.today() - timedelta(days=1)).isoformat()
# The subject is never named; Hendel and Eisenkot are.
_BODY = (
    "The lists were submitted on Monday. Eisenkot and Bennett arrived separately, and "
    "Yoaz Hendel's Reservists list was among the first five to file. "
) * 3

GANTZ = SubjectCard(actors=[
    SubjectActor(name_en="Benny Gantz", type="person", surface_forms=["Benny Gantz", "Gantz", "בני גנץ", "גנץ"]),
])
HENDEL = SubjectCard(actors=[
    SubjectActor(name_en="Yoaz Hendel", type="person", surface_forms=["Yoaz Hendel", "Hendel", "יועז הנדל"]),
])
CARD = ArticleCard(named_actors=[
    ArticleCardActor(span="Eisenkot", name_en="Gadi Eisenkot"),
    ArticleCardActor(span="Yoaz Hendel", name_en="Yoaz Hendel"),
], bears_on_question=False)


def _claim() -> PredictionExtraction:
    return PredictionExtraction(
        quote="Eisenkot and Bennett arrived separately",
        claim="Gantz's party will run alone.",
        stance=0.9, certainty=0.8, settled=False,
    )


def _patch(monkeypatch, *, card: SubjectCard | None, article_card: ArticleCard | None = CARD,
           predictions: bool = True):
    derive_calls: list[str] = []

    async def fake_gate(**kwargs):
        return (GatekeeperOutput(is_prediction=True, reason="fixture gate", relevance_score=1.0),
                {"total_tokens": 0})

    async def fake_extract(**kwargs):
        return (ExtractionOutput(predictions=[_claim()] if predictions else [],
                                 article_card=article_card), {"total_tokens": 0})

    async def fake_derive(question, criteria, **kw):
        derive_calls.append(question)
        return card

    monkeypatch.setattr(forecaster, "check_is_prediction", fake_gate)
    monkeypatch.setattr(forecaster, "extract_predictions", fake_extract)
    monkeypatch.setattr(forecaster, "derive_subject_card", fake_derive)
    return derive_calls


@pytest.fixture(autouse=True)
def _gate_settings(monkeypatch, tmp_path):
    # Every test asks the same question with the same article set, and the response
    # cache is keyed on exactly that — clear it so each test exercises the gate.
    forecaster.forecast_cache.clear()
    monkeypatch.setattr(api_settings, "subject_gate_enabled", True)
    monkeypatch.setattr(api_settings, "subject_gate_enforce", False)
    monkeypatch.setattr(api_settings, "subject_gate_cache_enabled", True)
    monkeypatch.setattr(api_settings, "subject_gate_cache_path", tmp_path / "subject_cards")


async def _run(n_articles: int = 2):
    return await forecaster.run_forecast(ForecastRequest(
        question="Benny Gantz's party will run in the 2026 elections without an electoral alliance.",
        resolution_criteria="Resolves Yes if Benny Gantz's party registers without a joint list.",
        prediction_id="clx_gantz",
        debug=True,
        articles=[
            ArticleInput(
                url=f"https://source.example.test/story-{i}",
                title="Fixture headline", snippet="Fixture snippet long enough to be usable.",
                source="source", published_date=_FRESH, text=_BODY,
            )
            for i in range(n_articles)
        ],
    ))


def _outcomes(resp) -> list[str]:
    return [a.outcome for a in resp.debug.per_article]


class TestShadowDefault:
    def test_ships_in_shadow(self):
        assert api_settings.subject_gate_enabled is True
        assert api_settings.subject_gate_enforce is False, "enforce is a post-shadow decision"
        assert api_settings.subject_gate_trust_gloss is False

    async def test_fired_gate_is_logged_but_nothing_dropped(self, monkeypatch, caplog):
        derive_calls = _patch(monkeypatch, card=GANTZ)
        with caplog.at_level(logging.INFO, logger="forecast_api.forecaster"):
            resp = await _run()
        assert _outcomes(resp) == ["ok", "ok"]
        assert len(derive_calls) == 1, "one derivation per request, not per article"
        lines = [r.message for r in caplog.records if "event=subject_gate " in r.message]
        assert len(lines) == 2
        assert all("fired=True enforce=False matched_via=none" in l for l in lines)
        assert all("prediction_id=clx_gantz" in l and "url=https://source.example.test/story-" in l for l in lines)
        assert "verified_spans=['Eisenkot', 'Yoaz Hendel']" in lines[0]

    async def test_second_request_reuses_cached_card(self, monkeypatch):
        derive_calls = _patch(monkeypatch, card=GANTZ)
        await _run(1)
        forecaster.forecast_cache.clear()  # force the second request through the pipeline
        await _run(2)
        assert len(derive_calls) == 1

    async def test_control_passes(self, monkeypatch, caplog):
        _patch(monkeypatch, card=HENDEL)
        with caplog.at_level(logging.INFO, logger="forecast_api.forecaster"):
            resp = await _run(1)
        assert _outcomes(resp) == ["ok"]
        line = next(r.message for r in caplog.records if "event=subject_gate " in r.message)
        assert "fired=False" in line and "matched_via=surface_form" in line and "matched_actor='Yoaz Hendel'" in line


class TestEnforce:
    async def test_fired_gate_drops_article_with_subject_absent(self, monkeypatch):
        monkeypatch.setattr(api_settings, "subject_gate_enforce", True)
        _patch(monkeypatch, card=GANTZ)
        resp = await _run()
        assert _outcomes(resp) == ["subject_absent", "subject_absent"]
        assert resp.debug.per_article[0].gate_passed is True
        assert resp.sources == []

    async def test_control_still_passes_under_enforce(self, monkeypatch):
        monkeypatch.setattr(api_settings, "subject_gate_enforce", True)
        _patch(monkeypatch, card=HENDEL)
        resp = await _run(1)
        assert _outcomes(resp) == ["ok"]

    async def test_empty_extraction_keeps_no_predictions_outcome(self, monkeypatch):
        monkeypatch.setattr(api_settings, "subject_gate_enforce", True)
        _patch(monkeypatch, card=GANTZ, predictions=False)
        resp = await _run(1)
        assert _outcomes(resp) == ["no_predictions"]


class TestFailOpen:
    @pytest.mark.parametrize("enforce", [False, True])
    async def test_derivation_failure_skips_gate(self, monkeypatch, enforce, caplog):
        monkeypatch.setattr(api_settings, "subject_gate_enforce", enforce)
        _patch(monkeypatch, card=None)
        with caplog.at_level(logging.INFO, logger="forecast_api.forecaster"):
            resp = await _run(1)
        assert _outcomes(resp) == ["ok"]
        assert not any("event=subject_gate " in r.message for r in caplog.records)

    async def test_no_article_card_skips_gate_under_enforce(self, monkeypatch, caplog):
        monkeypatch.setattr(api_settings, "subject_gate_enforce", True)
        _patch(monkeypatch, card=GANTZ, article_card=None)
        with caplog.at_level(logging.INFO, logger="forecast_api.forecaster"):
            resp = await _run(1)
        assert _outcomes(resp) == ["ok"]
        line = next(r.message for r in caplog.records if "event=subject_gate " in r.message)
        assert "fired=False" in line and "skip=no_article_card" in line

    async def test_empty_subject_card_skips_gate_under_enforce(self, monkeypatch):
        monkeypatch.setattr(api_settings, "subject_gate_enforce", True)
        _patch(monkeypatch, card=SubjectCard(actors=[]))
        resp = await _run(1)
        assert _outcomes(resp) == ["ok"]

    async def test_disabled_never_derives(self, monkeypatch):
        monkeypatch.setattr(api_settings, "subject_gate_enabled", False)
        monkeypatch.setattr(api_settings, "subject_gate_enforce", True)
        derive_calls = _patch(monkeypatch, card=GANTZ)
        resp = await _run(1)
        assert _outcomes(resp) == ["ok"] and derive_calls == []

    async def test_gate_exception_is_swallowed(self, monkeypatch, caplog):
        monkeypatch.setattr(api_settings, "subject_gate_enforce", True)
        _patch(monkeypatch, card=GANTZ)

        def boom(*a, **kw):
            raise RuntimeError("gate bug")
        monkeypatch.setattr(forecaster, "evaluate_subject_gate", boom)
        with caplog.at_level(logging.WARNING, logger="forecast_api.forecaster"):
            resp = await _run(1)
        assert _outcomes(resp) == ["ok"]
        assert any("event=subject_gate_error" in r.message for r in caplog.records)
