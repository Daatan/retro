"""retro#590 — `claim_archetype` reaches `audit_scheduled_deadline_unconfirmed`.

Same shape as `test_claim_window_reaches_extractor.py` (retro#704): the rule itself
is tested in `pipeline/tests/test_scheduled_deadline_audit.py`. What this file pins
is the *plumbing* through the two wrappers (`_process_article_bounded` ->
`_process_article` -> the audit call) — a dropped kwarg anywhere in that chain would
leave every unit test on the rule itself green while the live path never sees it.
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from forecast_api import forecaster
from tm.models import GatekeeperOutput, PredictionExtraction
from tm.web_search import SearchResult

_QUESTION = "Will the referendum pass by the scheduled date?"


def _sr():
    return SearchResult(
        title="Coverage of the lead-up to the vote",
        url="https://example.com/referendum-coverage",
        snippet="Analysts weigh in ahead of the vote.",
        source="example.com", published_date="2026-08-15",
    )


async def _run(monkeypatch, *, claim_archetype):
    monkeypatch.setattr(forecaster, "check_is_prediction", AsyncMock(return_value=(
        GatekeeperOutput(is_prediction=True, reason="judged", relevance_score=0.9), {},
    )))
    monkeypatch.setattr(forecaster, "extract_predictions", AsyncMock(return_value=(
        SimpleNamespace(
            predictions=[PredictionExtraction(
                quote="analysts weigh in ahead of the vote",
                claim="The referendum will pass",
                stance=0.8, certainty=0.8, specificity=1.0,
            )],
            author_lean=None, author_lean_certainty=None, consensus_view=None,
            claim_actor=None, claim_predicate=None, claim_scope=None,
        ), {},
    )))
    monkeypatch.setattr(forecaster, "_fetch_article_text",
                        Mock(return_value="A long article body about the referendum. " * 4))
    return await forecaster._process_article_bounded(
        _sr(), _QUESTION, max_article_chars=4000, timings=[], article_debugs=[],
        timeout_s=30.0, claim_deadline="2026-08-01", claim_archetype=claim_archetype,
    )


@pytest.mark.asyncio
class TestClaimArchetypeReachesTheExtractor:
    async def test_scheduled_archetype_past_deadline_fires_the_shadow_audit(self, monkeypatch, caplog):
        with caplog.at_level("WARNING", logger="tm.extractor"):
            out = await _run(monkeypatch, claim_archetype="scheduled")
        assert out is not None
        assert any(
            "event=scheduled_deadline_unconfirmed " in r.message for r in caplog.records
        )

    async def test_missing_archetype_does_not_fire(self, monkeypatch, caplog):
        """Fail-open control — proves the test above is measuring the plumbing
        for claim_archetype specifically, not some other trigger."""
        with caplog.at_level("WARNING", logger="tm.extractor"):
            out = await _run(monkeypatch, claim_archetype=None)
        assert out is not None
        assert not any(
            "event=scheduled_deadline_unconfirmed " in r.message for r in caplog.records
        )

    async def test_the_flagged_claim_still_votes_unmutated(self, monkeypatch):
        """Shadow-only: the audit must never touch stance/claim_strength/settled."""
        out = await _run(monkeypatch, claim_archetype="scheduled")
        preds = out[2]
        assert preds[0].stance == 0.8
        assert preds[0].claim_strength == 0.8
        assert preds[0].settled is None
