"""retro#704 — `claim_created_at` reaches the extractor's settlement enforcement.

The rule itself is tested in `pipeline/tests/test_settlement_event_date.py`. What
this file pins is the *plumbing*, which is where it was broken: the value existed
on `ForecastRequest`, was threaded into `aggregate_pool` for vote-time
revalidation, and never reached `_process_article` at all — so the extractor kept
writing `settled=true` on rows the pooling layer would discount. A dropped kwarg
somewhere in the two wrappers restores that silently, with every unit test green.
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from forecast_api import forecaster
from tm.models import GatekeeperOutput, PredictionExtraction
from tm.web_search import SearchResult

_QUESTION = "Will Netanyahu win the 2026 Israeli general election?"
_CREATED = "2026-04-18T09:31:00+00:00"


def _sr():
    return SearchResult(
        title="Netanyahu's bloc won the general elections with a clear majority",
        url="https://chinadailyhk.example/hk/article/298215",
        snippet="The right-wing bloc won the general elections with a clear majority.",
        source="chinadailyhk.example", published_date="2022-11-04",
    )


async def _run(monkeypatch, *, claim_created_at):
    """One article, one settled claim dated 2022-11-01 — the retro#704 shape."""
    monkeypatch.setattr(forecaster, "check_is_prediction", AsyncMock(return_value=(
        GatekeeperOutput(is_prediction=True, reason="judged", relevance_score=0.9), {},
    )))
    monkeypatch.setattr(forecaster, "extract_predictions", AsyncMock(return_value=(
        SimpleNamespace(
            predictions=[PredictionExtraction(
                quote="won the general elections with a clear majority",
                claim="Netanyahu's right-wing bloc won the general election",
                stance=1.0, certainty=0.95, specificity=1.0,
                settled=True, event_date="2022-11-01",
            )],
            author_lean=None, author_lean_certainty=None,
        ), {},
    )))
    monkeypatch.setattr(forecaster, "enforce_deadline_arithmetic", lambda p, dl, d: p)
    monkeypatch.setattr(forecaster, "_fetch_article_text",
                        Mock(return_value="A long article body about the election result. " * 4))
    out = await forecaster._process_article_bounded(
        _sr(), _QUESTION, max_article_chars=4000, timings=[], article_debugs=[],
        timeout_s=30.0, claim_deadline="2026-12-31", claim_created_at=claim_created_at,
    )
    assert out is not None, "the article should still be evidence, only not a settlement"
    return out[2]


@pytest.mark.asyncio
class TestClaimWindowReachesTheExtractor:
    async def test_a_pre_creation_event_does_not_settle(self, monkeypatch):
        preds = await _run(monkeypatch, claim_created_at=_CREATED)
        assert [p.settled for p in preds] == [False]

    async def test_the_demoted_claim_still_votes(self, monkeypatch):
        """Demotion clears the settlement bit only — the article is still evidence,
        which is why this is safe to apply at extraction time."""
        preds = await _run(monkeypatch, claim_created_at=_CREATED)
        assert preds[0].stance == 1.0
        assert preds[0].claim_strength == 0.95

    async def test_without_the_value_the_old_behaviour_stands(self, monkeypatch):
        """Fail-open, matching aggregation — and the control that proves the test
        above is measuring the plumbing and not something else."""
        preds = await _run(monkeypatch, claim_created_at=None)
        assert [p.settled for p in preds] == [True]
