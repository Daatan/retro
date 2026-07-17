"""Tests for run_pool_aggregate() / POST /pool/aggregate (retro
docs/ORACLE_VARIABLES.md, recompute-over-pool). No search, no LLM — given a
caller-supplied set of already-extracted per-source signals, it must
reproduce exactly what a fresh /forecast run's pooling math would produce
for the same evidence (see aggregate_pool() in aggregation.py, shared by
both). Tested against the async function directly, matching this suite's
convention for authed business logic (see test_evidence_class.py /
test_settlement_hardening.py) rather than through TestClient.
"""

from __future__ import annotations

import pytest

from forecast_api import forecaster
from forecast_api.config import settings as api_settings
from forecast_api.models import PoolAggregateRequest, PoolSourceInput


def _prob(stance: float) -> float:
    return (stance + 1.0) / 2.0


def _source(**over) -> PoolSourceInput:
    return PoolSourceInput(**{
        "stance": 0.5,
        "certainty": 0.6,
        "credibility_weight": 1.0,
        "relevance_score": 1.0,
        "evidence_weight": 0.6,
        "published_date": "2026-07-10",
        "settled": False,
        **over,
    })


class TestEmptyAndInsufficient:
    async def test_no_sources_is_insufficient(self):
        resp = await forecaster.run_pool_aggregate(PoolAggregateRequest(sources=[]))
        assert resp.insufficient_data is True
        assert resp.reason == "no_sources"
        assert resp.articles_used == 0

    async def test_all_off_topic_is_insufficient(self):
        resp = await forecaster.run_pool_aggregate(PoolAggregateRequest(
            sources=[_source(relevance_score=0.05), _source(relevance_score=0.05)],
        ))
        assert resp.insufficient_data is True
        assert resp.reason == "all_articles_off_topic"


class TestBasicPooling:
    async def test_pools_a_single_source_toward_its_own_stance(self):
        resp = await forecaster.run_pool_aggregate(PoolAggregateRequest(
            sources=[_source(stance=0.8, evidence_weight=1.0, published_date=None)],
        ))
        assert resp.insufficient_data is False
        assert resp.articles_used == 1
        assert _prob(resp.mean) > 0.85  # near-boundary evidence pools decisively

    async def test_evidence_weight_none_falls_back_to_certainty(self):
        with_weight = await forecaster.run_pool_aggregate(PoolAggregateRequest(
            sources=[_source(evidence_weight=0.9, certainty=0.9, published_date=None)],
        ))
        # Same numeric value via the certainty fallback (evidence_weight unset).
        via_fallback = await forecaster.run_pool_aggregate(PoolAggregateRequest(
            sources=[_source(evidence_weight=None, certainty=0.9, published_date=None)],
        ))
        assert with_weight.mean == pytest.approx(via_fallback.mean)

    async def test_relevance_squared_down_weights_a_tangential_source(self):
        # Two sources disagree; the low-relevance one should barely move the
        # pool off the high-relevance one's stance.
        resp = await forecaster.run_pool_aggregate(PoolAggregateRequest(
            sources=[
                _source(stance=0.9, relevance_score=1.0, evidence_weight=1.0, published_date=None),
                _source(stance=-0.9, relevance_score=0.2, evidence_weight=1.0, published_date=None),
            ],
        ))
        assert resp.mean > 0.5  # dominated by the on-topic source


class TestRecency:
    async def test_older_article_pulls_less_than_a_fresh_one(self):
        # Two disagreeing sources, one dated far in the past (relative to
        # "now", recomputed fresh by run_pool_aggregate — not a stored value).
        resp = await forecaster.run_pool_aggregate(PoolAggregateRequest(
            sources=[
                _source(stance=0.9, published_date="2026-01-01", evidence_weight=1.0),
                _source(stance=-0.9, published_date=None, evidence_weight=1.0),  # no date -> neutral weight
            ],
        ))
        # The undated (neutral-weight) dissenter outweighs the stale one.
        assert resp.mean < 0.5


class TestSettlement:
    async def test_settlement_override_pins_through_the_endpoint(self):
        # Dated anchors: the endpoint runs with settlement_revalidate on (the
        # prod default), so an undated positive vote no longer counts — see
        # test_settlement_revalidation.py for the demotion cases.
        resp = await forecaster.run_pool_aggregate(PoolAggregateRequest(
            sources=[
                _source(stance=0.95, settled=True, settlement_event_date="2026-07-09"),
                _source(stance=0.9, settled=True, settlement_event_date="2026-07-09"),
            ],
        ))
        assert resp.settled is True
        assert resp.mean == pytest.approx(api_settings.settlement_stance)

    async def test_undated_positive_votes_are_demoted_through_the_endpoint(self):
        resp = await forecaster.run_pool_aggregate(PoolAggregateRequest(
            sources=[
                _source(stance=0.95, settled=True, published_date=None),
                _source(stance=0.9, settled=True, published_date=None),
            ],
        ))
        assert resp.settled is False
        assert resp.settlement_votes_demoted == 2

    async def test_early_undated_negative_settlement_is_still_suppressed(self):
        # Same observable as the old pin-level direction guard, new mechanism:
        # each undated pre-deadline negative is demoted per-vote
        # (undated_foreclosure) instead of the pin being blocked wholesale.
        resp = await forecaster.run_pool_aggregate(PoolAggregateRequest(
            sources=[
                _source(stance=-0.95, settled=True, published_date=None),
                _source(stance=-0.9, settled=True, published_date=None),
            ],
            claim_direction="arrival",
            claim_deadline="2099-01-01",
        ))
        assert resp.settled is False
        assert resp.settlement_votes_demoted == 2
