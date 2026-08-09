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

from datetime import date, timedelta

import pytest

from forecast_api import forecaster
from forecast_api.config import settings as api_settings
from forecast_api.models import PoolAggregateRequest, PoolSourceInput


def _prob(stance: float) -> float:
    return (stance + 1.0) / 2.0


# Relative, not hard-coded: see the note in test_settlement.py.
_FRESH = (date.today() - timedelta(days=1)).isoformat()


def _source(**over) -> PoolSourceInput:
    return PoolSourceInput(**{
        "stance": 0.5,
        "certainty": 0.6,
        "credibility_weight": 1.0,
        "relevance_score": 1.0,
        "evidence_weight": 0.6,
        "published_date": _FRESH,
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


class TestZeroWeightPool:
    async def test_all_zero_weight_sources_abstain(self):
        """R3/F14 through the endpoint: every row blocked by credibility, so
        nothing carries weight. The recompute must abstain rather than answer
        from the rows it just valued at zero."""
        resp = await forecaster.run_pool_aggregate(PoolAggregateRequest(
            sources=[
                _source(stance=0.9, credibility_weight=0.0),
                _source(stance=0.7, credibility_weight=0.0),
            ],
        ))
        assert resp.insufficient_data is True
        assert resp.reason == "no_usable_weight"
        assert resp.articles_used == 2

    async def test_legacy_row_without_evidence_weight_is_capped(self):
        """R3/F10 on the recompute path: a pre-S2-cutover row with no stored
        evidence_weight falls back to certainty under the same cap the live path
        applies to an unclassified claim, so the rows we know least about cannot
        weigh most. certainty 0.9 → 0.25, so this pool lands below the
        decisiveness floor and self-reports with a widened CI."""
        resp = await forecaster.run_pool_aggregate(PoolAggregateRequest(
            sources=[_source(stance=0.6, certainty=0.9, evidence_weight=None)],
        ))
        assert resp.insufficient_data is False
        assert (resp.ci_high - resp.ci_low) > 0.5


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


class TestSourceMassCap:
    """cap_source_mass() (retro#458 Phase 1) through the full recompute path
    — run_pool_aggregate() reads PoolSourceInput.source_id (previously
    persisted-and-ignored, see the whitelist comment in run_pool_aggregate)
    purely as a cap_source_mass grouping key. Ships inert at the default
    max_source_share=1.0."""

    async def test_default_ships_inert_even_with_a_dominant_source_id(self):
        sources = [
            _source(stance=0.9, source_id="dominant", evidence_weight=1.0),
            _source(stance=0.9, source_id="dominant", evidence_weight=1.0),
            _source(stance=0.9, source_id="dominant", evidence_weight=1.0),
            _source(stance=0.9, source_id="dominant", evidence_weight=1.0),
            _source(stance=-0.9, source_id="other", evidence_weight=1.0),
        ]
        with_ids = await forecaster.run_pool_aggregate(PoolAggregateRequest(sources=sources))
        without_ids = await forecaster.run_pool_aggregate(PoolAggregateRequest(
            sources=[s.model_copy(update={"source_id": None}) for s in sources],
        ))
        assert with_ids.mean == pytest.approx(without_ids.mean)

    async def test_dominant_source_id_is_capped_once_configured(self, monkeypatch):
        # Same shape as the prod finding (one aggregator dominating the
        # pool): 4 agreeing rows from one source_id vs 1 independent
        # dissenter, equal per-row weight.
        sources = [
            _source(stance=0.9, source_id="dominant", evidence_weight=1.0),
            _source(stance=0.9, source_id="dominant", evidence_weight=1.0),
            _source(stance=0.9, source_id="dominant", evidence_weight=1.0),
            _source(stance=0.9, source_id="dominant", evidence_weight=1.0),
            _source(stance=-0.9, source_id="other", evidence_weight=1.0),
        ]
        uncapped = await forecaster.run_pool_aggregate(PoolAggregateRequest(sources=sources))
        monkeypatch.setattr(api_settings, "max_source_share", 0.3)
        capped = await forecaster.run_pool_aggregate(PoolAggregateRequest(sources=sources))
        assert capped.mean < uncapped.mean

    async def test_rows_without_a_source_id_are_never_capped_together(self, monkeypatch):
        # Legacy/anonymous rows (source_id unset) must not be treated as one
        # dominant "unknown" source just because none of them carry an id.
        monkeypatch.setattr(api_settings, "max_source_share", 0.3)
        sources = [_source(stance=0.9, evidence_weight=1.0) for _ in range(5)]
        resp = await forecaster.run_pool_aggregate(PoolAggregateRequest(sources=sources))
        assert resp.insufficient_data is False
        assert _prob(resp.mean) > 0.85  # unaffected — pools as decisively as with no cap at all


class TestPoolReportingFields:
    """evidence_mass/n_eff/age_adjusted_mass on PoolAggregateResponse
    (retro#458 Phase 2) — reporting-only visibility, not new estimator
    behaviour: mean/std/ci are unaffected by any of this class."""

    async def test_evidence_mass_and_n_eff_populated_on_a_normal_pool(self):
        resp = await forecaster.run_pool_aggregate(PoolAggregateRequest(
            sources=[
                _source(stance=0.6, evidence_weight=1.0),
                _source(stance=0.5, evidence_weight=1.0),
                _source(stance=0.4, evidence_weight=1.0),
            ],
        ))
        assert resp.insufficient_data is False
        assert resp.evidence_mass > 0.0
        # Equal-weight rows: n_eff == articles_used (Kish's ESS is exact here).
        assert resp.n_eff == pytest.approx(resp.articles_used, abs=1e-6)

    async def test_n_eff_shrinks_when_one_row_dominates(self):
        dominated = await forecaster.run_pool_aggregate(PoolAggregateRequest(
            sources=[
                _source(stance=0.6, evidence_weight=1.0),
                _source(stance=0.5, evidence_weight=0.001),
                _source(stance=0.4, evidence_weight=0.001),
            ],
        ))
        assert dominated.insufficient_data is False
        assert dominated.articles_used == 3
        assert dominated.n_eff < 1.5  # near-1, one row carries almost all the mass

    async def test_evidence_mass_and_n_eff_still_reported_on_an_insufficient_pool(self):
        # F14/no_usable_weight: the pool that abstains still had a shape (see
        # aggregation.PoolAggregateResult's docstring) — evidence_mass reads
        # 0.0 (every row was blocked), but n_eff is still Kish's ESS of that
        # same zero-weight vector, not silently dropped.
        resp = await forecaster.run_pool_aggregate(PoolAggregateRequest(
            sources=[
                _source(stance=0.9, credibility_weight=0.0),
                _source(stance=0.7, credibility_weight=0.0),
            ],
        ))
        assert resp.insufficient_data is True
        assert resp.reason == "no_usable_weight"
        assert resp.evidence_mass == 0.0

    async def test_age_adjusted_mass_equals_evidence_mass_with_no_decay_in_play(self):
        # published today: recency_weight's age gap is exactly zero, so
        # switching off decay changes nothing for this pool.
        today = date.today().isoformat()
        resp = await forecaster.run_pool_aggregate(PoolAggregateRequest(
            sources=[
                _source(stance=0.6, evidence_weight=1.0, published_date=today),
                _source(stance=0.4, evidence_weight=1.0, published_date=today),
            ],
        ))
        assert resp.insufficient_data is False
        assert resp.age_adjusted_mass == pytest.approx(resp.evidence_mass, rel=1e-6)

    async def test_age_adjusted_mass_exceeds_evidence_mass_for_a_stale_pool(self):
        # The sanity invariant this phase adds: removing recency decay can
        # only ever raise a pool's mass relative to the recency-discounted
        # sum, never lower it. Exercised with a real decay gap (well past one
        # recency half-life), not a same-day pool where the two trivially tie.
        stale = (date.today() - timedelta(days=250)).isoformat()
        resp = await forecaster.run_pool_aggregate(PoolAggregateRequest(
            sources=[
                _source(stance=0.6, evidence_weight=1.0, published_date=stale),
                _source(stance=0.4, evidence_weight=1.0, published_date=stale),
            ],
        ))
        assert resp.insufficient_data is False
        assert resp.age_adjusted_mass >= resp.evidence_mass
        assert resp.age_adjusted_mass > resp.evidence_mass  # decay actually bit here
