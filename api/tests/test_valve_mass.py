"""retro#397 — the abstention/CI valves read un-floored recency-weighted mass.

system-model §6.1 states the intended consequence of recency decay: as a pool's
evidence ages its total mass shrinks and the thin-evidence valve widens the CI
toward "we barely know", until the system declares ignorance rather than
whispering its last headline forever. Two things stopped that happening, both
in ``aggregate_pool``:

1. the abstention gate reads *relevance* mass, which time never touches; and
2. ``recency_floor`` (0.02) is applied to the per-source weight and the valves
   reused that floored mass — so 50 fully-decayed rows still sum to 1.0 and a
   large enough pool clears ``decisiveness_floor`` indefinitely.

The floor is right for *voting* weight (an old row's influence should not go to
exactly zero) and wrong for the valves. ``valve_weights`` carries the un-floored
twin; ``PoolAggregateResult.valve_mass`` reports what the valves actually read.

The R8 matrix pins the published consequence (C18, and the movement on B12/C6).
What lives here is the mechanism itself plus the glide carve-out, which needs
``defer_on_thin_evidence`` — off at the shipped config, so unreachable from a
fixture.
"""
from __future__ import annotations

import pytest

from forecast_api.aggregation import aggregate_pool
from forecast_api.config import settings as api_settings


def _kwargs(**overrides):
    kw = dict(
        relevance_weight_floor=api_settings.relevance_weight_floor,
        decisiveness_floor=api_settings.decisiveness_floor,
        thin_evidence_ci_inflation=api_settings.thin_evidence_ci_inflation,
        defer_on_thin_evidence=False,
        settlement_min_sources=2,
        settlement_stance=api_settings.settlement_stance,
        logit_clamp=api_settings.logit_clamp,
        pool_dispersion_floor=api_settings.pool_dispersion_floor,
        settlement_revalidate=True,
    )
    kw.update(overrides)
    return kw


def _agg(weights, valve_weights=None, stances=None, **overrides):
    n = len(weights)
    return aggregate_pool(
        stances if stances is not None else [0.6] * n,
        weights, [1.0] * n, [False] * n,
        valve_weights=valve_weights,
        **_kwargs(**overrides),
    )


class TestValveMassIsSeparateFromVotingMass:
    def test_absent_valve_weights_fall_back_to_the_voting_mass(self):
        """The compatibility contract: a caller that has not been taught to
        compute the un-floored twin gets exactly today's behaviour, not a
        silently stricter rule."""
        agg = _agg([0.3, 0.3])
        assert agg.valve_mass == pytest.approx(agg.evidence_mass)
        assert agg.valve_mass == pytest.approx(0.6)

    def test_floor_propped_mass_clears_the_floor_but_the_valve_mass_does_not(self):
        # The issue's own shape: 50 rows decayed to the 0.02 recency floor sum
        # to 1.0 and clear decisiveness_floor (0.5) forever, while the real
        # recency-weighted mass is ~nothing.
        floored = [0.02] * 50
        unfloored = [1e-9] * 50
        agg = _agg(floored, valve_weights=unfloored)
        assert agg.evidence_mass == pytest.approx(1.0)
        assert agg.valve_mass == pytest.approx(5e-8)
        assert agg.thin_evidence is True, "the pool must now read as thin"

    def test_the_same_pool_without_valve_weights_reads_as_decisive(self):
        """The before-picture, pinned so the change cannot be mistaken for a
        pre-existing behaviour."""
        agg = _agg([0.02] * 50)
        assert agg.thin_evidence is False

    def test_voting_weights_are_untouched_so_the_mean_does_not_move(self):
        with_valve = _agg([0.02] * 50, valve_weights=[1e-9] * 50)
        without = _agg([0.02] * 50)
        assert with_valve.mean == pytest.approx(without.mean)
        # ...but the interval is wider, which is the whole point.
        assert (with_valve.ci_high - with_valve.ci_low) > (without.ci_high - without.ci_low)

    def test_a_fresh_pool_is_unaffected(self):
        """Recency at age 0 is 1.0 either way, so the floor never bound and
        nothing about a current pool changes."""
        agg = _agg([0.4, 0.4], valve_weights=[0.4, 0.4])
        assert agg.thin_evidence is False
        assert agg.valve_mass == pytest.approx(agg.evidence_mass)


class TestAgingAbstention:
    def test_a_decayed_pool_abstains_once_deferral_is_enabled(self):
        agg = aggregate_pool(
            [0.6] * 50, [0.02] * 50, [1.0] * 50, [False] * 50,
            valve_weights=[1e-9] * 50,
            **_kwargs(defer_on_thin_evidence=True),
        )
        assert agg.insufficient_reason == "no_decisive_signal"

    def test_the_same_pool_does_not_abstain_on_floored_mass(self):
        agg = aggregate_pool(
            [0.6] * 50, [0.02] * 50, [1.0] * 50, [False] * 50,
            **_kwargs(defer_on_thin_evidence=True),
        )
        assert agg.insufficient_reason is None


class TestGlideCarveOut:
    """§6.1's one carve-out, load-bearing against §6.2: on a glide-eligible
    question decayed mass widens the CI but never aborts an ACTIVE glide into
    abstention. The glide is the deadline clock pricing the silence, and it
    converges on the boundary the impossibility pin will declare from metadata
    alone — so falling silent mid-glide would discard information, not withhold
    a guess."""

    def _decayed(self, **overrides):
        return aggregate_pool(
            [0.6] * 50, [0.02] * 50, [1.0] * 50, [False] * 50,
            valve_weights=[1e-9] * 50,
            **_kwargs(defer_on_thin_evidence=True, **overrides),
        )

    def test_an_active_glide_is_not_aborted_into_abstention(self):
        agg = self._decayed(claim_deadline="2099-12-31")
        assert agg.insufficient_reason is None
        # The CI still widens — the carve-out suppresses the abstention, not
        # the thinness that justified it.
        assert agg.thin_evidence is True
        assert (agg.ci_high - agg.ci_low) > 1.2

    def test_a_passed_deadline_leaves_no_glide_to_protect(self):
        agg = self._decayed(claim_deadline="2020-01-01")
        assert agg.insufficient_reason == "no_decisive_signal"

    def test_a_question_with_no_deadline_is_not_glide_eligible(self):
        agg = self._decayed()
        assert agg.insufficient_reason == "no_decisive_signal"

    def test_the_carve_out_does_not_rescue_an_off_topic_pool(self):
        """Abstention outranks a glide in its §6.2 sense — relevance mass ≈ 0
        means no valid anchor ever existed, or the pool was killed. That gate
        reads relevance, which the carve-out deliberately does not touch."""
        agg = aggregate_pool(
            [0.6, 0.6], [0.4, 0.4], [0.1, 0.1], [False, False],
            valve_weights=[1e-9, 1e-9],
            **_kwargs(defer_on_thin_evidence=True, claim_deadline="2099-12-31"),
        )
        assert agg.insufficient_reason == "all_articles_off_topic"
