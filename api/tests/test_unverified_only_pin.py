"""retro#449 Stage B — the `event=unverified_only_pin` detector.

Stage A measured this shape at zero live instances (2026-08-12: 987 settlement
votes over 176 forecasts), which is why it ships as a detector and not a guard.
These tests pin the detector's *selectivity* rather than a threshold: it must
fire on the fixture-B21 shape and stay silent on everything a real pin looks
like today, or the first prod instance will be lost in noise.
"""

from forecast_api.aggregation import PoolAggregateResult
from forecast_api.forecaster import unverified_only_pin_votes


def _agg(settled: bool, vote_indices: tuple) -> PoolAggregateResult:
    return PoolAggregateResult(
        mean=0.94, std=0.1, ci_low=0.82, ci_high=0.99, settled=settled, n=2,
        n_eff=2.0, evidence_mass=0.656, thin_evidence=False, valve_mass=0.656,
        age_adjusted_mass=0.656, insufficient_reason=None,
        settled_sources=len(vote_indices), settlement_suppressed=False,
        settlement_vote_indices=vote_indices,
    )


def _verified(mapping: dict):
    return lambda i: mapping.get(i)


def test_fires_when_every_winning_vote_is_unverified():
    """B21's shape: two verified=null votes carry the pin."""
    votes = unverified_only_pin_votes(_agg(True, (0, 1)), _verified({0: None, 1: None}))
    assert votes == (0, 1)


def test_silent_when_one_winning_vote_is_verified():
    """One corroborated vote is the ordinary case — 25 of the 176 measured
    forecasts had a material unverified mass AND a verified vote. Firing on
    those would bury the signal this exists to catch."""
    assert unverified_only_pin_votes(_agg(True, (0, 1)), _verified({0: None, 1: True})) == ()


def test_silent_when_a_winning_vote_is_explicitly_unverified():
    """verified=False is F20's territory — those claims are clamped below the
    settlement grade gate and never reached the pin at all in prod (0 of 987)."""
    assert unverified_only_pin_votes(_agg(True, (0, 1)), _verified({0: None, 1: False})) == ()


def test_silent_when_no_pin_fired():
    assert unverified_only_pin_votes(_agg(False, ()), _verified({0: None})) == ()


def test_silent_when_pin_fired_but_votes_were_not_recorded():
    """The legacy non-revalidation path leaves `settlement_vote_indices` empty.
    The detector reads the recorded winning direction and never re-derives one,
    so it stays quiet rather than guessing which rows carried the pin."""
    assert unverified_only_pin_votes(_agg(True, ()), _verified({0: None})) == ()


def test_does_not_consult_rows_outside_the_winning_direction():
    """Only the winning direction's votes decide. A losing-direction row (index
    2 here) is never queried — asserted by making the lookup raise on it."""
    def verified_for_index(i: int):
        assert i in (0, 1), f"detector consulted non-winning row {i}"
        return None

    assert unverified_only_pin_votes(_agg(True, (0, 1)), verified_for_index) == (0, 1)
