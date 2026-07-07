"""Unit tests for :mod:`forecast_api.aggregation`.

These are pure-function tests — no LLM, no network. They lock in the behaviour
that fixes the "73% on a decided series" dilution bug:
  * logit pooling extremizes toward consensus (vs the old arithmetic mean),
  * recency weighting decays older articles,
  * within-article claim aggregation weights decisive claims, and
  * a lone weak dissenter cannot drag a confident forecast back to the middle.
"""
from __future__ import annotations

import pytest

from forecast_api.aggregation import (
    claim_weighted_stance,
    logit,
    pool_sources,
    prob_to_stance,
    quantitative_anchor_multiplier,
    recency_weight,
    sigmoid,
    stance_to_prob,
    weighted_mean,
    widen_ci_for_thin_evidence,
)


class TestWidenCiForThinEvidence:
    def test_noop_when_no_deficit(self):
        out = widen_ci_for_thin_evidence(0.2, 0.1, 0.3, 0.05, deficit=0.0, max_inflation=0.45)
        assert out == (0.1, 0.3, 0.05)

    def test_noop_when_inflation_zero(self):
        out = widen_ci_for_thin_evidence(0.2, 0.1, 0.3, 0.05, deficit=1.0, max_inflation=0.0)
        assert out == (0.1, 0.3, 0.05)

    def test_widens_band_and_brackets_mean(self):
        ci_low, ci_high, std = widen_ci_for_thin_evidence(
            0.2, 0.15, 0.25, 0.05, deficit=0.5, max_inflation=0.45
        )
        assert ci_low < 0.15
        assert ci_high > 0.25
        assert ci_low <= 0.2 <= ci_high
        assert std >= 0.05

    def test_full_deficit_spans_nearly_everything(self):
        ci_low, ci_high, _std = widen_ci_for_thin_evidence(
            0.0, -0.1, 0.1, 0.1, deficit=1.0, max_inflation=0.45
        )
        assert (ci_high - ci_low) > 1.5

    def test_widening_grows_monotonically_with_deficit(self):
        widths = []
        for d in (0.1, 0.4, 0.8):
            lo, hi, _ = widen_ci_for_thin_evidence(0.0, -0.1, 0.1, 0.1, deficit=d, max_inflation=0.45)
            widths.append(hi - lo)
        assert widths[0] < widths[1] < widths[2]


def _prob(mean_stance: float) -> float:
    """Convert a pooled stance back to the displayed probability."""
    return (mean_stance + 1.0) / 2.0


class TestLogitSigmoidRoundTrip:
    @pytest.mark.parametrize("p", [0.01, 0.1, 0.5, 0.73, 0.9, 0.99])
    def test_round_trip(self, p):
        assert sigmoid(logit(p)) == pytest.approx(p, abs=1e-9)

    def test_stance_prob_round_trip(self):
        for s in (-1.0, -0.5, 0.0, 0.4521, 1.0):
            assert prob_to_stance(stance_to_prob(s)) == pytest.approx(s, abs=1e-12)

    def test_sigmoid_stable_for_large_inputs(self):
        assert sigmoid(1000) == pytest.approx(1.0, abs=1e-9)
        assert sigmoid(-1000) == pytest.approx(0.0, abs=1e-9)


class TestRecencyWeight:
    def test_today_is_full_weight(self):
        assert recency_weight("2026-06-15", "2026-06-15", 7.0) == pytest.approx(1.0)

    def test_one_half_life_halves_weight(self):
        assert recency_weight("2026-06-08", "2026-06-15", 7.0) == pytest.approx(0.5, abs=1e-9)

    def test_old_article_decays_toward_floor(self):
        # 30 days at 7d half-life ≈ 0.0517, above the 0.02 floor
        w = recency_weight("2026-05-16", "2026-06-15", 7.0, floor=0.02)
        assert w == pytest.approx(0.5 ** (30 / 7), abs=1e-6)

    def test_floor_is_respected(self):
        # 120 days is well past where decay < floor
        assert recency_weight("2026-02-15", "2026-06-15", 7.0, floor=0.02) == 0.02

    def test_missing_date_is_neutral(self):
        assert recency_weight(None, "2026-06-15", 7.0) == 1.0
        assert recency_weight("2026-06-15", None, 7.0) == 1.0
        assert recency_weight("not-a-date", "2026-06-15", 7.0) == 1.0

    def test_future_article_is_not_boosted(self):
        # age is floored at 0, so a future date is treated as "today"
        assert recency_weight("2026-07-01", "2026-06-15", 7.0) == pytest.approx(1.0)


class TestClaimWeightedStance:
    def test_decisive_high_certainty_claim_dominates(self):
        # One decisive claim (stance +1, certainty 1) + three tangential hedged
        # claims (stance 0.2, certainty 0.1). Flat mean ≈ 0.4; certainty-weighted
        # should sit much closer to the decisive claim.
        stances = [1.0, 0.2, 0.2, 0.2]
        certainties = [1.0, 0.1, 0.1, 0.1]
        flat = sum(stances) / len(stances)
        weighted = claim_weighted_stance(stances, certainties)
        assert weighted > flat
        assert weighted > 0.7

    def test_falls_back_to_plain_mean_when_all_certainty_zero(self):
        assert claim_weighted_stance([0.2, 0.8], [0.0, 0.0]) == pytest.approx(0.5)

    def test_specificity_multiplies_weight(self):
        # Equal certainty, but the +1 claim is far more specific → pulls up.
        stances = [1.0, -1.0]
        certainties = [1.0, 1.0]
        specificities = [1.0, 0.0]
        assert claim_weighted_stance(stances, certainties, specificities) == pytest.approx(1.0)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            claim_weighted_stance([], [])


class TestPoolSources:
    def test_more_decisive_than_arithmetic_for_one_sided_consensus(self):
        # 5 sources ~80% YES (stance 0.6), 1 says 40% (stance -0.2), equal weights.
        stances = [0.6, 0.6, 0.6, 0.6, 0.6, -0.2]
        weights = [1.0] * 6
        arithmetic = weighted_mean(stances, weights)  # the OLD behaviour
        mean, std, ci_low, ci_high = pool_sources(stances, weights)
        # logit pool is at least as decisive as the arithmetic mean and stays
        # bounded by the inputs (it can't be flipped by, nor overshoot, members).
        assert _prob(mean) >= _prob(arithmetic)
        assert 0.4 < _prob(mean) <= 0.8 + 1e-9  # within [min, max] member probs
        assert ci_low <= mean <= ci_high

    def test_lone_weak_dissenter_cannot_flip_confident_consensus(self):
        # Five confident YES sources + one weak NO source.
        stances = [0.9, 0.9, 0.9, 0.9, 0.9, -0.9]
        weights = [1.0, 1.0, 1.0, 1.0, 1.0, 0.1]
        mean, _std, _lo, _hi = pool_sources(stances, weights)
        assert _prob(mean) > 0.9

    def test_unanimous_high_certainty_is_near_certain(self):
        stances = [0.95, 0.97, 0.96]
        weights = [1.0, 1.0, 1.0]
        mean, _std, ci_low, ci_high = pool_sources(stances, weights, clamp_eps=0.01)
        assert _prob(mean) > 0.95
        assert ci_high <= 1.0 and ci_low >= -1.0

    def test_single_source(self):
        mean, std, ci_low, ci_high = pool_sources([0.8], [2.0])
        assert _prob(mean) == pytest.approx(0.9, abs=1e-6)
        assert std == pytest.approx(0.0, abs=1e-9)

    def test_zero_total_weight_falls_back_to_equal(self):
        mean, _std, _lo, _hi = pool_sources([0.5, -0.5], [0.0, 0.0])
        assert mean == pytest.approx(0.0, abs=1e-9)

    def test_clamp_bounds_output(self):
        mean, _std, ci_low, ci_high = pool_sources([1.0, 1.0], [1.0, 1.0], clamp_eps=0.01)
        # p clamped to 0.99 → stance 0.98
        assert mean <= 0.98 + 1e-9
        assert ci_high <= 0.98 + 1e-9

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            pool_sources([], [])


class TestQuantitativeAnchorMultiplier:
    def test_neutral_when_no_estimates_present(self):
        assert quantitative_anchor_multiplier([None, None], multiplier=4.0) == 1.0

    def test_neutral_on_empty_list(self):
        assert quantitative_anchor_multiplier([], multiplier=4.0) == 1.0

    def test_applies_multiplier_when_one_estimate_present(self):
        assert quantitative_anchor_multiplier([None, 0.1883, None], multiplier=4.0) == 4.0

    def test_applies_multiplier_even_when_estimate_is_zero(self):
        # 0.0 is a valid cited probability (not falsy-None) — must still count.
        assert quantitative_anchor_multiplier([0.0], multiplier=4.0) == 4.0


class TestFranceWorldCupRegression:
    """Reproduces the pooled-overconfidence bug and proves the fix.

    Five match-recap articles frame France as the "favorite"/"strong candidate"
    entering later knockout rounds (stance +0.3 to +0.6, per the multi-stage
    guidance in extractor.py), pooling to ~75% — while one of those very articles
    also cites an explicit Opta model estimate of 18.83% for the tournament
    outcome itself. Without a weight premium, the quantitative source is just one
    of six equal-weight votes and gets outvoted by the qualitative majority.
    """

    # (stance, certainty, quantitative_estimate)
    SOURCES = [
        (0.30, 0.30, None),    # "favorite entering the Round of 16"
        (0.40, 0.50, None),    # "beats Paraguay to reach the quarter-finals"
        (0.46, 0.64, None),    # "strongest candidate to win the title"
        (0.61, 0.60, None),    # "superior head-to-head record"
        (0.43, 0.48, None),    # "historical coincidences" narrative piece
        (-0.62, 0.85, 0.1883),  # cites Opta: 18.83% to win the tournament
    ]

    def _pool(self, *, apply_premium: bool, multiplier: float = 4.0):
        stances = [s for s, _c, _q in self.SOURCES]
        weights = []
        for _s, cert, q in self.SOURCES:
            mult = quantitative_anchor_multiplier([q], multiplier=multiplier) if apply_premium else 1.0
            weights.append(cert * mult)
        return pool_sources(stances, weights)

    def test_without_premium_reproduces_overconfidence(self):
        mean, *_ = self._pool(apply_premium=False)
        # The bug: qualitative volume pools well above the cited 18.83% baseline
        # (roughly 3x higher — the real incident pooled all the way to 75%).
        assert _prob(mean) > 0.55

    def test_premium_pulls_pooled_estimate_toward_the_cited_baseline(self):
        without_mean, *_ = self._pool(apply_premium=False)
        with_mean, *_ = self._pool(apply_premium=True)
        assert _prob(with_mean) < _prob(without_mean)
        # Still not literally 18.83% (qualitative sources retain some pull), but
        # materially closer to the cited baseline than the unpremiumed pool.
        assert _prob(with_mean) < 0.5


class TestKnicksRegression:
    """Reproduces the full two-stage dilution and proves the fix.

    The stored snapshot read ≈73% on a decided series. Two compounding causes:
      1. Within-article flat mean — a decisive "they won" claim averaged with
         tangential hedged claims, so even a confirming source scored only ~0.5.
      2. No recency — stale pre-clinch "anyone's series" articles weighted the
         same as today's "they clinched it" report.

    Layer A (certainty-weighted claims) + Layer B (recency) + Layer C (logit
    pooling) together must lift the probability into the decisive range.

    Each source is modelled as raw per-claim ``(stance, certainty)`` pairs plus a
    credibility and an article date, so both stages of the fix are exercised.
    """

    SOURCES = [
        # (claims=[(stance, certainty)], credibility, article_date)
        ([(0.98, 0.97), (0.30, 0.30), (0.20, 0.25)], 1.2, "2026-06-14"),  # "clinched" + tangents
        ([(0.95, 0.95), (0.10, 0.20)], 1.0, "2026-06-14"),                # verbatim confirmation + aside
        ([(0.90, 0.85)], 0.9, "2026-06-13"),                              # recent, decisive
        ([(0.30, 0.50)], 1.0, "2026-05-20"),                              # stale speculation
        ([(0.20, 0.45)], 0.8, "2026-05-10"),                              # stale
        ([(-0.10, 0.40)], 0.7, "2026-05-01"),                            # stale doubt
    ]
    REF_DATE = "2026-06-15"

    def _old_prob(self) -> float:
        # OLD: per-source flat claim mean, weight = credibility×certainty, no recency.
        stances, weights = [], []
        for claims, cr, _d in self.SOURCES:
            flat = sum(s for s, _c in claims) / len(claims)
            cert = sum(c for _s, c in claims) / len(claims)
            stances.append(flat)
            weights.append(cr * cert)
        return _prob(weighted_mean(stances, weights))

    def _new(self):
        stances, weights = [], []
        for claims, cr, d in self.SOURCES:
            stance = claim_weighted_stance(
                [s for s, _c in claims], [c for _s, c in claims]
            )
            cert = sum(c for _s, c in claims) / len(claims)
            rw = recency_weight(d, self.REF_DATE, 7.0, floor=0.02)
            stances.append(stance)
            weights.append(cr * cert * rw)
        return pool_sources(stances, weights)

    def test_old_path_reproduces_dilution(self):
        assert self._old_prob() < 0.78  # the wishy-washy ~73-78% result

    def test_fix_climbs_into_decisive_range(self):
        mean, _std, ci_low, ci_high = self._new()
        assert _prob(mean) > 0.90
        assert _prob(mean) > self._old_prob() + 0.15  # substantial improvement
        assert ci_low <= mean <= ci_high
