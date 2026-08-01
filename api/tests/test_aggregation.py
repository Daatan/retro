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
    aggregate_pool,
    claim_weighted_stance,
    effective_sample_size,
    evidence_class_weight,
    logit,
    pool_sources,
    prob_to_stance,
    recency_weight,
    resolve_stance_certainty,
    sigmoid,
    stance_to_prob,
    weighted_mean,
    widen_ci_for_thin_evidence,
)
from forecast_api.config import settings as api_settings


def _aggregate_kwargs(**overrides):
    kw = dict(
        relevance_weight_floor=api_settings.relevance_weight_floor,
        decisiveness_floor=api_settings.decisiveness_floor,
        thin_evidence_ci_inflation=api_settings.thin_evidence_ci_inflation,
        defer_on_thin_evidence=False,
        settlement_min_sources=api_settings.settlement_min_sources,
        settlement_stance=api_settings.settlement_stance,
        logit_clamp=api_settings.logit_clamp,
        pool_dispersion_floor=api_settings.pool_dispersion_floor,
    )
    kw.update(overrides)
    return kw


class TestWidenCiForThinEvidence:
    def test_noop_when_no_deficit(self):
        out = widen_ci_for_thin_evidence(0.2, 0.1, 0.3, 0.05, deficit=0.0, max_inflation=0.45, clamp_eps=0.01)
        assert out == (0.1, 0.3, 0.05)

    def test_noop_when_inflation_zero(self):
        out = widen_ci_for_thin_evidence(0.2, 0.1, 0.3, 0.05, deficit=1.0, max_inflation=0.0, clamp_eps=0.01)
        assert out == (0.1, 0.3, 0.05)

    def test_widens_band_and_brackets_mean(self):
        ci_low, ci_high, std = widen_ci_for_thin_evidence(
            0.2, 0.15, 0.25, 0.05, deficit=0.5, max_inflation=0.45, clamp_eps=0.01
        )
        assert ci_low < 0.15
        assert ci_high > 0.25
        assert ci_low <= 0.2 <= ci_high
        assert std >= 0.05

    def test_full_deficit_spans_nearly_everything(self):
        ci_low, ci_high, _std = widen_ci_for_thin_evidence(
            0.0, -0.1, 0.1, 0.1, deficit=1.0, max_inflation=0.45, clamp_eps=0.01
        )
        assert (ci_high - ci_low) > 1.5

    def test_widening_grows_monotonically_with_deficit(self):
        widths = []
        for d in (0.1, 0.4, 0.8):
            lo, hi, _ = widen_ci_for_thin_evidence(0.0, -0.1, 0.1, 0.1, deficit=d, max_inflation=0.45, clamp_eps=0.01)
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

    def test_missing_article_date_decays_to_the_floor(self):
        # R3/F13: an undated article is treated as maximally stale, not maximally
        # fresh. It used to return a neutral 1.0 — the best multiplier the term can
        # produce — so knowing less about a source made it worth more.
        assert recency_weight(None, "2026-06-15", 7.0, floor=0.02) == 0.02
        assert recency_weight("", "2026-06-15", 7.0, floor=0.02) == 0.02
        assert recency_weight("not-a-date", "2026-06-15", 7.0, floor=0.02) == 0.02

    def test_undated_never_outweighs_a_dated_article(self):
        # The invariant behind F13, stated directly: whatever date an article
        # carries, having one can only help it relative to having none.
        undated = recency_weight(None, "2026-06-15", 7.0, floor=0.02)
        for dated in ("2026-06-15", "2026-06-08", "2026-05-16", "2026-02-15"):
            assert recency_weight(dated, "2026-06-15", 7.0, floor=0.02) >= undated

    def test_recency_disabled_stays_neutral_for_everyone(self):
        # A missing REFERENCE date, or a non-positive half-life, means the caller
        # has switched recency off — that is not an article-level absence, and it
        # must not be confused with one.
        assert recency_weight("2026-06-15", None, 7.0) == 1.0
        assert recency_weight(None, None, 7.0) == 1.0
        assert recency_weight("2026-06-08", "2026-06-15", 0.0) == 1.0

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


class TestResolveStanceCertainty:
    def test_unchanged_when_no_estimate(self):
        assert resolve_stance_certainty(
            0.3, 0.5, None, evidence_class="cited_probability"
        ) == (0.3, 0.5)

    def test_overrides_stance_to_match_estimate(self):
        stance, _certainty = resolve_stance_certainty(
            0.3, 0.5, 0.1883, evidence_class="cited_probability"
        )
        assert stance == pytest.approx(prob_to_stance(0.1883))
        assert stance < 0  # correctly flips the misaligned positive stance

    def test_raises_certainty_to_floor(self):
        _stance, certainty = resolve_stance_certainty(
            0.3, 0.5, 0.1883, evidence_class="cited_probability"
        )
        assert certainty == 0.9

    def test_does_not_lower_an_already_high_certainty(self):
        _stance, certainty = resolve_stance_certainty(
            0.3, 0.97, 0.1883, evidence_class="cited_probability"
        )
        assert certainty == 0.97

    def test_estimate_of_zero_still_overrides(self):
        # 0.0 is a valid cited probability (not falsy-None) — must still apply.
        stance, certainty = resolve_stance_certainty(
            0.8, 0.2, 0.0, evidence_class="cited_probability"
        )
        assert stance == pytest.approx(-1.0)
        assert certainty == 0.9

    # retro#362 — the rewrite is a category error for any figure that is not a
    # probability of the event itself, so every class except cited_probability
    # (and unclassified) must pass through untouched.

    def test_cited_share_is_never_rewritten(self):
        # "Likud polls at 22 seats" → qe 0.183: a seat SHARE. 2×0.183−1 = −0.63
        # at certainty 0.9 would be a fabricated probability judgment.
        assert resolve_stance_certainty(
            -0.4, 0.6, 0.183, evidence_class="cited_share"
        ) == (-0.4, 0.6)

    def test_unclassified_is_never_rewritten(self):
        # Missing data must not increase influence: no class → no forced 0.9.
        assert resolve_stance_certainty(0.3, 0.5, 0.1883) == (0.3, 0.5)
        assert resolve_stance_certainty(
            0.3, 0.5, 0.1883, evidence_class=None
        ) == (0.3, 0.5)

    @pytest.mark.parametrize(
        "klass", ["reported_fact", "reporting", "opinion"]
    )
    def test_other_classes_are_never_rewritten(self, klass):
        assert resolve_stance_certainty(
            0.3, 0.5, 0.45, evidence_class=klass
        ) == (0.3, 0.5)


class TestQuantitativeMisalignmentRegression:
    """Reproduces the risk flagged in review of the fix-1/2 PR: the prompt tells
    the LLM to align stance/certainty with a cited quantitative_estimate, but
    nothing enforces it. An extractor that cites Opta's 18.83% correctly but
    leaves a stale positive stance (a plausible LLM inconsistency) would, under
    evidence_class_weight's cited_probability premium, amplify the
    overconfidence bug instead of fixing it — resolve_stance_certainty must
    close that gap.
    """

    # 5 qualitative "favorite" match-recap sources, unchanged from
    # TestFranceWorldCupRegression, plus one MISALIGNED quantitative source
    # (cites 18.83% but the extractor left stance positive).
    QUALITATIVE = [(0.30, 0.30), (0.40, 0.50), (0.46, 0.64), (0.61, 0.60), (0.43, 0.48)]
    MISALIGNED_STANCE, MISALIGNED_CERTAINTY, CITED_ESTIMATE = 0.3, 0.5, 0.1883

    def test_without_the_fix_misalignment_amplifies_overconfidence(self):
        stances = [s for s, _c in self.QUALITATIVE] + [self.MISALIGNED_STANCE]
        # cited_probability's class_weight applied to the misaligned *positive*
        # stance — exactly what evidence_class_weight does today without
        # resolution.
        weights = [c for _s, c in self.QUALITATIVE] + [
            api_settings.evidence_class_weight["cited_probability"],
        ]
        mean, *_ = pool_sources(stances, weights)
        assert _prob(mean) > 0.6  # bug amplified, not fixed

    def test_resolve_stance_certainty_prevents_the_amplification(self):
        resolved_stance, resolved_certainty = resolve_stance_certainty(
            self.MISALIGNED_STANCE, self.MISALIGNED_CERTAINTY, self.CITED_ESTIMATE,
            evidence_class="cited_probability",
        )
        assert resolved_stance < 0  # flipped to match the cited 18.83%

        stances = [s for s, _c in self.QUALITATIVE] + [resolved_stance]
        weights = [c for _s, c in self.QUALITATIVE] + [
            api_settings.evidence_class_weight["cited_probability"],
        ]
        mean, *_ = pool_sources(stances, weights)
        assert _prob(mean) < 0.5


class TestEvidenceClassWeight:
    WEIGHTS = {
        "cited_probability": 4.0,
        "reported_fact": 1.0,
        "cited_share": 1.5,
        "reporting": 0.6,
        "opinion": 0.25,
    }

    def test_unclassified_falls_back_to_certainty_under_a_cap(self):
        # evidence_class omitted by the extractor — still fall back to certainty so
        # partial classifier coverage doesn't flatten everything, but cap it (R3/F10):
        # certainty and the class table are different scales sharing one slot.
        assert evidence_class_weight(None, 0.73, weights=self.WEIGHTS, unclassified_cap=0.25) == 0.25
        # Below the cap the fallback is unchanged — the conservative direction is
        # preserved, not flattened.
        assert evidence_class_weight(None, 0.1, weights=self.WEIGHTS, unclassified_cap=0.25) == 0.1
        assert evidence_class_weight(None, 0.0, weights=self.WEIGHTS, unclassified_cap=0.25) == 0.0

    def test_unclassified_never_outweighs_any_classified_claim(self):
        # The invariant behind F10, stated directly, at the prod cap.
        cap = api_settings.evidence_class_weight_unclassified_cap
        weakest = min(self.WEIGHTS.values())
        assert cap <= weakest
        for certainty in (0.0, 0.25, 0.6, 0.9, 1.0):
            unclassified = evidence_class_weight(None, certainty, weights=self.WEIGHTS, unclassified_cap=cap)
            for cls in self.WEIGHTS:
                assert unclassified <= evidence_class_weight(cls, certainty, weights=self.WEIGHTS)

    def test_known_classes_use_the_lookup_table(self):
        for cls, expected in self.WEIGHTS.items():
            # certainty is ignored once evidence_class is known.
            assert evidence_class_weight(cls, 0.05, weights=self.WEIGHTS) == expected

    def test_unrecognized_class_uses_the_default(self):
        # Defensive only — PredictionExtraction.evidence_class is a Literal of
        # the WEIGHTS keys, so pydantic already rejects anything else.
        assert evidence_class_weight("mystery", 0.05, weights=self.WEIGHTS, default=0.42) == 0.42

    def test_matches_the_configured_production_defaults(self):
        # Locks in that config.py's ApiSettings.evidence_class_weight is what
        # actually ships, not a value that drifted from this test's WEIGHTS.
        assert api_settings.evidence_class_weight == self.WEIGHTS


class TestFranceWorldCupRegression:
    """Reproduces the pooled-overconfidence bug and proves the fix.

    Five match-recap articles frame France as the "favorite"/"strong candidate"
    entering later knockout rounds (stance +0.3 to +0.6, per the multi-stage
    guidance in extractor.py), pooling to ~75% — while one of those very articles
    also cites an explicit Opta model estimate of 18.83% for the tournament
    outcome itself. Without evidence-class weighting, the quantitative source is
    just one of six equal-weight votes and gets outvoted by the qualitative
    majority.
    """

    # (stance, certainty, evidence_class) — unclassified qualitative "favorite"
    # framing vs. the one explicit cited_probability claim (S2, retro
    # docs/ORACLE_VARIABLES.md §5).
    SOURCES = [
        (0.30, 0.30, None),    # "favorite entering the Round of 16"
        (0.40, 0.50, None),    # "beats Paraguay to reach the quarter-finals"
        (0.46, 0.64, None),    # "strongest candidate to win the title"
        (0.61, 0.60, None),    # "superior head-to-head record"
        (0.43, 0.48, None),    # "historical coincidences" narrative piece
        (-0.62, 0.85, "cited_probability"),  # cites Opta: 18.83% to win the tournament
    ]

    def _pool(self, *, classify: bool):
        stances = [s for s, _c, _e in self.SOURCES]
        weights = [
            evidence_class_weight(
                cls if classify else None, cert,
                weights=api_settings.evidence_class_weight,
            )
            for _s, cert, cls in self.SOURCES
        ]
        return pool_sources(stances, weights)

    def test_without_classification_reproduces_overconfidence(self):
        mean, *_ = self._pool(classify=False)
        # The bug: qualitative volume pools well above the cited 18.83% baseline
        # (roughly 3x higher — the real incident pooled all the way to 75%).
        assert _prob(mean) > 0.55

    def test_classification_pulls_pooled_estimate_toward_the_cited_baseline(self):
        without_mean, *_ = self._pool(classify=False)
        with_mean, *_ = self._pool(classify=True)
        assert _prob(with_mean) < _prob(without_mean)
        # Still not literally 18.83% (qualitative sources retain some pull), but
        # materially closer to the cited baseline than the unclassified pool.
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


class TestAggregatePool:
    """aggregate_pool() is every post-extraction-loop step of
    forecaster.run_forecast() extracted into a pure function, so a recompute
    over an already-extracted evidence pool can never drift from a fresh
    run. These tests exercise it directly, independent of the live pipeline —
    see test_forecaster_helpers.py / the endpoint's own tests for the wiring.
    """

    def test_empty_input_returns_none(self):
        assert aggregate_pool([], [], [], [], **_aggregate_kwargs()) is None

    def test_all_off_topic_returns_insufficient_reason(self):
        result = aggregate_pool(
            [0.5, 0.5], [1.0, 1.0], [0.05, 0.05], [False, False],
            **_aggregate_kwargs(),
        )
        assert result is not None
        assert result.insufficient_reason == "all_articles_off_topic"

    def test_zero_total_weight_abstains(self):
        # R3/F14: every source weighs nothing (blocked by credibility, zeroed by
        # relevance, or both). Pooling anyway would fall through pool_sources'
        # zero-total guard, which replaces the weights with a flat 1.0 each — the
        # answer would come, unweighted, from exactly the rows the weighting judged
        # worthless. Abstain instead.
        result = aggregate_pool(
            [0.9, 0.7], [0.0, 0.0], [1.0, 1.0], [False, False],
            **_aggregate_kwargs(),
        )
        assert result is not None
        assert result.insufficient_reason == "no_usable_weight"
        assert result.n == 2
        assert result.evidence_mass == 0.0

    def test_zero_total_weight_abstains_even_with_defer_off(self):
        # Distinct from thin evidence: defer_on_thin_evidence controls whether a
        # THIN pool answers with a wide CI, and it is off in prod. A pool with no
        # weight at all is not thin evidence, it is no evidence.
        result = aggregate_pool(
            [0.9], [0.0], [1.0], [False],
            **_aggregate_kwargs(defer_on_thin_evidence=False),
        )
        assert result is not None
        assert result.insufficient_reason == "no_usable_weight"

    def test_one_weighted_source_among_zeroes_still_pools(self):
        # The guard is for a pool with NO usable weight — a single surviving source
        # is thin, not absent, and must still produce an estimate.
        result = aggregate_pool(
            [0.9, -0.9], [0.0, 0.6], [1.0, 1.0], [False, False],
            **_aggregate_kwargs(),
        )
        assert result is not None
        assert result.insufficient_reason is None
        assert result.mean < 0

    def test_off_topic_takes_precedence_over_zero_weight(self):
        # Both conditions hold (relevance 0 zeroes the weights too); the off-topic
        # reason is the more informative one and is reported first.
        result = aggregate_pool(
            [0.5, 0.5], [0.0, 0.0], [0.0, 0.0], [False, False],
            **_aggregate_kwargs(),
        )
        assert result is not None
        assert result.insufficient_reason == "all_articles_off_topic"

    def test_thin_evidence_deferred_returns_insufficient_reason(self):
        result = aggregate_pool(
            [0.5], [0.01], [1.0], [False],
            **_aggregate_kwargs(defer_on_thin_evidence=True, decisiveness_floor=0.5),
        )
        assert result is not None
        assert result.insufficient_reason == "no_decisive_signal"

    def test_thin_evidence_not_deferred_still_pools_with_widened_ci(self):
        thin = aggregate_pool(
            [0.5], [0.01], [1.0], [False],
            **_aggregate_kwargs(defer_on_thin_evidence=False, decisiveness_floor=0.5),
        )
        solid = aggregate_pool(
            [0.5], [1.0], [1.0], [False],
            **_aggregate_kwargs(defer_on_thin_evidence=False, decisiveness_floor=0.5),
        )
        assert thin is not None and solid is not None
        assert thin.insufficient_reason is None
        assert thin.thin_evidence is True
        assert solid.thin_evidence is False
        assert (thin.ci_high - thin.ci_low) > (solid.ci_high - solid.ci_low)

    def test_matches_pool_sources_directly_when_no_settlement(self):
        stances = [0.3, -0.1, 0.6]
        weights = [0.8, 0.5, 1.2]
        relevances = [0.9, 0.8, 1.0]
        expected_mean, expected_std, expected_ci_low, expected_ci_high = pool_sources(
            stances, weights, clamp_eps=api_settings.logit_clamp,
        )
        result = aggregate_pool(
            stances, weights, relevances, [False, False, False],
            **_aggregate_kwargs(),
        )
        assert result is not None
        assert result.insufficient_reason is None
        assert result.mean == pytest.approx(expected_mean)
        assert result.std == pytest.approx(expected_std)
        assert result.ci_low == pytest.approx(expected_ci_low)
        assert result.ci_high == pytest.approx(expected_ci_high)
        assert result.settled is False

    def test_settlement_override_pins_when_enough_agreeing_sources(self):
        result = aggregate_pool(
            [0.9, 0.85, -0.2], [1.0, 1.0, 0.3], [1.0, 1.0, 0.5],
            [True, True, False],
            **_aggregate_kwargs(settlement_min_sources=2),
        )
        assert result is not None
        assert result.settled is True
        assert result.mean == pytest.approx(api_settings.settlement_stance)
        assert result.settled_sources == 2
        assert result.settlement_suppressed is False

    def test_settlement_tie_does_not_pin(self):
        result = aggregate_pool(
            [0.9, -0.9], [1.0, 1.0], [1.0, 1.0], [True, True],
            **_aggregate_kwargs(settlement_min_sources=1),
        )
        assert result is not None
        assert result.settled is False

    def test_settlement_respects_direction_guard(self):
        # Early "won't happen" settlement on an unexpired arrival claim (the
        # F-35 false-pin class) must be suppressed even with enough sources.
        result = aggregate_pool(
            [-0.95, -0.9], [1.0, 1.0], [1.0, 1.0], [True, True],
            **_aggregate_kwargs(
                settlement_min_sources=2,
                claim_direction="arrival",
                claim_deadline="2099-01-01",
            ),
        )
        assert result is not None
        assert result.settled is False
        assert result.settlement_suppressed is True

    def test_below_settlement_threshold_does_not_pin(self):
        result = aggregate_pool(
            [0.9], [1.0], [1.0], [True],
            **_aggregate_kwargs(settlement_min_sources=2),
        )
        assert result is not None
        assert result.settled is False


class TestEffectiveSampleSize:
    """Kish's n_eff — defect (a) of F16 / retro#365."""

    def test_equal_weights_give_exactly_n(self):
        assert effective_sample_size([0.6] * 4) == pytest.approx(4.0)

    def test_single_source_is_one(self):
        assert effective_sample_size([0.7]) == pytest.approx(1.0)

    def test_near_weightless_rows_buy_almost_nothing(self):
        # The attack (a) exists to stop: pad a real pool with rows that carry
        # no weight and the raw count would shrink the standard error by ~sqrt.
        padded = effective_sample_size([1.0, 1.0] + [1e-9] * 98)
        assert padded == pytest.approx(2.0, abs=1e-3)

    def test_a_blocked_row_does_not_count(self):
        # B15's shape: one real source, one zeroed by credibility.
        assert effective_sample_size([0.0, 0.6]) == pytest.approx(1.0)

    def test_never_exceeds_n_and_never_below_one(self):
        for weights in ([0.1, 0.2, 0.3], [5.0, 0.001], [1.0] * 7, [0.0, 0.0]):
            assert 1.0 <= effective_sample_size(weights) <= len(weights)

    def test_degenerate_inputs_fall_back_to_the_row_count(self):
        # Mirrors pool_sources' zero-total guard, which flattens to 1.0 each.
        assert effective_sample_size([0.0, 0.0, 0.0]) == pytest.approx(3.0)
        assert effective_sample_size([]) == 0.0


class TestDispersionFloor:
    """The unanimity floor — defect (b) of F16 / retro#365."""

    def test_unanimous_pool_no_longer_publishes_a_point(self):
        result = aggregate_pool([0.4] * 4, [0.6] * 4, [1.0] * 4, [False] * 4,
                                **_aggregate_kwargs())
        assert result is not None
        assert result.ci_high - result.ci_low > 0.15
        assert result.ci_low < result.mean < result.ci_high

    def test_single_strong_source_pays_the_floor_undivided(self):
        one = aggregate_pool([0.75], [1.0], [1.0], [False], **_aggregate_kwargs())
        four = aggregate_pool([0.75] * 4, [1.0] * 4, [1.0] * 4, [False] * 4,
                              **_aggregate_kwargs())
        assert one is not None and four is not None
        # n_eff 1 vs 4 ⇒ the floor halves. This is the decay property; without
        # it the floor would be a flat minimum and unanimity would never pay off.
        assert (one.ci_high - one.ci_low) == pytest.approx(
            2.0 * (four.ci_high - four.ci_low), rel=1e-3
        )

    def test_the_floor_never_narrows_a_dispersed_pool(self):
        stances = [0.9, 0.1, -0.4, 0.6]
        floored = aggregate_pool(stances, [1.0] * 4, [1.0] * 4, [False] * 4,
                                 **_aggregate_kwargs())
        off = aggregate_pool(stances, [1.0] * 4, [1.0] * 4, [False] * 4,
                             **_aggregate_kwargs(pool_dispersion_floor=0.0))
        assert floored is not None and off is not None
        assert floored.ci_low == off.ci_low
        assert floored.ci_high == off.ci_high

    def test_widths_decay_monotonically_with_effective_n(self):
        widths = []
        for k in range(1, 9):
            r = aggregate_pool([0.4] * k, [0.6] * k, [1.0] * k, [False] * k,
                               **_aggregate_kwargs())
            assert r is not None
            widths.append(r.ci_high - r.ci_low)
        assert widths == sorted(widths, reverse=True)
        assert all(a > b for a, b in zip(widths, widths[1:]))

    def test_zero_disables_the_floor(self):
        r = aggregate_pool([0.4] * 4, [0.6] * 4, [1.0] * 4, [False] * 4,
                           **_aggregate_kwargs(pool_dispersion_floor=0.0))
        assert r is not None
        # The pre-F16 behaviour, ULP noise aside: unanimity reads as precision.
        assert r.ci_high - r.ci_low < 1e-9

    def test_the_floor_never_moves_the_mean(self):
        """The load-bearing safety property: F16 is an interval change only."""
        cases = [
            ([0.4] * 4, [0.6] * 4),
            ([0.75], [1.0]),
            ([-0.9, 0.6], [0.0, 0.6]),
            ([0.8, 0.8, 0.75, 0.8], [1.0] * 4),
            ([0.98, -0.98], [2.0, 0.001]),
            ([0.0] * 3, [0.3, 0.3, 0.3]),
        ]
        for stances, weights in cases:
            n = len(stances)
            on = aggregate_pool(stances, weights, [1.0] * n, [False] * n,
                                **_aggregate_kwargs())
            off = aggregate_pool(stances, weights, [1.0] * n, [False] * n,
                                 **_aggregate_kwargs(pool_dispersion_floor=0.0))
            assert on is not None and off is not None
            assert on.mean == off.mean, f"mean moved for {stances}/{weights}"

    def test_the_floor_does_not_stack_on_thin_evidence_widening(self):
        """A thin *and* unanimous pool must not be charged twice — the floor is
        a min/max against the widened band, not an addition to it."""
        thin = _aggregate_kwargs()
        r = aggregate_pool([0.4], [0.05], [1.0], [False], **thin)
        off = aggregate_pool([0.4], [0.05], [1.0], [False],
                             **_aggregate_kwargs(pool_dispersion_floor=0.0))
        assert r is not None and off is not None
        assert r.thin_evidence is True
        # Thin widening already dominates the floor here, so the band is
        # untouched by it.
        assert r.ci_low == off.ci_low
        assert r.ci_high == off.ci_high

    def test_widened_band_respects_the_pooling_clamp(self):
        """Defect (c): the widening term used to be the only path to a literal
        0%–100% interval, endpoints the estimator's own clamp calls unreachable."""
        r = aggregate_pool([0.0], [0.001], [1.0], [False], **_aggregate_kwargs())
        assert r is not None
        assert r.ci_low >= -0.98 - 1e-9
        assert r.ci_high <= 0.98 + 1e-9
