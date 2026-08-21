"""retro#356 — shadow hazard prior for diffuse by-deadline arrival claims.

Every assertion here is about a REPORTING-ONLY field. The load-bearing test is
`test_hazard_never_moves_the_published_mean`: if that ever fails, the shadow has
stopped being a shadow.
"""
from datetime import date

import pytest

from forecast_api.aggregation import (
    aggregate_pool,
    shadow_hazard_mean,
    stance_to_prob,
)
from forecast_api.resolution_scorer import archetype_base_rate


CREATED = "2026-01-01"
DEADLINE = "2026-12-31"           # 364-day window
MIDPOINT = date(2026, 7, 1)       # 181 days elapsed ≈ 0.497 of the window


def _hazard(mean, **over):
    kw = dict(
        claim_archetype="diffuse",
        claim_created_at=CREATED,
        claim_deadline=DEADLINE,
        base_rate=0.15,
        half_life_fraction=0.5,
        occurrence_seen=False,
        today=MIDPOINT,
    )
    kw.update(over)
    return shadow_hazard_mean(mean, **kw)


# ── the decay itself ────────────────────────────────────────────────────────

def test_drifts_toward_base_rate_not_past_it():
    """An elevated P moves DOWN toward the base rate, and never overshoots."""
    live = 0.90
    out = _hazard(2 * live - 1)
    assert out is not None
    p = stance_to_prob(out)
    assert 0.15 < p < 0.90


def test_half_the_excess_is_gone_at_the_half_life():
    """half_life_fraction=0.5 at ~the window midpoint ⇒ ~half the excess left."""
    live = 0.90
    p = stance_to_prob(_hazard(2 * live - 1))
    # excess 0.75 → ~0.375 left, i.e. ~0.525. Tolerance covers the 181/364 ≈
    # 0.497 elapsed fraction rather than an exact 0.5.
    assert p == pytest.approx(0.15 + 0.75 * 0.5, abs=0.02)


def test_no_decay_at_the_window_start():
    """Nothing has elapsed, so there is no absence to price yet."""
    p = stance_to_prob(_hazard(2 * 0.90 - 1, today=date(2026, 1, 1)))
    assert p == pytest.approx(0.90, abs=1e-9)


def test_a_claim_already_below_the_base_rate_drifts_UP_toward_it():
    """Shrinkage is symmetric — the target is a base rate, not a floor. This is
    the behaviour that makes 0.5 the wrong target and 0.15 the right one."""
    p = stance_to_prob(_hazard(2 * 0.02 - 1))
    assert 0.02 < p < 0.15


def test_elapsed_is_clamped_past_the_deadline():
    """A claim read after its deadline decays no further than at the deadline."""
    at_deadline = stance_to_prob(_hazard(2 * 0.9 - 1, today=date(2026, 12, 31)))
    long_after = stance_to_prob(_hazard(2 * 0.9 - 1, today=date(2027, 12, 31)))
    assert at_deadline == pytest.approx(long_after, abs=1e-9)


# ── the gates: None means "no shadow opinion", not "no decay" ───────────────

@pytest.mark.parametrize("archetype", ["scheduled", "threshold", "none", None, ""])
def test_only_diffuse_is_hazard_shaped(archetype):
    assert _hazard(0.8, claim_archetype=archetype) is None


def test_occurrence_evidence_disables_the_hazard():
    """The hazard prices ABSENCE; once occurrence evidence exists there is none."""
    assert _hazard(0.8, occurrence_seen=True) is None


def test_feature_off_by_default():
    assert _hazard(0.8, base_rate=None) is None
    assert _hazard(0.8, half_life_fraction=0.0) is None


@pytest.mark.parametrize("created,deadline", [
    (None, DEADLINE), (CREATED, None), ("garbage", DEADLINE),
    ("2026-12-31", "2026-01-01"),   # inverted window
    ("2026-05-05", "2026-05-05"),   # zero-length window
])
def test_unusable_window_fails_open(created, deadline):
    assert _hazard(0.8, claim_created_at=created, claim_deadline=deadline) is None


def test_output_stays_in_stance_range():
    for live in (0.0, 0.001, 0.5, 0.999, 1.0):
        out = _hazard(2 * live - 1)
        assert -1.0 <= out <= 1.0


# ── the shrunk base rate ───────────────────────────────────────────────────

def _write(tmp_path, records):
    import json
    p = tmp_path / "resolution_feedback.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in records))
    return p


def test_base_rate_is_exactly_the_prior_with_no_matching_records(tmp_path):
    p = _write(tmp_path, [{"prediction_id": "a", "outcome": True}])  # no archetype
    rate, n = archetype_base_rate(p, "diffuse", prior_p=0.15, prior_n=10.0)
    assert (rate, n) == (0.15, 0)


def test_base_rate_moves_toward_evidence_without_a_cliff(tmp_path):
    p = _write(tmp_path, [
        {"prediction_id": str(i), "outcome": True, "claim_archetype": "diffuse"}
        for i in range(5)
    ])
    rate, n = archetype_base_rate(p, "diffuse", prior_p=0.15, prior_n=10.0)
    assert n == 5
    # (5 + 10*0.15) / 15 = 0.4333 — pulled up by evidence, still short of 1.0.
    assert rate == pytest.approx((5 + 1.5) / 15)


def test_base_rate_ignores_other_archetypes(tmp_path):
    p = _write(tmp_path, [
        {"prediction_id": "a", "outcome": True, "claim_archetype": "scheduled"},
        {"prediction_id": "b", "outcome": False, "claim_archetype": "diffuse"},
    ])
    rate, n = archetype_base_rate(p, "diffuse", prior_p=0.15, prior_n=10.0)
    assert n == 1
    assert rate == pytest.approx(1.5 / 11)


def test_base_rate_counts_resolutions_with_no_scoreable_sources(tmp_path):
    """A resolution still resolved even if no source row is scoreable — dropping
    it would bias the rate toward claims that attracted usable coverage."""
    p = _write(tmp_path, [
        {"prediction_id": "a", "outcome": False, "claim_archetype": "diffuse",
         "sources": []},
    ])
    _rate, n = archetype_base_rate(p, "diffuse", prior_p=0.15, prior_n=10.0)
    assert n == 1


# ── the contract: it is a SHADOW ───────────────────────────────────────────

_POOL = dict(
    relevance_weight_floor=0.0,
    decisiveness_floor=0.0,
    thin_evidence_ci_inflation=1.0,
    defer_on_thin_evidence=False,
    settlement_min_sources=2,
    settlement_stance=0.95,
    logit_clamp=0.999,
    pool_dispersion_floor=0.0,
    claim_created_at=CREATED,
    claim_deadline=DEADLINE,
    claim_archetype="diffuse",
)


def test_hazard_never_moves_the_published_mean():
    """The load-bearing guarantee. Same inputs, hazard off vs on: every
    published field must be byte-identical, and only the shadow field differs."""
    args = ([0.8, 0.9], [1.0, 1.0], [0.9, 0.9], [False, False])
    off = aggregate_pool(*args, **_POOL)
    on = aggregate_pool(*args, **_POOL, hazard_shadow_base_rate=0.15,
                        hazard_shadow_half_life_fraction=0.5)
    assert off.hazard_shadow_mean is None
    assert on.hazard_shadow_mean is not None
    published = lambda r: r._replace(hazard_shadow_mean=None)
    assert published(off) == published(on)


def test_abstaining_pool_reports_no_shadow():
    """No pooled mean was computed, so there is nothing to drift — None rather
    than a hazard applied to the placeholder zeros."""
    res = aggregate_pool(
        [0.8], [0.0], [0.0], [False],
        **{**_POOL, "relevance_weight_floor": 0.5},
        hazard_shadow_base_rate=0.15, hazard_shadow_half_life_fraction=0.5,
    )
    assert res.insufficient_reason is not None
    assert res.hazard_shadow_mean is None
