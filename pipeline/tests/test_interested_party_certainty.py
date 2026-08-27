"""The weight-side half of the interested-party rule (retro#378, F20 family).

The prompt's VERIFIED vs CLAIMED section has said for months that an unverified
interested-party claim "carries certainty no higher than 0.5, however declaratively
it reads". Nothing in code checked it, and the prompt did not hold it: measured on
prod (2026-08-01/02), **56 of 185 `verified=false` rows — 30.3% — exceed the cap**,
max 0.733, ten sitting at exactly 0.70 with an average |stance| of 0.76. Same shape
as the precursor cap (retro#367, 24.4% breach): a numeric taught only by the prompt.

Separate from `enforce_interested_party_stance_cap` (retro#368) on purpose. Stance is
a vote's LOCATION in the pool, certainty its WEIGHT — different consequences, and the
two must produce separately attributable R8 movement. The weight side matters in the
38 live mixed pools where an unverified claim competes with verified ones; in a
wholly-unverified pool a weight-only discount cancels under normalization, which is
why the stance half exists (and that case has 0 live instances out of 71).

Unlike the stance cap, the number here is NOT new policy — it is the prompt's own
literal. `test_extractor_prompt.py` pins the two together so they cannot drift.
"""
import pytest

from tm.config import settings
from tm.extractor import enforce_interested_party_certainty
from tm.models import PredictionExtraction

CAP = settings.interested_party_certainty_cap


def pred(
    certainty: float,
    verified: bool | None = False,
    stance: float = 0.8,
    evidence_class: str | None = "reporting",
):
    return PredictionExtraction(
        quote="q", claim="c", stance=stance, certainty=certainty,
        evidence_class=evidence_class, verified=verified,
    )


# ── the clamp itself ──────────────────────────────────────────────────────────


def test_an_unverified_claim_above_the_cap_is_clamped():
    [out] = enforce_interested_party_certainty([pred(0.9)])
    assert out.claim_strength == pytest.approx(CAP)


def test_the_measured_worst_case_is_clamped():
    """0.733 was the highest certainty on a verified=false row in the prod audit."""
    [out] = enforce_interested_party_certainty([pred(0.733)])
    assert out.claim_strength == pytest.approx(CAP)


def test_the_modal_violation_is_clamped():
    """Ten rows sat at exactly 0.70 carrying avg |stance| 0.76 — the single
    largest cluster of violations, and the one this cap is really about."""
    [out] = enforce_interested_party_certainty([pred(0.70, stance=0.76)])
    assert out.claim_strength == pytest.approx(CAP)
    assert out.stance == pytest.approx(0.76), "the location axis is #368's, not this one's"


def test_an_unverified_claim_at_the_cap_is_untouched():
    [out] = enforce_interested_party_certainty([pred(CAP)])
    assert out.claim_strength == pytest.approx(CAP)


def test_an_unverified_claim_below_the_cap_is_untouched():
    """Unverified rows sit LOWER on both axes than verified ones (avg certainty
    0.456 vs 0.547) — most are already in contract and must not be disturbed."""
    [out] = enforce_interested_party_certainty([pred(0.3)])
    assert out.claim_strength == pytest.approx(0.3)


def test_the_r8_b4_shape_a_minimal_verified_pair():
    """B4 is two articles identical but for `verified`, whose weights are
    bit-identical at 1.0 today. After this cap they must stop being a
    bit-identical pair on the certainty axis — that IS the finding."""
    unverified, verified = enforce_interested_party_certainty(
        [pred(0.9, verified=False), pred(0.9, verified=True)]
    )
    assert unverified.claim_strength == pytest.approx(CAP)
    assert verified.claim_strength == pytest.approx(0.9)
    assert unverified.claim_strength != verified.claim_strength


# ── what it must not touch ────────────────────────────────────────────────────


def test_a_verified_claim_is_never_touched():
    [out] = enforce_interested_party_certainty([pred(0.9, verified=True)])
    assert out.claim_strength == pytest.approx(0.9)


def test_an_unjudged_claim_is_never_touched():
    """verified=None is ~87% of live pool rows — populated only on extractions
    since 2026-07-09, never backfilled. Fail open: this cap is a no-op on
    historical rows and can only be validated forward."""
    [out] = enforce_interested_party_certainty([pred(0.9, verified=None)])
    assert out.claim_strength == pytest.approx(0.9)


def test_stance_and_class_are_untouched():
    """Only the weight axis moves here. Expect R8 movement in sources[*].certainty
    and the pooled mean/CI — never in evidence_class."""
    [out] = enforce_interested_party_certainty([pred(0.9, stance=-0.7)])
    assert out.stance == pytest.approx(-0.7)
    assert out.evidence_class == "reporting"


def test_an_empty_list_is_a_no_op():
    assert enforce_interested_party_certainty([]) == []


# ── config and observability ──────────────────────────────────────────────────


def test_the_cap_is_config_driven(monkeypatch):
    monkeypatch.setattr(settings, "interested_party_certainty_cap", 0.25)
    [out] = enforce_interested_party_certainty([pred(0.9)])
    assert out.claim_strength == pytest.approx(0.25)


def test_a_cap_above_every_certainty_is_inert(monkeypatch):
    """The R8 attribution baseline: with the cap above 1.0 the whole matrix must
    be unchanged, which is what proves the movement at the real cap belongs to
    this function and nothing else (the trick that worked for F20)."""
    monkeypatch.setattr(settings, "interested_party_certainty_cap", 1.1)
    [out] = enforce_interested_party_certainty([pred(1.0)])
    assert out.claim_strength == pytest.approx(1.0)


def test_default_cap_matches_the_prompt_literal():
    """Unlike the stance cap, this number is the prompt's own — it must not drift
    away from the sentence the model is still being taught (same pin as F9's)."""
    from tm.extractor import PROMPT_PREFIX
    assert settings.interested_party_certainty_cap == 0.5
    assert "carries claim_strength no higher than 0.5" in PROMPT_PREFIX


def test_the_clamp_is_logged_with_the_resolved_class(caplog):
    with caplog.at_level("WARNING"):
        enforce_interested_party_certainty([pred(0.9)])
    assert "event=interested_party_certainty_clamped" in caplog.text
    assert "evidence_class=reporting" in caplog.text


# ── the two halves compose ────────────────────────────────────────────────────


def test_both_halves_clamp_the_same_claim_on_their_own_axis():
    """The chain runs the stance cap then this one. An unverified claim that is
    over-cap on both axes must end up clamped on both, each by its own number —
    that is what makes the two halves separately attributable."""
    from tm.extractor import enforce_interested_party_stance_cap

    [out] = enforce_interested_party_certainty(
        enforce_interested_party_stance_cap([pred(0.9, stance=0.95)])
    )
    assert out.stance == pytest.approx(settings.interested_party_stance_cap)
    assert out.claim_strength == pytest.approx(CAP)
