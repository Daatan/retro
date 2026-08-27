"""A decider's stated future intent may not vote at full stance magnitude.

The fact lane already treats a decider's on-record commitment as a capped
precursor (`enforce_precursor_cap`, ±0.3); the stance lane — where a vote's
location actually comes from — had no guardrail for the same rows, because F20's
`enforce_interested_party_stance_cap` keys on `verified=false` and a decider's
own statement is usually `verified=true` or unjudged. Prod audit 2026-08-15:
71 of 119 rows with `is_occurrence=false` + facet announcement/denial voted
above 0.3, to |0.85|, every one against a fact lane capped at ±0.3 (retro#518,
the Netanyahu/Le Monde asymmetry, elections#141).

Keyed on the extractor's own markers, never an inferred signature (the retro#368
lesson). Forward-only: `facet` ships 2026-08-10 and was never backfilled.
"""
import pytest

from tm.config import settings
from tm.extractor import enforce_decider_intent_stance_cap
from tm.models import PredictionExtraction

CAP = settings.decider_intent_stance_cap


def pred(
    stance: float,
    *,
    is_occurrence: bool | None = False,
    facet: str | None = "announcement",
    verified: bool | None = True,
    fact_signal: float | None = 0.3,
    certainty: float = 0.8,
):
    return PredictionExtraction(
        quote="q", claim="c", stance=stance, certainty=certainty,
        fact_signal=fact_signal, is_occurrence=is_occurrence, facet=facet,
        verified=verified,
    )


# ── the clamp itself ──────────────────────────────────────────────────────────


def test_an_over_cap_announcement_is_clamped():
    [out] = enforce_decider_intent_stance_cap([pred(0.9)])
    assert out.stance == pytest.approx(CAP)


def test_an_over_cap_denial_is_clamped_with_sign_preserved():
    """A decider ruling something out is still evidence ABOUT the event — only
    the magnitude is policy, the direction is the model's judgement."""
    [out] = enforce_decider_intent_stance_cap([pred(-0.9, facet="denial")])
    assert out.stance == pytest.approx(-CAP)


def test_a_verified_true_row_is_in_scope():
    """The whole point: F20 keys on verified=false, so a decider's on-record
    statement — usually verified=true, the statement demonstrably happened —
    passed the stance lane unguarded."""
    [out] = enforce_decider_intent_stance_cap([pred(0.85, verified=True)])
    assert out.stance == pytest.approx(CAP)


def test_at_or_below_the_cap_is_untouched():
    [a, b] = enforce_decider_intent_stance_cap([pred(0.3), pred(-0.25, facet="denial")])
    assert a.stance == pytest.approx(0.3)
    assert b.stance == pytest.approx(-0.25)


def test_only_stance_moves():
    [out] = enforce_decider_intent_stance_cap([pred(0.9, fact_signal=0.25, certainty=0.77)])
    assert out.fact_signal == pytest.approx(0.25)
    assert out.claim_strength == pytest.approx(0.77)
    assert out.facet == "announcement"
    assert out.is_occurrence is False


# ── fail-open: only an explicitly-marked decider-intent claim is in scope ─────


def test_an_unjudged_is_occurrence_is_left_alone():
    [out] = enforce_decider_intent_stance_cap([pred(0.9, is_occurrence=None)])
    assert out.stance == pytest.approx(0.9)


def test_an_occurrence_is_left_alone():
    """The fact IS the event — an announcement of a completed occurrence is the
    strongest evidence there is, not a decider's forward-looking say-so."""
    [out] = enforce_decider_intent_stance_cap([pred(0.9, is_occurrence=True)])
    assert out.stance == pytest.approx(0.9)


def test_a_missing_facet_is_left_alone():
    [out] = enforce_decider_intent_stance_cap([pred(0.9, facet=None)])
    assert out.stance == pytest.approx(0.9)


def test_a_neither_facet_is_left_alone():
    [out] = enforce_decider_intent_stance_cap([pred(0.9, facet="neither")])
    assert out.stance == pytest.approx(0.9)


# ── shipped policy ────────────────────────────────────────────────────────────


def test_the_shipped_cap_matches_the_fact_lane_precursor_cap():
    """Decision on retro#518 (2026-08-15): symmetric with the fact lane unless
    an audit produces evidence for a different number. Changing either side of
    this equality is a policy decision, not a tidy-up."""
    assert settings.decider_intent_stance_cap == pytest.approx(0.3)
    assert settings.decider_intent_stance_cap == pytest.approx(
        settings.fact_signal_precursor_cap
    )
