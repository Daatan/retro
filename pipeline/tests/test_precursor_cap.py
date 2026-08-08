"""A precursor's fact_signal may not exceed the precursor cap.

The prompt taught the rule since the fact lane shipped: a fact that only precedes
the event — a mobilisation, a capability, an escalation — is capped "no matter how
sustained, repeated, or intensifying it is" (originally stated as a |0.3| numeral;
retro#354 D1 later deleted it once enforcement moved to code). Advisory-only wasn't
enough: of the 1101 rows carrying `is_occurrence=false` (prod audit 2026-08-01,
retro#367), **269 — 24.4% — sat above the cap**, the worst at |0.90|. Since the stored
number is the claim-weighted MEAN over an article's claims, that figure is a floor on
the per-claim breach rate, not the rate itself.

enforce_precursor_cap is the enforcement: the model says whether the fact IS the event,
code decides how far a fact that isn't may move the estimate.
"""
import pytest

from tm.config import settings
from tm.extractor import enforce_precursor_cap
from tm.models import PredictionExtraction

CAP = 0.3


def pred(
    fact_signal: float | None,
    is_occurrence: bool | None,
    stance: float = 0.8,
    certainty: float = 0.9,
    verified: bool | None = True,
):
    return PredictionExtraction(
        quote="q", claim="c", stance=stance, certainty=certainty,
        fact_signal=fact_signal, is_occurrence=is_occurrence, verified=verified,
    )


# ── the measured breaches ─────────────────────────────────────────────────────


def test_the_worst_live_row_a_precursor_at_090_is_clamped():
    """ynet.co.il, 2026-07-23: |fact_signal| 0.90 with is_occurrence=false — the
    largest precursor magnitude in the pool at audit time."""
    [out] = enforce_precursor_cap([pred(0.90, False)])
    assert out.fact_signal == CAP


def test_a_negative_precursor_keeps_its_sign():
    """dw.com, 2026-08-01: −0.82. The direction a precursor points is a genuine
    judgement; only the magnitude is policy."""
    [out] = enforce_precursor_cap([pred(-0.82, False)])
    assert out.fact_signal == -CAP


@pytest.mark.parametrize("fs", [0.31, 0.5, 0.71, 1.0, -0.31, -0.75, -1.0])
def test_every_over_cap_precursor_lands_exactly_on_the_cap(fs):
    [out] = enforce_precursor_cap([pred(fs, False)])
    assert abs(out.fact_signal) == CAP
    assert (out.fact_signal > 0) == (fs > 0)


# ── what the clamp must not touch ─────────────────────────────────────────────


@pytest.mark.parametrize("fs", [0.3, 0.2, 0.0, -0.2, -0.3])
def test_an_in_contract_precursor_is_left_alone(fs):
    """Including exactly at the cap — the comparison is `<=`, so a claim the model
    already placed on the boundary is not nudged by float wobble."""
    [out] = enforce_precursor_cap([pred(fs, False)])
    assert out.fact_signal == fs


def test_an_occurrence_may_carry_full_magnitude():
    """is_occurrence=true means the fact IS the event. |1.0| is the correct answer
    there, and the cap has nothing to say about it."""
    [out] = enforce_precursor_cap([pred(1.0, True)])
    assert out.fact_signal == 1.0


def test_an_unjudged_occurrence_flag_fails_open():
    """is_occurrence=null: the extractor declined to judge. The clamp never invents
    a judgement the model didn't make."""
    [out] = enforce_precursor_cap([pred(0.9, None)])
    assert out.fact_signal == 0.9


def test_a_claim_with_no_fact_signal_is_untouched():
    [out] = enforce_precursor_cap([pred(None, False)])
    assert out.fact_signal is None


def test_a_clamped_claim_keeps_every_other_field():
    """It still votes in the stance lane exactly as it did before, and its facets
    still describe the same fact."""
    p = pred(0.9, False, stance=0.75, certainty=0.85, verified=False)
    p.settled = True
    p.event_actors, p.event_target = "Russia", "Ukraine"
    [out] = enforce_precursor_cap([p])
    assert out.fact_signal == CAP
    assert (out.stance, out.certainty) == (0.75, 0.85)
    assert out.settled is True
    assert out.verified is False
    assert (out.event_actors, out.event_target) == ("Russia", "Ukraine")


def test_an_empty_list_is_a_no_op():
    assert enforce_precursor_cap([]) == []


# ── the second-order effect on fusion ─────────────────────────────────────────


def test_an_over_cap_precursor_no_longer_outranks_a_real_occurrence():
    """Fusion stores the facets of the DOMINANT (max |fact_signal|) claim for the
    whole article (forecaster.py). Before the clamp, a precursor at 0.8 outranked a
    confirmed occurrence at 0.5 and the article was filed as is_occurrence=false.
    Running the cap first restores the intended ordering."""
    precursor = pred(0.8, False)
    occurrence = pred(0.5, True)
    out = enforce_precursor_cap([precursor, occurrence])
    dominant = max(out, key=lambda p: abs(p.fact_signal))
    assert dominant is occurrence
    assert dominant.is_occurrence is True


# ── the number lives in config, not in the code ───────────────────────────────


def test_the_cap_is_read_from_config_at_call_time(monkeypatch):
    """Magnitude policy is config (retro#354's D1 direction), so a policy change is a
    config change — not a prompt edit and not a code edit."""
    monkeypatch.setattr(settings, "fact_signal_precursor_cap", 0.5)
    [out] = enforce_precursor_cap([pred(0.9, False)])
    assert out.fact_signal == 0.5


def test_default_cap_matches_the_prompt_qualitative_rule():
    """retro#354 D1's follow-up cleanup: now that enforce_precursor_cap enforces the
    ceiling in code regardless of what the model emits, the |0.3| numeral was deleted
    from the prompt — a number in prose was policy the estimator, not the prompt,
    should carry. Only the qualitative rule remains there; the config default here is
    what actually sets the ceiling."""
    from tm.extractor import PROMPT_PREFIX
    assert settings.fact_signal_precursor_cap == CAP
    assert "capped at |0.3|" not in PROMPT_PREFIX
    assert "A precursor never scores as the event occurring" in PROMPT_PREFIX
