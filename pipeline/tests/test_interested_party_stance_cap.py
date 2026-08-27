"""An interested party's unverified assertion may not vote at full magnitude.

The prompt's interested-party rule caps `certainty` at 0.5 while stating that "full
stance magnitude still applies". But a vote's LOCATION in the pool is stance alone
(`stance_to_prob = (stance + 1) / 2`, aggregation.py), so a certainty-only discount is
weight-only and cancels under normalization: N unverified claims at stance +1 still
pool to +0.99. The one rule written to discount self-serving claims had no effect in
exactly the case it exists for (retro#368, lane-soundness F20).

Keyed on `verified` directly. The issue proposed keying on the prompt's implied
signature — high |stance| with low certainty — and warned not to assume it held. The
prod audit (2026-08-01) found it inverted: across 1,418 rows carrying the marker,
|stance| and certainty are COUPLED on unverified claims (corr +0.68 vs +0.79 on
verified), unverified rows sit LOWER on both axes, and the proposed key fired on 1 row
in 1,418 — which was verified=true.

Unlike the precursor cap (retro#367) and the provenance check (retro#369), this cap
enforces no pre-existing prompt promise: the prompt says the opposite. The number is a
deliberate new policy value, which is why it lives in config and is pinned here.
"""
import pytest

from tm.config import settings
from tm.extractor import enforce_interested_party_stance_cap
from tm.models import PredictionExtraction

CAP = settings.interested_party_stance_cap


def pred(
    stance: float,
    verified: bool | None = False,
    certainty: float = 0.85,
    evidence_class: str | None = "reporting",
):
    return PredictionExtraction(
        quote="q", claim="c", stance=stance, certainty=certainty,
        evidence_class=evidence_class, verified=verified,
    )


# ── the clamp itself ──────────────────────────────────────────────────────────


def test_an_unverified_claim_above_the_cap_is_clamped():
    [out] = enforce_interested_party_stance_cap([pred(0.9)])
    assert out.stance == pytest.approx(CAP)


def test_the_sign_is_preserved():
    """A company denying a merger is still evidence ABOUT the merger — only the
    magnitude is policy, the direction is the model's judgement."""
    [out] = enforce_interested_party_stance_cap([pred(-0.9)])
    assert out.stance == pytest.approx(-CAP)


def test_an_unverified_claim_at_the_cap_is_untouched():
    [out] = enforce_interested_party_stance_cap([pred(CAP)])
    assert out.stance == pytest.approx(CAP)


def test_an_unverified_claim_below_the_cap_is_untouched():
    [out] = enforce_interested_party_stance_cap([pred(0.1)])
    assert out.stance == pytest.approx(0.1)


def test_the_r8_b3_shape_four_echoes_of_one_interested_party():
    """B3: four outlets relaying one interested party's assertion, every claim
    verified=false. Each is clamped independently — the collapse of four echoes
    into one observation is F2's clustering job, not this function's."""
    out = enforce_interested_party_stance_cap(
        [pred(0.8), pred(0.8), pred(0.75), pred(0.8)]
    )
    assert [p.stance for p in out] == pytest.approx([CAP] * 4)


# ── what it must not touch ────────────────────────────────────────────────────


def test_a_verified_claim_is_never_touched():
    [out] = enforce_interested_party_stance_cap([pred(0.9, verified=True)])
    assert out.stance == pytest.approx(0.9)


def test_an_unjudged_claim_is_never_touched():
    """verified=None is 87% of live pool rows — the marker is populated only on
    extractions since 2026-07-09 and was never backfilled. Fail open: this must
    never invent a judgement the model declined to make."""
    [out] = enforce_interested_party_stance_cap([pred(0.9, verified=None)])
    assert out.stance == pytest.approx(0.9)


def test_certainty_and_class_are_untouched():
    """Only the location axis moves here. The certainty cap the prompt already
    promises — violated on 30.3% of live unverified rows — is retro#378."""
    [out] = enforce_interested_party_stance_cap([pred(0.9, certainty=0.85)])
    assert out.claim_strength == pytest.approx(0.85)
    assert out.evidence_class == "reporting"


def test_an_empty_list_is_a_no_op():
    assert enforce_interested_party_stance_cap([]) == []


# ── config and observability ──────────────────────────────────────────────────


def test_the_cap_is_config_driven(monkeypatch):
    monkeypatch.setattr(settings, "interested_party_stance_cap", 0.5)
    [out] = enforce_interested_party_stance_cap([pred(0.9)])
    assert out.stance == pytest.approx(0.5)


def test_a_cap_above_every_stance_is_inert(monkeypatch):
    """The R8 sweep's baseline: with the cap above 1.0 the whole matrix is
    unchanged, which is what proves the movement at a real cap is this function's
    and nothing else's."""
    monkeypatch.setattr(settings, "interested_party_stance_cap", 1.1)
    [out] = enforce_interested_party_stance_cap([pred(1.0)])
    assert out.stance == pytest.approx(1.0)


def test_the_clamp_is_logged_with_the_resolved_class(caplog):
    with caplog.at_level("WARNING"):
        enforce_interested_party_stance_cap([pred(0.9)])
    assert "event=interested_party_stance_clamped" in caplog.text
    assert "evidence_class=reporting" in caplog.text
