"""Log-only: does a strong-stance claim's fact_signal point the opposite way
from its own stance (retro#602, follow-up of #545)?

Unlike `enforce_settlement_fact_signal_agreement` (settled=True only, 0.5
anchor, neutralises the row), this covers every claim and never mutates —
retro#602's 2026-08-23 sweep found ~90% per-row precision at |stance|>=0.7,
certainty>=0.7, |fact_signal|>=0.3, so it's promoted from shadow to a warning
but not to enforcement.
"""
from tm.extractor import (
    _FACT_SIGNAL_SIGN_CERTAINTY_GATE,
    _FACT_SIGNAL_SIGN_MAGNITUDE_GATE,
    _FACT_SIGNAL_SIGN_STANCE_GATE,
    audit_fact_signal_sign_mismatch,
)
from tm.models import PredictionExtraction


def pred(
    stance: float = 0.9,
    fact_signal: float | None = -0.9,
    *,
    certainty: float = 0.9,
    settled: bool | None = None,
    claim: str = "Andy Burnham will remain Prime Minister until 2028",
):
    return PredictionExtraction(
        quote="q", claim=claim, stance=stance, certainty=certainty,
        settled=settled, fact_signal=fact_signal,
    )


# ── the mismatch itself ─────────────────────────────────────────────────────

def test_opposing_signs_log_a_warning(caplog):
    preds = [pred(0.9, -0.9)]
    with caplog.at_level("WARNING", logger="tm.extractor"):
        audit_fact_signal_sign_mismatch(preds)
    assert any(
        "event=fact_signal_sign_mismatch" in r.message for r in caplog.records
    )


def test_opposing_signs_in_the_other_direction_also_log(caplog):
    preds = [pred(-0.85, 0.8)]
    with caplog.at_level("WARNING", logger="tm.extractor"):
        audit_fact_signal_sign_mismatch(preds)
    assert any(
        "event=fact_signal_sign_mismatch" in r.message for r in caplog.records
    )


def test_never_mutates_anything():
    preds = [pred(0.9, -0.9, certainty=0.95, settled=True)]
    audit_fact_signal_sign_mismatch(preds)
    assert (preds[0].stance, preds[0].fact_signal, preds[0].certainty, preds[0].settled) == (
        0.9, -0.9, 0.95, True,
    )


def test_returns_the_same_list_object():
    preds = [pred(0.9, -0.9)]
    assert audit_fact_signal_sign_mismatch(preds) is preds


def test_fires_regardless_of_settled_state(caplog):
    """Unlike enforce_settlement_fact_signal_agreement, this isn't settled-gated."""
    for settled in (True, False, None):
        caplog.clear()
        preds = [pred(0.9, -0.9, settled=settled)]
        with caplog.at_level("WARNING", logger="tm.extractor"):
            audit_fact_signal_sign_mismatch(preds)
        assert any(
            "event=fact_signal_sign_mismatch" in r.message for r in caplog.records
        )


# ── everything that must be left alone ──────────────────────────────────────

def test_agreeing_signs_are_silent(caplog):
    preds = [pred(0.9, 0.9)]
    with caplog.at_level("WARNING", logger="tm.extractor"):
        audit_fact_signal_sign_mismatch(preds)
    assert not any(
        "event=fact_signal_sign_mismatch" in r.message for r in caplog.records
    )


def test_a_missing_fact_signal_is_silent(caplog):
    """Legitimately omitted for opinion/advocacy rows — the null is not a zero."""
    preds = [pred(0.9, None)]
    with caplog.at_level("WARNING", logger="tm.extractor"):
        audit_fact_signal_sign_mismatch(preds)
    assert not any(
        "event=fact_signal_sign_mismatch" in r.message for r in caplog.records
    )


def test_below_the_magnitude_gate_is_silent(caplog):
    """A near-zero fact_signal bears on the event without establishing it —
    not a real polarity flip (retro#602's 1/20 borderline case)."""
    preds = [pred(0.9, -(_FACT_SIGNAL_SIGN_MAGNITUDE_GATE - 0.01))]
    with caplog.at_level("WARNING", logger="tm.extractor"):
        audit_fact_signal_sign_mismatch(preds)
    assert not any(
        "event=fact_signal_sign_mismatch" in r.message for r in caplog.records
    )


def test_exactly_at_the_magnitude_gate_fires(caplog):
    preds = [pred(0.9, -_FACT_SIGNAL_SIGN_MAGNITUDE_GATE)]
    with caplog.at_level("WARNING", logger="tm.extractor"):
        audit_fact_signal_sign_mismatch(preds)
    assert any(
        "event=fact_signal_sign_mismatch" in r.message for r in caplog.records
    )


def test_below_the_stance_gate_is_silent(caplog):
    preds = [pred(_FACT_SIGNAL_SIGN_STANCE_GATE - 0.01, -0.9)]
    with caplog.at_level("WARNING", logger="tm.extractor"):
        audit_fact_signal_sign_mismatch(preds)
    assert not any(
        "event=fact_signal_sign_mismatch" in r.message for r in caplog.records
    )


def test_below_the_certainty_gate_is_silent(caplog):
    preds = [pred(0.9, -0.9, certainty=_FACT_SIGNAL_SIGN_CERTAINTY_GATE - 0.01)]
    with caplog.at_level("WARNING", logger="tm.extractor"):
        audit_fact_signal_sign_mismatch(preds)
    assert not any(
        "event=fact_signal_sign_mismatch" in r.message for r in caplog.records
    )


def test_a_zero_stance_has_no_sign_to_contradict(caplog):
    preds = [pred(0.0, -0.9)]
    with caplog.at_level("WARNING", logger="tm.extractor"):
        audit_fact_signal_sign_mismatch(preds)
    assert not any(
        "event=fact_signal_sign_mismatch" in r.message for r in caplog.records
    )


def test_mixed_batch_flags_only_the_conflicting_row(caplog):
    conflicting = pred(0.9, -0.9, claim="conflicting")
    agreeing = pred(0.9, 0.9, claim="agreeing")
    missing = pred(0.9, None, claim="missing")
    with caplog.at_level("WARNING", logger="tm.extractor"):
        audit_fact_signal_sign_mismatch([conflicting, agreeing, missing])
    messages = [r.message for r in caplog.records if "event=fact_signal_sign_mismatch" in r.message]
    assert len(messages) == 1
    assert "conflicting" in messages[0]


def test_empty_batch():
    assert audit_fact_signal_sign_mismatch([]) == []
