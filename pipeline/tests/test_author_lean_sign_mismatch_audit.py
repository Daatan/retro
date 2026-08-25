"""Log-only: does author_lean's sign disagree with the article's own
claim-weighted stance (retro#326)?

A 2026-08-25 prod sweep of author_lean rows added since the PR#314
sentiment-leak fix deployed found ~26-30/1467 (~2%) rows where author_lean
reads the opposite sign of what the article's own claims affirm — not just
the narrow "Behrendt-class" residual PR#314 tracked, but a broader ongoing
leak. Same gate shape/thresholds as `audit_fact_signal_sign_mismatch`
(retro#602) since that guard's 0.7/0.7/0.3 bar is the only precision-sized
precedent in this codebase.
"""
from tm.extractor import (
    _AUTHOR_LEAN_SIGN_CERTAINTY_GATE,
    _AUTHOR_LEAN_SIGN_MAGNITUDE_GATE,
    _AUTHOR_LEAN_SIGN_STANCE_GATE,
    audit_author_lean_sign_mismatch,
)


# ── the mismatch itself ─────────────────────────────────────────────────────

def test_opposing_signs_log_a_warning(caplog):
    with caplog.at_level("WARNING", logger="tm.extractor"):
        audit_author_lean_sign_mismatch(-0.9, 0.85, 1.0, 0.9)
    assert any(
        "event=author_lean_sign_mismatch" in r.message for r in caplog.records
    )


def test_opposing_signs_in_the_other_direction_also_log(caplog):
    with caplog.at_level("WARNING", logger="tm.extractor"):
        audit_author_lean_sign_mismatch(0.8, 0.75, -0.9, 0.85)
    assert any(
        "event=author_lean_sign_mismatch" in r.message for r in caplog.records
    )


def test_returns_none():
    assert audit_author_lean_sign_mismatch(-0.9, 0.85, 1.0, 0.9) is None


def test_real_prod_case_hnaftali_lebanon_withdrawal(caplog):
    """retro#326 spot-check: byline declares Israel stays (avg_stance=1.0)
    but the piece's outrage about a separate US-Iran deal leaked author_lean
    to -0.9."""
    with caplog.at_level("WARNING", logger="tm.extractor"):
        audit_author_lean_sign_mismatch(-0.9, 0.85, 1.0, 0.85)
    assert any(
        "event=author_lean_sign_mismatch" in r.message for r in caplog.records
    )


# ── everything that must be left alone ──────────────────────────────────────

def test_agreeing_signs_are_silent(caplog):
    with caplog.at_level("WARNING", logger="tm.extractor"):
        audit_author_lean_sign_mismatch(0.7, 0.8, 0.9, 0.9)
    assert not any(
        "event=author_lean_sign_mismatch" in r.message for r in caplog.records
    )


def test_genuine_disagreement_is_silent(caplog):
    """When the author's own claims ALSO read negative (they dispute the
    fact), author_lean agreeing with them is correct, not a leak."""
    with caplog.at_level("WARNING", logger="tm.extractor"):
        audit_author_lean_sign_mismatch(-0.7, 0.8, -0.9, 0.9)
    assert not any(
        "event=author_lean_sign_mismatch" in r.message for r in caplog.records
    )


def test_a_null_author_lean_is_silent(caplog):
    """Null means the byline took no position — most reporting — never a leak."""
    with caplog.at_level("WARNING", logger="tm.extractor"):
        audit_author_lean_sign_mismatch(None, None, 1.0, 0.9)
    assert not any(
        "event=author_lean_sign_mismatch" in r.message for r in caplog.records
    )


def test_below_the_author_lean_magnitude_gate_is_silent(caplog):
    with caplog.at_level("WARNING", logger="tm.extractor"):
        audit_author_lean_sign_mismatch(
            -(_AUTHOR_LEAN_SIGN_MAGNITUDE_GATE - 0.01), 0.5, 1.0, 0.9,
        )
    assert not any(
        "event=author_lean_sign_mismatch" in r.message for r in caplog.records
    )


def test_exactly_at_the_magnitude_gate_fires(caplog):
    with caplog.at_level("WARNING", logger="tm.extractor"):
        audit_author_lean_sign_mismatch(
            -_AUTHOR_LEAN_SIGN_MAGNITUDE_GATE, 0.5, 1.0, 0.9,
        )
    assert any(
        "event=author_lean_sign_mismatch" in r.message for r in caplog.records
    )


def test_below_the_stance_gate_is_silent(caplog):
    """retro#326's own finding: several real leaks sat at stance 0.3-0.66,
    below this gate — the gate trades recall for the same precision bar the
    fact_signal precedent used; a broader hand-check to loosen it is a
    follow-up, not a claim that these are non-leaks."""
    with caplog.at_level("WARNING", logger="tm.extractor"):
        audit_author_lean_sign_mismatch(
            -0.9, 0.85, _AUTHOR_LEAN_SIGN_STANCE_GATE - 0.01, 0.9,
        )
    assert not any(
        "event=author_lean_sign_mismatch" in r.message for r in caplog.records
    )


def test_below_the_avg_certainty_gate_is_silent(caplog):
    with caplog.at_level("WARNING", logger="tm.extractor"):
        audit_author_lean_sign_mismatch(
            -0.9, 0.85, 1.0, _AUTHOR_LEAN_SIGN_CERTAINTY_GATE - 0.01,
        )
    assert not any(
        "event=author_lean_sign_mismatch" in r.message for r in caplog.records
    )


def test_zero_avg_stance_has_no_sign_to_contradict(caplog):
    with caplog.at_level("WARNING", logger="tm.extractor"):
        audit_author_lean_sign_mismatch(-0.9, 0.85, 0.0, 0.9)
    assert not any(
        "event=author_lean_sign_mismatch" in r.message for r in caplog.records
    )


def test_a_missing_author_lean_certainty_does_not_crash(caplog):
    """author_lean_certainty is Optional independent of author_lean per the
    model, even though the prompt asks for both together — the audit must
    not assume it's populated."""
    with caplog.at_level("WARNING", logger="tm.extractor"):
        audit_author_lean_sign_mismatch(-0.9, None, 1.0, 0.9)
    assert any(
        "event=author_lean_sign_mismatch" in r.message for r in caplog.records
    )
