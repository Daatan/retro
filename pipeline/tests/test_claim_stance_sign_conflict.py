"""retro#298: the extracted claim text and stance can disagree with each other in the
same row — e.g. row 6205 from the 2026-07-19/20 audit, a claim stating a withdrawal
"is mandatory" scored stance -0.136. The issue scoped a full semantic fix as needing a
second LLM call or a verifier stage; its own suggested "cheap partial" is a deterministic
marker check as an observability signal, which is what flag_claim_stance_sign_conflicts
implements. It never corrects predictions — only logs.
"""
from tm.extractor import flag_claim_stance_sign_conflicts
from tm.models import PredictionExtraction


def pred(claim: str, stance: float):
    return PredictionExtraction(quote="q", claim=claim, stance=stance, certainty=0.5)


def test_returns_predictions_unchanged():
    preds = [pred("Withdrawal is mandatory under the memorandum", -0.136)]
    out = flag_claim_stance_sign_conflicts(preds)
    assert out is preds
    assert out[0].stance == -0.136


def test_the_row_6205_case_logs_a_support_marker_conflict(caplog):
    """"is mandatory" (support marker) with a negative stance."""
    preds = [pred(
        "Iran's foreign minister asserts that Israeli withdrawal from Lebanon "
        "is mandatory under the U.S.-Iran memorandum", -0.136,
    )]
    with caplog.at_level("WARNING", logger="tm.extractor"):
        flag_claim_stance_sign_conflicts(preds)
    assert any(
        "event=claim_stance_sign_conflict marker=support" in r.message
        for r in caplog.records
    )


def test_oppose_marker_with_positive_stance_logs_a_conflict(caplog):
    preds = [pred("The government refuses to withdraw its troops", 0.4)]
    with caplog.at_level("WARNING", logger="tm.extractor"):
        flag_claim_stance_sign_conflicts(preds)
    assert any(
        "event=claim_stance_sign_conflict marker=oppose" in r.message
        for r in caplog.records
    )


def test_support_marker_with_positive_stance_is_consistent_no_log(caplog):
    preds = [pred("Withdrawal is mandatory under the memorandum", 0.6)]
    with caplog.at_level("WARNING", logger="tm.extractor"):
        flag_claim_stance_sign_conflicts(preds)
    assert not any("event=claim_stance_sign_conflict" in r.message for r in caplog.records)


def test_no_marker_present_is_never_flagged(caplog):
    preds = [pred("Rebel forces are closing in on the capital", -0.5)]
    with caplog.at_level("WARNING", logger="tm.extractor"):
        flag_claim_stance_sign_conflicts(preds)
    assert not any("event=claim_stance_sign_conflict" in r.message for r in caplog.records)


def test_both_markers_present_is_ambiguous_and_not_flagged(caplog):
    """A claim containing both a support and an oppose marker is ambiguous by this
    heuristic's own design — it only fires on a clean single-direction marker match."""
    preds = [pred("Withdrawal is mandatory but the minister rejects the memorandum's demand", -0.4)]
    with caplog.at_level("WARNING", logger="tm.extractor"):
        flag_claim_stance_sign_conflicts(preds)
    assert not any("event=claim_stance_sign_conflict" in r.message for r in caplog.records)


def test_small_magnitude_stance_is_not_flagged(caplog):
    """The 0.1 tolerance band avoids flagging near-zero, low-signal stances."""
    preds = [pred("Withdrawal is mandatory under the memorandum", -0.05)]
    with caplog.at_level("WARNING", logger="tm.extractor"):
        flag_claim_stance_sign_conflicts(preds)
    assert not any("event=claim_stance_sign_conflict" in r.message for r in caplog.records)
