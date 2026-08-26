"""Log-only: is a claim's `quote` actually the event's own event_name/
event_description restated, not real article text (retro#545)?

retro#545's 2026-08-25 cross-model survey found 2 of 10 flagged articles carried a
fabricated quote of exactly this shape. Precision on this shape is unmeasured — same
rollout shape as audit_named_entity_dyad_mismatch: ship the shadow log, review real
trigger/precision rate before any promotion.
"""
from tm.extractor import (
    _QUOTE_PROVENANCE_MIN_LEN,
    audit_quote_provenance_mismatch,
)
from tm.models import PredictionExtraction

EVENT_NAME = "Global oil price drops below $70/barrel"
EVENT_DESCRIPTION = "Will the price of Brent crude fall below $70 a barrel this month?"


def pred(quote: str, *, stance: float = 0.9, certainty: float = 0.9, claim: str = "c"):
    return PredictionExtraction(quote=quote, claim=claim, stance=stance, certainty=certainty)


# ── the mismatch itself ─────────────────────────────────────────────────────

def test_quote_verbatim_event_name_logs_a_warning(caplog):
    preds = [pred(EVENT_NAME)]
    with caplog.at_level("WARNING", logger="tm.extractor"):
        audit_quote_provenance_mismatch(preds, EVENT_NAME, EVENT_DESCRIPTION)
    assert any(
        "event=quote_provenance_mismatch " in r.message for r in caplog.records
    )


def test_quote_verbatim_event_description_logs_a_warning(caplog):
    preds = [pred(EVENT_DESCRIPTION)]
    with caplog.at_level("WARNING", logger="tm.extractor"):
        audit_quote_provenance_mismatch(preds, EVENT_NAME, EVENT_DESCRIPTION)
    assert any(
        "event=quote_provenance_mismatch " in r.message for r in caplog.records
    )


def test_formatting_differences_still_match(caplog):
    """Casefold + whitespace-collapse + trailing punctuation — the two known real
    examples were restatements, not byte-identical strings."""
    preds = [pred(f"  {EVENT_NAME.upper()}.  ")]
    with caplog.at_level("WARNING", logger="tm.extractor"):
        audit_quote_provenance_mismatch(preds, EVENT_NAME, EVENT_DESCRIPTION)
    assert any(
        "event=quote_provenance_mismatch " in r.message for r in caplog.records
    )


def test_never_mutates_anything():
    preds = [pred(EVENT_NAME, stance=0.9, certainty=0.95)]
    audit_quote_provenance_mismatch(preds, EVENT_NAME, EVENT_DESCRIPTION)
    assert (preds[0].stance, preds[0].certainty, preds[0].quote) == (0.9, 0.95, EVENT_NAME)


def test_returns_the_same_list_object():
    preds = [pred(EVENT_NAME)]
    assert audit_quote_provenance_mismatch(preds, EVENT_NAME, EVENT_DESCRIPTION) is preds


# ── everything that must be left alone ──────────────────────────────────────

def test_a_real_article_quote_is_silent(caplog):
    preds = [pred(
        "Brent crude futures fell 3% in early trading Tuesday amid oversupply concerns."
    )]
    with caplog.at_level("WARNING", logger="tm.extractor"):
        audit_quote_provenance_mismatch(preds, EVENT_NAME, EVENT_DESCRIPTION)
    assert not any(
        "event=quote_provenance_mismatch " in r.message for r in caplog.records
    )


def test_short_quote_below_the_length_floor_is_silent(caplog):
    """A short quote could coincidentally overlap event text without being
    fabricated — below _QUOTE_PROVENANCE_MIN_LEN this is a no-op even if it
    happens to equal a (short) event_name."""
    short_event = "Fed cuts rates"
    preds = [pred(short_event)]
    assert len(short_event) < _QUOTE_PROVENANCE_MIN_LEN
    with caplog.at_level("WARNING", logger="tm.extractor"):
        audit_quote_provenance_mismatch(preds, short_event, "Will the Fed cut rates?")
    assert not any(
        "event=quote_provenance_mismatch " in r.message for r in caplog.records
    )


def test_empty_event_name_and_description_is_silent(caplog):
    preds = [pred("Some long enough quote text from the article body itself.")]
    with caplog.at_level("WARNING", logger="tm.extractor"):
        audit_quote_provenance_mismatch(preds, "", "")
    assert not any(
        "event=quote_provenance_mismatch " in r.message for r in caplog.records
    )


def test_mixed_batch_flags_only_the_fabricated_row(caplog):
    fabricated = pred(EVENT_NAME, claim="fabricated")
    real = pred(
        "Brent crude futures fell 3% in early trading Tuesday amid oversupply concerns.",
        claim="real",
    )
    with caplog.at_level("WARNING", logger="tm.extractor"):
        audit_quote_provenance_mismatch([fabricated, real], EVENT_NAME, EVENT_DESCRIPTION)
    messages = [
        r.message for r in caplog.records if "event=quote_provenance_mismatch " in r.message
    ]
    assert len(messages) == 1
    assert "fabricated" in messages[0]


def test_empty_batch():
    assert audit_quote_provenance_mismatch([], EVENT_NAME, EVENT_DESCRIPTION) == []


def test_shadow_summary_line_always_logs(caplog):
    preds = [pred(EVENT_NAME), pred("a real article sentence long enough to pass the floor")]
    with caplog.at_level("INFO", logger="tm.extractor"):
        audit_quote_provenance_mismatch(preds, EVENT_NAME, EVENT_DESCRIPTION)
    summaries = [
        r.message for r in caplog.records if "event=quote_provenance_mismatch_shadow" in r.message
    ]
    assert len(summaries) == 1
    assert "eligible=2 fired=1 n=2" in summaries[0]
