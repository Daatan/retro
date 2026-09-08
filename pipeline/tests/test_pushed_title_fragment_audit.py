"""Log-only: is a pushed (Telegram) source's whole text a colon-terminated lead-in
fragment or a very short teaser, with the extractor still emitting a claim (retro#770
suggestion 3)?

PR#773 taught the live (Haiku) extractor to emit nothing for exactly this shape
(the Yinon Magal TV-billing case, "הערב בפטריוטים:"), but the fix is behavioural and
Haiku-gated only — nothing checks in production whether it holds, and the Nova/batch
lane was explicitly excluded from PR#773's measurement. This audits, it never mutates
stance/claim/anything else.
"""
from tm.extractor import audit_pushed_title_fragment
from tm.models import PredictionExtraction

TELEGRAM_URL = "https://t.me/yinonews/61432"
ARTICLE_URL = "https://www.jpost.com/some/real/article"


def pred(*, stance: float = 1.0, claim: str = "Benjamin Netanyahu is the Prime Minister"):
    return PredictionExtraction(quote="q", claim=claim, stance=stance, certainty=0.9)


# ── the shape itself ────────────────────────────────────────────────────────

def test_colon_terminated_telegram_fragment_logs_a_warning(caplog):
    preds = [pred()]
    with caplog.at_level("WARNING", logger="tm.extractor"):
        audit_pushed_title_fragment(preds, TELEGRAM_URL, "הערב בפטריוטים:")
    assert any(
        "event=pushed_title_fragment " in r.message for r in caplog.records
    )


def test_very_short_telegram_text_without_a_colon_also_fires(caplog):
    preds = [pred()]
    with caplog.at_level("WARNING", logger="tm.extractor"):
        audit_pushed_title_fragment(preds, TELEGRAM_URL, "Tonight on Patriots")
    assert any(
        "event=pushed_title_fragment " in r.message for r in caplog.records
    )


def test_fired_flag_never_mutates_predictions():
    preds = [pred()]
    out = audit_pushed_title_fragment(preds, TELEGRAM_URL, "הערב בפטריוטים:")
    assert out is preds
    assert out[0].stance == 1.0


# ── the non-firing paths ─────────────────────────────────────────────────────

def test_no_predictions_does_not_log(caplog):
    with caplog.at_level("WARNING", logger="tm.extractor"):
        audit_pushed_title_fragment([], TELEGRAM_URL, "הערב בפטריוטים:")
    assert not caplog.records


def test_real_article_page_url_does_not_fire_even_if_text_is_short(caplog):
    """`has_no_article_page` is false for a real site — a short body there is a
    fetch/extraction question, not this shape."""
    preds = [pred()]
    with caplog.at_level("WARNING", logger="tm.extractor"):
        audit_pushed_title_fragment(preds, ARTICLE_URL, "Short:")
    assert not caplog.records


def test_long_telegram_text_without_a_colon_does_not_fire(caplog):
    long_text = "A" * 150
    preds = [pred()]
    with caplog.at_level("WARNING", logger="tm.extractor"):
        audit_pushed_title_fragment(preds, TELEGRAM_URL, long_text)
    assert not caplog.records


def test_empty_text_does_not_fire(caplog):
    preds = [pred()]
    with caplog.at_level("WARNING", logger="tm.extractor"):
        audit_pushed_title_fragment(preds, TELEGRAM_URL, "   ")
    assert not caplog.records


def test_none_url_does_not_fire(caplog):
    preds = [pred()]
    with caplog.at_level("WARNING", logger="tm.extractor"):
        audit_pushed_title_fragment(preds, None, "הערב בפטריוטים:")
    assert not caplog.records
