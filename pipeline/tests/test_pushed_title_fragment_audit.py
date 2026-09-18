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


# ── log-line fields added after the 2026-09-18 precision review ─────────────

def _line(caplog) -> str:
    return next(r.getMessage() for r in caplog.records if "event=pushed_title_fragment " in r.getMessage())


def test_extreme_stance_is_logged_for_the_candidate_rule(caplog):
    with caplog.at_level("WARNING", logger="tm.extractor"):
        audit_pushed_title_fragment([pred(stance=0.3), pred(stance=-0.95)], TELEGRAM_URL, "Short teaser")
    assert "extreme_stance=True" in _line(caplog)


def test_a_moderate_stance_is_not_extreme(caplog):
    with caplog.at_level("WARNING", logger="tm.extractor"):
        audit_pushed_title_fragment([pred(stance=0.3)], TELEGRAM_URL, "Likud primaries turnout: 36%")
    assert "extreme_stance=False" in _line(caplog)


def test_effective_len_counts_a_doubled_title_once(caplog):
    # forecaster.py's fallback text is "title — snippet"; for a short post the two are identical.
    title = "IAEA: we identified activity at Pickaxe Mountain"
    with caplog.at_level("WARNING", logger="tm.extractor"):
        audit_pushed_title_fragment([pred(stance=0.3)], TELEGRAM_URL, f"{title} — {title}")
    line = _line(caplog)
    assert f"len={len(title) * 2 + 3} " in line
    assert f"effective_len={len(title)} " in line


def test_effective_len_leaves_a_title_with_its_own_dash_alone(caplog):
    text = "Bennett — we will defeat Hezbollah"
    with caplog.at_level("WARNING", logger="tm.extractor"):
        audit_pushed_title_fragment([pred(stance=0.3)], TELEGRAM_URL, text)
    assert f"effective_len={len(text)} " in _line(caplog)


def test_effective_len_does_not_count_a_bare_link(caplog):
    text = "https://www.youtube.com/watch?v=abcdefghijk Watch to the end."
    with caplog.at_level("WARNING", logger="tm.extractor"):
        audit_pushed_title_fragment([pred()], TELEGRAM_URL, text)
    assert f"effective_len={len('Watch to the end.')} " in _line(caplog)


def test_firing_still_keys_on_the_text_the_extractor_saw(caplog):
    # 130 chars of link + 20 of prose: effective_len is short, but the flag must not newly fire.
    text = "https://example.com/" + "a" * 110 + " twenty chars of text"
    with caplog.at_level("WARNING", logger="tm.extractor"):
        audit_pushed_title_fragment([pred()], TELEGRAM_URL, text)
    assert not any("event=pushed_title_fragment " in r.getMessage() for r in caplog.records)
