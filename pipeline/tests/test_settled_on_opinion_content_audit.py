"""Log-only: is `settled=true` asserted at an extreme stance on a URL whose own path
marks it as opinion/commentary rather than reporting (retro#770 class 3)?

The flagship case: `jpost.com/opinion/article-906779`, an election-math opinion column,
scored `settled=true` at stance -1 on a live forecast. Arguing a position is not reporting
an accomplished fact, however confidently it's phrased. This audits, it never mutates
settled/stance/anything else.
"""
from tm.extractor import audit_settled_on_opinion_content
from tm.models import PredictionExtraction

OPINION_URL = "https://www.jpost.com/opinion/article-906779"
NEWS_URL = "https://www.jpost.com/israel-news/article-906779"


def pred(*, settled=True, stance=-1.0, claim="Only two blocs remain viable"):
    return PredictionExtraction(quote="q", claim=claim, stance=stance, certainty=0.9, settled=settled)


# ── the shape itself ────────────────────────────────────────────────────────

def test_settled_extreme_stance_on_opinion_url_logs_a_warning(caplog):
    preds = [pred()]
    with caplog.at_level("WARNING", logger="tm.extractor"):
        audit_settled_on_opinion_content(preds, OPINION_URL)
    assert any(
        "event=settled_on_opinion_content " in r.message for r in caplog.records
    )


def test_fired_flag_never_mutates_predictions():
    preds = [pred()]
    out = audit_settled_on_opinion_content(preds, OPINION_URL)
    assert out is preds
    assert out[0].settled is True
    assert out[0].stance == -1.0


def test_positive_stance_on_opinion_url_also_fires(caplog):
    preds = [pred(stance=1.0)]
    with caplog.at_level("WARNING", logger="tm.extractor"):
        audit_settled_on_opinion_content(preds, OPINION_URL)
    assert any(
        "event=settled_on_opinion_content " in r.message for r in caplog.records
    )


def test_other_opinion_path_markers_also_fire(caplog):
    for path in ("/opinions/", "/oped/", "/op-ed/", "/column/", "/columnists/", "/editorial/"):
        preds = [pred()]
        with caplog.at_level("WARNING", logger="tm.extractor"):
            audit_settled_on_opinion_content(preds, f"https://example.com{path}article-1")
        assert any(
            "event=settled_on_opinion_content " in r.message for r in caplog.records
        ), f"expected a fire for path marker {path!r}"


# ── the non-firing paths ─────────────────────────────────────────────────────

def test_no_predictions_does_not_log(caplog):
    with caplog.at_level("WARNING", logger="tm.extractor"):
        audit_settled_on_opinion_content([], OPINION_URL)
    assert not caplog.records


def test_none_url_does_not_fire(caplog):
    preds = [pred()]
    with caplog.at_level("WARNING", logger="tm.extractor"):
        audit_settled_on_opinion_content(preds, None)
    assert not caplog.records


def test_non_opinion_url_does_not_fire(caplog):
    preds = [pred()]
    with caplog.at_level("WARNING", logger="tm.extractor"):
        audit_settled_on_opinion_content(preds, NEWS_URL)
    assert not caplog.records


def test_settled_false_does_not_fire(caplog):
    preds = [pred(settled=False)]
    with caplog.at_level("WARNING", logger="tm.extractor"):
        audit_settled_on_opinion_content(preds, OPINION_URL)
    assert not caplog.records


def test_settled_none_does_not_fire(caplog):
    preds = [pred(settled=None)]
    with caplog.at_level("WARNING", logger="tm.extractor"):
        audit_settled_on_opinion_content(preds, OPINION_URL)
    assert not caplog.records


def test_stance_below_gate_does_not_fire(caplog):
    preds = [pred(stance=0.5)]
    with caplog.at_level("WARNING", logger="tm.extractor"):
        audit_settled_on_opinion_content(preds, OPINION_URL)
    assert not caplog.records
