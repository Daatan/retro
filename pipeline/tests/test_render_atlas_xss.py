"""Stored-XSS regression coverage for render_atlas.py (retro#427).

headline/description/event-name/quote/claim all originate from scraped
third-party article content or Polymarket market metadata — none of it is
trusted. Every HTML-fragment renderer must html.escape() it before
interpolation, and every value embedded inside a <script> block via
json.dumps() must have '<' escaped so a payload containing '</script' can't
prematurely close the tag (the HTML tokenizer runs before JS parsing, so
this has to happen at the JSON-string level)."""

import json

from tm.render_atlas import (
    MVP_EVENTS,
    SOURCES,
    render,
    _json_for_script,
    _render_event_sections,
    _render_matrix,
    _render_scoring,
)

PAYLOAD = '<script>alert(1)</script>'
ESCAPED_PAYLOAD = '&lt;script&gt;alert(1)&lt;/script&gt;'


def test_json_for_script_neutralizes_script_close_tag():
    out = _json_for_script({"quote": "</script><script>alert(1)</script>"})
    assert '</script>' not in out
    # Only '<' needs escaping — a lone '>' can't start a tag, so the HTML
    # tokenizer never recognizes '\u003c/script>' as a closing tag.
    assert '\\u003c/script>' in out


def test_render_matrix_escapes_event_name_and_polymarket_fields():
    matrix_rows = [{
        "id": "A01",
        "name": PAYLOAD,
        "outcome": True,
        "outcome_date": "2024-06-10",
        "polymarket": {"url": f'"><script>x</script>', "question": PAYLOAD, "match_quality": "exact"},
        "cells": [],
    }]
    out = _render_matrix(matrix_rows, search_status={}, competitive_scores={}, cell_articles={})
    assert PAYLOAD not in out
    assert ESCAPED_PAYLOAD in out
    assert '"><script>' not in out


def test_render_event_sections_escapes_description_headline_and_name():
    events = {"A01": {
        "outcome": True,
        "outcome_date": "2024-06-10",
        "name": PAYLOAD,
        "description": PAYLOAD,
    }}
    cell_articles = {
        ("A01", "ynet"): [{"headline": PAYLOAD, "url": "https://example.com", "date": "2024-06-01", "pred_count": 1}],
    }
    out = _render_event_sections(["A01"], events, polymarket={}, cell_articles=cell_articles)
    assert PAYLOAD not in out
    assert out.count(ESCAPED_PAYLOAD) == 3  # name, description, headline


def test_render_event_sections_escapes_article_url_in_href():
    events = {"A01": {"outcome": True, "outcome_date": "2024-06-10", "name": "Event", "description": ""}}
    cell_articles = {
        ("A01", "ynet"): [{"headline": "h", "url": "javascript:alert(1)'><script>x</script>", "date": "2024-06-01", "pred_count": 1}],
    }
    out = _render_event_sections(["A01"], events, polymarket={}, cell_articles=cell_articles)
    assert '<script>x</script>' not in out


def test_render_scoring_escapes_event_name_and_does_not_crash():
    # Regression test for a shadowing bug introduced (and fixed) alongside the
    # XSS fix: this function used a local variable literally named `html`,
    # which shadowed the module-level `import html` and made `html.escape()`
    # raise AttributeError the moment it was called from inside this function.
    scores = {
        "overall": {"brier": 0.2, "skill": 0.1, "n": 5},
        "by_event": {"A01": {"n": 5, "implied_p": 60.0, "brier": 0.2, "outcome": 1}},
        "by_source": {"ynet": {"n": 5, "brier": 0.2}},
        "calibration": None,
    }
    events = {"A01": {"name": PAYLOAD}}
    out = _render_scoring(scores, events)
    assert PAYLOAD not in out
    assert ESCAPED_PAYLOAD in out


def test_render_scoring_calibration_json_is_script_safe():
    scores = {
        "overall": {"brier": 0.2, "skill": 0.1, "n": 5},
        "by_event": {},
        "by_source": {},
        "calibration": {
            "labels": ["</script><script>alert(1)</script>"],
            "predicted": [0.5],
            "actual": [0.5],
            "counts": [5],
        },
    }
    out = _render_scoring(scores, events={})
    assert '</script><script>alert(1)</script>' not in out
    assert '\\u003c/script>\\u003cscript>alert(1)\\u003c/script>' in out


def test_render_end_to_end_escapes_chart_data_and_predictions_json(tmp_path):
    # _json_for_script's '<' -> '\u003c' protects the <script> tag boundary,
    # but the JS engine decodes '\u003c' back to a literal '<' the moment it
    # parses `const PREDICTIONS = {predictions_js};` as an object literal —
    # so quote/claim/headline/event_name/url must ALSO be html.escape()'d
    # individually before json.dumps(), or the frontend's innerHTML sinks
    # (renderPredictions/openCellPopup in atlas.html) stay exploitable.
    # chart_data/all_preds/cell_articles are built inline inside render(),
    # not as separately-testable functions, so this drives it end-to-end.
    eid, sid = MVP_EVENTS[0], SOURCES[0]
    data_dir = tmp_path / "data"

    events_dir = data_dir / "events"
    events_dir.mkdir(parents=True)
    (events_dir / f"{eid}.json").write_text(json.dumps({
        "outcome": True, "outcome_date": "2024-06-10",
        "name": PAYLOAD, "description": "",
    }))

    sources_dir = data_dir / "sources"
    sources_dir.mkdir(parents=True)
    (sources_dir / f"{sid}.json").write_text(json.dumps({"name": "Test Source"}))

    cell_dir = data_dir / "atlas" / eid / sid
    cell_dir.mkdir(parents=True)
    (cell_dir / "entry_0001.json").write_text(json.dumps({
        "headline": PAYLOAD,
        "article_date": "2024-06-01",
        "article_hash": "abc123",
        "predictions": [{"stance": 0.5, "certainty": 0.8, "quote": PAYLOAD, "claim": PAYLOAD}],
    }))

    vault_dir = data_dir / "vault2" / "articles"
    vault_dir.mkdir(parents=True)
    (vault_dir / "abc123.json").write_text(json.dumps({"url": 'https://x/"><script>alert(2)</script>'}))

    output_path = tmp_path / "out.html"
    render(data_dir, output_path)
    out = output_path.read_text()

    assert PAYLOAD not in out
    assert '<script>alert(2)</script>' not in out
    # The escaped forms must survive into the embedded JSON (event_name,
    # quote, claim, headline all carry PAYLOAD; url carries the second one).
    assert ESCAPED_PAYLOAD in out
    assert '&lt;script&gt;alert(2)&lt;/script&gt;' in out
