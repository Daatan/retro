"""retro#851 — Jev as the evidence_class corrector: one call per claim with the whole article,
fail-open to Haiku's class, and the wiring in _process_article (off / shadow / enforce)."""
import asyncio
import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from forecast_api import forecaster
from forecast_api import jev_class as jc
from forecast_api.config import settings as api_settings
from tm.models import GatekeeperOutput, PredictionExtraction
from tm.web_search import SearchResult

ARTICLE = (
    "The coalition bloc fell by one seat to just 49 in the Maariv poll. "
    "Overall, the opposition bloc gained one seat, rising to 55.\n"
    "The weather in Tel Aviv was sunny on Monday."
)
QUESTION = "Will the opposition bloc win 61 seats?"


def _choice(label: str) -> dict:
    return {"type": "choice", "probabilities": {label: 0.8, "reporting": 0.2}}


def _transport(calls: list, answer=lambda state: "cited_share", fail=False, delay=0.0):
    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body)
        if delay:
            await asyncio.sleep(delay)
        if fail:
            return httpx.Response(500, json={"error": "boom"})
        return httpx.Response(200, json={"answers": {"evidence_class": _choice(answer(body["state"]))},
                                         "usage": {"input_tokens": 100}})
    return httpx.MockTransport(handler)


async def test_one_call_per_located_quote_with_the_whole_article():
    calls: list = []
    out = await jc.jev_evidence_classes(
        text=ARTICLE, question=QUESTION, api_key="k",
        quotes=["Overall, the opposition bloc gained one seat, rising to 55.",
                "A paraphrase that is not in the article at all.",
                None],
        transport=_transport(calls))
    assert out["classes"] == ["cited_share", None, None]
    assert out["p"] == [0.8, 0.0, 0.0] and out["tok_in"] == 100 and out["err_n"] == 0
    assert len(calls) == 1
    state = calls[0]["state"]
    assert state["sentence"] == "Overall, the opposition bloc gained one seat, rising to 55."
    assert state["related_event"] == QUESTION
    assert "fell by one seat to just 49" in state["article"] and "sunny" in state["article"]
    assert calls[0]["questions"] == jc.CLASS_QUESTION


async def test_multi_sentence_quote_is_sent_whole():
    calls: list = []
    await jc.jev_evidence_classes(
        text=ARTICLE, question=QUESTION, api_key="k",
        quotes=["The coalition bloc fell by one seat to just 49 in the Maariv poll. "
                "Overall, the opposition bloc gained one seat, rising to 55."],
        transport=_transport(calls))
    assert calls[0]["state"]["sentence"].count("seat") == 2


async def test_http_error_keeps_none_and_never_raises():
    out = await jc.jev_evidence_classes(
        text=ARTICLE, question=QUESTION, api_key="k",
        quotes=["Overall, the opposition bloc gained one seat, rising to 55."],
        transport=_transport([], fail=True))
    assert out["classes"] == [None] and out["err_n"] == 1


async def test_timeout_keeps_none():
    out = await jc.jev_evidence_classes(
        text=ARTICLE, question=QUESTION, api_key="k", timeout_s=0.05,
        quotes=["Overall, the opposition bloc gained one seat, rising to 55."],
        transport=_transport([], delay=0.5))
    assert out["classes"] == [None] and "err" in out


async def test_label_outside_the_enum_is_ignored():
    out = await jc.jev_evidence_classes(
        text=ARTICLE, question=QUESTION, api_key="k",
        quotes=["Overall, the opposition bloc gained one seat, rising to 55."],
        transport=_transport([], answer=lambda s: "rumour"))
    assert out["classes"] == [None]


async def test_no_key_skips(monkeypatch):
    monkeypatch.setattr(jc, "resolve_api_key", lambda: None)
    out = await jc.jev_evidence_classes(text=ARTICLE, question=QUESTION, quotes=["x"])
    assert out["skip"] == "no_key" and out["classes"] == [None]


def _preds(*classes):
    return [SimpleNamespace(evidence_class=c) for c in classes]


def test_apply_enforce_replaces_only_where_jev_answered(caplog):
    preds = _preds("reported_fact", "opinion", None)
    with caplog.at_level(logging.INFO, logger="forecast_api.jev_class"):
        jc.apply_jev_classes(preds, {"classes": ["cited_share", None, "reporting"], "p": [0.8, 0.0, 0.7],
                                     "ms": 500, "tok_in": 300}, enforce=True, url="u")
    assert [p.evidence_class for p in preds] == ["cited_share", "opinion", "reporting"]
    line = next(r.getMessage() for r in caplog.records if "event=jev_class" in r.getMessage())
    payload = json.loads(line.split("payload=", 1)[1])
    assert payload["mode"] == "enforce" and payload["changed"] == 2
    assert payload["rows"] == [["reported_fact", "cited_share", 0.8], ["opinion", None, 0.0], [None, "reporting", 0.7]]


def test_apply_shadow_changes_nothing():
    preds = _preds("reported_fact")
    jc.apply_jev_classes(preds, {"classes": ["cited_share"], "p": [0.8]}, enforce=False)
    assert preds[0].evidence_class == "reported_fact"


# ---------------------------------------------------------------- wiring in _process_article

def _sr():
    return SearchResult(
        title="A clear title about the event", url="http://x.com/1",
        snippet="A snippet long enough to clear the twenty-char fallback guard.",
        source="x.com", published_date="2026-07-15", _prefetched_text=ARTICLE,
    )


def _extractor(cls="reported_fact"):
    async def _extract(**kwargs):
        preds = [PredictionExtraction(quote="Overall, the opposition bloc gained one seat, rising to 55.",
                                      claim="c", stance=0.3, certainty=0.8, specificity=1.0,
                                      settled=None, evidence_class=cls)]
        return SimpleNamespace(predictions=preds, author_lean=None, author_lean_certainty=None,
                               consensus_view=None, claim_actor=None, claim_predicate=None,
                               claim_scope=None), {}
    return AsyncMock(side_effect=_extract)


async def _process(monkeypatch, *, enabled, enforce, jev):
    monkeypatch.setattr(api_settings, "jev_class_enabled", enabled)
    monkeypatch.setattr(api_settings, "jev_class_enforce", enforce)
    monkeypatch.setattr(api_settings, "jev_gate_enabled", False)
    monkeypatch.setattr(api_settings, "jev_shadow_enabled", False)
    monkeypatch.setattr(forecaster, "check_is_prediction", AsyncMock(return_value=(
        GatekeeperOutput(is_prediction=True, reason="judged", relevance_score=0.9), {})))
    monkeypatch.setattr(forecaster, "extract_predictions", _extractor())
    monkeypatch.setattr(forecaster, "enforce_deadline_arithmetic", lambda preds, dl, direction: preds)
    monkeypatch.setattr(forecaster, "jev_evidence_classes", jev)
    seen: list = []

    def provenance(preds):
        seen.extend(p.evidence_class for p in preds)
        return preds
    monkeypatch.setattr(forecaster, "enforce_anchor_provenance", provenance)
    shadow: list = []
    monkeypatch.setattr(forecaster, "fire_jev_class_shadow", lambda preds, **kw: shadow.append(kw))
    await forecaster._process_article(_sr(), QUESTION, max_article_chars=4000, timings=[],
                                      article_debugs=[], prediction_id="pid-1")
    return seen, shadow


@pytest.mark.parametrize("enabled", [False, True])
async def test_off_and_shadow_leave_haikus_class(monkeypatch, enabled):
    jev = AsyncMock(return_value={"classes": ["cited_share"], "p": [0.9]})
    seen, shadow = await _process(monkeypatch, enabled=enabled, enforce=False, jev=jev)
    assert seen == ["reported_fact"]
    jev.assert_not_awaited()
    assert len(shadow) == (1 if enabled else 0)
    if enabled:
        assert shadow[0]["quotes"] == ["Overall, the opposition bloc gained one seat, rising to 55."]


async def test_enforce_replaces_before_anchor_provenance(monkeypatch):
    jev = AsyncMock(return_value={"classes": ["cited_share"], "p": [0.9]})
    seen, shadow = await _process(monkeypatch, enabled=True, enforce=True, jev=jev)
    assert seen == ["cited_share"] and shadow == []
    assert jev.await_args.kwargs["question"] == QUESTION


async def test_enforce_fails_open_to_haiku(monkeypatch):
    jev = AsyncMock(return_value={"classes": [None], "p": [0.0], "err": "TimeoutError()"})
    seen, _ = await _process(monkeypatch, enabled=True, enforce=True, jev=jev)
    assert seen == ["reported_fact"]
