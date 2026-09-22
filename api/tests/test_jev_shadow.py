"""retro#840 — Jev shadow extraction: segmentation, quote location, both passes, and that it
never raises or blocks /forecast."""
import asyncio
import json
import logging

import httpx
import pytest

from forecast_api import jev_shadow as js

ARTICLE = (
    "Analysts expect the Bank of Israel to cut rates in October. "
    "The weather in Tel Aviv was sunny on Monday.\n"
    "הנגיד רמז כי הריבית תרד עוד לפני סוף השנה."
)
QUESTION = "Will the Bank of Israel cut its interest rate before the end of 2026?"


def test_segment_splits_latin_hebrew_and_newlines():
    _, spans = js.segment(ARTICLE)
    assert len(spans) == 3


def test_segment_drops_fragments_shorter_than_a_claim():
    _, spans = js.segment("Yes. No. Analysts expect a cut in October this year.")
    assert len(spans) == 1


def test_locate_quote_exact_and_multi_sentence():
    text, spans = js.segment(ARTICLE)
    assert js.locate_quote(text, spans, "Analysts expect the Bank of Israel to cut rates in October.") == [0]
    both = "Analysts expect the Bank of Israel to cut rates in October. The weather in Tel Aviv was sunny on Monday."
    assert js.locate_quote(text, spans, both) == [0, 1]


def test_locate_quote_normalises_curly_quotes_and_gershayim():
    text, spans = js.segment('The IDF said “operations continue” in Rafah today. צה"ל הודיע על כך.')
    assert js.locate_quote(text, spans, 'The IDF said "operations continue" in Rafah today.') == [0]
    assert js.locate_quote(text, spans, "צה״ל הודיע על כך.") == [1]


def test_locate_quote_paraphrase_is_empty():
    text, spans = js.segment(ARTICLE)
    assert js.locate_quote(text, spans, "Economists foresee a rate reduction soon.") == []


def _score(level: int, n: int) -> dict:
    return {"type": "score", "score": float(level), "confidence": 0.9,
            "probabilities": {str(k): (1.0 if k == level else 0.0) for k in range(n)}}


def _transport(calls: list, *, fail: bool = False, urls: list | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body)
        urls.append(str(request.url)) if urls is not None else None
        assert request.headers["Authorization"] == "Bearer k"
        if fail:
            return httpx.Response(500, json={"error": "boom"})
        qs = body["questions"]
        if "neg" in qs:
            answers = {"neg": {"type": "noul", "noul": 0.03}}
        elif "stance" in qs:
            answers = {"stance": _score(5, 7), "strength": _score(3, 5),
                       "settled": {"type": "noul", "noul": 0.1}}
        else:  # selection: sentence 0 and 2 bear on the question, 1 does not
            answers = {k: {"type": "noul", "noul": {"s0": 0.9, "s1": 0.05, "s2": 0.8}[k]} for k in qs}
        return httpx.Response(200, json={"model": "jev-1.13.0", "answers": answers,
                                         "usage": {"input_tokens": 100, "output_tokens": 10}})
    return httpx.MockTransport(handler)


@pytest.fixture(autouse=True)
def _clear_caches():
    js._NEG_CACHE.clear()
    js._KEY.clear()
    yield
    js._NEG_CACHE.clear()
    js._KEY.clear()


def test_run_selects_scores_and_logs(caplog):
    calls: list = []
    haiku = [{"quote": "Analysts expect the Bank of Israel to cut rates in October.",
              "stance": 0.7, "settled": False, "claim_strength": 0.6}]
    with caplog.at_level(logging.INFO, logger="forecast_api.jev_shadow"):
        p = asyncio.run(js.run_jev_shadow(text=ARTICLE, question=QUESTION, url="u", haiku_predictions=haiku,
                                          api_key="k", transport=_transport(calls)))
    assert "err" not in p
    assert p["n"] == 3 and p["neg"] == 0.03 and p["max_noul"] == 0.9
    assert [c[0] for c in p["cand"]] == [0, 2, 1]         # top-3 by noul, best first
    assert p["cand"][0][2] == pytest.approx(0.7)           # stance expected, level 5
    assert p["cand"][0][3] == 0.7                          # stance argmax
    assert p["cand"][0][5] == pytest.approx(0.7)           # claim_strength level 3
    assert p["haiku"] == [[[0], 0.7, False, 0.6]]
    assert p["cand"][0][6:] == [0.7, 0.0]                  # stance over non-zero levels, p(no signal)
    assert p["tok_in"] == 500                              # selection + neg + 3 scorings
    assert len(calls) == 5
    line = next(r.getMessage() for r in caplog.records if "event=jev_shadow" in r.getMessage())
    assert json.loads(line.split("payload=", 1)[1])["cand"] == p["cand"]


def test_negation_verdict_is_cached_per_question():
    calls: list = []
    for _ in range(2):
        asyncio.run(js.run_jev_shadow(text=ARTICLE, question=QUESTION, url="u", haiku_predictions=[],
                                      api_key="k", transport=_transport(calls)))
    assert sum("neg" in c["questions"] for c in calls) == 1


def test_select_bar_and_max_candidates_limit_pass_two():
    calls: list = []
    p = asyncio.run(js.run_jev_shadow(text=ARTICLE, question=QUESTION, url="u", haiku_predictions=[],
                                      api_key="k", select_bar=0.5, min_top=0, max_candidates=1, transport=_transport(calls)))
    assert [c[0] for c in p["cand"]] == [0]
    assert sum("stance" in c["questions"] for c in calls) == 1


def test_api_failure_is_swallowed_and_logged(caplog):
    with caplog.at_level(logging.INFO, logger="forecast_api.jev_shadow"):
        p = asyncio.run(js.run_jev_shadow(text=ARTICLE, question=QUESTION, url="u", haiku_predictions=[],
                                          api_key="k", transport=_transport([], fail=True)))
    assert "err" in p and "cand" not in p
    assert any("event=jev_shadow" in r.getMessage() for r in caplog.records)


def test_empty_article_skips_without_calls():
    calls: list = []
    p = asyncio.run(js.run_jev_shadow(text="ok.", question=QUESTION, url="u", haiku_predictions=[],
                                      api_key="k", transport=_transport(calls)))
    assert p["skip"] == "no_sentences" and calls == []


def test_fire_outside_event_loop_is_a_noop():
    assert js.fire_jev_shadow(text=ARTICLE, question=QUESTION, url="u", haiku_predictions=[], api_key="k") is None


def test_fire_keeps_a_strong_reference_until_done():
    async def go():
        task = js.fire_jev_shadow(text=ARTICLE, question=QUESTION, url="u", haiku_predictions=[],
                                  api_key="k", transport=_transport([]))
        assert task in js._TASKS
        await task
        await asyncio.sleep(0)
        assert task not in js._TASKS
    asyncio.run(go())


def test_disabled_by_default():
    from forecast_api.config import ApiSettings
    s = ApiSettings(_env_file=None)
    assert s.jev_shadow_enabled is False and s.typesafe_api_key == ""


def test_key_falls_back_to_ssm_once(monkeypatch):
    import tm.web_search
    asked = []
    monkeypatch.setattr(tm.web_search, "_secret", lambda env, name: asked.append(name) or "k")
    for _ in range(2):
        calls = []
        p = asyncio.run(js.run_jev_shadow(text=ARTICLE, question=QUESTION, url="u", haiku_predictions=[],
                                          transport=_transport(calls)))
        assert "err" not in p and calls
    assert asked == [js.KEY_SSM_NAME]


def test_missing_key_skips_loudly(monkeypatch, caplog):
    import tm.web_search
    monkeypatch.setattr(tm.web_search, "_secret", lambda env, name: None)
    calls = []
    with caplog.at_level(logging.INFO, logger="forecast_api.jev_shadow"):
        p = asyncio.run(js.run_jev_shadow(text=ARTICLE, question=QUESTION, url="u", haiku_predictions=[],
                                          transport=_transport(calls)))
    assert p["skip"] == "no_key" and calls == []
    assert '"skip":"no_key"' in caplog.text


def test_api_url_is_configurable():
    calls, urls = [], []
    asyncio.run(js.run_jev_shadow(text=ARTICLE, question=QUESTION, url="u", haiku_predictions=[], api_key="k",
                                  api_url="https://openrouter.ai/api/v1/systemone",
                                  transport=_transport(calls, urls=urls)))
    assert urls and all(u == "https://openrouter.ai/api/v1/systemone" for u in urls)


def test_bar_adds_below_top_k_and_top_k_floor_applies():
    calls: list = []
    p = asyncio.run(js.run_jev_shadow(text=ARTICLE, question=QUESTION, url="u", haiku_predictions=[],
                                      api_key="k", select_bar=0.5, min_top=1, transport=_transport(calls)))
    assert [c[0] for c in p["cand"]] == [0, 2]            # top-1 (s0) plus s2 at 0.8 >= bar; s1 dropped
    p = asyncio.run(js.run_jev_shadow(text=ARTICLE, question=QUESTION, url="u", haiku_predictions=[],
                                      api_key="k", select_bar=0.95, min_top=2, transport=_transport(calls)))
    assert [c[0] for c in p["cand"]] == [0, 2]            # nothing clears 0.95; top-2 floor still scored


def test_nonzero_stance_splits_no_signal_mass():
    ans = {"probabilities": {"3": 0.5, "5": 0.25, "1": 0.25}}   # half "no signal", rest split +0.7 / -0.7
    ex, p0 = js._nonzero(ans, js.STANCE_VALUES)
    assert p0 == 0.5 and ex == pytest.approx(0.0)
    ex, p0 = js._nonzero({"probabilities": {"3": 0.6, "5": 0.4}}, js.STANCE_VALUES)
    assert ex == pytest.approx(0.7) and p0 == 0.6
    assert js._nonzero({"probabilities": {"3": 1.0}}, js.STANCE_VALUES) == (0.0, 1.0)
