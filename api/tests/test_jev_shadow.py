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
        elif "when" in qs:  # retro#873: pick the past-Monday candidate if offered, else "none"
            pick = next((k for k, v in qs["when"]["criteria"].items() if "Monday (past)" in v), "none")
            answers = {"dated": {"type": "noul", "noul": 0.9},
                       "when": {"type": "choice", "probabilities": {pick: 0.8, "pub": 0.2}}}
        elif "stance" in qs:
            answers = {"stance": _score(5, 7), "strength": _score(3, 5),
                       "settled": {"type": "noul", "noul": 0.1},
                       "topic": {"type": "noul", "noul": 0.8}, "refs": {"type": "noul", "noul": 0.2},
                       "evidence_class": {"type": "choice", "probabilities": {
                           "reported_fact": 0.1, "cited_probability": 0.05, "cited_share": 0.05,
                           "reporting": 0.7, "opinion": 0.1}}}
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
              "stance": 0.7, "settled": False, "claim_strength": 0.6, "evidence_class": "reporting"}]
    with caplog.at_level(logging.INFO, logger="forecast_api.jev_shadow"):
        p = asyncio.run(js.run_jev_shadow(text=ARTICLE, question=QUESTION, url="u", haiku_predictions=haiku,
                                          api_key="k", transport=_transport(calls)))
    assert "err" not in p
    assert p["n"] == 3 and p["neg"] == 0.03 and p["max_noul"] == 0.9
    assert [c[0] for c in p["cand"]] == [0, 2, 1]         # top-3 by noul, best first
    assert p["cand"][0][2] == pytest.approx(0.7)           # stance expected, level 5
    assert p["cand"][0][3] == 0.7                          # stance argmax
    assert p["cand"][0][5] == pytest.approx(0.7)           # claim_strength level 3
    assert p["haiku"] == [[[0], 0.7, False, 0.6, "reporting", None]]
    assert "dates" not in p                                # unsettled: no date call
    assert p["cand"][0][6:10] == [0.7, 0.0, 0.8, 0.2]      # stance over non-zero levels, p(no signal), topic, refs
    assert p["cand"][0][10:] == ["reporting", 0.7]         # Jev evidence_class + its probability (retro#851)
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


def test_haiku_quoted_sentences_are_scored_beyond_the_cap():
    calls: list = []
    haiku = [{"quote": "Analysts expect the Bank of Israel to cut rates in October.", "stance": 0.7,
              "settled": False, "claim_strength": 0.6},
             {"quote": "not in the article", "stance": 0.1, "settled": False, "claim_strength": 0.3}]
    p = asyncio.run(js.run_jev_shadow(text=ARTICLE, question=QUESTION, url="u", haiku_predictions=haiku,
                                      api_key="k", select_bar=0.95, min_top=0, max_candidates=0,
                                      transport=_transport(calls)))
    assert [c[0] for c in p["cand"]] == [0]                # only the located Haiku sentence
    assert sum("stance" in c["questions"] for c in calls) == 1


# --- retro#850: pass 1 on its own, the skip-gate verdict, and reuse by the shadow ---------

def test_pass1_returns_nouls_and_max_noul():
    calls: list = []
    p1 = asyncio.run(js.jev_pass1(text=ARTICLE, question=QUESTION, api_key="k", transport=_transport(calls)))
    assert "err" not in p1 and "skip" not in p1
    assert p1["nouls"] == [0.9, 0.05, 0.8] and p1["max_noul"] == 0.9 and p1["neg"] == 0.03
    assert len(p1["sentences"]) == 3 and p1["tok_in"] == 200 and "ms" in p1
    assert len(calls) == 2                                  # selection + neg, no scoring


def test_pass1_never_raises():
    p1 = asyncio.run(js.jev_pass1(text=ARTICLE, question=QUESTION, api_key="k", transport=_transport([], fail=True)))
    assert "err" in p1 and "nouls" not in p1
    assert asyncio.run(js.jev_pass1(text="ok.", question=QUESTION, api_key="k"))["skip"] == "no_sentences"


def test_shadow_reuses_a_precomputed_pass1():
    calls: list = []
    p1 = asyncio.run(js.jev_pass1(text=ARTICLE, question=QUESTION, api_key="k", transport=_transport(calls)))
    before = len(calls)
    p = asyncio.run(js.run_jev_shadow(text=ARTICLE, question=QUESTION, url="u", haiku_predictions=[],
                                      api_key="k", transport=_transport(calls), pass1=p1))
    assert [c[0] for c in p["cand"]] == [0, 2, 1] and p["max_noul"] == 0.9
    assert not any("s0" in c["questions"] for c in calls[before:])   # selection asked exactly once
    assert len(calls) - before == 3                                  # only the three scorings


def test_shadow_with_a_failed_pass1_logs_err_without_calls(caplog):
    calls: list = []
    with caplog.at_level(logging.INFO, logger="forecast_api.jev_shadow"):
        p = asyncio.run(js.run_jev_shadow(text=ARTICLE, question=QUESTION, url="u", haiku_predictions=[],
                                          api_key="k", transport=_transport(calls),
                                          pass1={"norm_text": "x", "spans": [], "sentences": [], "err": "boom"}))
    assert p["err"] == "boom" and calls == []


def test_gate_verdict_and_fail_open():
    assert js.evaluate_jev_gate({"nouls": [0.05, 0.1], "max_noul": 0.1}, 0.15) == (True, 0.1)
    assert js.evaluate_jev_gate({"nouls": [0.05, 0.9], "max_noul": 0.9}, 0.15) == (False, 0.9)
    assert js.evaluate_jev_gate({"max_noul": 0.15, "nouls": [0.15]}, 0.15) == (False, 0.15)  # threshold is inclusive
    assert js.evaluate_jev_gate(None, 0.15) == (False, None)
    assert js.evaluate_jev_gate({"err": "boom"}, 0.15) == (False, None)
    assert js.evaluate_jev_gate({"skip": "no_key"}, 0.15) == (False, None)


def test_fire_pass1_keeps_a_strong_reference():
    async def go():
        t = js.fire_jev_pass1(text=ARTICLE, question=QUESTION, api_key="k", transport=_transport([]))
        assert t in js._TASKS
        r = await t
        assert r["max_noul"] == 0.9
    asyncio.run(go())
    assert js.fire_jev_pass1(text=ARTICLE, question=QUESTION, api_key="k") is None   # no loop


def test_script_of():
    assert js.script_of("Analysts expect the Bank of Israel to cut rates.") == "latin"
    assert js.script_of("הנגיד רמז כי הריבית תרד עוד לפני סוף השנה.") == "he"
    assert js.script_of("قال المحافظ إن الفائدة ستنخفض قبل نهاية العام.") == "ar"
    assert js.script_of("Аналитики ожидают снижения ставки в октябре.") == "cyr"
    assert js.script_of("Reuters: הנגיד רמז כי הריבית תרד עוד לפני סוף השנה (Bank of Israel).") == "he"
    assert js.script_of("12345 !!!") == "other"


def test_gate_flags_off_by_default():
    from forecast_api.config import ApiSettings
    s = ApiSettings(_env_file=None)
    assert s.jev_gate_enabled is False and s.jev_gate_enforce is False
    assert s.jev_gate_threshold == 0.15 and s.jev_gate_timeout_seconds == 8.0
    assert s.jev_gate_shadow_wait_seconds == 0.5


class TestSimplifyQuestion:
    """retro#849: the question rewrite that feeds the pass-1 A/B."""

    async def test_caches_per_question_and_strips(self):
        from forecast_api import jev_shadow as js
        js._SIMPLIFIED.clear()
        calls = []

        async def fake(model, prompt, *, system, max_tokens, temperature):
            calls.append(prompt)
            return ('  "Likud wins fewer than 20 Knesset seats"  \nignored second line', {})

        q = "Will Likud win fewer than 20 seats in the 2026 Israeli legislative election?"
        out = await js.simplify_question(q, model="m", completer=fake)
        assert out == "Likud wins fewer than 20 Knesset seats"
        again = await js.simplify_question(q, model="m", completer=fake)
        assert again == out and len(calls) == 1          # one cheap call per question, ever

    async def test_failures_return_empty_and_are_not_cached(self):
        from forecast_api import jev_shadow as js
        js._SIMPLIFIED.clear()

        async def boom(*a, **k):
            raise RuntimeError("bedrock down")

        async def empty(*a, **k):
            return ("", {})

        async def huge(*a, **k):
            return ("x" * 400, {})

        q = "Will Ra'am win a seat in the 2026 Knesset elections?"
        assert await js.simplify_question(q, model="m", completer=boom) == ""
        assert await js.simplify_question(q, model="m", completer=empty) == ""
        assert await js.simplify_question(q, model="m", completer=huge) == ""
        assert js._SIMPLIFIED == {}

    async def test_blank_question_short_circuits(self):
        from forecast_api import jev_shadow as js

        async def never(*a, **k):
            raise AssertionError("should not be called")

        assert await js.simplify_question("   ", model="m", completer=never) == ""

    def test_fingerprint_is_stable_and_distinct(self):
        from forecast_api.jev_shadow import question_fingerprint as fp
        assert fp("a") == fp("a") and len(fp("a")) == 8
        assert fp("a") != fp("b") and fp("") == ""


def test_evidence_class_question_and_missing_answer():
    """retro#851: the class question rides the existing pass-2 call (no extra request), with
    the offline-measured wording; a response lacking it logs (None, 0.0) instead of failing."""
    q = js._scoring_questions()["evidence_class"]
    assert q["type"] == "choice" and set(q["criteria"]) == {
        "reported_fact", "cited_probability", "cited_share", "reporting", "opinion"}
    assert js._top_choice(None) == (None, 0.0)
    assert js._top_choice({"probabilities": {"opinion": 0.6, "reporting": 0.4}}) == ("opinion", 0.6)


# --- retro#873: event_date shadow on Haiku-settled claims ---------------------------------

SETTLED = {"quote": "The weather in Tel Aviv was sunny on Monday.", "stance": 0.0, "settled": True,
           "claim_strength": 0.9, "event_date": "2026-09-16"}


def test_settled_claim_gets_a_jev_date_beside_haikus():
    calls: list = []
    p = asyncio.run(js.run_jev_shadow(text=ARTICLE, question=QUESTION, url="u", haiku_predictions=[SETTLED],
                                      api_key="k", transport=_transport(calls), article_date="2026-09-16T08:00:00Z"))
    assert "err" not in p
    date_calls = [c for c in calls if "when" in c["questions"]]
    assert len(date_calls) == 1
    state = date_calls[0]["state"]
    assert state["sentence"] == SETTLED["quote"] and state["publication_date"] == "2026-09-16"
    assert set(date_calls[0]["questions"]["when"]["criteria"]) >= {"pub", "none"}
    [row] = p["dates"]
    # Haiku said the publication date (a Wednesday); Jev picks the Monday before it.
    assert row[:4] == [0, "2026-09-16", "2026-09-14", "2026-09-14"]
    assert row[5:7] == [0.8, 0.9] and row[7] >= 2
    assert p["tok_in"] == 600                              # the date call is counted


def test_date_shadow_needs_a_publication_date_and_a_located_quote():
    for kw, haiku in (({}, [SETTLED]), ({"article_date": "2026-09-16"}, [dict(SETTLED, quote="not in the article")]),
                      ({"article_date": "garbage"}, [SETTLED])):
        calls: list = []
        p = asyncio.run(js.run_jev_shadow(text=ARTICLE, question=QUESTION, url="u", haiku_predictions=haiku,
                                          api_key="k", transport=_transport(calls), **kw))
        assert "dates" not in p and not any("when" in c["questions"] for c in calls)


def test_date_pick_none_and_failed_date_call_do_not_break_the_shadow():
    calls: list = []
    text = "Rates rose. The Bank of Israel cut rates."
    p = asyncio.run(js.run_jev_shadow(text=text, question=QUESTION, url="u", api_key="k", article_date="2026-09-16",
                                      haiku_predictions=[dict(SETTLED, quote="The Bank of Israel cut rates.")],
                                      transport=_transport(calls)))
    assert p["dates"][0][2:5] == [None, None, "none"]

    real = js._ask

    async def flaky(client, api_key, state, questions, api_url):
        if "when" in questions:
            raise httpx.ReadTimeout("slow")
        return await real(client, api_key, state, questions, api_url)
    js._ask, calls = flaky, []
    try:
        p = asyncio.run(js.run_jev_shadow(text=ARTICLE, question=QUESTION, url="u", haiku_predictions=[SETTLED],
                                          api_key="k", transport=_transport(calls), article_date="2026-09-16"))
    finally:
        js._ask = real
    assert "err" not in p and p["cand"]                    # pass 2 survives a failed date call
    assert p["dates"][0][4] == "err:ReadTimeout"


def test_one_failed_scoring_call_drops_only_its_row():
    calls: list = []
    real = js._ask
    seen = {"n": 0}

    async def flaky(client, api_key, state, questions, api_url):
        if "stance" in questions:
            seen["n"] += 1
            if seen["n"] == 1:
                raise httpx.HTTPStatusError("429", request=None, response=None)
        return await real(client, api_key, state, questions, api_url)
    js._ask = flaky
    try:
        p = asyncio.run(js.run_jev_shadow(text=ARTICLE, question=QUESTION, url="u", haiku_predictions=[SETTLED],
                                          api_key="k", transport=_transport(calls), article_date="2026-09-16"))
    finally:
        js._ask = real
    assert "err" not in p and p["score_err"] == 1
    assert len(p["cand"]) == 2 and p["dates"]
