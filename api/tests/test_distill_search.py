"""The forecaster must distill the NL question to keywords BEFORE searching.

GDELT (the usual chain winner) is a keyword matcher; feeding it a verbose
question ("Will X happen by 2027?") returns off-topic junk. Distilling first
restores relevance. If distilled keywords find nothing, fall back to verbatim.
"""

import asyncio

from forecast_api import forecaster
from forecast_api.forecaster import run_forecast
from forecast_api.models import ForecastRequest


def test_distilled_keywords_are_searched_first(monkeypatch):
    calls: list[str] = []

    def fake_search(q, limit):
        calls.append(q)
        return []  # force the no-results path; we only assert the query used

    async def fake_distill(_q):
        return "russia ukraine ceasefire"

    monkeypatch.setattr(forecaster, "search_articles", fake_search)
    monkeypatch.setattr(forecaster, "_distill_query", fake_distill)

    req = ForecastRequest(question="Will Russia and Ukraine reach a lasting ceasefire by 2027? [uniq-7k]")
    resp = asyncio.run(run_forecast(req))

    # Distilled keywords searched first…
    assert calls[0] == "russia ukraine ceasefire"
    # …then a verbatim fallback because distilled found nothing.
    assert calls[1] == req.question
    assert resp.reason == "no_search_results"


def test_no_verbatim_retry_when_distill_is_noop(monkeypatch):
    calls: list[str] = []

    def fake_search(q, limit):
        calls.append(q)
        return []

    async def fake_distill(q):
        return q  # distillation no-op (e.g. error fallback)

    monkeypatch.setattr(forecaster, "search_articles", fake_search)
    monkeypatch.setattr(forecaster, "_distill_query", fake_distill)

    req = ForecastRequest(question="Some question [uniq-9z]")
    asyncio.run(run_forecast(req))

    # distilled == verbatim → no redundant second search.
    assert calls == [req.question]
