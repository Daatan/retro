"""Tests for the /search wrapper and provider-health checks (searcher.py).

Covers two observability fixes:
  * run_search must capture the thread-local search provider *inside the worker
    thread* (reading it in the event-loop thread always returned "none").
  * _check_gdelt must surface GDELT's runtime circuit/cooldown state instead of a
    hardcoded "ok" that hid the Doc-API throttling.
"""

import time

import pytest

from forecast_api import searcher
from forecast_api.models import SearchRequest
from tm.web_search import SearchResult


class TestRunSearchProviderAttribution:
    async def test_captures_provider_from_worker_thread(self, monkeypatch):
        def fake_search(query, limit, date_from, date_to):
            # search_articles sets these thread-locals in the worker thread.
            searcher._ws._provider_local.name = "gdelt"
            searcher._ws._provider_local.chain = ["gdelt", "serpapi"]
            return [SearchResult(title="t", url="http://x", snippet="s")]

        monkeypatch.setattr(searcher._ws, "search_articles", fake_search)
        resp = await searcher.run_search(SearchRequest(query="ceasefire talks", limit=3))

        assert resp.provider == "gdelt"           # was "none" before the fix
        assert resp.provider_chain == ["gdelt", "serpapi"]
        assert resp.count == 1


class TestCheckGdeltHealth:
    async def test_reports_cooldown(self, monkeypatch):
        monkeypatch.setattr(searcher._ws, "_GDELT_DOC_BROKEN_UNTIL", 0.0)
        monkeypatch.setattr(searcher._ws, "_GDELT_COOLDOWN_UNTIL", time.time() + 30)
        st = await searcher._check_gdelt()
        assert st.status == "cooldown"
        assert st.exhausted is True

    async def test_reports_circuit_open(self, monkeypatch):
        monkeypatch.setattr(searcher._ws, "_GDELT_DOC_BROKEN_UNTIL", time.time() + 120)
        monkeypatch.setattr(searcher._ws, "_GDELT_COOLDOWN_UNTIL", 0.0)
        st = await searcher._check_gdelt()
        assert st.status == "circuit_open"
        assert st.exhausted is True

    async def test_ok_when_idle(self, monkeypatch):
        monkeypatch.setattr(searcher._ws, "_GDELT_DOC_BROKEN_UNTIL", 0.0)
        monkeypatch.setattr(searcher._ws, "_GDELT_COOLDOWN_UNTIL", 0.0)
        st = await searcher._check_gdelt()
        assert st.status == "ok"
        assert st.exhausted is False
