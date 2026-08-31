"""Per-key API clients and the max_articles cap (docs#57 item 1).

Staging shared the prod ``ORACLE_API_KEY`` and drove ~64 unmetered ``/forecast``
calls/day into the prod Oracul; stopping the caller fixed the traffic but not
the structure. These tests lock the structural half: named keys resolve to a
capped ``ApiKeyClient``, the cap clamps both the search limit and the
caller-supplied ``articles`` list (which used to bypass every cap), and a
malformed ``ORACLE_API_KEYS`` fails closed instead of open.
"""

import json

import pytest
from fastapi import HTTPException

from forecast_api import forecaster
from forecast_api.auth import ApiKeyClient, verify_api_key
from forecast_api.config import settings as api_settings
from forecast_api.models import ArticleInput, ForecastRequest

NAMED = json.dumps({"staging": {"key": "staging-key", "max_articles": 3}})


class TestVerifyApiKey:
    async def test_primary_key_resolves_to_uncapped_default(self):
        client = await verify_api_key(api_settings.oracle_api_key)
        assert client == ApiKeyClient(name="default", max_articles=None)

    async def test_named_key_resolves_to_capped_client(self, monkeypatch):
        monkeypatch.setattr(api_settings, "oracle_api_keys", NAMED)
        client = await verify_api_key("staging-key")
        assert client == ApiKeyClient(name="staging", max_articles=3)

    async def test_wrong_key_is_401(self, monkeypatch):
        monkeypatch.setattr(api_settings, "oracle_api_keys", NAMED)
        with pytest.raises(HTTPException) as exc:
            await verify_api_key("not-a-key")
        assert exc.value.status_code == 401

    async def test_malformed_config_fails_closed(self, monkeypatch):
        # Named keys 401 (never an uncapped bypass); the primary key still works.
        monkeypatch.setattr(api_settings, "oracle_api_keys", "{not json")
        with pytest.raises(HTTPException):
            await verify_api_key("staging-key")
        assert (await verify_api_key(api_settings.oracle_api_key)).name == "default"

    async def test_invalid_entry_disables_all_named_keys(self, monkeypatch):
        bad = json.dumps({"a": {"key": "k1", "max_articles": 0}, "b": {"key": "k2"}})
        monkeypatch.setattr(api_settings, "oracle_api_keys", bad)
        with pytest.raises(HTTPException):
            await verify_api_key("k2")


class _CaptureInner:
    """Stub for _run_forecast_inner that records the resolved limit/request."""

    def __init__(self):
        self.limit = None
        self.req = None

    async def __call__(self, req, cache_key, limit, total_start):
        self.req = req
        self.limit = limit
        return forecaster._empty_response(req.question)


@pytest.fixture
def inner(monkeypatch):
    capture = _CaptureInner()
    monkeypatch.setattr(forecaster, "_run_forecast_inner", capture)
    return capture


class TestPerKeyCap:
    async def test_cap_clamps_requested_limit(self, inner):
        await forecaster.run_forecast(
            ForecastRequest(question="per-key cap clamps limit — unique q1", max_articles=10),
            client=ApiKeyClient(name="staging", max_articles=3),
        )
        assert inner.limit == 3

    async def test_cap_clamps_server_default_too(self, inner, monkeypatch):
        monkeypatch.setattr(api_settings, "max_articles", 10)
        await forecaster.run_forecast(
            ForecastRequest(question="per-key cap clamps default — unique q2"),
            client=ApiKeyClient(name="staging", max_articles=3),
        )
        assert inner.limit == 3

    async def test_uncapped_client_keeps_requested_limit(self, inner):
        await forecaster.run_forecast(
            ForecastRequest(question="uncapped default client — unique q3", max_articles=10),
            client=ApiKeyClient(name="default"),
        )
        assert inner.limit == 10

    async def test_no_client_keeps_requested_limit(self, inner):
        # MCP lane calls run_forecast without a client — behavior unchanged.
        await forecaster.run_forecast(
            ForecastRequest(question="clientless call — unique q4", max_articles=10),
        )
        assert inner.limit == 10

    async def test_cap_truncates_supplied_articles(self, inner):
        articles = [ArticleInput(url=f"https://example.com/{i}") for i in range(5)]
        await forecaster.run_forecast(
            ForecastRequest(question="supplied articles truncated — unique q5", articles=articles),
            client=ApiKeyClient(name="staging", max_articles=3),
        )
        assert len(inner.req.articles) == 3
        assert [a.url for a in inner.req.articles] == [a.url for a in articles[:3]]

    async def test_uncapped_client_keeps_all_supplied_articles(self, inner):
        articles = [ArticleInput(url=f"https://example.com/{i}") for i in range(5)]
        await forecaster.run_forecast(
            ForecastRequest(question="supplied articles kept — unique q6", articles=articles),
            client=ApiKeyClient(name="default"),
        )
        assert len(inner.req.articles) == 5
