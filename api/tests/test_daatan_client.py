"""Tests for daatan_client.find_similar_forecasts / is_open.

httpx.MockTransport keeps these offline (pipeline/tests/test_net_guard.py's
convention). The key contract under test: a real failure (network error, non-200,
malformed body) must raise DaatanLookupError, never silently collapse to the same
`[]` a genuine zero-result response returns — retro#608's precursor-match shadow
step needs to tell a Daatan outage apart from "no match exists" in its logs.
"""

from datetime import datetime, timezone

import httpx
import pytest

from forecast_api import daatan_client


def _transport(handler):
    return httpx.MockTransport(handler)


class TestFindSimilarForecasts:
    async def test_empty_query_makes_no_call(self):
        def handler(request):
            raise AssertionError("should not be called for an empty query")

        result = await daatan_client.find_similar_forecasts("   ", transport=_transport(handler))

        assert result == []

    async def test_query_truncated_at_200_chars(self):
        seen = {}

        def handler(request):
            seen["q"] = request.url.params.get("q")
            return httpx.Response(200, json={"similar": []})

        long_query = "x" * 500
        await daatan_client.find_similar_forecasts(long_query, transport=_transport(handler))

        assert len(seen["q"]) == 200

    async def test_200_with_results_returns_similar_list(self):
        candidates = [
            {"id": "abc", "slug": "will-x-happen", "claimText": "Will X happen?", "status": "ACTIVE",
             "resolveByDatetime": "2027-01-01T00:00:00.000Z", "author": {"name": "A", "username": "a"}, "score": 0.91},
        ]

        def handler(request):
            return httpx.Response(200, json={"similar": candidates})

        result = await daatan_client.find_similar_forecasts("will x happen", transport=_transport(handler))

        assert result == candidates

    async def test_200_with_empty_similar_is_a_true_negative_not_an_error(self):
        def handler(request):
            return httpx.Response(200, json={"similar": []})

        result = await daatan_client.find_similar_forecasts("nothing matches this", transport=_transport(handler))

        assert result == []

    async def test_network_error_raises_lookup_error(self):
        def handler(request):
            raise httpx.ConnectError("connection refused")

        with pytest.raises(daatan_client.DaatanLookupError):
            await daatan_client.find_similar_forecasts("q", transport=_transport(handler))

    async def test_non_200_raises_lookup_error(self):
        def handler(request):
            return httpx.Response(500, text="internal error")

        with pytest.raises(daatan_client.DaatanLookupError):
            await daatan_client.find_similar_forecasts("q", transport=_transport(handler))

    async def test_malformed_json_raises_lookup_error(self):
        def handler(request):
            return httpx.Response(200, content=b"not json")

        with pytest.raises(daatan_client.DaatanLookupError):
            await daatan_client.find_similar_forecasts("q", transport=_transport(handler))

    async def test_missing_similar_key_raises_lookup_error(self):
        def handler(request):
            return httpx.Response(200, json={"unexpected": "shape"})

        with pytest.raises(daatan_client.DaatanLookupError):
            await daatan_client.find_similar_forecasts("q", transport=_transport(handler))

    async def test_error_and_true_negative_are_distinguishable(self):
        """The whole point of raising rather than swallowing: these two must not
        look the same to a caller (regression test mirroring
        test_net_guard.py's dead-link-vs-ssrf distinction)."""
        def not_found_handler(request):
            return httpx.Response(200, json={"similar": []})

        def error_handler(request):
            return httpx.Response(503, text="unavailable")

        not_found = await daatan_client.find_similar_forecasts("q", transport=_transport(not_found_handler))
        assert not_found == []

        with pytest.raises(daatan_client.DaatanLookupError):
            await daatan_client.find_similar_forecasts("q", transport=_transport(error_handler))


class TestIsOpen:
    def test_non_active_status_is_not_open(self):
        assert daatan_client.is_open({"status": "RESOLVED_YES", "resolveByDatetime": "2099-01-01T00:00:00Z"}) is False

    def test_active_with_future_deadline_is_open(self):
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        candidate = {"status": "ACTIVE", "resolveByDatetime": "2027-01-01T00:00:00Z"}

        assert daatan_client.is_open(candidate, now=now) is True

    def test_active_with_past_deadline_is_not_open(self):
        now = datetime(2028, 1, 1, tzinfo=timezone.utc)
        candidate = {"status": "ACTIVE", "resolveByDatetime": "2027-01-01T00:00:00Z"}

        assert daatan_client.is_open(candidate, now=now) is False

    def test_active_with_no_deadline_is_open(self):
        assert daatan_client.is_open({"status": "ACTIVE"}) is True

    def test_unparseable_deadline_defaults_open(self):
        assert daatan_client.is_open({"status": "ACTIVE", "resolveByDatetime": "not-a-date"}) is True
