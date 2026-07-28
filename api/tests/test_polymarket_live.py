"""Tests for polymarket_live.resolve_market's dispatch logic.

Only covers the wiring specific to this module (URL/id/slug branching, and the
prefer_open flag passed to the shared tm.polymarket keyword fallback) — the
search/ranking behavior itself is tested in pipeline's
test_polymarket_keyword_relevance.py, against the same tm.polymarket functions.
"""

from forecast_api import polymarket_live


class TestResolveMarketDispatch:
    async def test_url_input_goes_to_lookup_by_url(self, monkeypatch):
        seen = {}

        async def fake_lookup_by_url(url):
            seen["url"] = url
            return {"question": "from url"}

        monkeypatch.setattr(polymarket_live, "_lookup_by_url", fake_lookup_by_url)

        result = await polymarket_live.resolve_market("https://polymarket.com/event/some-slug")

        assert result == {"question": "from url"}
        assert seen["url"] == "https://polymarket.com/event/some-slug"

    async def test_numeric_input_goes_to_lookup_by_id(self, monkeypatch):
        async def fake_lookup_by_id(market_id):
            return {"question": "from id", "id": market_id}

        monkeypatch.setattr(polymarket_live, "_lookup_by_id", fake_lookup_by_id)

        result = await polymarket_live.resolve_market("618949")

        assert result == {"question": "from id", "id": "618949"}

    async def test_natural_language_falls_through_to_keyword_search_with_prefer_open(
        self, monkeypatch
    ):
        """The regression this file exists to guard: the live MCP path must ask
        for prefer_open=True, not the batch fetcher's default. Losing this kwarg
        silently reintroduces picking a stale closed market on a relevance tie
        (see tm.polymarket._lookup_by_keywords's docstring / #339 follow-up)."""
        seen = {}

        async def fake_lookup_by_url(url):
            return None  # synthesized slug URL never matches

        async def fake_lookup_by_keywords(keywords, event_name, *, prefer_open=False):
            seen["keywords"] = keywords
            seen["event_name"] = event_name
            seen["prefer_open"] = prefer_open
            return {"question": "from keyword search"}

        monkeypatch.setattr(polymarket_live, "_lookup_by_url", fake_lookup_by_url)
        monkeypatch.setattr(polymarket_live, "_lookup_by_keywords", fake_lookup_by_keywords)

        result = await polymarket_live.resolve_market("will bitcoin reach 200k")

        assert result == {"question": "from keyword search"}
        assert seen["prefer_open"] is True
        assert seen["keywords"] == ["will bitcoin reach 200k"]
        assert seen["event_name"] == "will bitcoin reach 200k"

    async def test_slug_match_short_circuits_keyword_search(self, monkeypatch):
        async def fake_lookup_by_url(url):
            return {"question": "matched by slug"}

        async def fake_lookup_by_keywords(*args, **kwargs):
            raise AssertionError("should not reach keyword search when slug matches")

        monkeypatch.setattr(polymarket_live, "_lookup_by_url", fake_lookup_by_url)
        monkeypatch.setattr(polymarket_live, "_lookup_by_keywords", fake_lookup_by_keywords)

        result = await polymarket_live.resolve_market("some-event-slug")

        assert result == {"question": "matched by slug"}

    async def test_empty_input_returns_none(self):
        assert await polymarket_live.resolve_market("") is None
        assert await polymarket_live.resolve_market("   ") is None
