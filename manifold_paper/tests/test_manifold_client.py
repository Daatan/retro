import ast
from pathlib import Path

import httpx
import pytest

from conftest import raw_market
from manifold_client import ManifoldClient, parse_market


class TestParseMarket:
    def test_open_binary(self):
        m = parse_market(raw_market("a", prob=0.37))
        assert (m.market_id, m.yes_price, m.unique_bettors, m.volume) == ("a", 0.37, 40, 2000.0)
        assert m.resolved_outcome is None and m.resolution_raw is None
        assert m.end_date.endswith("+00:00")

    @pytest.mark.parametrize("resolution,scoreable", [("YES", "YES"), ("NO", "NO"), ("MKT", None), ("CANCEL", None)])
    def test_resolution_mapping(self, resolution, scoreable):
        m = parse_market(raw_market(resolved=True, resolution=resolution))
        assert m.closed is True
        assert m.resolution_raw == resolution
        assert m.resolved_outcome == scoreable

    def test_non_binary_is_dropped(self):
        assert parse_market(raw_market(outcome_type="MULTIPLE_CHOICE")) is None
        assert parse_market({"outcomeType": "BINARY"}) is None  # no id

    def test_past_close_time_marks_closed_without_resolution(self):
        m = parse_market(raw_market(close_in_days=-1.0))
        assert m.closed is True and m.resolution_raw is None

    def test_out_of_range_probability_is_no_price(self):
        assert parse_market(raw_market(prob=1.5)).yes_price is None

    def test_description_prefers_plain_text(self):
        assert "ECB" in parse_market(raw_market(full=True)).description
        assert parse_market(raw_market()).description == ""


class TestClient:
    def _client(self, handler):
        return ManifoldClient(transport=httpx.MockTransport(handler))

    def test_search_dedupes_pages_and_filters_non_binary(self, mk):
        pages = {0: [mk("a"), mk("b"), mk("x", outcome_type="POLL")], 3: [mk("b"), mk("c")]}

        def handler(req: httpx.Request) -> httpx.Response:
            assert req.method == "GET" and req.url.path == "/v0/search-markets"
            assert req.url.params["contractType"] == "BINARY"
            return httpx.Response(200, json=pages.get(int(req.url.params["offset"]), []))

        with self._client(handler) as c:
            got = c.search_open_binary(limit=3, pages=3)
        assert [m.market_id for m in got] == ["a", "b", "c"]

    def test_market_404_is_none(self):
        with self._client(lambda r: httpx.Response(404, json={"message": "nope"})) as c:
            assert c.market("gone") is None

    def test_only_get_requests_are_ever_sent(self, mk):
        seen = []

        def handler(req: httpx.Request) -> httpx.Response:
            seen.append(req.method)
            return httpx.Response(200, json=[mk("a")] if "search" in req.url.path else mk("a", full=True))

        with self._client(handler) as c:
            c.search_open_binary(pages=1)
            c.market("a")
        assert seen and set(seen) == {"GET"}


class TestStructuralNoWrite:
    """Phase 1 must be unable to place a Mana bet: no write verbs, no auth."""

    TREE = ast.parse((Path(__file__).resolve().parent.parent / "manifold_client.py").read_text())

    def _strings(self):
        docstrings = {id(n.body[0].value) for n in ast.walk(self.TREE)
                      if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef))
                      and n.body and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
        return [n.value for n in ast.walk(self.TREE)
                if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docstrings]

    def test_only_get_is_ever_called(self):
        verbs = {n.func.attr for n in ast.walk(self.TREE) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        assert not verbs & {"post", "put", "delete", "patch", "request", "send", "stream"}
        assert "get" in verbs

    def test_only_two_read_endpoints_are_referenced(self):
        paths = {s for s in self._strings() if s.startswith("/v0/") or "/v0/" in s}
        assert paths == {"/v0/search-markets", "/v0/market/"}  # the f-string prefix of /v0/market/{id}

    def test_no_auth_header_or_key(self):
        joined = " ".join(self._strings()).lower()
        assert "authorization" not in joined and "key " not in joined and "bearer" not in joined
