"""Unit tests for web_search.py — no network calls, no API keys required.

Source-level tests (SQL text, OAuth scope, global declaration order) read the
file as plain text and require no dependencies beyond the stdlib.

Runtime tests (Tavily 432 flag, GDELT circuit breaker) import the module and
are skipped automatically when optional deps (httpx, google-cloud-bigquery,
boto3) are not installed in the test environment.
"""

import ast
import importlib
import inspect
import re
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Locate the source file — works regardless of cwd
# ---------------------------------------------------------------------------

_SRC = (Path(__file__).parent.parent / "src" / "tm" / "web_search.py").resolve()
assert _SRC.exists(), f"web_search.py not found at {_SRC}"
_SOURCE = _SRC.read_text()


# ---------------------------------------------------------------------------
# Marker: skip if the module can't be imported (missing deps)
# ---------------------------------------------------------------------------

def _can_import() -> bool:
    try:
        import httpx  # noqa: F401
        return True
    except ImportError:
        return False


needs_deps = pytest.mark.skipif(not _can_import(), reason="httpx / pipeline deps not installed")


def _fresh_ws():
    """Import web_search with reset globals (clears sys.modules cache)."""
    for key in list(sys.modules):
        if "tm.web_search" in key:
            del sys.modules[key]
    import tm.web_search as ws
    return ws


# ---------------------------------------------------------------------------
# Syntax & import checks
# ---------------------------------------------------------------------------

class TestSyntax:
    def test_file_parses_as_valid_python(self):
        """Source must parse without SyntaxError."""
        ast.parse(_SOURCE, filename=str(_SRC))

    def test_global_before_use_in_search_articles(self):
        """
        Regression (PR #128): _GDELT_DOC_BROKEN_UNTIL and _GDELT_DOC_FAIL_COUNT
        were declared global inside an else-block *after* being read earlier in
        search_articles(). Python 3.12+ raises SyntaxError for that pattern.

        Verify that the global declaration for these names appears before any read
        of _GDELT_DOC_BROKEN_UNTIL within search_articles().
        """
        tree = ast.parse(_SOURCE)
        func = next(
            n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "search_articles"
        )
        func_src_lines = _SOURCE.splitlines()[func.lineno - 1 : func.end_lineno]
        func_src = "\n".join(func_src_lines)

        m_global = re.search(r"global\s+[^\n]*_GDELT_DOC_BROKEN_UNTIL", func_src)
        m_read   = re.search(r"_GDELT_DOC_BROKEN_UNTIL\s*-\s*time", func_src)
        assert m_global, "search_articles must declare _GDELT_DOC_BROKEN_UNTIL global"
        assert m_read,   "search_articles must read _GDELT_DOC_BROKEN_UNTIL"
        assert m_global.start() < m_read.start(), (
            "global declaration must appear before first read of _GDELT_DOC_BROKEN_UNTIL"
        )


# ---------------------------------------------------------------------------
# GDELT BQ SQL correctness (source-level, no import needed)
# ---------------------------------------------------------------------------

class TestGdeltBqSql:
    def _bq_function_src(self) -> str:
        tree = ast.parse(_SOURCE)
        func = next(
            n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "_search_gdelt_bq"
        )
        lines = _SOURCE.splitlines()[func.lineno - 1 : func.end_lineno]
        return "\n".join(lines)

    def _sql_block(self) -> str:
        """Extract only the SQL string literal from _search_gdelt_bq."""
        src = self._bq_function_src()
        # Grab everything between the triple-quoted sql = f""" ... """
        m = re.search(r'sql\s*=\s*f"""(.*?)"""', src, re.DOTALL)
        assert m, "_search_gdelt_bq must contain a sql = f\"\"\"...\"\"\" block"
        return m.group(1)

    def test_uses_gkg_partitioned_with_partition_time(self):
        """
        Regression (PRs #127-#130): only gkg_partitioned is DAY-partitioned by
        ingestion time. The legacy `gkg` table lacks _PARTITIONTIME; the `gkg_*`
        wildcard has no matching shards. Must use gkg_partitioned + _PARTITIONTIME.
        """
        sql = self._sql_block()
        assert "gkg_partitioned" in sql, "SQL must query gkg_partitioned"
        assert "_PARTITIONTIME" in sql, "SQL must use _PARTITIONTIME for partition pruning"
        assert "_PARTITIONDATE" not in sql, "SQL must NOT use _PARTITIONDATE"
        assert "gkg_*" not in sql, "SQL must NOT use wildcard gkg_*"

    def test_partition_bounds_use_timestamp(self):
        """_PARTITIONTIME bounds must use TIMESTAMP() with ISO date strings."""
        sql = self._sql_block()
        assert "TIMESTAMP(" in sql, "Partition bounds must use TIMESTAMP()"
        assert "DATE(" not in sql, "Partition bounds must not use DATE()"

    def test_uses_full_bigquery_oauth_scope(self):
        """
        Regression (PR #127): scope was bigquery.readonly — jobs.create
        requires the full bigquery scope.
        """
        tree = ast.parse(_SOURCE)
        func = next(
            n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "_get_bq_client"
        )
        lines = _SOURCE.splitlines()[func.lineno - 1 : func.end_lineno]
        src = "\n".join(lines)
        assert "bigquery.readonly" not in src, "Must not use read-only scope"
        assert re.search(r"auth/bigquery[\"']", src), "Must use full bigquery scope"


# ---------------------------------------------------------------------------
# Tavily 432 quota exhaustion (requires import)
# ---------------------------------------------------------------------------

@needs_deps
class TestTavily432:
    def _make_response(self, status_code: int, text: str = ""):
        resp = MagicMock()
        resp.status_code = status_code
        resp.text = text
        resp.raise_for_status.side_effect = Exception(f"HTTP {status_code}")
        return resp

    def test_432_sets_quota_exhausted_flag(self):
        """Tavily 432 (plan limit) must set _TAVILY_QUOTA_EXHAUSTED and raise."""
        ws = _fresh_ws()
        ws.TAVILY_API_KEY = "fake-key"
        ws._TAVILY_QUOTA_EXHAUSTED = False

        with patch("httpx.post", return_value=self._make_response(432)):
            with pytest.raises(RuntimeError, match="432"):
                ws._search_tavily("test query", limit=5)

        assert ws._TAVILY_QUOTA_EXHAUSTED is True

    def test_432_checked_before_generic_raise_for_status(self):
        """432 must be an explicit branch, not fall through to raise_for_status."""
        src = inspect.getsource(_fresh_ws()._search_tavily)
        assert src.index("432") < src.index("raise_for_status"), \
            "432 check must precede raise_for_status"

    def test_exhausted_flag_skips_provider_in_chain(self):
        """When _TAVILY_QUOTA_EXHAUSTED is True, search_articles must not call _search_tavily."""
        ws = _fresh_ws()
        ws._TAVILY_QUOTA_EXHAUSTED = True
        ws.TAVILY_API_KEY = "fake-key"

        called = []
        real_fn = ws._search_tavily

        def _spy(*a, **kw):
            called.append(True)
            return real_fn(*a, **kw)

        skip = RuntimeError("skip")
        with patch.multiple(ws,
            _search_tavily=_spy,
            _search_gdelt=MagicMock(side_effect=skip),
            _search_gdelt_bq=MagicMock(side_effect=skip),
            _search_serpapi_news=MagicMock(side_effect=skip),
            _search_serper_news=MagicMock(side_effect=skip),
            _search_brave_news=MagicMock(side_effect=skip),
            _search_brightdata=MagicMock(side_effect=skip),
            _search_nimbleway=MagicMock(side_effect=skip),
            _search_scrapingbee=MagicMock(side_effect=skip),
            _search_newsdata_io=MagicMock(side_effect=skip),
            _search_dataforseo=MagicMock(side_effect=skip),
            _search_ddg_news=MagicMock(return_value=[]),
        ):
            ws.search_articles("test")

        assert not called, "_search_tavily must not be called when quota is exhausted"


# ---------------------------------------------------------------------------
# Google Custom Search provider (requires import)
# ---------------------------------------------------------------------------

class _FakeResp:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}
    def raise_for_status(self):
        pass
    def json(self):
        return self._payload


@needs_deps
class TestGoogleCseProvider:
    def test_inert_when_unconfigured(self):
        """Keys absent → _search_google_cse is never called and not in the chain.
        This is the load-bearing guarantee: shipping CSE changes nothing until the
        Google credentials are configured."""
        ws = _fresh_ws()
        ws.GOOGLE_CSE_API_KEY = None
        ws.GOOGLE_CSE_CX = None
        cse = MagicMock(return_value=[])
        skip = RuntimeError("skip")
        with patch.multiple(ws,
            _search_google_cse=cse,
            _search_gdelt=MagicMock(side_effect=skip),
            _search_gdelt_bq=MagicMock(side_effect=skip),
            _search_serpapi_news=MagicMock(side_effect=skip),
            _search_serper_news=MagicMock(side_effect=skip),
            _search_brave_news=MagicMock(side_effect=skip),
            _search_tavily=MagicMock(side_effect=skip),
            _search_brightdata=MagicMock(side_effect=skip),
            _search_nimbleway=MagicMock(side_effect=skip),
            _search_scrapingbee=MagicMock(side_effect=skip),
            _search_newsdata_io=MagicMock(side_effect=skip),
            _search_dataforseo=MagicMock(side_effect=skip),
            _search_ddg_news=MagicMock(return_value=[]),
        ):
            ws.search_articles("test query")
        assert not cse.called, "_search_google_cse must not run when unconfigured"
        assert "google_cse" not in ws.get_last_search_provider_chain()

    def test_runs_before_serpapi_when_configured(self):
        ws = _fresh_ws()
        ws.GOOGLE_CSE_API_KEY = "k"
        ws.GOOGLE_CSE_CX = "cx"
        ws._GOOGLE_CSE_QUOTA_EXHAUSTED = False
        ws.SERPAPI_API_KEY = "s"
        ws._SERPAPI_QUOTA_EXHAUSTED = False
        from tm.web_search import SearchResult
        hit = [SearchResult(title="t", url="http://x", snippet="s")]
        serp_spy = MagicMock(return_value=hit)
        with patch.multiple(ws,
            _search_gdelt=MagicMock(return_value=[]),
            _search_gdelt_bq=MagicMock(side_effect=RuntimeError("skip")),
            _search_google_cse=MagicMock(return_value=hit),
            _search_serpapi_news=serp_spy,
            _search_ddg_news=MagicMock(return_value=[]),
        ):
            res = ws.search_articles("Some live query")
        assert res
        assert ws.get_last_search_provider() == "google_cse"
        assert not serp_spy.called, "CSE must short-circuit before SerpAPI"
        chain = ws.get_last_search_provider_chain()
        assert chain.index("google_cse") < chain.index("serpapi") if "serpapi" in chain else True

    def test_parses_response_defensively(self, monkeypatch):
        ws = _fresh_ws()
        ws.GOOGLE_CSE_API_KEY = "k"
        ws.GOOGLE_CSE_CX = "cx"
        payload = {"items": [
            {"title": "T1", "link": "https://a.com/1", "snippet": "S1", "displayLink": "a.com",
             "pagemap": {"metatags": [{"article:published_time": "2026-06-01T10:00:00Z"}]}},
            {"title": "T2", "link": "https://b.com/2", "snippet": "S2"},  # no pagemap/displayLink
            {"title": "no link — dropped"},
        ]}
        monkeypatch.setattr(ws.httpx, "get", lambda *a, **k: _FakeResp(200, payload))
        out = ws._search_google_cse("q", 5)
        assert [r.url for r in out] == ["https://a.com/1", "https://b.com/2"]
        assert out[0].published_date == "2026-06-01"
        assert out[1].snippet == "S2"

    def test_429_sets_quota_exhausted(self, monkeypatch):
        ws = _fresh_ws()
        ws.GOOGLE_CSE_API_KEY = "k"
        ws.GOOGLE_CSE_CX = "cx"
        monkeypatch.setattr(ws.httpx, "get", lambda *a, **k: _FakeResp(429))
        monkeypatch.setattr(ws, "_persist_quota_state", lambda: None)
        with pytest.raises(RuntimeError, match="quota"):
            ws._search_google_cse("q", 5)
        assert ws._GOOGLE_CSE_QUOTA_EXHAUSTED is True


# ---------------------------------------------------------------------------
# GDELT BigQuery fallback ordering (requires import)
# ---------------------------------------------------------------------------

@needs_deps
class TestGdeltBqFallbackOrder:
    """gdelt_bq is low-relevance (URL-slug, recency-ranked). It must run EARLY
    only for historical queries and as a LAST resort for live/recent ones, so it
    doesn't short-circuit the chain before the real news providers."""

    def _ws_with_one_result_provider(self, ws, winner: str):
        """Configure keys and patch the chain so only *winner* returns a result;
        every other provider returns []. Returns the SearchResult list it yields."""
        from tm.web_search import SearchResult
        ws.GCP_SA_KEY_JSON = "fake"
        ws.SERPAPI_API_KEY = "fake"
        ws._SERPAPI_QUOTA_EXHAUSTED = False
        hit = [SearchResult(title="t", url="http://x", snippet="s")]

        def mk(name):
            return MagicMock(return_value=(hit if name == winner else []))

        return patch.multiple(ws,
            _search_gdelt=mk("gdelt"),
            _search_gdelt_bq=mk("gdelt_bq"),
            _search_serpapi_news=mk("serpapi"),
            _search_serper_news=mk("serper"),
            _search_tavily=mk("tavily"),
            _search_brave_news=mk("brave"),
            _search_brightdata=mk("brightdata"),
            _search_nimbleway=mk("nimbleway"),
            _search_scrapingbee=mk("scrapingbee"),
            _search_newsdata_io=mk("newsdata"),
            _search_dataforseo=mk("dataforseo"),
            _search_ddg_news=mk("ddg"),
        )

    def test_live_query_runs_gdelt_bq_after_ddg(self):
        """date_from=None (live forecast): gdelt_bq must be the absolute last
        resort — after the SERP providers AND after DDG, which returns relevant
        results (incl. Hebrew). gdelt_bq's low-relevance results must not
        short-circuit the chain before DDG."""
        ws = _fresh_ws()
        with self._ws_with_one_result_provider(ws, winner="gdelt_bq"):
            results = ws.search_articles("Israel Hamas ceasefire")
        assert results  # gdelt_bq still serves as the last resort when all else is empty
        chain = ws.get_last_search_provider_chain()
        assert ws.get_last_search_provider() == "gdelt_bq"
        assert "serpapi" in chain, "SERP providers must be attempted on a live query"
        assert "ddg" in chain, "DDG must be attempted on a live query"
        assert chain.index("gdelt_bq") > chain.index("serpapi"), \
            "gdelt_bq must come AFTER the SERP providers"
        assert chain.index("gdelt_bq") > chain.index("ddg"), \
            "gdelt_bq must come AFTER DDG (it is the absolute last resort)"

    def test_historical_query_uses_gdelt_bq_before_serp(self):
        """date_from older than the Doc window: gdelt_bq is the historical
        specialist and runs early, short-circuiting before the SERP providers."""
        from datetime import datetime, timedelta
        ws = _fresh_ws()
        old = datetime.utcnow() - timedelta(days=ws._GDELT_DOC_WINDOW_DAYS + 60)
        with self._ws_with_one_result_provider(ws, winner="gdelt_bq"):
            results = ws.search_articles("Assad regime falls", date_from=old)
        assert results
        chain = ws.get_last_search_provider_chain()
        assert ws.get_last_search_provider() == "gdelt_bq"
        assert "serpapi" not in chain, \
            "historical early gdelt_bq must short-circuit before the SERP providers"

    def test_gdelt_bq_not_tried_twice(self):
        """When the historical early path runs and returns empty, the late
        last-resort path must not call gdelt_bq again."""
        from datetime import datetime, timedelta
        ws = _fresh_ws()
        old = datetime.utcnow() - timedelta(days=ws._GDELT_DOC_WINDOW_DAYS + 60)
        # No provider returns anything → chain runs to the end.
        with self._ws_with_one_result_provider(ws, winner="__none__"):
            ws.search_articles("nothing matches", date_from=old)
        chain = ws.get_last_search_provider_chain()
        assert chain.count("gdelt_bq") == 1, "gdelt_bq must be attempted at most once"


# ---------------------------------------------------------------------------
# GDELT Doc circuit breaker (requires import)
# ---------------------------------------------------------------------------

@needs_deps
class TestGdeltDocCircuitBreaker:
    def _patched_chain(self, ws, gdelt_side_effect):
        """Context manager: patches the full provider chain, only GDELT is configurable."""
        skip = RuntimeError("skip")
        return patch.multiple(ws,
            _search_gdelt=MagicMock(side_effect=gdelt_side_effect),
            _search_gdelt_bq=MagicMock(side_effect=skip),
            _search_serpapi_news=MagicMock(side_effect=skip),
            _search_serper_news=MagicMock(side_effect=skip),
            _search_tavily=MagicMock(side_effect=skip),
            _search_brave_news=MagicMock(side_effect=skip),
            _search_brightdata=MagicMock(side_effect=skip),
            _search_nimbleway=MagicMock(side_effect=skip),
            _search_scrapingbee=MagicMock(side_effect=skip),
            _search_newsdata_io=MagicMock(side_effect=skip),
            _search_dataforseo=MagicMock(side_effect=skip),
            _search_ddg_news=MagicMock(return_value=[]),
        )

    def test_circuit_opens_after_threshold_failures(self):
        """After _GDELT_DOC_FAIL_THRESHOLD consecutive failures circuit opens."""
        ws = _fresh_ws()
        ws._GDELT_DOC_FAIL_COUNT = 0
        ws._GDELT_DOC_BROKEN_UNTIL = 0.0
        threshold = ws._GDELT_DOC_FAIL_THRESHOLD

        call_count = [0]

        def _fail(*a, **kw):
            call_count[0] += 1
            raise ConnectionError("simulated timeout")

        with self._patched_chain(ws, _fail):
            for _ in range(threshold):
                ws.search_articles(f"query")

            assert ws._GDELT_DOC_BROKEN_UNTIL > time.time(), "Circuit must be open"

            before = call_count[0]
            ws.search_articles("after circuit open")
            assert call_count[0] == before, "GDELT must not be called when circuit is open"

    def test_circuit_resets_on_success(self):
        """A successful GDELT result must reset the failure counter to 0."""
        ws = _fresh_ws()
        ws._GDELT_DOC_FAIL_COUNT = 1
        ws._GDELT_DOC_BROKEN_UNTIL = 0.0

        result = [ws.SearchResult(title="t", url="http://x.com", snippet="s")]

        with self._patched_chain(ws, result):
            ws.search_articles("query")

        assert ws._GDELT_DOC_FAIL_COUNT == 0
