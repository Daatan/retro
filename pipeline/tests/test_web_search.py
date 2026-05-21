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

    def test_uses_partition_date_not_partition_time(self):
        """
        Regression (PR #127): SQL used _PARTITIONTIME — unrecognised on
        date-partitioned tables. Must use _PARTITIONDATE.
        """
        src = self._bq_function_src()
        assert "_PARTITIONDATE" in src, "SQL must use _PARTITIONDATE"
        assert "_PARTITIONTIME" not in src, "SQL must NOT use _PARTITIONTIME"

    def test_bounds_cast_with_date_not_timestamp(self):
        """Partition bounds must use DATE(), not TIMESTAMP(), for a DATE-type column."""
        src = self._bq_function_src()
        assert re.search(r"DATE\(['\"]?\{ts_", src), "Bounds must use DATE({ts_...})"
        assert not re.search(r"TIMESTAMP\(['\"]?\{ts_", src), "Must not cast bounds with TIMESTAMP()"

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
