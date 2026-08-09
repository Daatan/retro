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
import logging
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
        # The provider chain (and the GDELT global) now lives in _search_articles_chain;
        # search_articles is a thin wrapper that also warms the news-indexer cache.
        func = next(
            n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "_search_articles_chain"
        )
        func_src_lines = _SOURCE.splitlines()[func.lineno - 1 : func.end_lineno]
        func_src = "\n".join(func_src_lines)

        m_global = re.search(r"global\s+[^\n]*_GDELT_DOC_BROKEN_UNTIL", func_src)
        m_read   = re.search(r"_GDELT_DOC_BROKEN_UNTIL\s*-\s*time", func_src)
        assert m_global, "_search_articles_chain must declare _GDELT_DOC_BROKEN_UNTIL global"
        assert m_read,   "_search_articles_chain must read _GDELT_DOC_BROKEN_UNTIL"
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
# Trusted-sites last-resort fallback (requires import)
# ---------------------------------------------------------------------------

@needs_deps
class TestTrustedSitesFallback:
    def test_builds_batched_site_queries_and_dedups(self, monkeypatch):
        ws = _fresh_ws()
        from tm.web_search import SearchResult
        calls = []

        def fake_ddg(q, limit, date_from=None, date_to=None):
            calls.append(q)
            n = len(calls)
            # a unique result per batch + one shared URL across batches (dedup probe)
            return [SearchResult(title=f"t{n}", url=f"http://x/{n}", snippet="s"),
                    SearchResult(title="dup", url="http://dup", snippet="s")]

        monkeypatch.setattr(ws, "_search_ddg_news", fake_ddg)
        out = ws._search_trusted_sites("ceasefire talks", limit=5)

        assert all(q.startswith("ceasefire talks (") and "site:" in q and " OR " in q for q in calls)
        urls = [r.url for r in out]
        assert urls.count("http://dup") == 1, "results must be deduped across batches"
        assert len(out) <= 5

    def test_stops_early_once_limit_reached(self, monkeypatch):
        ws = _fresh_ws()
        from tm.web_search import SearchResult
        calls = []

        def fake_ddg(q, limit, date_from=None, date_to=None):
            calls.append(q)
            n = len(calls)
            return [SearchResult(title=f"{n}-{i}", url=f"http://x/{n}/{i}", snippet="s") for i in range(5)]

        monkeypatch.setattr(ws, "_search_ddg_news", fake_ddg)
        out = ws._search_trusted_sites("q", limit=5)
        assert len(calls) == 1, "first batch already met the limit → no further DDG calls"
        assert len(out) == 5

    def test_respects_max_batches(self, monkeypatch):
        ws = _fresh_ws()
        calls = {"n": 0}

        def fake_ddg(q, limit, date_from=None, date_to=None):
            calls["n"] += 1
            return []

        monkeypatch.setattr(ws, "_search_ddg_news", fake_ddg)
        out = ws._search_trusted_sites("q", limit=5)
        assert calls["n"] == ws._TRUSTED_SITES_MAX_BATCHES
        assert out == []

    def test_runs_after_ddg_before_gdelt_bq_when_all_empty(self):
        ws = _fresh_ws()
        ws.GCP_SA_KEY_JSON = "fake"  # so gdelt_bq would otherwise be tried
        from tm.web_search import SearchResult
        hit = [SearchResult(title="t", url="http://x", snippet="s")]
        bq_spy = MagicMock(return_value=hit)
        # Patch the FULL provider set empty (real keys may be loaded from secrets,
        # so an unpatched provider could actually serve and pre-empt the fallback).
        with patch.multiple(ws,
            _search_gdelt=MagicMock(return_value=[]),
            _search_google_cse=MagicMock(return_value=[]),
            _search_serpapi_news=MagicMock(return_value=[]),
            _search_serper_news=MagicMock(return_value=[]),
            _search_brave_news=MagicMock(return_value=[]),
            _search_tavily=MagicMock(return_value=[]),
            _search_brightdata=MagicMock(return_value=[]),
            _search_nimbleway=MagicMock(return_value=[]),
            _search_scrapingbee=MagicMock(return_value=[]),
            _search_newsdata_io=MagicMock(return_value=[]),
            _search_dataforseo=MagicMock(return_value=[]),
            _search_ddg_news=MagicMock(return_value=[]),       # plain DDG empty
            _search_trusted_sites=MagicMock(return_value=hit),  # trusted serves
            _search_gdelt_bq=bq_spy,
        ):
            res = ws.search_articles("live query")
        assert res
        assert ws.get_last_search_provider() == "trusted_sites"
        assert not bq_spy.called, "trusted_sites must short-circuit before gdelt_bq"
        chain = ws.get_last_search_provider_chain()
        assert chain.index("trusted_sites") > chain.index("ddg")

    def test_skipped_when_earlier_provider_serves(self):
        ws = _fresh_ws()
        from tm.web_search import SearchResult
        hit = [SearchResult(title="t", url="http://x", snippet="s")]
        ts_spy = MagicMock(return_value=hit)
        with patch.multiple(ws,
            _search_gdelt=MagicMock(return_value=hit),  # serves first
            _search_ddg_news=MagicMock(return_value=[]),
            _search_trusted_sites=ts_spy,
        ):
            ws.search_articles("q")
        assert not ts_spy.called


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
# news-indexer provider (requires import)
# ---------------------------------------------------------------------------

@needs_deps
class TestNewsIndexerProvider:
    def test_inert_when_unconfigured(self, monkeypatch):
        """Both env vars absent → httpx never called, 'news_indexer' not in chain.
        This is the load-bearing guarantee: the block is a no-op until secrets
        are configured in Secrets Manager."""
        ws = _fresh_ws()
        ws.NEWS_INDEXER_URL = None
        ws.NEWS_INDEXER_API_KEY = None
        called = []
        monkeypatch.setattr(ws.httpx, "get", lambda *a, **k: called.append(a) or (_ for _ in ()).throw(RuntimeError("must not be called")))
        with patch.multiple(ws,
            _search_gdelt=MagicMock(return_value=[]),
            _search_gdelt_bq=MagicMock(return_value=[]),
            _search_serpapi_news=MagicMock(return_value=[]),
            _search_serper_news=MagicMock(return_value=[]),
            _search_brave_news=MagicMock(return_value=[]),
            _search_tavily=MagicMock(return_value=[]),
            _search_brightdata=MagicMock(return_value=[]),
            _search_nimbleway=MagicMock(return_value=[]),
            _search_scrapingbee=MagicMock(return_value=[]),
            _search_newsdata_io=MagicMock(return_value=[]),
            _search_dataforseo=MagicMock(return_value=[]),
            _search_ddg_news=MagicMock(return_value=[]),
        ):
            ws.search_articles("test query")
        assert not called, "httpx.get must not be called when news-indexer is unconfigured"
        assert "news_indexer" not in ws.get_last_search_provider_chain()

    def test_runs_before_gdelt_when_configured(self, monkeypatch):
        """Configured + non-empty response → returns before GDELT; provider is 'news_indexer'."""
        ws = _fresh_ws()
        ws.NEWS_INDEXER_URL = "http://ni.local"
        ws.NEWS_INDEXER_API_KEY = "secret"
        from tm.web_search import SearchResult
        hit = [{"title": "T", "url": "http://x.com/1", "snippet": "S", "source": "x.com", "published_date": "2026-06-01"}]
        monkeypatch.setattr(ws.httpx, "get", lambda *a, **k: _FakeResp(200, hit))
        gdelt_spy = MagicMock(return_value=[])
        with patch.multiple(ws, _search_gdelt=gdelt_spy):
            res = ws.search_articles("some query")
        assert res
        assert res[0].url == "http://x.com/1"
        assert ws.get_last_search_provider() == "news_indexer"
        assert not gdelt_spy.called, "GDELT must not run when news-indexer returns hits"

    def test_falls_through_on_empty_list(self, monkeypatch):
        """Empty list response → fall through to GDELT; 'news_indexer' still appears in chain."""
        ws = _fresh_ws()
        ws.NEWS_INDEXER_URL = "http://ni.local"
        ws.NEWS_INDEXER_API_KEY = "secret"
        from tm.web_search import SearchResult
        gdelt_hit = [SearchResult(title="G", url="http://g.com/1", snippet="gdelt")]
        monkeypatch.setattr(ws.httpx, "get", lambda *a, **k: _FakeResp(200, []))
        with patch.multiple(ws,
            _search_gdelt=MagicMock(return_value=gdelt_hit),
            _search_gdelt_bq=MagicMock(return_value=[]),
        ):
            res = ws.search_articles("some query")
        assert res
        assert res[0].url == "http://g.com/1"
        assert ws.get_last_search_provider() == "gdelt"
        assert "news_indexer" in ws.get_last_search_provider_chain()

    # ── The /search contract (retro#459) ──────────────────────────────────────────────
    #
    # `SearchResult(**h)` turned news-indexer's response into a strict schema enforced by
    # TypeError, inside a bare `except Exception`. One added key upstream would take the
    # free first-in-chain provider offline for every retro caller, move the Oracle onto
    # paid SERP, and swap the local semantic index for GDELT keyword matching — with no
    # counter moving and only a WARNING line to show for it. news-indexer has already
    # widened the sibling `/context` payload twice (relevance/isPrediction, then
    # personId/outletId); daatan survived both only because Zod strips unknown keys.

    def test_documented_search_payload_shape_is_the_contract(self):
        """The five keys news-indexer's /search returns today. This test is the tripwire:
        if SearchResult's public fields change, check the other side of the wire before
        editing this set — the two repos have no shared schema to enforce it."""
        from tm.web_search import _SEARCH_RESULT_WIRE_FIELDS
        assert _SEARCH_RESULT_WIRE_FIELDS == {
            "title", "url", "snippet", "source", "published_date",
        }

    def test_unknown_key_does_not_take_the_provider_offline(self, monkeypatch, caplog):
        """The actual regression. An added upstream key must be ignored, not fatal —
        and must still be visible in the log, because it means the contract moved."""
        ws = _fresh_ws()
        ws.NEWS_INDEXER_URL = "http://ni.local"
        ws.NEWS_INDEXER_API_KEY = "secret"
        hit = [{
            "title": "T", "url": "http://x.com/1", "snippet": "S",
            "source": "x.com", "published_date": "2026-06-01",
            "relevance": 0.82, "personId": "p-1",     # the next widening, whatever it is
        }]
        monkeypatch.setattr(ws.httpx, "get", lambda *a, **k: _FakeResp(200, hit))
        gdelt_spy = MagicMock(return_value=[])
        with caplog.at_level(logging.WARNING), patch.multiple(ws, _search_gdelt=gdelt_spy):
            res = ws.search_articles("some query")

        assert res and res[0].url == "http://x.com/1"
        assert ws.get_last_search_provider() == "news_indexer"
        assert not gdelt_spy.called, "an unknown key must not push us onto a paid provider"
        assert any("personId" in r.getMessage() for r in caplog.records), \
            "the widening must still be logged"

    def test_provider_drop_logs_at_error_not_warning(self, monkeypatch, caplog):
        """Losing this provider is a silent cost + quality change, so the one signal that
        exists has to be alarm-able. WARNING was not."""
        ws = _fresh_ws()
        ws.NEWS_INDEXER_URL = "http://ni.local"
        ws.NEWS_INDEXER_API_KEY = "secret"

        def _boom(*a, **k):
            raise RuntimeError("connection reset")

        monkeypatch.setattr(ws.httpx, "get", _boom)
        with caplog.at_level(logging.DEBUG), patch.multiple(ws,
            _search_gdelt=MagicMock(return_value=[]),
            _search_gdelt_bq=MagicMock(return_value=[]),
            _search_serpapi_news=MagicMock(return_value=[]),
            _search_serper_news=MagicMock(return_value=[]),
            _search_brave_news=MagicMock(return_value=[]),
            _search_tavily=MagicMock(return_value=[]),
            _search_brightdata=MagicMock(return_value=[]),
            _search_nimbleway=MagicMock(return_value=[]),
            _search_scrapingbee=MagicMock(return_value=[]),
            _search_newsdata_io=MagicMock(return_value=[]),
            _search_dataforseo=MagicMock(return_value=[]),
            _search_ddg_news=MagicMock(return_value=[]),
        ):
            ws.search_articles("some query")

        drops = [r for r in caplog.records if "news_indexer failed" in r.getMessage()]
        assert drops, "the provider drop must be logged"
        assert all(r.levelno >= logging.ERROR for r in drops), \
            f"expected ERROR, got {[r.levelname for r in drops]}"

    def test_falls_through_on_http_error(self, monkeypatch):
        """Network/HTTP error → logged (at ERROR since retro#459), fall through to GDELT;
        no exception raised."""
        ws = _fresh_ws()
        ws.NEWS_INDEXER_URL = "http://ni.local"
        ws.NEWS_INDEXER_API_KEY = "secret"
        from tm.web_search import SearchResult
        gdelt_hit = [SearchResult(title="G", url="http://g.com/1", snippet="gdelt")]
        monkeypatch.setattr(ws.httpx, "get", lambda *a, **k: (_ for _ in ()).throw(Exception("connection refused")))
        with patch.multiple(ws,
            _search_gdelt=MagicMock(return_value=gdelt_hit),
            _search_gdelt_bq=MagicMock(return_value=[]),
        ):
            res = ws.search_articles("some query")
        assert res
        assert ws.get_last_search_provider() == "gdelt"

    def test_passes_correct_params(self, monkeypatch):
        """Verifies URL, query param, limit, and x-api-key header are sent correctly."""
        ws = _fresh_ws()
        ws.NEWS_INDEXER_URL = "https://scrapper.daatan.com"
        ws.NEWS_INDEXER_API_KEY = "mykey"
        captured = {}
        def fake_get(url, *, params=None, headers=None, timeout=None):
            captured["url"] = url
            captured["params"] = params
            captured["headers"] = headers
            return _FakeResp(200, [{"title": "T", "url": "http://x.com", "snippet": "S", "source": "", "published_date": ""}])
        monkeypatch.setattr(ws.httpx, "get", fake_get)
        ws.search_articles("ukraine war", limit=5)
        assert captured["url"] == "https://scrapper.daatan.com/search"
        assert captured["params"] == {"q": "ukraine war", "limit": 5}
        assert captured["headers"].get("x-api-key") == "mykey"


# ---------------------------------------------------------------------------
# news-indexer cache-fill (warm /enqueue after a paid hit)
# ---------------------------------------------------------------------------

@needs_deps
class TestNewsIndexerWarm:
    """search_articles() feeds paid-provider results back into the news-indexer on-demand
    cache via POST /enqueue, so a future identical query is served locally with no SERP cost."""

    @staticmethod
    def _inline_threads(ws, monkeypatch):
        # Run the daemon-thread body synchronously so the POST is observable in-test.
        class _Inline:
            def __init__(self, target=None, **kw):
                self._t = target

            def start(self):
                if self._t:
                    self._t()

        monkeypatch.setattr(ws.threading, "Thread", _Inline)

    def test_warms_with_paid_provider_results(self, monkeypatch):
        ws = _fresh_ws()
        ws.NEWS_INDEXER_URL = "http://ni.local"
        ws.NEWS_INDEXER_API_KEY = "secret"
        self._inline_threads(ws, monkeypatch)
        monkeypatch.setattr(ws, "get_last_search_provider", lambda: "gdelt")
        posts = []
        monkeypatch.setattr(ws.httpx, "post", lambda *a, **k: posts.append(k) or _FakeResp(200, {}))
        from tm.web_search import SearchResult
        results = [
            SearchResult(title="A", url="http://a.com/1", snippet="s"),
            SearchResult(title="B", url="http://b.com/2", snippet="s"),
        ]
        ws._warm_news_indexer(results)
        assert len(posts) == 1
        assert posts[0]["json"] == {"urls": ["http://a.com/1", "http://b.com/2"]}
        assert posts[0]["headers"]["x-api-key"] == "secret"

    def test_no_warm_when_news_indexer_served(self, monkeypatch):
        ws = _fresh_ws()
        ws.NEWS_INDEXER_URL = "http://ni.local"
        ws.NEWS_INDEXER_API_KEY = "secret"
        self._inline_threads(ws, monkeypatch)
        monkeypatch.setattr(ws, "get_last_search_provider", lambda: "news_indexer")
        posts = []
        monkeypatch.setattr(ws.httpx, "post", lambda *a, **k: posts.append(1))
        from tm.web_search import SearchResult
        ws._warm_news_indexer([SearchResult(title="A", url="http://a.com/1", snippet="s")])
        assert not posts, "already-indexed results must not be re-enqueued"

    def test_inert_when_unconfigured(self, monkeypatch):
        ws = _fresh_ws()
        ws.NEWS_INDEXER_URL = None
        ws.NEWS_INDEXER_API_KEY = None
        posts = []
        monkeypatch.setattr(ws.httpx, "post", lambda *a, **k: posts.append(1))
        from tm.web_search import SearchResult
        ws._warm_news_indexer([SearchResult(title="A", url="http://a.com/1", snippet="s")])
        assert not posts

    def test_post_failure_is_swallowed(self, monkeypatch):
        ws = _fresh_ws()
        ws.NEWS_INDEXER_URL = "http://ni.local"
        ws.NEWS_INDEXER_API_KEY = "secret"
        self._inline_threads(ws, monkeypatch)
        monkeypatch.setattr(ws, "get_last_search_provider", lambda: "gdelt")

        def boom(*a, **k):
            raise RuntimeError("enqueue down")

        monkeypatch.setattr(ws.httpx, "post", boom)
        from tm.web_search import SearchResult
        ws._warm_news_indexer([SearchResult(title="A", url="http://a.com/1", snippet="s")])  # must not raise

    def test_non_list_results_never_break_search(self, monkeypatch):
        # A bare (non-iterable) result must not raise out of warming.
        ws = _fresh_ws()
        ws.NEWS_INDEXER_URL = "http://ni.local"
        ws.NEWS_INDEXER_API_KEY = "secret"
        monkeypatch.setattr(ws, "get_last_search_provider", lambda: "gdelt")
        from tm.web_search import SearchResult
        ws._warm_news_indexer(SearchResult(title="A", url="http://a.com/1", snippet="s"))


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
        # Test isolation: null the news-indexer provider so a resolvable secret on the dev box
        # doesn't let it intercept the live query (it runs first in the chain). Unset in CI.
        ws.NEWS_INDEXER_URL = None
        ws.NEWS_INDEXER_API_KEY = None
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


# ---------------------------------------------------------------------------
# Quota flag TTL — a sticky flag must not be a permanent one
# ---------------------------------------------------------------------------

@needs_deps
class TestQuotaFlagTTL:
    """Regression: a quota flag, once set, was persisted and never cleared by any code
    path — so one transient 429/401 disabled a provider permanently, across restarts."""

    def _isolate_state(self, ws, tmp_path):
        ws._QUOTA_STATE_PATH = tmp_path / "quota_exhausted.json"
        ws._QUOTA_SET_AT.clear()

    def test_flag_older_than_ttl_is_cleared(self, tmp_path):
        ws = _fresh_ws()
        self._isolate_state(ws, tmp_path)
        ws._SERPAPI_QUOTA_EXHAUSTED = True
        ws._QUOTA_SET_AT["serpapi"] = time.time() - ws._QUOTA_TTL_SECONDS - 1

        ws._expire_stale_quota_flags()

        assert ws._SERPAPI_QUOTA_EXHAUSTED is False
        assert "serpapi" not in ws._QUOTA_SET_AT

    def test_fresh_flag_is_kept(self, tmp_path):
        ws = _fresh_ws()
        self._isolate_state(ws, tmp_path)
        ws._SERPAPI_QUOTA_EXHAUSTED = True
        ws._QUOTA_SET_AT["serpapi"] = time.time()

        ws._expire_stale_quota_flags()

        assert ws._SERPAPI_QUOTA_EXHAUSTED is True

    def test_persist_stamps_set_at_for_newly_set_flag(self, tmp_path):
        """Callers assign the global directly; _persist_quota_state must supply the timestamp."""
        ws = _fresh_ws()
        self._isolate_state(ws, tmp_path)
        ws._BRAVE_QUOTA_EXHAUSTED = True

        ws._persist_quota_state()

        assert ws._QUOTA_SET_AT["brave"] == pytest.approx(time.time(), abs=5)

    def test_clearing_flag_drops_its_timestamp(self, tmp_path):
        ws = _fresh_ws()
        self._isolate_state(ws, tmp_path)
        ws._BRAVE_QUOTA_EXHAUSTED = True
        ws._persist_quota_state()

        ws._BRAVE_QUOTA_EXHAUSTED = False
        ws._persist_quota_state()

        assert "brave" not in ws._QUOTA_SET_AT

    def test_legacy_bool_file_gets_a_timestamp_so_it_can_expire(self, tmp_path):
        """The pre-TTL on-disk format has no timestamp. It must still be honoured on load
        (a real exhaustion shouldn't be forgotten), but must be stamped so the TTL can
        eventually clear it — the old format could pin a provider off forever."""
        ws = _fresh_ws()
        state = tmp_path / "quota_exhausted.json"
        state.write_text('{"serpapi": true, "brave": false}')
        ws._QUOTA_STATE_PATH = state
        ws._QUOTA_SET_AT.clear()

        ws._load_quota_state()

        assert ws._SERPAPI_QUOTA_EXHAUSTED is True
        assert ws._BRAVE_QUOTA_EXHAUSTED is False
        # Stamped as first-seen now — so it survives this sweep...
        assert ws._QUOTA_SET_AT["serpapi"] == pytest.approx(time.time(), abs=5)
        ws._expire_stale_quota_flags()
        assert ws._SERPAPI_QUOTA_EXHAUSTED is True

        # ...but is cleared once it ages past the TTL, which the old format never did.
        ws._QUOTA_SET_AT["serpapi"] = time.time() - ws._QUOTA_TTL_SECONDS - 1
        ws._expire_stale_quota_flags()
        assert ws._SERPAPI_QUOTA_EXHAUSTED is False

    def test_state_survives_a_round_trip(self, tmp_path):
        ws = _fresh_ws()
        self._isolate_state(ws, tmp_path)
        ws._TAVILY_QUOTA_EXHAUSTED = True
        ws._persist_quota_state()

        ws._TAVILY_QUOTA_EXHAUSTED = False
        ws._QUOTA_SET_AT.clear()
        ws._load_quota_state()

        assert ws._TAVILY_QUOTA_EXHAUSTED is True
        assert ws._QUOTA_SET_AT["tavily"] > 0

    def test_chain_sweeps_stale_flags(self, tmp_path):
        """search_articles must run the sweep, so a stale flag can't skip a provider."""
        src = inspect.getsource(_fresh_ws()._search_articles_chain)
        assert "_expire_stale_quota_flags()" in src


# ---------------------------------------------------------------------------
# Secret loading — _secret() fails open by design, but must fail LOUDLY
# ---------------------------------------------------------------------------

class TestSecretLoadingLogsLoudly:
    @needs_deps
    def test_secret_failure_logs_at_warning_not_debug(self, caplog, monkeypatch):
        """A Secrets Manager failure must be visible without debug logging enabled —
        this used to be logger.debug(), which silently hid missing providers."""
        ws = _fresh_ws()
        monkeypatch.delenv("SOME_TEST_ENV_VAR", raising=False)

        class _FailingClient:
            def get_secret_value(self, SecretId):
                raise RuntimeError("simulated Secrets Manager failure")

        with patch("boto3.client", return_value=_FailingClient()):
            with caplog.at_level("WARNING", logger="tm.web_search"):
                result = ws._secret("SOME_TEST_ENV_VAR", "daatan/some-test-key")

        assert result is None
        assert any(
            r.levelname == "WARNING" and "daatan/some-test-key" in r.message
            for r in caplog.records
        ), "secret-load failure must log at WARNING with the secret name"

    @needs_deps
    def test_log_unresolved_secrets_reports_missing_keys(self, caplog):
        """_log_unresolved_secrets() must name every unresolved provider secret in one line."""
        ws = _fresh_ws()
        ws.DATAFORSEO_API_KEY = None
        ws.SERPAPI_API_KEY = "present"

        caplog.clear()  # _fresh_ws() import may itself have logged real unresolved-secret warnings
        with caplog.at_level("WARNING", logger="tm.web_search"):
            ws._log_unresolved_secrets()

        messages = [r.message for r in caplog.records if r.levelname == "WARNING"]
        assert any("daatan/dataforseo-key" in m for m in messages)
        assert not any("daatan/serpapi-key" in m for m in messages)

    @needs_deps
    def test_log_unresolved_secrets_silent_when_all_present(self, caplog):
        """No warning at all once every provider secret has resolved."""
        ws = _fresh_ws()
        for name in (
            "DATAFORSEO_API_KEY", "SERPAPI_API_KEY", "SERPER_API_KEY", "BRAVE_API_KEY",
            "BRIGHTDATA_API_KEY", "NIMBLEWAY_API_KEY", "SCRAPINGBEE_API_KEY",
            "NEWSDATA_API_KEY", "TAVILY_API_KEY", "GOOGLE_CSE_API_KEY", "GOOGLE_CSE_CX",
            "GCP_SA_KEY_JSON",
        ):
            setattr(ws, name, "present")

        caplog.clear()  # _fresh_ws() import may itself have logged real unresolved-secret warnings
        with caplog.at_level("WARNING", logger="tm.web_search"):
            ws._log_unresolved_secrets()

        assert not any(r.levelname == "WARNING" for r in caplog.records)
