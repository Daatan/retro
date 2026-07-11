"""Unit tests for the GDELT-BigQuery ingestor's pure logic and the BQ domain filter.

No network, no BigQuery, no API keys: these cover host normalisation, the
source-domain map, the Wayback raw-URL transform, and the SQL ``domains`` clause
added to ``_search_gdelt_bq``. Skipped only when core deps are absent.
"""

import json
from datetime import datetime

import pytest

pytest.importorskip("httpx")
pytest.importorskip("boto3")

from tm import gdelt_bq_ingest as g
from tm import web_search as ws


class TestNormHost:
    @pytest.mark.parametrize("raw,expected", [
        ("https://www.ynet.co.il", "ynet.co.il"),
        ("http://example.com/path/to/x", "example.com"),
        ("ynet.co.il", "ynet.co.il"),
        ("WWW.Foo.COM", "foo.com"),
        ("https://sub.haaretz.co.il/a?b=c", "sub.haaretz.co.il"),
        ("", ""),
    ])
    def test_normalises(self, raw, expected):
        assert g._norm_host(raw) == expected


class TestBuildDomainMap:
    def test_maps_known_sources_and_excludes_others(self, tmp_path):
        sources = tmp_path / "sources"
        sources.mkdir()
        (sources / "ynet.json").write_text(json.dumps(
            {"id": "ynet", "name": "Ynet", "url": "https://www.ynet.co.il"}))
        (sources / "toi.json").write_text(json.dumps(
            {"id": "toi", "name": "ToI", "url": "https://www.timesofisrael.com"}))
        # Excluded: web_search is a meta-source, not in KNOWN_SOURCE_IDS.
        (sources / "web_search.json").write_text(json.dumps(
            {"id": "web_search", "name": "WS", "url": "https://web.search"}))
        # Excluded: unknown id not in KNOWN_SOURCE_IDS.
        (sources / "randomx.json").write_text(json.dumps(
            {"id": "randomx", "name": "R", "url": "https://randomx.com"}))

        m = g.build_domain_map(sources)
        assert m == {"ynet.co.il": "ynet", "timesofisrael.com": "toi"}

    def test_ignores_corrupt_files(self, tmp_path):
        sources = tmp_path / "sources"
        sources.mkdir()
        (sources / "bad.json").write_text("not json")
        (sources / "ynet.json").write_text(json.dumps(
            {"id": "ynet", "name": "Ynet", "url": "https://www.ynet.co.il"}))
        assert g.build_domain_map(sources) == {"ynet.co.il": "ynet"}


class TestRawSnapshotUrl:
    def test_inserts_id_modifier(self):
        snap = "http://web.archive.org/web/20231004093000/https://www.ynet.co.il/a"
        assert g._raw_snapshot_url(snap) == \
            "http://web.archive.org/web/20231004093000id_/https://www.ynet.co.il/a"

    def test_leaves_non_wayback_url_unchanged(self):
        assert g._raw_snapshot_url("https://ynet.co.il/a") == "https://ynet.co.il/a"


class TestSqlDomainFilter:
    def test_none_and_empty_yield_no_clause(self):
        assert ws._sql_domain_filter(None) == ""
        assert ws._sql_domain_filter([]) == ""

    def test_builds_in_clause_and_normalises(self):
        clause = ws._sql_domain_filter(["https://www.ynet.co.il", "haaretz.co.il"])
        assert clause == "AND SourceCommonName IN ('ynet.co.il','haaretz.co.il')"

    def test_dedups_after_normalisation(self):
        clause = ws._sql_domain_filter(["ynet.co.il", "https://www.ynet.co.il"])
        assert clause == "AND SourceCommonName IN ('ynet.co.il')"

    def test_strips_single_quotes_defensively(self):
        # A quote can only ever appear via bad input; it must not reach the SQL.
        clause = ws._sql_domain_filter(["ev'il.com"])
        assert "'" not in clause.split("IN (", 1)[1].replace("'evil.com'", "")
        assert "evil.com" in clause


class TestSpreadSample:
    def test_empty_and_nonpositive_k(self):
        assert g._spread_sample([1, 2, 3], 0) == []
        assert g._spread_sample([1, 2, 3], -1) == []

    def test_returns_all_when_fewer_than_k(self):
        assert g._spread_sample([1, 2], 5) == [1, 2]

    def test_k_one_takes_middle(self):
        assert g._spread_sample(list(range(10)), 1) == [5]

    def test_spreads_across_full_range(self):
        # First and last are always included; picks are spread, not clustered.
        out = g._spread_sample(list(range(10)), 3)
        assert out == [0, 4, 9]
        assert out[0] == 0 and out[-1] == 9

    def test_no_duplicate_indices_on_small_n(self):
        out = g._spread_sample(list(range(4)), 3)
        assert out == sorted(set(out))  # no repeats
        assert out[0] == 0 and out[-1] == 3


class TestIngestWindowAlignment:
    """The ingest window must match the backtest's scored window
    [outcome − MAX_DAYS, outcome − MIN_DAYS], not run up to the outcome."""

    async def _capture_window(self, monkeypatch, tmp_path, t_days):
        captured = {}

        async def fake_to_thread(fn, *args, **kw):
            # ingest_event calls to_thread(_search_gdelt_bq, query, cap, start, end, domains, cap)
            captured["start"], captured["end"] = args[2], args[3]
            return []  # empty → ingest_event returns before any fetch

        monkeypatch.setattr(g.asyncio, "to_thread", fake_to_thread)
        event = {"id": "E1", "outcome_date": "2024-12-08",
                 "name": "Test", "search_keywords": ["Netanyahu"]}
        await g.ingest_event(event, {"ynet.co.il": "ynet"}, tmp_path,
                             per_source_limit=5, force=False, t_days=t_days, allow_live=False)
        return captured

    async def test_default_window_ends_3d_before_outcome(self, monkeypatch, tmp_path):
        c = await self._capture_window(monkeypatch, tmp_path, t_days=0)
        assert c["end"] == datetime(2024, 12, 5)    # outcome − MIN_DAYS_BEFORE_EVENT (3)
        assert c["start"] == datetime(2024, 11, 8)   # outcome − MAX_DAYS_BEFORE_EVENT (30)

    async def test_t_days_pushes_end_earlier(self, monkeypatch, tmp_path):
        c = await self._capture_window(monkeypatch, tmp_path, t_days=7)
        assert c["end"] == datetime(2024, 12, 1)     # outcome − 7 (t_days > MIN)


class TestSourceLevelInvariants:
    """Plain-text guards so a refactor can't silently drop the cost/safety controls."""

    def test_bq_query_has_domain_filter_and_byte_cap(self):
        import inspect
        src = inspect.getsource(ws._search_gdelt_bq)
        assert "SourceCommonName IN" in inspect.getsource(ws._sql_domain_filter)
        assert "domain_filter" in src
        assert "maximum_bytes_billed" in src
