"""Unit tests for the GDELT-BigQuery ingestor's pure logic and the BQ domain filter.

No network, no BigQuery, no API keys: these cover host normalisation, the
source-domain map, the Wayback raw-URL transform, and the SQL ``domains`` clause
added to ``_search_gdelt_bq``. Skipped only when core deps are absent.
"""

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytest.importorskip("httpx")
pytest.importorskip("boto3")
pytest.importorskip("google.cloud.bigquery")

from tm import gdelt_bq_ingest as g
from tm import web_search as ws

REPO_ROOT = Path(__file__).resolve().parents[2]


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

    def test_bq_query_has_time_fuse(self):
        """retro#655 — job.result() must not be called without a timeout, or a
        stalled BigQuery job hangs the ingestor indefinitely (observed: 0% CPU,
        alive 11+ minutes on a real event)."""
        import inspect
        assert "job.result(timeout=" in inspect.getsource(ws._search_gdelt_bq)
        assert "job.result(timeout=" in inspect.getsource(g.discover) \
            or "job.result, timeout=" in inspect.getsource(g.discover)


class TestSearchGdeltBqTimeout:
    """Behavioral counterpart to the source-inspection guard above."""

    def test_job_result_called_with_timeout(self, monkeypatch):
        fake_client = MagicMock()
        fake_job = MagicMock()
        fake_job.result.return_value = []
        fake_client.query.return_value = fake_job
        monkeypatch.setattr(ws, "_get_bq_client", lambda: fake_client)

        ws._search_gdelt_bq("Netanyahu", limit=10)

        fake_job.result.assert_called_once_with(timeout=ws._GDELT_BQ_QUERY_TIMEOUT_S)


class TestEntityRegexPatterns:
    """_entity_regex_patterns() builds the values bound as a BigQuery
    ArrayQueryParameter in discover() — it must never be interpolated into SQL."""

    def test_wraps_each_term_case_insensitively_and_escapes_regex_metachars(self):
        patterns = g._entity_regex_patterns(["Netanyahu", "a.b*c"])
        assert patterns[0] == "(?i)Netanyahu"
        assert patterns[1].startswith("(?i)")
        # re.escape neutralises regex metacharacters (not SQL syntax — that's
        # discover()'s job, by binding this value as a parameter).
        assert "\\." in patterns[1] and "\\*" in patterns[1]


class TestDiscoverSqlInjectionSafety:
    """retro#440 item 4 — discover() must bind entity terms and date bounds as
    BigQuery query parameters, never splice them into the SQL string. Regex-
    escaping (re.escape) is not SQL-escaping; only parameter binding is safe
    regardless of what characters a term or date string ever contains.
    """

    async def _run_discover(self, tmp_path, monkeypatch, terms, date_from="2024-01-01", date_to="2024-02-01"):
        sources_dir = tmp_path / "sources"
        sources_dir.mkdir()

        fake_client = MagicMock()
        fake_job = MagicMock()
        fake_job.result.return_value = []
        fake_job.total_bytes_processed = 0
        fake_client.query.return_value = fake_job

        monkeypatch.setattr(ws, "_get_bq_client", lambda: fake_client)
        monkeypatch.setattr(ws, "_extract_bq_terms", lambda keywords: terms)

        await g.discover("irrelevant keywords", date_from, date_to, tmp_path)
        return fake_client

    async def test_sql_injection_shaped_term_is_bound_not_interpolated(self, tmp_path, monkeypatch):
        evil_term = "' OR '1'='1"
        fake_client = await self._run_discover(tmp_path, monkeypatch, [evil_term])

        assert fake_client.query.call_count == 1
        sql, kwargs = fake_client.query.call_args
        sql_text = sql[0]

        # The malicious term must never appear in the SQL text itself.
        assert evil_term not in sql_text
        assert "1'='1" not in sql_text

        # It must instead be present, verbatim (only regex-escaped), inside a
        # bound query parameter.
        job_config = kwargs["job_config"]
        param_names = {p.name for p in job_config.query_parameters}
        assert param_names == {"date_from", "date_to", "patterns"}

        patterns_param = next(p for p in job_config.query_parameters if p.name == "patterns")
        assert patterns_param.array_type == "STRING"
        assert patterns_param.values == [f"(?i){__import__('re').escape(evil_term)}"]

    async def test_date_bounds_are_also_bound_parameters_not_interpolated(self, tmp_path, monkeypatch):
        evil_date = "2024-01-01') OR ('1'='1"
        fake_client = await self._run_discover(
            tmp_path, monkeypatch, ["Netanyahu"], date_from=evil_date,
        )

        sql, kwargs = fake_client.query.call_args
        sql_text = sql[0]
        assert evil_date not in sql_text

        job_config = kwargs["job_config"]
        date_from_param = next(p for p in job_config.query_parameters if p.name == "date_from")
        assert date_from_param.value == evil_date

    async def test_sql_text_uses_named_placeholders_not_literal_values(self, tmp_path, monkeypatch):
        fake_client = await self._run_discover(tmp_path, monkeypatch, ["Netanyahu"])
        sql, _kwargs = fake_client.query.call_args
        sql_text = sql[0]
        assert "@date_from" in sql_text
        assert "@date_to" in sql_text
        assert "@patterns" in sql_text
        # No raw TIMESTAMP('...') literal left over from the old f-string form.
        assert "TIMESTAMP('" not in sql_text


class TestNoEventEntityExtractability:
    """retro#509 — GDELT NO-event curation pass.

    A NO-outcome event whose keywords yield no extractable BQ entity terms
    (``_extract_bq_terms`` empty) can never be scanned at all — ``ingest_event``
    raises before the query even runs (see ``RuntimeError`` in
    ``_search_gdelt_bq``), so the event silently vanishes from any
    ``--all``/``--events`` backfill with no article ever fetched. This bit the
    batch-1 seeded NO-events (retro#281/#509): ``_extract_bq_terms`` filters
    generic single words like "Israel"/"Iran"/"Gaza" as too-common
    (``_common`` stopword set in ``web_search.py``), and 4 of the 6 events had
    no other proper noun in their keywords, so their queries never ran —
    indistinguishable from "GDELT has no coverage" unless you look closely.
    Live discovery (``--discover``) confirmed all 6 candidates actually have
    heavy genuine predictive-window GKG coverage (100s of articles across a
    dozen+ tracked outlets each) once given an extractable keyword — the gap
    was tooling, not missing news coverage. Fixed by adding ``duel_keywords``
    with a real named entity (e.g. "Netanyahu", "Amir Yaron") to each of the
    4 affected events. This test is the regression barrier: every committed
    NO-outcome event must keep at least one extractable entity term so a
    future NO-event addition can't reintroduce the same silent gap.
    """

    def _no_events(self):
        events_dir = REPO_ROOT / "data" / "events"
        for f in sorted(events_dir.glob("*.json")):
            event = json.loads(f.read_text())
            if event.get("outcome") is False:
                yield event

    def test_every_no_event_has_extractable_entity_terms(self):
        keywords_by_bad_event = {}
        for event in self._no_events():
            keywords = event.get("duel_keywords") or event.get("search_keywords") or []
            query = " ".join(keywords)
            terms = ws._extract_bq_terms(query)
            if not terms:
                keywords_by_bad_event[event["id"]] = keywords
        assert keywords_by_bad_event == {}, (
            f"NO-events with no extractable GDELT entity term (add duel_keywords "
            f"with a real proper noun): {keywords_by_bad_event}"
        )

    def test_at_least_one_no_event_exists(self):
        # A vacuous pass above (empty NO-event set) would hide a regression.
        assert len(list(self._no_events())) > 0
