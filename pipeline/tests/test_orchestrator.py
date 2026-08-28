"""Tests for tm.orchestrator — the TruthMachine pipeline run loop.

Previously had zero coverage. These tests exercise:
  - run_event(): the main per-event/per-source loop, its skip conditions
    (done/no_predictions/force_reextract/retry_empty), error handling
    (timeout, generic exception) and final cell-status derivation.
  - process_article(): stub-skip, near-duplicate reuse, extraction-cache
    reuse, force-reextract, article-level aggregation (and its failure
    path), vault writing.
  - search_articles()/local_file_search()/_oracle_search(): mode dispatch,
    date-window/liveblog/stub filtering, Oracle API-key gating, domain
    matching, and network-error handling.
  - _write_cell_signal() and create_atlas_link(): the atlas-writing helpers.

External calls (LLM extraction via runner.run_article, HTTP via httpx,
article-level aggregation) are mocked; nothing here hits real network
or LLM services.
"""

import json
from datetime import datetime
from types import SimpleNamespace

import httpx
import pytest

import tm.orchestrator as orch_mod
from tm.config import settings as tm_settings
from tm.models import CellStatus, ExtractionOutput, PredictionExtraction
from tm.orchestrator import EXTRACTION_PROMPT_VERSION, Orchestrator, SearchMode
from tm.progress import load_state
from tm.runner import ArticleInput, PipelineResult


@pytest.fixture(autouse=True)
def _isolated_data_dir(tmp_path, monkeypatch):
    """run_event()/process_article() call progress.update_cell()/load_state(),
    which read/write settings.progress_file (settings.data_dir/progress.json).
    Point that at the same tmp_path each test uses for its Orchestrator so
    state never touches the real /app/data or leaks across tests.

    Also reset settings.vault_dir: the repo's local .env pins VAULT_DIR to
    the real data/vault2 directory, which pydantic-settings loads into
    `settings.vault_dir` at Settings() construction time (independent of
    os.environ). Orchestrator.__init__ falls back to
    `str(settings.vault_dir)` whenever it's non-empty — bypassing the
    data_dir constructor arg entirely — so without this reset every test
    would read/write the real vault, including reusing real cached LLM
    extractions, which silently short-circuits process_article() before it
    ever calls run_article().
    """
    monkeypatch.setattr(tm_settings, "data_dir", tmp_path)
    monkeypatch.setattr(tm_settings, "vault_dir", tm_settings.__class__.model_fields["vault_dir"].default)
    monkeypatch.delenv("VAULT_DIR", raising=False)
    yield


def _write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def _make_event(data_dir, event_id="E1", outcome_date="2024-12-08", window_days=14):
    _write_json(data_dir / "events" / f"{event_id}.json", {
        "id": event_id,
        "name": "Test Event",
        "outcome_date": outcome_date,
        "predictive_window_days": window_days,
        "outcome": True,
        "llm_referee_criteria": "criteria",
    })


def _make_source(data_dir, source_id="ynet"):
    _write_json(data_dir / "sources" / f"{source_id}.json", {
        "id": source_id, "name": source_id.title(), "url": f"https://{source_id}.com",
    })


def _make_orch(tmp_path, mode=SearchMode.local_file, **kwargs):
    return Orchestrator(tmp_path, mode=mode, **kwargs)


def _fake_prediction(stance=0.5, certainty=0.8):
    return PredictionExtraction(
        quote="quote", claim="claim", stance=stance, certainty=certainty,
    )


# ─────────────────────────────────────────────
# get_article_hash
# ─────────────────────────────────────────────


class TestGetArticleHash:
    def test_stable_and_sha256(self, tmp_path):
        orch = _make_orch(tmp_path)
        h1 = orch.get_article_hash("hello world")
        h2 = orch.get_article_hash("hello world")
        assert h1 == h2
        assert len(h1) == 64

    def test_different_text_different_hash(self, tmp_path):
        orch = _make_orch(tmp_path)
        assert orch.get_article_hash("a") != orch.get_article_hash("b")


# ─────────────────────────────────────────────
# get_full_text
# ─────────────────────────────────────────────


def _fake_async_client(response=None, exc=None):
    class _Client:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, *a, **kw):
            if exc is not None:
                raise exc
            return response

        async def post(self, *a, **kw):
            if exc is not None:
                raise exc
            if response is not None:
                try:
                    response.request
                except RuntimeError:
                    response.request = httpx.Request("POST", "https://oracle.daatan.com/search")
            return response

    return _Client


class TestGetFullText:
    # IP-literal hosts (not "example.com") so safe_get_async's DNS-resolution
    # check (net_guard.unsafe_reason) doesn't make a real query — matches
    # test_net_guard.py's own convention.
    async def test_extracts_text_from_html(self, tmp_path, monkeypatch):
        html = "<html><head><style>x</style></head><body><nav>Skip</nav><p>Real content here.</p></body></html>"
        monkeypatch.setattr(
            orch_mod.httpx, "AsyncClient",
            _fake_async_client(response=httpx.Response(200, text=html)),
        )
        orch = _make_orch(tmp_path)
        text = await orch.get_full_text("https://93.184.216.34/a")
        assert "Real content here." in text
        assert "Skip" not in text

    async def test_non_200_returns_empty_string(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            orch_mod.httpx, "AsyncClient",
            _fake_async_client(response=httpx.Response(404, text="nope")),
        )
        orch = _make_orch(tmp_path)
        assert await orch.get_full_text("https://93.184.216.34/missing") == ""

    async def test_network_exception_returns_empty_string(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            orch_mod.httpx, "AsyncClient",
            _fake_async_client(exc=httpx.ConnectTimeout("timeout")),
        )
        orch = _make_orch(tmp_path)
        assert await orch.get_full_text("https://93.184.216.34/a") == ""

    async def test_private_ip_url_is_rejected_without_network_call(self, tmp_path, monkeypatch):
        # The SSRF guard must reject before any client.get() happens — assert
        # via a client whose .get() would raise if it were ever reached.
        monkeypatch.setattr(
            orch_mod.httpx, "AsyncClient",
            _fake_async_client(exc=AssertionError("client.get() should not be called")),
        )
        orch = _make_orch(tmp_path)
        assert await orch.get_full_text("https://127.0.0.1/admin") == ""

    async def test_bracketed_ipv6_loopback_is_rejected(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            orch_mod.httpx, "AsyncClient",
            _fake_async_client(exc=AssertionError("client.get() should not be called")),
        )
        orch = _make_orch(tmp_path)
        assert await orch.get_full_text("https://[::1]/admin") == ""


# ─────────────────────────────────────────────
# local_file_search
# ─────────────────────────────────────────────


class TestLocalFileSearch:
    def _art(self, **overrides):
        art = {
            "url": "https://ynet.com/article1",
            "text": "x" * 600,
            "published_at": "2024-12-01",
            "headline": "Some headline",
        }
        art.update(overrides)
        return art

    def test_missing_search_path_returns_empty(self, tmp_path):
        orch = _make_orch(tmp_path)
        assert orch.local_file_search("ynet", "E1") == []

    def test_includes_valid_article_within_window(self, tmp_path):
        orch = _make_orch(tmp_path)
        _write_json(orch.raw_ingest_dir / "ynet" / "E1" / "a.json", self._art())
        result = orch.local_file_search(
            "ynet", "E1",
            start=datetime(2024, 11, 20), end=datetime(2024, 12, 8),
        )
        assert len(result) == 1

    def test_drops_post_outcome_article(self, tmp_path):
        orch = _make_orch(tmp_path)
        _write_json(
            orch.raw_ingest_dir / "ynet" / "E1" / "a.json",
            self._art(published_at="2024-12-20"),
        )
        result = orch.local_file_search(
            "ynet", "E1",
            start=datetime(2024, 11, 20), end=datetime(2024, 12, 8),
        )
        assert result == []

    def test_drops_pre_window_article(self, tmp_path):
        orch = _make_orch(tmp_path)
        _write_json(
            orch.raw_ingest_dir / "ynet" / "E1" / "a.json",
            self._art(published_at="2024-10-01"),
        )
        result = orch.local_file_search(
            "ynet", "E1",
            start=datetime(2024, 11, 20), end=datetime(2024, 12, 8),
        )
        assert result == []

    def test_skips_liveblog(self, tmp_path):
        orch = _make_orch(tmp_path)
        _write_json(
            orch.raw_ingest_dir / "ynet" / "E1" / "a.json",
            self._art(url="https://ynet.com/liveblog/1"),
        )
        assert orch.local_file_search("ynet", "E1") == []

    def test_skips_short_stub(self, tmp_path):
        orch = _make_orch(tmp_path)
        _write_json(orch.raw_ingest_dir / "ynet" / "E1" / "a.json", self._art(text="short"))
        assert orch.local_file_search("ynet", "E1") == []

    def test_skips_known_stub_page(self, tmp_path, monkeypatch):
        monkeypatch.setattr(orch_mod, "_is_stub_page", lambda text: True)
        orch = _make_orch(tmp_path)
        _write_json(orch.raw_ingest_dir / "ynet" / "E1" / "a.json", self._art())
        assert orch.local_file_search("ynet", "E1") == []

    def test_unparseable_published_at_is_kept(self, tmp_path):
        orch = _make_orch(tmp_path)
        _write_json(
            orch.raw_ingest_dir / "ynet" / "E1" / "a.json",
            self._art(published_at="not-a-date"),
        )
        result = orch.local_file_search(
            "ynet", "E1",
            start=datetime(2024, 11, 20), end=datetime(2024, 12, 8),
        )
        assert len(result) == 1


# ─────────────────────────────────────────────
# search_articles / _oracle_search
# ─────────────────────────────────────────────


class TestSearchArticlesDispatch:
    async def test_mock_mode_returns_empty(self, tmp_path):
        orch = _make_orch(tmp_path, mode=SearchMode.mock)
        result = await orch.search_articles({"id": "ynet", "url": "https://ynet.com"}, {"id": "E1"}, datetime(2024, 1, 1), datetime(2024, 1, 2))
        assert result == []

    async def test_local_file_mode_delegates(self, tmp_path, monkeypatch):
        orch = _make_orch(tmp_path, mode=SearchMode.local_file)
        called = {}

        def fake_local(source_id, event_id, start=None, end=None):
            called["args"] = (source_id, event_id, start, end)
            return ["article"]

        monkeypatch.setattr(orch, "local_file_search", fake_local)
        result = await orch.search_articles(
            {"id": "ynet", "url": "https://ynet.com"}, {"id": "E1"},
            datetime(2024, 1, 1), datetime(2024, 1, 2),
        )
        assert result == ["article"]
        assert called["args"][0] == "ynet" and called["args"][1] == "E1"


class TestOracleSearch:
    async def test_no_api_key_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ORACLE_API_KEY", raising=False)
        orch = _make_orch(tmp_path)
        orch._oracle_key = ""
        result = await orch._oracle_search("query", 10, datetime(2024, 1, 1), datetime(2024, 1, 2))
        assert result == []

    async def test_success_transforms_results(self, tmp_path, monkeypatch):
        orch = _make_orch(tmp_path)
        orch._oracle_key = "test-key"
        payload = {
            "results": [
                {"title": "Headline 1", "published_date": "2024-01-01", "url": "https://ynet.com/1"},
                {"title": "Headline 2", "published_date": "2024-01-02", "url": "https://ynet.com/2"},
            ]
        }
        monkeypatch.setattr(
            orch_mod.httpx, "AsyncClient",
            _fake_async_client(response=httpx.Response(200, json=payload)),
        )
        result = await orch._oracle_search("query", 10, datetime(2024, 1, 1), datetime(2024, 1, 2))
        assert len(result) == 2
        assert result[0]["headline"] == "Headline 1"
        assert result[0]["author"] == "Unknown"

    async def test_http_error_returns_empty(self, tmp_path, monkeypatch):
        orch = _make_orch(tmp_path)
        orch._oracle_key = "test-key"
        monkeypatch.setattr(
            orch_mod.httpx, "AsyncClient",
            _fake_async_client(exc=httpx.ConnectError("boom")),
        )
        result = await orch._oracle_search("query", 10, datetime(2024, 1, 1), datetime(2024, 1, 2))
        assert result == []


class TestSearchArticlesApiMode:
    async def test_domain_filtering_and_full_text_length_gate(self, tmp_path, monkeypatch):
        orch = _make_orch(tmp_path, mode=SearchMode.api)

        async def fake_oracle_search(query, limit, date_from, date_to):
            return [
                {"url": "https://ynet.com/a", "published_at": "2024-01-01"},   # matches domain, will pass
                {"url": "https://other.com/b", "published_at": "2024-01-01"},  # wrong domain
                {"url": "https://ynet.com/c", "published_at": None},           # missing date
            ]

        async def fake_get_full_text(url):
            return "x" * 300 if url.endswith("/a") else ""

        monkeypatch.setattr(orch, "_oracle_search", fake_oracle_search)
        monkeypatch.setattr(orch, "get_full_text", fake_get_full_text)

        source = {"id": "ynet", "url": "https://www.ynet.com", "name": "Ynet"}
        event = {"id": "E1", "search_keywords": ["kw1", "kw2"]}
        result = await orch.search_articles(source, event, datetime(2024, 1, 1), datetime(2024, 1, 2))
        assert len(result) == 1
        assert result[0]["url"] == "https://ynet.com/a"
        assert result[0]["text"] == "x" * 300

    async def test_all_wrong_domain_prints_but_returns_empty(self, tmp_path, monkeypatch):
        orch = _make_orch(tmp_path, mode=SearchMode.api)

        async def fake_oracle_search(query, limit, date_from, date_to):
            return [{"url": "https://other.com/b", "published_at": "2024-01-01"}]

        monkeypatch.setattr(orch, "_oracle_search", fake_oracle_search)
        source = {"id": "ynet", "url": "https://www.ynet.com", "name": "Ynet"}
        event = {"id": "E1", "search_keywords": ["kw1"]}
        result = await orch.search_articles(source, event, datetime(2024, 1, 1), datetime(2024, 1, 2))
        assert result == []

    async def test_short_full_text_is_excluded(self, tmp_path, monkeypatch):
        orch = _make_orch(tmp_path, mode=SearchMode.api)

        async def fake_oracle_search(query, limit, date_from, date_to):
            return [{"url": "https://ynet.com/a", "published_at": "2024-01-01"}]

        async def fake_get_full_text(url):
            return "short"

        monkeypatch.setattr(orch, "_oracle_search", fake_oracle_search)
        monkeypatch.setattr(orch, "get_full_text", fake_get_full_text)
        source = {"id": "ynet", "url": "https://www.ynet.com", "name": "Ynet"}
        event = {"id": "E1", "search_keywords": ["kw1"]}
        result = await orch.search_articles(source, event, datetime(2024, 1, 1), datetime(2024, 1, 2))
        assert result == []


# ─────────────────────────────────────────────
# process_article
# ─────────────────────────────────────────────


class TestProcessArticle:
    def _event(self):
        return {"id": "E1", "name": "Test Event", "llm_referee_criteria": "crit", "outcome_date": "2024-12-08"}

    def _source(self):
        return {"id": "ynet", "name": "Ynet"}

    def _raw_art(self, text="x" * 600):
        return {"text": text, "published_at": "2024-12-01", "author": "Jane", "url": "https://ynet.com/a", "headline": "Headline"}

    async def test_short_article_is_skipped(self, tmp_path):
        orch = _make_orch(tmp_path)
        result = await orch.process_article(self._raw_art(text="short"), self._event(), self._source())
        assert result is None
        assert not any((orch.vault_dir / "articles").iterdir())

    async def test_near_duplicate_with_existing_canonical_reuses_extraction(self, tmp_path, monkeypatch):
        orch = _make_orch(tmp_path)
        canonical_hash = "c" * 64
        monkeypatch.setattr(orch._simhash_idx, "find_near_duplicate", lambda text: canonical_hash)
        extract_path = orch.vault_dir / "extractions" / f"{canonical_hash}_E1_{EXTRACTION_PROMPT_VERSION}.json"
        _write_json(extract_path, {
            "extraction": {"predictions": []},
            "extractor_model": "m", "gatekeeper_model": "g", "gatekeeper_reason": "",
        })
        link_called = {}
        monkeypatch.setattr(
            orch, "create_atlas_link",
            lambda eid, sid, art_hash, ep, raw, event_date="": link_called.update(hash=art_hash)
        )
        result = await orch.process_article(self._raw_art(), self._event(), self._source())
        assert result is None
        assert link_called["hash"] == canonical_hash

    async def test_near_duplicate_without_canonical_falls_through(self, tmp_path, monkeypatch):
        orch = _make_orch(tmp_path)
        monkeypatch.setattr(orch._simhash_idx, "find_near_duplicate", lambda text: "c" * 64)

        async def fake_run_article(article_input, extractor_model=None):
            return PipelineResult(article=article_input, is_prediction=True, gatekeeper_reason="", extraction=ExtractionOutput(predictions=[]))

        monkeypatch.setattr(orch_mod, "run_article", fake_run_article)
        result = await orch.process_article(self._raw_art(), self._event(), self._source())
        assert result is not None
        art_hash = orch.get_article_hash(self._raw_art()["text"])
        assert (orch.vault_dir / "articles" / f"{art_hash}.json").exists()

    async def test_existing_extraction_is_reused_without_calling_runner(self, tmp_path, monkeypatch):
        orch = _make_orch(tmp_path)
        art_hash = orch.get_article_hash(self._raw_art()["text"])
        extract_path = orch.vault_dir / "extractions" / f"{art_hash}_E1_{EXTRACTION_PROMPT_VERSION}.json"
        _write_json(extract_path, {
            "extraction": {"predictions": []}, "extractor_model": "m",
            "gatekeeper_model": "g", "gatekeeper_reason": "",
        })

        called = {"run_article": False}

        async def fake_run_article(article_input, extractor_model=None):
            called["run_article"] = True
            raise AssertionError("should not be called")

        monkeypatch.setattr(orch_mod, "run_article", fake_run_article)
        result = await orch.process_article(self._raw_art(), self._event(), self._source())
        assert result is None
        assert called["run_article"] is False
        assert (orch.atlas_dir / "E1" / "ynet" / f"entry_{art_hash[:8]}.json").exists()

    async def test_force_reextract_calls_runner_even_if_cached(self, tmp_path, monkeypatch):
        orch = _make_orch(tmp_path, force_reextract=True)
        art_hash = orch.get_article_hash(self._raw_art()["text"])
        extract_path = orch.vault_dir / "extractions" / f"{art_hash}_E1_{EXTRACTION_PROMPT_VERSION}.json"
        _write_json(extract_path, {
            "extraction": {"predictions": []}, "extractor_model": "m",
            "gatekeeper_model": "g", "gatekeeper_reason": "",
        })

        async def fake_run_article(article_input, extractor_model=None):
            return PipelineResult(article=article_input, is_prediction=True, gatekeeper_reason="", extraction=ExtractionOutput(predictions=[_fake_prediction()]))

        monkeypatch.setattr(orch_mod, "run_article", fake_run_article)
        result = await orch.process_article(self._raw_art(), self._event(), self._source())
        assert result is not None
        assert len(result.extraction.predictions) == 1

    async def test_runner_error_does_not_raise(self, tmp_path, monkeypatch):
        orch = _make_orch(tmp_path)

        async def fake_run_article(article_input, extractor_model=None):
            return PipelineResult(article=article_input, is_prediction=False, gatekeeper_reason="", error="boom")

        monkeypatch.setattr(orch_mod, "run_article", fake_run_article)
        result = await orch.process_article(self._raw_art(), self._event(), self._source())
        assert result.error == "boom"

    async def test_no_extraction_writes_nothing(self, tmp_path, monkeypatch):
        orch = _make_orch(tmp_path)

        async def fake_run_article(article_input, extractor_model=None):
            return PipelineResult(article=article_input, is_prediction=False, gatekeeper_reason="not a prediction", extraction=None)

        monkeypatch.setattr(orch_mod, "run_article", fake_run_article)
        result = await orch.process_article(self._raw_art(), self._event(), self._source())
        assert result.extraction is None
        assert not orch.atlas_dir.exists() or not any(orch.atlas_dir.rglob("entry_*.json"))

    async def test_multiple_predictions_with_wide_spread_are_aggregated(self, tmp_path, monkeypatch):
        orch = _make_orch(tmp_path)
        preds = [_fake_prediction(stance=0.9), _fake_prediction(stance=-0.9)]

        async def fake_run_article(article_input, extractor_model=None):
            return PipelineResult(article=article_input, is_prediction=True, gatekeeper_reason="", extraction=ExtractionOutput(predictions=preds))

        aggregated = _fake_prediction(stance=0.1)

        async def fake_aggregate(predictions, event_name, source_name, article_date):
            return aggregated

        monkeypatch.setattr(orch_mod, "run_article", fake_run_article)
        monkeypatch.setattr(orch_mod, "aggregate_article_predictions", fake_aggregate)
        result = await orch.process_article(self._raw_art(), self._event(), self._source())
        assert len(result.extraction.predictions) == 1
        assert result.extraction.predictions[0].stance == 0.1

    async def test_aggregation_failure_keeps_original_predictions(self, tmp_path, monkeypatch):
        orch = _make_orch(tmp_path)
        preds = [_fake_prediction(stance=0.9), _fake_prediction(stance=-0.9)]

        async def fake_run_article(article_input, extractor_model=None):
            return PipelineResult(article=article_input, is_prediction=True, gatekeeper_reason="", extraction=ExtractionOutput(predictions=preds))

        async def fake_aggregate(*a, **kw):
            raise RuntimeError("LLM down")

        monkeypatch.setattr(orch_mod, "run_article", fake_run_article)
        monkeypatch.setattr(orch_mod, "aggregate_article_predictions", fake_aggregate)
        result = await orch.process_article(self._raw_art(), self._event(), self._source())
        assert len(result.extraction.predictions) == 2

    async def test_narrow_spread_predictions_are_not_aggregated(self, tmp_path, monkeypatch):
        orch = _make_orch(tmp_path)
        preds = [_fake_prediction(stance=0.5), _fake_prediction(stance=0.55)]

        async def fake_run_article(article_input, extractor_model=None):
            return PipelineResult(article=article_input, is_prediction=True, gatekeeper_reason="", extraction=ExtractionOutput(predictions=preds))

        called = {"agg": False}

        async def fake_aggregate(*a, **kw):
            called["agg"] = True
            raise AssertionError("should not aggregate")

        monkeypatch.setattr(orch_mod, "run_article", fake_run_article)
        monkeypatch.setattr(orch_mod, "aggregate_article_predictions", fake_aggregate)
        result = await orch.process_article(self._raw_art(), self._event(), self._source())
        assert called["agg"] is False
        assert len(result.extraction.predictions) == 2


# ─────────────────────────────────────────────
# _write_cell_signal
# ─────────────────────────────────────────────


class TestWriteCellSignal:
    def test_missing_cell_dir_is_noop(self, tmp_path):
        orch = _make_orch(tmp_path)
        orch._write_cell_signal("E1", "ynet")  # no exception

    def test_writes_aggregated_signal(self, tmp_path):
        orch = _make_orch(tmp_path)
        cell_dir = orch.atlas_dir / "E1" / "ynet"
        _write_json(cell_dir / "entry_aaaaaaaa.json", {
            "predictions": [{"quote": "q", "claim": "c", "stance": 0.5, "certainty": 0.8}],
        })
        orch._write_cell_signal("E1", "ynet")
        signal = json.loads((cell_dir / "cell_signal.json").read_text())
        assert signal["claim_count"] == 1
        assert signal["stance"] == 0.5

    def test_no_predictions_across_entries_writes_nothing(self, tmp_path):
        orch = _make_orch(tmp_path)
        cell_dir = orch.atlas_dir / "E1" / "ynet"
        _write_json(cell_dir / "entry_aaaaaaaa.json", {"predictions": []})
        orch._write_cell_signal("E1", "ynet")
        assert not (cell_dir / "cell_signal.json").exists()

    def test_corrupt_entry_file_is_skipped_not_fatal(self, tmp_path):
        orch = _make_orch(tmp_path)
        cell_dir = orch.atlas_dir / "E1" / "ynet"
        cell_dir.mkdir(parents=True, exist_ok=True)
        (cell_dir / "entry_bad.json").write_text("not json")
        _write_json(cell_dir / "entry_good.json", {
            "predictions": [{"quote": "q", "claim": "c", "stance": 0.2, "certainty": 0.5}],
        })
        orch._write_cell_signal("E1", "ynet")
        signal = json.loads((cell_dir / "cell_signal.json").read_text())
        assert signal["claim_count"] == 1


# ─────────────────────────────────────────────
# create_atlas_link
# ─────────────────────────────────────────────


class TestCreateAtlasLink:
    def test_builds_expected_link_data(self, tmp_path):
        orch = _make_orch(tmp_path)
        art_hash = "a" * 64
        extract_path = orch.vault_dir / "extractions" / f"{art_hash}_E1_{EXTRACTION_PROMPT_VERSION}.json"
        _write_json(extract_path, {
            "extraction": {"predictions": [{"quote": "q", "claim": "c", "stance": 0.5, "certainty": 0.8}]},
            "extractor_model": "nova-lite",
            "gatekeeper_model": "nova-micro",
            "gatekeeper_reason": "looks predictive",
        })
        raw_art = {"headline": "Head", "url": "https://ynet.com/a", "author": "Jane", "published_at": "2024-12-01"}
        orch.create_atlas_link("E1", "ynet", art_hash, extract_path, raw_art, event_date="2024-12-08")

        link_path = orch.atlas_dir / "E1" / "ynet" / f"entry_{art_hash[:8]}.json"
        data = json.loads(link_path.read_text())
        assert data["headline"] == "Head"
        assert data["article_date"] == "2024-12-01"
        assert data["event_date"] == "2024-12-08"
        assert data["extractor_model"] == "nova-lite"
        assert len(data["predictions"]) == 1

    def test_post_outcome_article_is_not_written(self, tmp_path):
        # Anti-lookahead backstop: an article dated after the outcome must not
        # reach the atlas even via create_atlas_link (near-dup / cached-extract
        # paths that bypass local_file_search's window filter).
        orch = _make_orch(tmp_path)
        art_hash = "d" * 64
        extract_path = orch.vault_dir / "extractions" / f"{art_hash}_E1_{EXTRACTION_PROMPT_VERSION}.json"
        _write_json(extract_path, {
            "extraction": {"predictions": [{"quote": "q", "claim": "c", "stance": 0.5, "certainty": 0.8}]},
            "extractor_model": "m", "gatekeeper_model": "g", "gatekeeper_reason": "",
        })
        raw_art = {"headline": "Leak", "url": "https://ynet.com/x", "author": "A", "published_at": "2024-12-20"}
        orch.create_atlas_link("E1", "ynet", art_hash, extract_path, raw_art, event_date="2024-12-08")
        assert not (orch.atlas_dir / "E1" / "ynet" / f"entry_{art_hash[:8]}.json").exists()

    def test_missing_article_date_is_conservatively_written(self, tmp_path):
        # predates_outcome is permissive on missing dates (ingest filters undated),
        # so create_atlas_link still writes — matching existing behavior.
        orch = _make_orch(tmp_path)
        art_hash = "e" * 64
        extract_path = orch.vault_dir / "extractions" / f"{art_hash}_E1_{EXTRACTION_PROMPT_VERSION}.json"
        _write_json(extract_path, {
            "extraction": {"predictions": []},
            "extractor_model": "m", "gatekeeper_model": "g", "gatekeeper_reason": "",
        })
        raw_art = {"headline": "H", "url": "https://ynet.com/y", "author": "A", "published_at": ""}
        orch.create_atlas_link("E1", "ynet", art_hash, extract_path, raw_art, event_date="2024-12-08")
        assert (orch.atlas_dir / "E1" / "ynet" / f"entry_{art_hash[:8]}.json").exists()


# ─────────────────────────────────────────────
# run_event — main run loop
# ─────────────────────────────────────────────


class TestRunEvent:
    async def test_missing_event_file_returns_early(self, tmp_path):
        orch = _make_orch(tmp_path)
        await orch.run_event("does-not-exist")  # no exception

    async def test_unknown_source_id_is_skipped(self, tmp_path, monkeypatch):
        orch = _make_orch(tmp_path)
        _make_event(tmp_path)
        _write_json(tmp_path / "sources" / "unknown.json", {"id": "unknown", "name": "Unknown", "url": "https://unknown.com"})

        called = {"search": False}

        async def fake_search(*a, **kw):
            called["search"] = True
            return []

        monkeypatch.setattr(orch, "search_articles", fake_search)
        await orch.run_event("E1")
        assert called["search"] is False

    async def test_done_cell_is_skipped_by_default(self, tmp_path, monkeypatch):
        from tm.progress import update_cell
        orch = _make_orch(tmp_path)
        _make_event(tmp_path)
        _make_source(tmp_path)
        update_cell("E1", "ynet", CellStatus.done, prediction_count=3)

        called = {"search": False}

        async def fake_search(*a, **kw):
            called["search"] = True
            return []

        monkeypatch.setattr(orch, "search_articles", fake_search)
        await orch.run_event("E1")
        assert called["search"] is False

    async def test_no_predictions_cell_is_skipped_by_default(self, tmp_path, monkeypatch):
        from tm.progress import update_cell
        orch = _make_orch(tmp_path)
        _make_event(tmp_path)
        _make_source(tmp_path)
        update_cell("E1", "ynet", CellStatus.no_predictions)

        called = {"search": False}

        async def fake_search(*a, **kw):
            called["search"] = True
            return []

        monkeypatch.setattr(orch, "search_articles", fake_search)
        await orch.run_event("E1")
        assert called["search"] is False

    async def test_retry_empty_reprocesses_no_predictions_cell(self, tmp_path, monkeypatch):
        from tm.progress import update_cell
        orch = _make_orch(tmp_path, retry_empty=True)
        _make_event(tmp_path)
        _make_source(tmp_path)
        update_cell("E1", "ynet", CellStatus.no_predictions)

        called = {"search": False}

        async def fake_search(*a, **kw):
            called["search"] = True
            return []

        monkeypatch.setattr(orch, "search_articles", fake_search)
        await orch.run_event("E1")
        assert called["search"] is True

    async def test_no_articles_marks_no_predictions(self, tmp_path, monkeypatch):
        orch = _make_orch(tmp_path)
        _make_event(tmp_path)
        _make_source(tmp_path)

        async def fake_search(*a, **kw):
            return []

        monkeypatch.setattr(orch, "search_articles", fake_search)
        await orch.run_event("E1")
        state = load_state()
        cell = state.get("E1", "ynet")
        assert cell.status == CellStatus.no_predictions

    async def test_no_articles_does_not_overwrite_failed_status(self, tmp_path, monkeypatch):
        from tm.progress import update_cell
        orch = _make_orch(tmp_path)
        _make_event(tmp_path)
        _make_source(tmp_path)
        update_cell("E1", "ynet", CellStatus.failed, error="prior failure")

        async def fake_search(*a, **kw):
            return []

        monkeypatch.setattr(orch, "search_articles", fake_search)
        await orch.run_event("E1")
        state = load_state()
        cell = state.get("E1", "ynet")
        assert cell.status == CellStatus.failed

    async def test_happy_path_marks_done_with_prediction_count(self, tmp_path, monkeypatch):
        orch = _make_orch(tmp_path)
        _make_event(tmp_path)
        _make_source(tmp_path)

        async def fake_search(*a, **kw):
            return [{"text": "x" * 600, "published_at": "2024-12-01", "url": "u", "author": "a", "headline": "h"}]

        async def fake_process(raw_art, event, source):
            cell_dir = orch.atlas_dir / event["id"] / source["id"]
            _write_json(cell_dir / "entry_aaaaaaaa.json", {
                "predictions": [{"quote": "q", "claim": "c", "stance": 0.5, "certainty": 0.8}],
                "article_date": "2024-12-01",
            })
            return PipelineResult(article=None, is_prediction=True, gatekeeper_reason="", extraction=ExtractionOutput(predictions=[_fake_prediction()]))

        monkeypatch.setattr(orch, "search_articles", fake_search)
        monkeypatch.setattr(orch, "process_article", fake_process)
        await orch.run_event("E1")

        state = load_state()
        cell = state.get("E1", "ynet")
        assert cell.status == CellStatus.done
        assert cell.prediction_count == 1
        assert (orch.atlas_dir / "E1" / "ynet" / "cell_signal.json").exists()

    async def test_article_timeout_marks_failed_when_no_predictions(self, tmp_path, monkeypatch):
        import asyncio

        orch = _make_orch(tmp_path)
        _make_event(tmp_path)
        _make_source(tmp_path)

        async def fake_search(*a, **kw):
            return [{"text": "x" * 600, "published_at": "2024-12-01", "url": "u", "author": "a", "headline": "h"}]

        async def fake_process(raw_art, event, source):
            # Simulate the wait_for(...) timeout without an actual 300s wait —
            # the run loop only distinguishes on the exception type it catches.
            raise asyncio.TimeoutError()

        monkeypatch.setattr(orch, "search_articles", fake_search)
        monkeypatch.setattr(orch, "process_article", fake_process)
        monkeypatch.setattr(orch_mod.asyncio, "sleep", _instant_sleep)

        await orch.run_event("E1")
        state = load_state()
        cell = state.get("E1", "ynet")
        assert cell.status == CellStatus.failed

    async def test_article_exception_marks_failed_when_no_predictions(self, tmp_path, monkeypatch):
        orch = _make_orch(tmp_path)
        _make_event(tmp_path)
        _make_source(tmp_path)

        async def fake_search(*a, **kw):
            return [{"text": "x" * 600, "published_at": "2024-12-01", "url": "u", "author": "a", "headline": "h"}]

        async def fake_process(raw_art, event, source):
            raise ValueError("boom")

        monkeypatch.setattr(orch, "search_articles", fake_search)
        monkeypatch.setattr(orch, "process_article", fake_process)
        monkeypatch.setattr(orch_mod.asyncio, "sleep", _instant_sleep)

        await orch.run_event("E1")
        state = load_state()
        cell = state.get("E1", "ynet")
        assert cell.status == CellStatus.failed

    async def test_partial_success_wins_over_had_errors(self, tmp_path, monkeypatch):
        """One article errors, another succeeds and writes a prediction —
        predictions-on-disk should win over had_errors (status == done)."""
        orch = _make_orch(tmp_path)
        _make_event(tmp_path)
        _make_source(tmp_path)

        async def fake_search(*a, **kw):
            return [
                {"text": "x" * 600, "published_at": "2024-12-01", "url": "u1", "author": "a", "headline": "h1"},
                {"text": "x" * 600, "published_at": "2024-12-01", "url": "u2", "author": "a", "headline": "h2"},
            ]

        call_count = {"n": 0}

        async def fake_process(raw_art, event, source):
            call_count["n"] += 1
            if raw_art["url"] == "u1":
                raise ValueError("boom")
            cell_dir = orch.atlas_dir / event["id"] / source["id"]
            _write_json(cell_dir / "entry_bbbbbbbb.json", {
                "predictions": [{"quote": "q", "claim": "c", "stance": 0.5, "certainty": 0.8}],
                "article_date": "2024-12-01",
            })
            return PipelineResult(article=None, is_prediction=True, gatekeeper_reason="", extraction=ExtractionOutput(predictions=[_fake_prediction()]))

        monkeypatch.setattr(orch, "search_articles", fake_search)
        monkeypatch.setattr(orch, "process_article", fake_process)
        monkeypatch.setattr(orch_mod.asyncio, "sleep", _instant_sleep)

        await orch.run_event("E1")
        assert call_count["n"] == 2
        state = load_state()
        cell = state.get("E1", "ynet")
        assert cell.status == CellStatus.done
        assert cell.prediction_count == 1


async def _instant_sleep(_seconds):
    return None
