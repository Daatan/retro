"""Negative cache for empty ingest cells (retro#836).

An empty cell leaves no ``article_*.json`` behind, so the ``existing_articles``
idiom could not tell "never searched" from "searched, nothing there", and the
batch loop re-walked the whole provider ladder — paid SERP legs included — every
time its offset came round. These tests pin the marker's contract: a second run
issues zero searches, ``--force`` and a changed window still search, an errored
run is never remembered, and the marker is invisible to the ``*.json`` globs
that ``ec2_run.sh`` and the orchestrator use to find articles."""

import json
from datetime import datetime, timedelta

import pytest

from tm import gnews_ingest, web_search_ingest
from tm.utils import EMPTY_MARKER_NAME, cell_marked_empty, mark_cell_empty

OLD = datetime(2024, 3, 1)
EVENT = {
    "id": "X01",
    "name": "test event",
    "outcome_date": "2024-03-01",
    "search_keywords": ["Israel recession 2024 GDP"],
}


# ── the helper pair ──────────────────────────────────────────────────────────

def test_marker_round_trip_and_not_a_json_glob_match(tmp_path):
    cell = tmp_path / "reuters" / "X01"
    assert not cell_marked_empty(cell, OLD)
    assert mark_cell_empty(cell, OLD, ingestor="gnews")
    assert cell_marked_empty(cell, OLD)
    # ec2_run.sh and Orchestrator.local_file_search treat every *.json in the
    # cell as an article; pathlib's glob matches dotfiles, so the name matters.
    assert list(cell.glob("*.json")) == []


def test_marker_expires(tmp_path):
    mark_cell_empty(tmp_path, OLD)
    marker = tmp_path / EMPTY_MARKER_NAME
    data = json.loads(marker.read_text())
    data["checked_at"] = (datetime.now() - timedelta(days=31)).isoformat()
    marker.write_text(json.dumps(data))
    assert not cell_marked_empty(tmp_path, OLD)
    assert cell_marked_empty(tmp_path, OLD, ttl_days=60)


def test_marker_ttl_from_env(tmp_path, monkeypatch):
    mark_cell_empty(tmp_path, OLD)
    monkeypatch.setenv("EMPTY_CELL_TTL_DAYS", "0")
    assert not cell_marked_empty(tmp_path, OLD)


def test_marker_for_another_window_is_ignored(tmp_path):
    mark_cell_empty(tmp_path, OLD)
    assert not cell_marked_empty(tmp_path, OLD - timedelta(days=30))


def test_open_or_just_closed_window_is_not_marked(tmp_path):
    assert not mark_cell_empty(tmp_path, datetime.now() + timedelta(days=10))
    assert not mark_cell_empty(tmp_path, datetime.now() - timedelta(days=2))
    assert not (tmp_path / EMPTY_MARKER_NAME).exists()


def test_corrupt_marker_counts_as_absent(tmp_path):
    (tmp_path / EMPTY_MARKER_NAME).write_text("not json")
    assert not cell_marked_empty(tmp_path, OLD)


# ── gnews_ingest.ingest_cell — the ingestor the batch loop actually runs ─────

@pytest.fixture
def gnews_calls(monkeypatch):
    """Stub every leg of the ladder to come back empty; count the calls."""
    calls = {"rss": 0, "web": 0, "newsdata": 0, "gdelt": 0}

    def rss(**kw):
        calls["rss"] += 1
        return []

    def web(*a, **kw):
        calls["web"] += 1
        return []

    async def newsdata(*a, **kw):
        calls["newsdata"] += 1
        return []

    async def gdelt(*a, **kw):
        calls["gdelt"] += 1
        return []

    monkeypatch.setattr(gnews_ingest, "search_gnews_rss", rss)
    monkeypatch.setattr(gnews_ingest, "_web_search", web)
    monkeypatch.setattr(gnews_ingest, "search_newsdata", newsdata)
    monkeypatch.setattr(gnews_ingest, "search_gdelt", gdelt)
    monkeypatch.delenv("ENABLE_CDX", raising=False)
    return calls


async def test_gnews_second_run_issues_no_searches(tmp_path, gnews_calls):
    assert await gnews_ingest.ingest_cell(EVENT, "reuters", tmp_path) == 0
    assert gnews_calls == {"rss": 1, "web": 1, "newsdata": 1, "gdelt": 1}

    assert await gnews_ingest.ingest_cell(EVENT, "reuters", tmp_path) == 0
    assert gnews_calls == {"rss": 1, "web": 1, "newsdata": 1, "gdelt": 1}


async def test_gnews_force_searches_again(tmp_path, gnews_calls):
    await gnews_ingest.ingest_cell(EVENT, "reuters", tmp_path)
    await gnews_ingest.ingest_cell(EVENT, "reuters", tmp_path, force=True)
    assert gnews_calls["web"] == 2


async def test_gnews_marker_is_per_cell(tmp_path, gnews_calls):
    await gnews_ingest.ingest_cell(EVENT, "reuters", tmp_path)
    await gnews_ingest.ingest_cell(EVENT, "bbc", tmp_path)
    assert gnews_calls["web"] == 2


async def test_gnews_error_is_not_remembered_as_empty(tmp_path, gnews_calls, monkeypatch):
    async def boom(*a, **kw):
        raise RuntimeError("provider outage")

    monkeypatch.setattr(gnews_ingest, "search_gdelt", boom)
    with pytest.raises(RuntimeError):
        await gnews_ingest.ingest_cell(EVENT, "reuters", tmp_path)
    assert not (tmp_path / "reuters" / "X01" / EMPTY_MARKER_NAME).exists()


# ── web_search_ingest.ingest_event — same hole, manual duel tool ─────────────

@pytest.fixture
def ws_calls(monkeypatch):
    calls = {"n": 0, "raise": False}

    def search(*a, **kw):
        calls["n"] += 1
        if calls["raise"]:
            raise RuntimeError("quota")
        return []

    async def no_sleep(_):
        return None

    monkeypatch.setattr(web_search_ingest._ws, "search_articles", search)
    monkeypatch.setattr(web_search_ingest.asyncio, "sleep", no_sleep)
    return calls


async def test_web_search_second_run_issues_no_searches(tmp_path, ws_calls):
    assert await web_search_ingest.ingest_event(EVENT, tmp_path, 10, False) == 0
    assert ws_calls["n"] == 1
    assert await web_search_ingest.ingest_event(EVENT, tmp_path, 10, False) == 0
    assert ws_calls["n"] == 1
    # a different --t-days is a different window, so it is searched
    await web_search_ingest.ingest_event(EVENT, tmp_path, 10, False, t_days=30)
    assert ws_calls["n"] == 2


async def test_web_search_errored_query_is_not_remembered(tmp_path, ws_calls):
    ws_calls["raise"] = True
    await web_search_ingest.ingest_event(EVENT, tmp_path, 10, False)
    assert not (tmp_path / "web_search" / "X01" / EMPTY_MARKER_NAME).exists()
