"""Tests for ingest_resolution() / POST /leaderboard/ingest (credibility
feedback loop, step 1 — docs/ORACLE_VARIABLES.md "Open, in suggested order").
Storage only: no scoring is wired up yet. Tested against the module function
directly, matching this suite's convention for authed business logic (see
test_pool_aggregate.py) rather than through TestClient.
"""

from __future__ import annotations

import asyncio
import json

import diskcache

from forecast_api import resolution_feedback
from forecast_api.models import AuthorSignalInput, IngestResolutionRequest, ResolutionSourceInput


def _req(**over) -> IngestResolutionRequest:
    return IngestResolutionRequest(**{
        "prediction_id": "pred-1",
        "outcome": True,
        "resolved_at": "2026-07-10",
        "sources": [
            ResolutionSourceInput(
                source="bbc",
                stance=0.6,
                evidence_class="reported_fact",
                credibility_weight=1.2,
                evidence_weight=1.0,
            ),
        ],
        **over,
    })


def _reset_module_state():
    """Simulate a fresh worker process: drop the in-process handles to the
    dedup diskcache stores (and the backfill-tracking set), but leave the
    on-disk diskcache directories themselves alone — that's the state a real
    process restart would see."""
    resolution_feedback._stores.clear()
    resolution_feedback._loaded_paths.clear()


class TestIngestResolution:
    async def test_first_ingest_is_accepted_and_persisted(self, tmp_path):
        _reset_module_state()
        path = tmp_path / "resolution_feedback.jsonl"

        resp = await resolution_feedback.ingest_resolution(path, _req())

        assert resp.accepted is True
        assert resp.already_ingested is False
        assert resp.sources_recorded == 1
        assert path.exists()
        lines = path.read_text().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["prediction_id"] == "pred-1"
        assert record["outcome"] is True
        assert record["sources"][0]["source"] == "bbc"
        assert record["sources"][0]["evidence_class"] == "reported_fact"

    async def test_duplicate_prediction_id_is_a_noop(self, tmp_path):
        _reset_module_state()
        path = tmp_path / "resolution_feedback.jsonl"

        await resolution_feedback.ingest_resolution(path, _req())
        resp = await resolution_feedback.ingest_resolution(path, _req())

        assert resp.already_ingested is True
        assert resp.sources_recorded == 0
        # still exactly one line on disk — the duplicate was never appended
        assert len(path.read_text().splitlines()) == 1

    async def test_duplicate_detection_survives_process_restart(self, tmp_path):
        """The in-memory guard is a cache, not the source of truth — a fresh
        process (module state cleared) must still dedupe from what's on disk."""
        path = tmp_path / "resolution_feedback.jsonl"

        _reset_module_state()
        await resolution_feedback.ingest_resolution(path, _req())

        _reset_module_state()  # simulate a fresh process, nothing loaded yet
        resp = await resolution_feedback.ingest_resolution(path, _req())

        assert resp.already_ingested is True
        assert len(path.read_text().splitlines()) == 1

    async def test_different_predictions_both_persist(self, tmp_path):
        _reset_module_state()
        path = tmp_path / "resolution_feedback.jsonl"

        r1 = await resolution_feedback.ingest_resolution(path, _req(prediction_id="pred-1"))
        r2 = await resolution_feedback.ingest_resolution(path, _req(prediction_id="pred-2", outcome=False))

        assert r1.already_ingested is False
        assert r2.already_ingested is False
        assert len(path.read_text().splitlines()) == 2
        assert resolution_feedback.ingested_count(path) == 2

    async def test_ingest_with_no_sources_still_records_the_resolution(self, tmp_path):
        _reset_module_state()
        path = tmp_path / "resolution_feedback.jsonl"

        resp = await resolution_feedback.ingest_resolution(path, _req(sources=[]))

        assert resp.already_ingested is False
        assert resp.sources_recorded == 0
        assert len(path.read_text().splitlines()) == 1

    async def test_malformed_line_on_disk_is_skipped_not_fatal(self, tmp_path):
        _reset_module_state()
        path = tmp_path / "resolution_feedback.jsonl"
        path.write_text("not json\n" + json.dumps({"prediction_id": "pred-existing"}) + "\n")

        resp = await resolution_feedback.ingest_resolution(path, _req(prediction_id="pred-existing"))

        assert resp.already_ingested is True  # loaded from the well-formed line despite the garbage one above it

    async def test_author_signals_persist_and_are_counted(self, tmp_path):
        _reset_module_state()
        path = tmp_path / "resolution_feedback.jsonl"
        req = _req(author_signals=[
            AuthorSignalInput(author="Ben Caspit", outlet_name="maariv", author_lean=0.9, author_lean_certainty=0.8, evidence_class="opinion"),
        ])

        resp = await resolution_feedback.ingest_resolution(path, req)

        assert resp.author_signals_recorded == 1
        record = json.loads(path.read_text().splitlines()[0])
        assert record["author_signals"][0]["author"] == "Ben Caspit"
        assert record["author_signals"][0]["author_lean"] == 0.9

    async def test_omitted_author_signals_default_to_empty(self, tmp_path):
        """An old-daatan payload with no author_signals key must keep
        working — the field is optional end to end."""
        _reset_module_state()
        path = tmp_path / "resolution_feedback.jsonl"

        resp = await resolution_feedback.ingest_resolution(path, _req())

        assert resp.author_signals_recorded == 0
        record = json.loads(path.read_text().splitlines()[0])
        assert record["author_signals"] == []

    async def test_opinion_class_is_recorded_not_filtered_at_ingest(self, tmp_path):
        """Exclusion of opinion-class articles is a scoring-time decision
        (step 2+), not an ingest-time filter — ingest keeps everything so the
        exclusion rule can change later without re-ingesting."""
        _reset_module_state()
        path = tmp_path / "resolution_feedback.jsonl"
        req = _req(sources=[
            ResolutionSourceInput(source="pundit-blog", stance=0.9, evidence_class="opinion"),
        ])

        resp = await resolution_feedback.ingest_resolution(path, req)

        assert resp.sources_recorded == 1
        record = json.loads(path.read_text().splitlines()[0])
        assert record["sources"][0]["evidence_class"] == "opinion"

    async def test_cross_worker_race_only_one_writer_wins(self, tmp_path):
        """retro#434 regression. Two gunicorn workers each hold their own
        diskcache.Cache handle onto the SAME on-disk dedup directory (this is
        exactly what _get_store() gives each worker process — separate Python
        objects, shared backing store). A concurrent add for the identical
        prediction_id from both must produce exactly one winner: the old
        in-memory-set guard couldn't provide this because each worker's set
        was invisible to the other."""
        path = tmp_path / "resolution_feedback.jsonl"
        dedup_dir = resolution_feedback._dedup_dir(path)
        worker_a = diskcache.Cache(str(dedup_dir))
        worker_b = diskcache.Cache(str(dedup_dir))

        results = await asyncio.gather(
            asyncio.to_thread(worker_a.add, "pred-race", True),
            asyncio.to_thread(worker_b.add, "pred-race", True),
        )

        assert sorted(results) == [False, True]

    async def test_ingest_from_a_different_worker_after_backfill_is_deduped(self, tmp_path):
        """A prediction_id ingested by 'worker A' (in-memory state reset to
        simulate a fresh process) must be recognized as already-ingested by
        'worker B', which never shared worker A's in-memory set — only the
        on-disk dedup store."""
        path = tmp_path / "resolution_feedback.jsonl"

        _reset_module_state()
        resp_a = await resolution_feedback.ingest_resolution(path, _req())

        _reset_module_state()
        resp_b = await resolution_feedback.ingest_resolution(path, _req())

        assert resp_a.already_ingested is False
        assert resp_b.already_ingested is True
        assert len(path.read_text().splitlines()) == 1
