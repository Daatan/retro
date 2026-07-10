"""Tests for ingest_resolution() / POST /leaderboard/ingest (credibility
feedback loop, step 1 — docs/ORACLE_VARIABLES.md "Open, in suggested order").
Storage only: no scoring is wired up yet. Tested against the module function
directly, matching this suite's convention for authed business logic (see
test_pool_aggregate.py) rather than through TestClient.
"""

from __future__ import annotations

import json

from forecast_api import resolution_feedback
from forecast_api.models import IngestResolutionRequest, ResolutionSourceInput


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
    resolution_feedback._ingested_ids.clear()
    resolution_feedback._loaded = False


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
        assert resolution_feedback.ingested_count() == 2

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
