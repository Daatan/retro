"""Tests for the settlement-pin ledger (retro#361 Phase 1): record_settlement_pin()
/ load_ledger() / contradicted_pins(), the storage + report layer behind
GET /leaderboard/settlement-pin-report. Tested against the module functions
directly, matching test_resolution_feedback.py's convention for authed
business logic (see test_pool_aggregate.py) rather than through TestClient.

Classifying WHY a contradicted pin was wrong is explicitly out of scope here
(Phase 2, deferred) — these tests only cover recording the pin-vs-outcome
pairing and reporting which pins disagree.
"""

from __future__ import annotations

import asyncio
import json

import diskcache

from forecast_api import settlement_pin_ledger
from forecast_api.models import IngestResolutionRequest, SettlementSnapshotInput


def _req(**over) -> IngestResolutionRequest:
    return IngestResolutionRequest(**{
        "prediction_id": "pred-1",
        "outcome": True,
        "resolved_at": "2026-07-10",
        **over,
    })


def _snapshot(**over) -> SettlementSnapshotInput:
    return SettlementSnapshotInput(**{
        "settled": True,
        "mean": 0.94,
        "ci_low": 0.85,
        "ci_high": 0.97,
        "settled_sources": 3,
        **over,
    })


def _reset_module_state():
    """Simulate a fresh worker process: drop the in-process handles to the
    dedup diskcache stores (and the backfill-tracking set), but leave the
    on-disk diskcache directories themselves alone — the state a real
    process restart would see."""
    settlement_pin_ledger._stores.clear()
    settlement_pin_ledger._loaded_paths.clear()


class TestRecordSettlementPin:
    async def test_no_snapshot_is_not_recorded(self, tmp_path):
        _reset_module_state()
        path = tmp_path / "settlement_pin_ledger.jsonl"

        wrote = await settlement_pin_ledger.record_settlement_pin(path, _req())

        assert wrote is False
        assert not path.exists()

    async def test_unsettled_snapshot_is_not_recorded(self, tmp_path):
        """A snapshot with settled=False means the pool never pinned — there
        is no pin to post-mortem, so the ledger records nothing."""
        _reset_module_state()
        path = tmp_path / "settlement_pin_ledger.jsonl"
        req = _req(settlement_snapshot=_snapshot(settled=False))

        wrote = await settlement_pin_ledger.record_settlement_pin(path, req)

        assert wrote is False
        assert not path.exists()

    async def test_contradicted_pin_is_recorded_and_flagged(self, tmp_path):
        """England-Argentina-97 shape (retro#360): pin fired YES, resolution
        went NO."""
        _reset_module_state()
        path = tmp_path / "settlement_pin_ledger.jsonl"
        req = _req(
            prediction_id="pred-contradicted",
            outcome=False,
            settlement_snapshot=_snapshot(mean=0.94, ci_low=0.85, ci_high=0.97),
        )

        wrote = await settlement_pin_ledger.record_settlement_pin(path, req)

        assert wrote is True
        record = json.loads(path.read_text().splitlines()[0])
        assert record["prediction_id"] == "pred-contradicted"
        assert record["pin_direction"] == "yes"
        assert record["outcome"] is False
        assert record["contradicted"] is True

    async def test_confirmed_pin_is_recorded_but_not_flagged(self, tmp_path):
        _reset_module_state()
        path = tmp_path / "settlement_pin_ledger.jsonl"
        req = _req(
            prediction_id="pred-confirmed",
            outcome=True,
            settlement_snapshot=_snapshot(mean=0.94),
        )

        wrote = await settlement_pin_ledger.record_settlement_pin(path, req)

        assert wrote is True
        record = json.loads(path.read_text().splitlines()[0])
        assert record["pin_direction"] == "yes"
        assert record["outcome"] is True
        assert record["contradicted"] is False

    async def test_negative_pin_direction_is_no(self, tmp_path):
        _reset_module_state()
        path = tmp_path / "settlement_pin_ledger.jsonl"
        req = _req(
            prediction_id="pred-negative",
            outcome=False,
            settlement_snapshot=_snapshot(mean=-0.9, ci_low=-0.97, ci_high=-0.85),
        )

        await settlement_pin_ledger.record_settlement_pin(path, req)

        record = json.loads(path.read_text().splitlines()[0])
        assert record["pin_direction"] == "no"
        assert record["contradicted"] is False

    async def test_duplicate_prediction_id_is_a_noop(self, tmp_path):
        _reset_module_state()
        path = tmp_path / "settlement_pin_ledger.jsonl"
        req = _req(settlement_snapshot=_snapshot())

        first = await settlement_pin_ledger.record_settlement_pin(path, req)
        second = await settlement_pin_ledger.record_settlement_pin(path, req)

        assert first is True
        assert second is False
        assert len(path.read_text().splitlines()) == 1

    async def test_duplicate_detection_survives_process_restart(self, tmp_path):
        path = tmp_path / "settlement_pin_ledger.jsonl"
        req = _req(settlement_snapshot=_snapshot())

        _reset_module_state()
        await settlement_pin_ledger.record_settlement_pin(path, req)

        _reset_module_state()  # simulate a fresh process, nothing backfilled yet
        second = await settlement_pin_ledger.record_settlement_pin(path, req)

        assert second is False
        assert len(path.read_text().splitlines()) == 1

    async def test_cross_worker_race_only_one_writer_wins(self, tmp_path):
        """retro#434-shaped regression, applied to this ledger's dedup store:
        two concurrent adds for the identical prediction_id must produce
        exactly one winner."""
        path = tmp_path / "settlement_pin_ledger.jsonl"
        dedup_dir = settlement_pin_ledger._dedup_dir(path)
        worker_a = diskcache.Cache(str(dedup_dir))
        worker_b = diskcache.Cache(str(dedup_dir))

        results = await asyncio.gather(
            asyncio.to_thread(worker_a.add, "pred-race", True),
            asyncio.to_thread(worker_b.add, "pred-race", True),
        )

        assert sorted(results) == [False, True]

    async def test_malformed_line_on_disk_is_skipped_not_fatal_for_backfill(self, tmp_path):
        _reset_module_state()
        path = tmp_path / "settlement_pin_ledger.jsonl"
        path.write_text("not json\n" + json.dumps({"prediction_id": "pred-existing"}) + "\n")
        req = _req(prediction_id="pred-existing", settlement_snapshot=_snapshot())

        wrote = await settlement_pin_ledger.record_settlement_pin(path, req)

        assert wrote is False  # backfilled from the well-formed line despite the garbage one above it


class TestReport:
    async def test_contradicted_pins_lists_only_disagreeing_entries(self, tmp_path):
        _reset_module_state()
        path = tmp_path / "settlement_pin_ledger.jsonl"
        await settlement_pin_ledger.record_settlement_pin(
            path, _req(prediction_id="pred-wrong", outcome=False, settlement_snapshot=_snapshot(mean=0.94)),
        )
        await settlement_pin_ledger.record_settlement_pin(
            path, _req(prediction_id="pred-right", outcome=True, settlement_snapshot=_snapshot(mean=0.94)),
        )

        contradicted = await settlement_pin_ledger.contradicted_pins(path)

        assert [e.prediction_id for e in contradicted] == ["pred-wrong"]

    async def test_load_ledger_returns_both_contradicted_and_confirmed(self, tmp_path):
        _reset_module_state()
        path = tmp_path / "settlement_pin_ledger.jsonl"
        await settlement_pin_ledger.record_settlement_pin(
            path, _req(prediction_id="pred-wrong", outcome=False, settlement_snapshot=_snapshot(mean=0.94)),
        )
        await settlement_pin_ledger.record_settlement_pin(
            path, _req(prediction_id="pred-right", outcome=True, settlement_snapshot=_snapshot(mean=0.94)),
        )

        all_entries = await settlement_pin_ledger.load_ledger(path)

        assert {e.prediction_id for e in all_entries} == {"pred-wrong", "pred-right"}

    async def test_empty_ledger_reports_nothing(self, tmp_path):
        path = tmp_path / "settlement_pin_ledger.jsonl"

        assert await settlement_pin_ledger.load_ledger(path) == []
        assert await settlement_pin_ledger.contradicted_pins(path) == []

    async def test_unsettled_and_snapshot_free_pins_never_reach_the_report(self, tmp_path):
        """Only settlement_snapshot.settled=True pins are ever written, so the
        report can never surface a claim that was never actually pinned."""
        _reset_module_state()
        path = tmp_path / "settlement_pin_ledger.jsonl"
        await settlement_pin_ledger.record_settlement_pin(path, _req(prediction_id="pred-no-snapshot"))
        await settlement_pin_ledger.record_settlement_pin(
            path, _req(prediction_id="pred-unsettled", settlement_snapshot=_snapshot(settled=False)),
        )

        assert await settlement_pin_ledger.load_ledger(path) == []

    async def test_malformed_line_on_disk_is_skipped_not_fatal_for_report(self, tmp_path):
        path = tmp_path / "settlement_pin_ledger.jsonl"
        path.write_text("not json\n")

        assert await settlement_pin_ledger.load_ledger(path) == []
