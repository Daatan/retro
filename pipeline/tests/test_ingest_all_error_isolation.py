"""Regression tests for retro#440 item 3 — per-event error isolation in ingest_all.run_all().

Before the fix, an exception raised while processing one event (in any of the
gdelt / site_search / web_search per-event loops) propagated out of run_all()
and aborted the whole batch, silently dropping every event that hadn't been
processed yet. Each loop must now catch, log, and continue.
"""

import json

import pytest

pytest.importorskip("httpx")

from tm import ingest_all


def _write_event(events_dir, eid):
    events_dir.mkdir(parents=True, exist_ok=True)
    (events_dir / f"{eid}.json").write_text(json.dumps({"id": eid, "name": f"Event {eid}"}))


class TestGdeltErrorIsolation:
    async def test_one_bad_event_does_not_abort_the_batch(self, tmp_path, monkeypatch, capsys):
        events_dir = tmp_path / "events"
        for eid in ("A", "B", "C"):
            _write_event(events_dir, eid)

        async def fake_ingest_event(event, raw_ingest_dir, limit, force):
            if event["id"] == "B":
                raise RuntimeError("simulated crash for B")
            return 3

        monkeypatch.setattr(ingest_all.gdelt_ingest, "ingest_event", fake_ingest_event)

        # Should not raise — B's failure must be caught and logged, not propagated.
        await ingest_all.run_all(
            tmp_path, ["A", "B", "C"], skip=["site_search", "web_search"], limit=10, force=False,
        )

        out = capsys.readouterr().out
        assert "B" in out  # the failure was logged somewhere


class TestSiteSearchErrorIsolation:
    async def test_one_bad_cell_does_not_abort_other_cells(self, tmp_path, monkeypatch):
        events_dir = tmp_path / "events"
        _write_event(events_dir, "A")

        monkeypatch.setattr(ingest_all.site_search, "SEARCH_FNS", {"src1": (None, ""), "src2": (None, "")})

        calls = []

        async def fake_ingest_cell(event, source_id, raw_ingest_dir, force):
            calls.append(source_id)
            if source_id == "src1":
                raise ValueError("simulated crash for src1")
            return 5

        monkeypatch.setattr(ingest_all.site_search, "ingest_cell", fake_ingest_cell)

        await ingest_all.run_all(
            tmp_path, ["A"], skip=["gdelt", "web_search"], limit=10, force=False,
        )

        # Both cells must have been attempted despite src1's failure.
        assert calls == ["src1", "src2"]


class TestWebSearchErrorIsolation:
    async def test_one_bad_event_does_not_abort_the_batch(self, tmp_path, monkeypatch):
        events_dir = tmp_path / "events"
        for eid in ("A", "B"):
            _write_event(events_dir, eid)

        calls = []

        async def fake_ingest_event(event, raw_ingest_dir, limit, force):
            calls.append(event["id"])
            if event["id"] == "A":
                raise RuntimeError("simulated crash for A")
            return 7

        monkeypatch.setattr(ingest_all.web_search_ingest, "ingest_event", fake_ingest_event)

        await ingest_all.run_all(
            tmp_path, ["A", "B"], skip=["gdelt", "site_search"], limit=10, force=False,
        )

        # Both events must have been attempted despite A's failure.
        assert calls == ["A", "B"]
