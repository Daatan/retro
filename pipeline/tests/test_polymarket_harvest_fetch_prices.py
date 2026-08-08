"""Regression tests for retro#440 item 2 — crash-safe fetch_prices().

Before the fix, fetch_prices() accumulated all CLOB price fetches into an
in-memory list and rewrote events.jsonl once at the very end. A crash partway
through (process killed, unhandled exception, OOM, ...) lost every fetch done
in that run. It must now persist after each event, mirroring harvest()'s
per-event append-write pattern, so a crash only loses the one in-flight fetch
and a re-run resumes via the existing `prices_fetched` flag.
"""

import json

import pytest

pytest.importorskip("httpx")

from tm import polymarket_harvest as pm


def _seed_events(tmp_path, n):
    harvest_dir = tmp_path / "pm_harvest"
    harvest_dir.mkdir()
    output_path = harvest_dir / "events.jsonl"
    events = [
        {
            "pm_id": str(i),
            "question": f"q{i}",
            "outcome": True,
            "outcome_date": "2024-01-01",
            "category": "Politics",
            "pm_url": "",
            "clob_token_yes": f"tok{i}",
            "prices": [],
        }
        for i in range(n)
    ]
    with open(output_path, "w") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")
    return output_path


class TestFetchPricesCrashSafety:
    def test_progress_persists_after_a_mid_run_crash(self, tmp_path, monkeypatch):
        output_path = _seed_events(tmp_path, 3)

        calls = {"n": 0}

        def fake_fetch(token, outcome_date, client):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("simulated crash mid-fetch")
            return [{"date": outcome_date, "probability": 0.5}]

        monkeypatch.setattr(pm, "_fetch_price_history", fake_fetch)

        with pytest.raises(RuntimeError, match="simulated crash"):
            pm.fetch_prices(tmp_path)

        # Reload straight from disk (not from any in-memory state) — this is
        # what a resumed run would see.
        written = [json.loads(line) for line in output_path.read_text().splitlines()]
        assert len(written) == 3  # no lines lost/corrupted

        assert written[0]["prices_fetched"] is True
        assert written[0]["prices"] == [{"date": "2024-01-01", "probability": 0.5}]

        # Event 1 crashed mid-fetch — its own progress was not marked fetched,
        # but the file must still be a complete, valid rewrite (not truncated).
        assert written[1].get("prices_fetched", False) is False

        # Event 2 was never reached.
        assert written[2].get("prices_fetched", False) is False

    def test_a_resumed_run_skips_already_fetched_events(self, tmp_path, monkeypatch):
        output_path = _seed_events(tmp_path, 2)

        calls = []

        def fake_fetch(token, outcome_date, client):
            calls.append(token)
            return [{"date": outcome_date, "probability": 0.9}]

        monkeypatch.setattr(pm, "_fetch_price_history", fake_fetch)
        pm.fetch_prices(tmp_path)
        assert calls == ["tok0", "tok1"]

        # Second run: nothing left to do — the flag from the first run persisted.
        calls.clear()
        pm.fetch_prices(tmp_path)
        assert calls == []

        written = [json.loads(line) for line in output_path.read_text().splitlines()]
        assert all(ev["prices_fetched"] for ev in written)

    def test_no_partial_or_truncated_file_after_each_event(self, tmp_path, monkeypatch):
        """Every intermediate rewrite must itself be valid, complete JSONL —
        not just the final state — since a crash can land between any two events."""
        output_path = _seed_events(tmp_path, 3)
        snapshots = []

        real_replace = pm.os.replace

        def spying_replace(src, dst):
            # Capture the fully-written temp file's content before the atomic
            # rename lands it, proving each intermediate write is complete.
            with open(src) as f:
                snapshots.append([json.loads(line) for line in f if line.strip()])
            real_replace(src, dst)

        monkeypatch.setattr(pm.os, "replace", spying_replace)
        monkeypatch.setattr(
            pm, "_fetch_price_history",
            lambda token, outcome_date, client: [{"date": outcome_date, "probability": 0.5}],
        )

        pm.fetch_prices(tmp_path)

        assert len(snapshots) == 3  # one atomic rewrite per event
        for snap in snapshots:
            assert len(snap) == 3  # always the full event set, never truncated
