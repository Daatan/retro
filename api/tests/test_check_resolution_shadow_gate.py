"""Tests for check_resolution_shadow_gate.py's gate_status() (retro#604)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from check_resolution_shadow_gate import gate_status  # noqa: E402


def _write_jsonl(path, records):
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


def _record(prediction_id, outcome, sources):
    return {"prediction_id": prediction_id, "outcome": outcome, "resolved_at": "2026-07-17", "sources": sources}


def _source(source, stance, evidence_class="reported_fact"):
    return {"source": source, "stance": stance, "evidence_class": evidence_class}


class TestGateStatus:
    def test_missing_ingest_file_reports_zero_and_gate_not_met(self, tmp_path):
        status = gate_status(tmp_path / "does-not-exist.jsonl", threshold=15)
        assert status == {"n": 0, "threshold": 15, "gate_met": False, "remaining": 15}

    def test_gate_not_met_below_threshold(self, tmp_path):
        ingest = tmp_path / "in.jsonl"
        _write_jsonl(ingest, [
            _record(f"p{i}", True, [_source("bbc", 0.8)]) for i in range(12)
        ])
        status = gate_status(ingest, threshold=15)
        assert status == {"n": 12, "threshold": 15, "gate_met": False, "remaining": 3}

    def test_gate_met_at_threshold(self, tmp_path):
        ingest = tmp_path / "in.jsonl"
        _write_jsonl(ingest, [
            _record(f"p{i}", True, [_source("bbc", 0.8)]) for i in range(15)
        ])
        status = gate_status(ingest, threshold=15)
        assert status == {"n": 15, "threshold": 15, "gate_met": True, "remaining": 0}

    def test_opinion_only_resolutions_are_not_counted(self, tmp_path):
        ingest = tmp_path / "in.jsonl"
        _write_jsonl(ingest, [
            _record("p1", True, [_source("pundit-blog", -0.8, evidence_class="opinion")]),
        ])
        status = gate_status(ingest, threshold=15)
        assert status["n"] == 0
        assert status["gate_met"] is False
