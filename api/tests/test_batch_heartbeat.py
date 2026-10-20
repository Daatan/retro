"""Batch-loop heartbeat on /health (retro#996)."""
import json

import pytest
from fastapi.testclient import TestClient

from forecast_api import main as api_main
from forecast_api.batch_heartbeat import read_batch_heartbeat

client = TestClient(api_main.app)

HB = {
    "last_cycle_at": "2026-10-20T20:53:00+00:00",
    "last_cycle_ok": True,
    "last_cycle_ok_at": "2026-10-20T20:53:00+00:00",
    "head": "3a1c1c2059",
    "oldest_unpushed_at": None,
    "cells": {"done": 1354, "no_predictions": 655, "failed": 0, "pending": 0},
}


def test_reads_whitelisted_fields(tmp_path):
    (tmp_path / "heartbeat.json").write_text(json.dumps({**HB, "secret": "x"}))
    assert read_batch_heartbeat(tmp_path) == HB


def test_missing_file_is_none(tmp_path):
    assert read_batch_heartbeat(tmp_path) is None


@pytest.mark.parametrize("content", ["{not json", "[]", "null", ""])
def test_unreadable_file_is_none(tmp_path, content):
    (tmp_path / "heartbeat.json").write_text(content)
    assert read_batch_heartbeat(tmp_path) is None


def test_wrong_typed_fields_become_none(tmp_path):
    (tmp_path / "heartbeat.json").write_text(json.dumps(
        {"last_cycle_at": 5, "last_cycle_ok": "yes", "head": "", "cells": {"done": True, "pending": 2}}
    ))
    out = read_batch_heartbeat(tmp_path)
    assert out["last_cycle_at"] is None
    assert out["last_cycle_ok"] is None
    assert out["head"] is None
    assert out["last_cycle_ok_at"] is None
    assert out["cells"] == {"pending": 2}


def test_health_exposes_batch_loop(tmp_path, monkeypatch):
    (tmp_path / "heartbeat.json").write_text(json.dumps(HB))
    monkeypatch.setattr(api_main.settings, "data_dir", tmp_path)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["batch_loop"] == HB


def test_health_ok_without_heartbeat(tmp_path, monkeypatch):
    (tmp_path / "heartbeat.json").write_text("{garbage")
    monkeypatch.setattr(api_main.settings, "data_dir", tmp_path)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["batch_loop"] is None
