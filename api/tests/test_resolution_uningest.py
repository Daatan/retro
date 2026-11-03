"""DELETE /leaderboard/ingest/{prediction_id}: undoing a reverted resolution (daatan#2077)."""
from __future__ import annotations

import json
import threading

import pytest
from fastapi.testclient import TestClient

from forecast_api import resolution_feedback, settlement_pin_ledger
from forecast_api.config import settings
from forecast_api.ledger_files import append_line, remove_prediction_lines
from forecast_api.main import app

client = TestClient(app)
HEADERS = {"x-api-key": "test-key"}


def _body(prediction_id: str, **over) -> dict:
    return {
        "prediction_id": prediction_id,
        "outcome": True,
        "resolved_at": "2026-11-02T00:00:00Z",
        "sources": [
            {"source": "s1", "stance": 0.6, "evidence_class": "reported_fact", "credibility_weight": 1.2, "evidence_weight": 1.0}
        ],
        "tags": ["politics"],
        "settlement_snapshot": {"settled": True, "mean": 0.9, "ci_low": 0.8, "ci_high": 1.0},
        **over,
    }


@pytest.fixture
def paths(tmp_path, monkeypatch):
    out = {
        "feedback": tmp_path / "feedback.jsonl",
        "pins": tmp_path / "pins.jsonl",
        "tags": tmp_path / "tags.json",
    }
    monkeypatch.setattr(settings, "resolution_feedback_path", out["feedback"])
    monkeypatch.setattr(settings, "settlement_pin_ledger_path", out["pins"])
    monkeypatch.setattr(settings, "resolution_tags_path", out["tags"])
    monkeypatch.setattr(settings, "resolution_leaderboard_path", tmp_path / "board.json")
    monkeypatch.setattr(settings, "resolution_author_leaderboard_path", tmp_path / "authors.json")
    for module in (resolution_feedback, settlement_pin_ledger):
        module._stores.clear()
        module._loaded_paths.clear()
    return out


def _ingest(prediction_id: str, **over) -> dict:
    r = client.post("/leaderboard/ingest", json=_body(prediction_id, **over), headers=HEADERS)
    assert r.status_code == 200, r.text
    return r.json()


def _ids(path) -> list[str]:
    return [json.loads(line)["prediction_id"] for line in path.read_text().splitlines()]


def test_removes_the_prediction_everywhere_and_leaves_others_alone(paths):
    _ingest("gone")
    _ingest("kept")
    kept_feedback = [line for line in paths["feedback"].read_text().splitlines() if '"kept"' in line]

    r = client.delete("/leaderboard/ingest/gone", headers=HEADERS)

    assert r.status_code == 200, r.text
    assert r.json() == {"feedback_removed": 1, "settlement_pins_removed": 1, "tags_removed": True}
    assert paths["feedback"].read_text().splitlines() == kept_feedback
    assert _ids(paths["pins"]) == ["kept"]
    assert list(json.loads(paths["tags"].read_text())) == ["kept"]


def test_a_re_resolution_after_the_revert_is_accepted_as_new(paths):
    _ingest("p")
    assert _ingest("p")["already_ingested"] is True

    client.delete("/leaderboard/ingest/p", headers=HEADERS)

    assert _ingest("p", outcome=False)["already_ingested"] is False
    assert [json.loads(line)["outcome"] for line in paths["feedback"].read_text().splitlines()] == [False]
    assert _ids(paths["pins"]) == ["p"]


def test_the_boards_are_replayed_without_the_prediction(paths):
    _ingest("gone")
    assert "s1" in json.dumps(json.loads((paths["feedback"].parent / "board.json").read_text()))

    client.delete("/leaderboard/ingest/gone", headers=HEADERS)

    assert "s1" not in json.dumps(json.loads((paths["feedback"].parent / "board.json").read_text()))


def test_never_ingested_is_a_harmless_zero_and_a_retry_is_too(paths):
    expected = {"feedback_removed": 0, "settlement_pins_removed": 0, "tags_removed": False}
    assert client.delete("/leaderboard/ingest/never", headers=HEADERS).json() == expected

    _ingest("p")
    client.delete("/leaderboard/ingest/p", headers=HEADERS)
    assert client.delete("/leaderboard/ingest/p", headers=HEADERS).json() == expected


def test_retry_rescores_and_clears_dedup_after_a_crash_mid_way(paths):
    """Line already removed, dedup key and board still stale: the retry must finish the job."""
    _ingest("p")
    assert remove_prediction_lines(paths["feedback"], "p") == 1

    assert client.delete("/leaderboard/ingest/p", headers=HEADERS).status_code == 200

    assert _ingest("p")["already_ingested"] is False


def test_requires_an_api_key(paths):
    assert client.delete("/leaderboard/ingest/p", headers={"x-api-key": "wrong"}).status_code in (401, 403)


def test_malformed_lines_survive_a_removal(tmp_path):
    path = tmp_path / "ledger.jsonl"
    path.write_text('{"prediction_id": "a"}\nnot json\n[1]\n{"prediction_id": "b"}\n')

    assert remove_prediction_lines(path, "a") == 1

    assert path.read_text() == 'not json\n[1]\n{"prediction_id": "b"}\n'


def test_appends_racing_a_removal_are_not_lost(tmp_path):
    path = tmp_path / "ledger.jsonl"
    path.write_text("".join(json.dumps({"prediction_id": f"old{i}"}) + "\n" for i in range(50)))

    def appender(n: int) -> None:
        for i in range(200):
            append_line(path, json.dumps({"prediction_id": f"new{n}-{i}"}))

    def remover() -> None:
        for i in range(50):
            remove_prediction_lines(path, f"old{i}")

    threads = [threading.Thread(target=appender, args=(n,)) for n in range(3)] + [threading.Thread(target=remover)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sorted(_ids(path)) == sorted(f"new{n}-{i}" for n in range(3) for i in range(200))
