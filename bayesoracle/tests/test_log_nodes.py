import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from series import log_nodes  # noqa: E402


def test_questions_cover_every_graph_node():
    qs = log_nodes.load_questions()
    assert len(qs) >= 45
    assert log_nodes.check_graph_coverage(qs) == []
    ids = [n for n, _ in qs]
    assert len(ids) == len(set(ids))
    assert "caseA.BLOC_61" in ids and "pm.BIBI_PM" in ids and "political.RIGHT_BLOC_61" in ids


def test_loader_rejects_bad_length(tmp_path):
    f = tmp_path / "q.json"
    f.write_text(json.dumps({"pm": {"X": "hi"}}))
    try:
        log_nodes.load_questions(f)
    except ValueError as e:
        assert "5–500" in str(e)
    else:
        raise AssertionError("expected ValueError")


def _fake(question):
    return {
        "probability": 0.42, "ci": {"lower": 0.3, "upper": 0.55}, "confidence": "medium",
        "articles_used": 3, "insufficient_data": False,
        "sources": [{"url": "https://example.com/a", "title": "A"}, {"title": "no url"}],
    }


def test_run_is_idempotent_per_day(tmp_path):
    out = tmp_path / "nodes.jsonl"
    qs = [("pm.A", "Will A happen by 2027?"), ("pm.B", "Will B happen by 2027?")]
    calls = []

    def fc(q):
        calls.append(q)
        return _fake(q)

    assert log_nodes.run(qs, out, fc, date="2026-08-21", sleep_s=0, log=lambda *_: None) == 2
    assert log_nodes.run(qs, out, fc, date="2026-08-21", sleep_s=0, log=lambda *_: None) == 0
    assert len(calls) == 2
    # a new day appends again
    assert log_nodes.run(qs, out, fc, date="2026-08-22", sleep_s=0, log=lambda *_: None) == 2
    lines = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(lines) == 4
    rec = lines[0]
    assert rec == {
        "date": "2026-08-21", "node_id": "pm.A", "question": "Will A happen by 2027?",
        "probability": 0.42, "ci": [0.3, 0.55], "articles_used": 3, "confidence": "medium",
        "insufficient_data": False, "sources": ["https://example.com/a"],
    }


def test_run_skips_failures_and_refills(tmp_path):
    out = tmp_path / "nodes.jsonl"
    qs = [("pm.A", "Will A happen by 2027?"), ("pm.B", "Will B happen by 2027?")]
    state = {"fail_b": True}

    def fc(q):
        if "B" in q and state["fail_b"]:
            raise RuntimeError("boom")
        return _fake(q)

    assert log_nodes.run(qs, out, fc, date="2026-08-21", sleep_s=0, log=lambda *_: None) == 1
    state["fail_b"] = False
    assert log_nodes.run(qs, out, fc, date="2026-08-21", sleep_s=0, log=lambda *_: None) == 1
    assert log_nodes.logged_today(out, "2026-08-21") == {"pm.A", "pm.B"}
