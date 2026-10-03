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
        "mean": -0.16, "std": 0.2, "ci_low": -0.4, "ci_high": 0.1, "n_eff": 2.5, "evidence_mass": 1.1,
        "articles_used": 3, "insufficient_data": False, "reason": None, "settled": False,
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
        "probability": 0.42, "ci": [0.3, 0.55], "articles_used": 3,
        "confidence": {"std": 0.2, "n_eff": 2.5, "evidence_mass": 1.1},
        "insufficient_data": False, "reason": None, "settled": False, "sources": ["https://example.com/a"],
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


def test_insufficient_data_has_null_probability():
    rec = log_nodes.make_record("2026-08-21", "pm.A", "q?", {"mean": 0.0, "ci_low": -1, "ci_high": 1,
                                "articles_used": 0, "insufficient_data": True, "reason": "no_search_results"})
    assert rec["probability"] is None and rec["ci"] is None and rec["confidence"] is None
    assert rec["insufficient_data"] and rec["reason"] == "no_search_results"


# ─── cadence guard (retro#896) ────────────────────────────────────────────────
def test_due_every_three_days():
    due = log_nodes.due
    assert due("2026-10-03", None, 3)                 # nothing logged yet
    assert due("2026-10-03", "2026-10-03", 3)         # same day: finish a partial run
    assert not due("2026-10-04", "2026-10-03", 3)
    assert not due("2026-10-05", "2026-10-03", 3)
    assert due("2026-10-06", "2026-10-03", 3)
    assert due("2026-10-09", "2026-10-03", 3)         # a missed run is caught up, not skipped
    # month / year boundaries are real 72h, unlike `*/3` in cron's day-of-month field
    assert not due("2026-11-01", "2026-10-30", 3)
    assert due("2026-11-02", "2026-10-30", 3)
    assert due("2027-01-01", "2026-12-29", 3)


def test_due_default_and_bypass_keep_daily_behaviour():
    assert log_nodes.due("2026-10-04", "2026-10-03", 1)
    assert log_nodes.due("2026-10-04", "2026-10-03", 0)


def test_last_logged_date(tmp_path):
    out = tmp_path / "nodes.jsonl"
    assert log_nodes.last_logged_date(out) is None
    out.write_text('{"date": "2026-10-01", "node_id": "a"}\nnot json\n\n'
                   '{"date": "2026-10-03", "node_id": "a"}\n{"date": "2026-10-02", "node_id": "b"}\n')
    assert log_nodes.last_logged_date(out) == "2026-10-03"


def test_main_skips_before_touching_the_api(tmp_path, capsys, monkeypatch):
    out = tmp_path / "nodes.jsonl"
    out.write_text('{"date": "2026-10-03", "node_id": "pm.A"}\n')

    def boom(*_a, **_k):
        raise AssertionError("must not build a forecaster on a skip day")

    monkeypatch.setattr(log_nodes, "http_forecaster", boom)
    monkeypatch.setattr(log_nodes, "api_key", boom)
    rc = log_nodes.main(["--out", str(out), "--date", "2026-10-05", "--min-interval-days", "3"])
    assert rc == 0
    assert "skip: last logged 2026-10-03" in capsys.readouterr().out
    assert out.read_text().count("\n") == 1
