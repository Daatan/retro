"""Tests for metaculus_backtest.py's self-resolve fallback (retro#737)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from metaculus_backtest import classify_self_resolution, cmd_score  # noqa: E402


class TestClassifySelfResolution:
    def test_high_probability_resolves_yes(self):
        assert classify_self_resolution(0.95, confidence=0.9) == "yes"

    def test_low_probability_resolves_no(self):
        assert classify_self_resolution(0.05, confidence=0.9) == "no"

    def test_ambiguous_probability_stays_unresolved(self):
        assert classify_self_resolution(0.6, confidence=0.9) is None

    def test_exactly_at_confidence_threshold_resolves(self):
        assert classify_self_resolution(0.9, confidence=0.9) == "yes"
        assert classify_self_resolution(0.1, confidence=0.9) == "no"

    def test_never_forces_a_verdict_from_a_coin_flip(self):
        assert classify_self_resolution(0.5, confidence=0.9) is None


class _Args:
    def __init__(self, results, questions, self_resolved=None):
        self.results = str(results)
        self.questions = str(questions)
        self.self_resolved = str(self_resolved) if self_resolved else None


def _question(post_id, resolution=None, title="Q"):
    return {
        "post_id": post_id,
        "title": title,
        "resolution_criteria": "",
        "resolution": resolution,
        "community_prediction_latest": None,
    }


def _oracle_result(post_id, probability):
    return {"post_id": post_id, "oracle_probability": probability}


class TestScoreSelfResolvedFallback:
    def test_metaculus_resolution_takes_priority_over_self_resolved(self, tmp_path, capsys):
        questions = tmp_path / "q.json"
        results = tmp_path / "r.json"
        self_resolved = tmp_path / "sr.json"
        questions.write_text(json.dumps([_question(1, resolution="yes")]))
        results.write_text(json.dumps([_oracle_result(1, 0.8)]))
        self_resolved.write_text(json.dumps([{"post_id": 1, "self_resolution": "no"}]))

        rc = cmd_score(_Args(results, questions, self_resolved))
        out = capsys.readouterr().out
        assert rc == 0
        assert "Scored 1/1 questions on Metaculus resolutions" in out
        assert "Self-resolved" not in out  # the real resolution wins; fallback never applied

    def test_falls_back_to_self_resolved_when_metaculus_resolution_missing(self, tmp_path, capsys):
        questions = tmp_path / "q.json"
        results = tmp_path / "r.json"
        self_resolved = tmp_path / "sr.json"
        questions.write_text(json.dumps([_question(2, resolution=None)]))
        results.write_text(json.dumps([_oracle_result(2, 0.9)]))
        self_resolved.write_text(json.dumps([{"post_id": 2, "self_resolution": "yes"}]))

        rc = cmd_score(_Args(results, questions, self_resolved))
        out = capsys.readouterr().out
        assert rc == 0
        assert "Self-resolved (weaker evidence" in out
        assert "n=1" in out

    def test_no_self_resolved_file_leaves_question_unscored(self, tmp_path, capsys):
        questions = tmp_path / "q.json"
        results = tmp_path / "r.json"
        questions.write_text(json.dumps([_question(3, resolution=None)]))
        results.write_text(json.dumps([_oracle_result(3, 0.9)]))

        rc = cmd_score(_Args(results, questions, self_resolved=None))
        out = capsys.readouterr().out
        assert rc == 1
        assert "Nothing scorable yet" in out

    def test_ambiguous_self_resolution_stays_unscored_not_guessed(self, tmp_path, capsys):
        questions = tmp_path / "q.json"
        results = tmp_path / "r.json"
        self_resolved = tmp_path / "sr.json"
        questions.write_text(json.dumps([_question(4, resolution=None)]))
        results.write_text(json.dumps([_oracle_result(4, 0.9)]))
        self_resolved.write_text(json.dumps([{"post_id": 4, "self_resolution": None, "reason": "ambiguous_post_close_evidence"}]))

        rc = cmd_score(_Args(results, questions, self_resolved))
        out = capsys.readouterr().out
        assert rc == 1
        assert "Nothing scorable yet" in out
