"""Tests for rescore_from_disk() / load_shadow_leaderboard() (credibility
feedback loop, step 3 — docs/ORACLE_VARIABLES.md §9). Replay-from-scratch
over an accumulated resolution_feedback.jsonl file, shadow-only: never reads
or writes leaderboard.json / get_credibility_weight().
"""

from __future__ import annotations

import json

from forecast_api.resolution_scorer import load_shadow_leaderboard, rescore_from_disk


def _write_jsonl(path, records):
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


def _record(prediction_id, outcome, sources):
    return {"prediction_id": prediction_id, "outcome": outcome, "resolved_at": "2026-07-17", "sources": sources}


def _source(source, stance, evidence_class="reported_fact"):
    return {"source": source, "stance": stance, "evidence_class": evidence_class, "credibility_weight": 1.0, "evidence_weight": 1.0}


class TestRescoreFromDisk:
    def test_missing_ingest_file_produces_empty_leaderboard(self, tmp_path):
        summary = rescore_from_disk(tmp_path / "does-not-exist.jsonl", tmp_path / "out.json")
        assert summary["resolutions_total"] == 0
        assert summary["sources_scored"] == 0
        assert json.loads((tmp_path / "out.json").read_text()) == []

    def test_correct_source_outranks_wrong_source(self, tmp_path):
        ingest = tmp_path / "in.jsonl"
        _write_jsonl(ingest, [
            _record("p1", True, [_source("bbc", 0.8), _source("blog", -0.8)]),
        ])
        summary = rescore_from_disk(ingest, tmp_path / "out.json")

        assert summary["resolutions_scored"] == 1
        board = {s["id"]: s for s in load_shadow_leaderboard(tmp_path / "out.json")}
        assert board["bbc"]["skill_conservative"] > board["blog"]["skill_conservative"]

    def test_opinion_class_articles_are_excluded_from_scoring(self, tmp_path):
        ingest = tmp_path / "in.jsonl"
        _write_jsonl(ingest, [
            _record("p1", True, [
                _source("bbc", 0.8, evidence_class="reported_fact"),
                _source("pundit-blog", -0.8, evidence_class="opinion"),
            ]),
        ])
        summary = rescore_from_disk(ingest, tmp_path / "out.json")

        assert summary["articles_skipped_opinion"] == 1
        board = load_shadow_leaderboard(tmp_path / "out.json")
        # only bbc remains usable — a single source has no ranking
        # counterparty, so it's not scored as a ranking event, but it IS
        # still recorded (brier) since resolutions_scored counts it.
        assert summary["resolutions_scored"] == 1
        assert summary["resolutions_single_source"] == 1
        ids = [s["id"] for s in board]
        assert "pundit-blog" not in ids
        assert "bbc" in ids

    def test_single_source_resolution_still_scores_brier_with_no_ranking_movement(self, tmp_path):
        ingest = tmp_path / "in.jsonl"
        _write_jsonl(ingest, [_record("p1", True, [_source("bbc", 0.9)])])
        summary = rescore_from_disk(ingest, tmp_path / "out.json")

        assert summary["resolutions_scored"] == 1
        assert summary["resolutions_single_source"] == 1
        board = {s["id"]: s for s in load_shadow_leaderboard(tmp_path / "out.json")}
        assert board["bbc"]["predictions"] == 1
        assert board["bbc"]["brier_score"] < 0.1  # stance 0.9 -> prob 0.95, outcome True: (0.95-1)^2 = 0.0025
        # no counterparty to rank against — rating stays at the untouched default
        assert board["bbc"]["skill_mu"] == 25.0

    def test_incomplete_records_are_skipped_not_fatal(self, tmp_path):
        ingest = tmp_path / "in.jsonl"
        _write_jsonl(ingest, [
            {"prediction_id": "p-no-outcome", "outcome": None, "sources": [_source("bbc", 0.5)]},
            {"prediction_id": "p-no-sources", "outcome": True, "sources": []},
            _record("p-good", True, [_source("bbc", 0.8), _source("cnn", 0.7)]),
        ])
        summary = rescore_from_disk(ingest, tmp_path / "out.json")

        assert summary["resolutions_total"] == 3
        assert summary["resolutions_skipped_incomplete"] == 2
        assert summary["resolutions_scored"] == 1

    def test_malformed_line_is_skipped_not_fatal(self, tmp_path):
        ingest = tmp_path / "in.jsonl"
        ingest.write_text("not json\n" + json.dumps(_record("p1", True, [_source("bbc", 0.8), _source("cnn", 0.7)])) + "\n")

        summary = rescore_from_disk(ingest, tmp_path / "out.json")

        assert summary["resolutions_total"] == 1
        assert summary["resolutions_scored"] == 1

    def test_leaderboard_is_replayed_fresh_each_call_not_accumulated_across_calls(self, tmp_path):
        """A source with a long winning streak followed by ingesting just a
        losing record on its own (single-source) must not retain the
        prior call's rating boost — every call replays from a brand-new
        model over the full file, so it's pure function of the file's
        current contents."""
        ingest = tmp_path / "in.jsonl"
        out = tmp_path / "out.json"
        _write_jsonl(ingest, [_record("p1", True, [_source("bbc", 0.9), _source("blog", -0.9)])])
        rescore_from_disk(ingest, out)
        board_first = {s["id"]: s for s in load_shadow_leaderboard(out)}
        assert board_first["bbc"]["skill_mu"] > 25.0

        # Same file, called again with no new data — must reproduce
        # identical output, not drift from being "called twice".
        rescore_from_disk(ingest, out)
        board_second = {s["id"]: s for s in load_shadow_leaderboard(out)}
        assert board_second["bbc"]["skill_mu"] == board_first["bbc"]["skill_mu"]


class TestLoadShadowLeaderboard:
    def test_missing_file_returns_empty_list(self, tmp_path):
        assert load_shadow_leaderboard(tmp_path / "nope.json") == []

    def test_reads_back_what_rescore_wrote(self, tmp_path):
        ingest = tmp_path / "in.jsonl"
        out = tmp_path / "out.json"
        _write_jsonl(ingest, [_record("p1", True, [_source("bbc", 0.8), _source("cnn", -0.5)])])
        rescore_from_disk(ingest, out)
        board = load_shadow_leaderboard(out)
        assert {s["id"] for s in board} == {"bbc", "cnn"}
