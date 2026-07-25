"""Tests for rescore_from_disk() / load_shadow_leaderboard() (credibility
feedback loop, step 3 — docs/ORACLE_VARIABLES.md §9). Replay-from-scratch
over an accumulated resolution_feedback.jsonl file, shadow-only: never reads
or writes leaderboard.json / get_credibility_weight().
"""

from __future__ import annotations

import json

from forecast_api.resolution_scorer import (
    load_shadow_leaderboard,
    rescore_authors_from_disk,
    rescore_from_disk,
)


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


def _author_record(prediction_id, outcome, author_signals):
    return {"prediction_id": prediction_id, "outcome": outcome, "resolved_at": "2026-07-25", "author_signals": author_signals}


def _signal(author, lean, outlet="ynet", evidence_class="opinion"):
    return {"author": author, "outlet_name": outlet, "author_lean": lean, "author_lean_certainty": 0.8, "evidence_class": evidence_class}


class TestRescoreAuthorsFromDisk:
    def test_missing_ingest_file_produces_empty_board(self, tmp_path):
        summary = rescore_authors_from_disk(tmp_path / "does-not-exist.jsonl", tmp_path / "out.json")
        assert summary["resolutions_total"] == 0
        assert summary["authors_scored"] == 0
        assert json.loads((tmp_path / "out.json").read_text()) == []

    def test_correct_author_outranks_wrong_author(self, tmp_path):
        ingest = tmp_path / "in.jsonl"
        _write_jsonl(ingest, [
            _author_record("p1", False, [_signal("Tehran Times desk", -0.7), _signal("Ben Caspit", 0.9, outlet="maariv")]),
        ])
        summary = rescore_authors_from_disk(ingest, tmp_path / "out.json")

        assert summary["resolutions_scored"] == 1
        board = {a["author"]: a for a in load_shadow_leaderboard(tmp_path / "out.json")}
        assert board["Tehran Times desk"]["skill_conservative"] > board["Ben Caspit"]["skill_conservative"]
        assert board["Tehran Times desk"]["brier_score"] < board["Ben Caspit"]["brier_score"]

    def test_opinion_class_is_scored_not_excluded(self, tmp_path):
        """The stance lane drops opinion; this lane exists FOR it — an
        op-ed author's lean is exactly the signal being scored."""
        ingest = tmp_path / "in.jsonl"
        _write_jsonl(ingest, [
            _author_record("p1", True, [_signal("Pundit", 0.8, evidence_class="opinion")]),
        ])
        rescore_authors_from_disk(ingest, tmp_path / "out.json")
        board = load_shadow_leaderboard(tmp_path / "out.json")
        assert [a["author"] for a in board] == ["Pundit"]

    def test_same_author_rows_average_within_one_resolution(self, tmp_path):
        """Three articles by one author on one prediction = ONE scored
        prediction at the mean lean, not three — mirrors the validated
        datapoint-#1 shape (P=(mean_lean+1)/2)."""
        ingest = tmp_path / "in.jsonl"
        _write_jsonl(ingest, [
            _author_record("p1", False, [_signal("Guy Bechor", 0.2), _signal("Guy Bechor", 0.4), _signal("Guy Bechor", 0.6)]),
        ])
        rescore_authors_from_disk(ingest, tmp_path / "out.json")
        board = {a["author"]: a for a in load_shadow_leaderboard(tmp_path / "out.json")}
        entry = board["Guy Bechor"]
        assert entry["predictions"] == 1
        assert entry["articles"] == 3
        # mean lean 0.4 -> P 0.7, outcome False: 0.7^2 = 0.49
        assert entry["brier_score"] == 0.49

    def test_no_byline_is_keyed_per_outlet(self, tmp_path):
        ingest = tmp_path / "in.jsonl"
        _write_jsonl(ingest, [
            _author_record("p1", True, [
                _signal(None, 0.5, outlet="tehran-times"),
                _signal("", -0.5, outlet="jpost"),
            ]),
        ])
        rescore_authors_from_disk(ingest, tmp_path / "out.json")
        board = load_shadow_leaderboard(tmp_path / "out.json")
        assert {(a["author"], a["outlet_name"]) for a in board} == {
            ("(no byline)", "tehran-times"),
            ("(no byline)", "jpost"),
        }

    def test_whitespace_variants_of_a_byline_merge(self, tmp_path):
        ingest = tmp_path / "in.jsonl"
        _write_jsonl(ingest, [
            _author_record("p1", True, [_signal("Ben  Caspit ", 0.6, outlet="maariv"), _signal(" Ben Caspit", 0.8, outlet="maariv")]),
        ])
        rescore_authors_from_disk(ingest, tmp_path / "out.json")
        board = load_shadow_leaderboard(tmp_path / "out.json")
        assert len(board) == 1
        assert board[0]["author"] == "Ben Caspit"
        assert board[0]["articles"] == 2

    def test_records_without_author_signals_are_skipped_not_fatal(self, tmp_path):
        """Pre-author-lane records have no author_signals key at all — the
        stance lane still scores them, this lane just passes over them."""
        ingest = tmp_path / "in.jsonl"
        _write_jsonl(ingest, [
            _record("p-old", True, [_source("bbc", 0.8)]),
            _author_record("p-new", True, [_signal("Author A", 0.5), _signal("Author B", -0.5)]),
        ])
        summary = rescore_authors_from_disk(ingest, tmp_path / "out.json")
        assert summary["resolutions_total"] == 2
        assert summary["resolutions_without_signals"] == 1
        assert summary["resolutions_scored"] == 1

    def test_non_ascii_bylines_survive_the_round_trip(self, tmp_path):
        ingest = tmp_path / "in.jsonl"
        _write_jsonl(ingest, [_author_record("p1", False, [_signal("אבי כאלו", -0.7, outlet="ynet")])])
        rescore_authors_from_disk(ingest, tmp_path / "out.json")
        board = load_shadow_leaderboard(tmp_path / "out.json")
        assert board[0]["author"] == "אבי כאלו"
        assert "אבי כאלו" in (tmp_path / "out.json").read_text()


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
