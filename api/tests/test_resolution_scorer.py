"""Tests for rescore_from_disk() / load_shadow_leaderboard() (credibility
feedback loop, step 3 — docs/ORACLE_VARIABLES.md §9). Replay-from-scratch
over an accumulated resolution_feedback.jsonl file, shadow-only: never reads
or writes leaderboard.json / get_credibility_weight().
"""

from __future__ import annotations

import json

import httpx
import pytest

from forecast_api import resolution_scorer
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

    def test_same_source_rows_average_within_one_resolution(self, tmp_path):
        """Three articles from one outlet on one prediction = ONE scored
        prediction at the mean stance, not three — same shape the author lane
        already uses, and what `predictions` must mean for the downstream
        shrinkage denominator to be honest."""
        ingest = tmp_path / "in.jsonl"
        _write_jsonl(ingest, [
            _record("p1", False, [_source("ynet", 0.2), _source("ynet", 0.4), _source("ynet", 0.6)]),
        ])
        rescore_from_disk(ingest, tmp_path / "out.json")
        board = {s["id"]: s for s in load_shadow_leaderboard(tmp_path / "out.json")}
        entry = board["ynet"]
        assert entry["predictions"] == 1
        assert entry["articles"] == 3
        # mean stance 0.4 -> P 0.7, outcome False: 0.7^2 = 0.49
        assert entry["brier_score"] == 0.49

    def test_mixed_sign_rows_never_make_a_source_compete_against_itself(self, tmp_path):
        """Regression: rows were fed to rate() individually, so an outlet with
        both a right and a wrong article landed in `winners` AND `losers`,
        competed against itself, and the loser write-back clobbered its winner
        update. Averaging first means it takes exactly one side."""
        ingest = tmp_path / "in.jsonl"
        _write_jsonl(ingest, [
            # ynet nets +0.5 (correct on a True outcome); blog is plainly wrong.
            _record("p1", True, [_source("ynet", 0.9), _source("ynet", -0.4), _source("blog", -0.9)]),
        ])
        rescore_from_disk(ingest, tmp_path / "out.json")
        board = {s["id"]: s for s in load_shadow_leaderboard(tmp_path / "out.json")}

        assert board["ynet"]["predictions"] == 1
        assert board["ynet"]["articles"] == 2
        # The net-correct outlet must gain, not be dragged below the loser by
        # its own contradictory row.
        assert board["ynet"]["skill_mu"] > 25.0
        assert board["blog"]["skill_mu"] < 25.0
        assert board["ynet"]["skill_conservative"] > board["blog"]["skill_conservative"]

    def test_single_source_is_counted_by_distinct_source_not_row_count(self, tmp_path):
        """Several rows from ONE outlet is still a single-source resolution —
        there is no ranking counterparty, however many articles it filed."""
        ingest = tmp_path / "in.jsonl"
        _write_jsonl(ingest, [
            _record("p1", True, [_source("ynet", 0.8), _source("ynet", 0.6), _source("ynet", 0.9)]),
        ])
        summary = rescore_from_disk(ingest, tmp_path / "out.json")
        assert summary["resolutions_single_source"] == 1


class TestCountResolutions:
    def test_missing_file_is_zero(self, tmp_path):
        assert resolution_scorer.count_resolutions(tmp_path / "nope.jsonl") == 0

    def test_counts_resolutions_not_sources_or_articles(self, tmp_path):
        ingest = tmp_path / "in.jsonl"
        _write_jsonl(ingest, [
            _record("p1", True, [_source("bbc", 0.8), _source("bbc", 0.7), _source("cnn", 0.6)]),
            _record("p2", False, [_source("bbc", -0.8)]),
        ])
        assert resolution_scorer.count_resolutions(ingest) == 2

    def test_matches_what_rescore_actually_scored(self, tmp_path):
        """The gate must count exactly the resolutions that produced scores —
        incomplete and opinion-only records are excluded by both."""
        ingest = tmp_path / "in.jsonl"
        _write_jsonl(ingest, [
            _record("p1", True, [_source("bbc", 0.8), _source("cnn", -0.8)]),
            _record("p2", None, [_source("bbc", 0.8)]),                                  # no outcome
            _record("p3", True, []),                                                     # no sources
            _record("p4", True, [_source("pundit", 0.9, evidence_class="opinion")]),      # opinion only
        ])
        summary = rescore_from_disk(ingest, tmp_path / "out.json")
        assert resolution_scorer.count_resolutions(ingest) == summary["resolutions_scored"] == 1


def _author_record(prediction_id, outcome, author_signals):
    return {"prediction_id": prediction_id, "outcome": outcome, "resolved_at": "2026-07-25", "author_signals": author_signals}


def _signal(author, lean, outlet="ynet", evidence_class="opinion"):
    return {"author": author, "outlet_name": outlet, "author_lean": lean, "author_lean_certainty": 0.8, "evidence_class": evidence_class}


class TestRescoreAuthorsFromDisk:
    @pytest.fixture(autouse=True)
    def _no_identity_map_by_default(self, monkeypatch):
        # Deterministic + network-free by default; tests below that care about
        # the identity map override this via their own monkeypatch.setattr.
        monkeypatch.setattr(resolution_scorer, "_load_identity_map", lambda: {})

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

    def test_aliased_bylines_merge_via_the_identity_map(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            resolution_scorer, "_load_identity_map",
            lambda: {"אבי כאלו": "Avi Kalo", "avi kalo (english byline)": "Avi Kalo"},
        )
        ingest = tmp_path / "in.jsonl"
        _write_jsonl(ingest, [
            _author_record("p1", True, [
                _signal("אבי כאלו", 0.6, outlet="ynet"),
                _signal("avi kalo (english byline)", 0.8, outlet="ynet"),
            ]),
        ])
        rescore_authors_from_disk(ingest, tmp_path / "out.json")
        board = load_shadow_leaderboard(tmp_path / "out.json")
        assert len(board) == 1
        assert board[0]["author"] == "Avi Kalo"
        assert board[0]["articles"] == 2

    def test_unmatched_byline_passes_through_unchanged(self, tmp_path, monkeypatch):
        monkeypatch.setattr(resolution_scorer, "_load_identity_map", lambda: {"someone else": "Someone Else"})
        ingest = tmp_path / "in.jsonl"
        _write_jsonl(ingest, [_author_record("p1", True, [_signal("Ben Caspit", 0.5, outlet="maariv")])])
        rescore_authors_from_disk(ingest, tmp_path / "out.json")
        board = load_shadow_leaderboard(tmp_path / "out.json")
        assert board[0]["author"] == "Ben Caspit"

    def test_falls_back_to_raw_grouping_when_identity_map_unavailable(self, tmp_path, monkeypatch):
        """Simulates news-indexer being unreachable: _load_identity_map already
        fails open to {} on its own (covered by TestLoadIdentityMap), so the
        caller-side behavior is just "an empty map changes nothing"."""
        monkeypatch.setattr(resolution_scorer, "_load_identity_map", lambda: {})
        ingest = tmp_path / "in.jsonl"
        _write_jsonl(ingest, [
            _author_record("p1", True, [_signal("Ben  Caspit ", 0.6, outlet="maariv"), _signal(" Ben Caspit", 0.8, outlet="maariv")]),
        ])
        rescore_authors_from_disk(ingest, tmp_path / "out.json")
        board = load_shadow_leaderboard(tmp_path / "out.json")
        assert len(board) == 1
        assert board[0]["author"] == "Ben Caspit"

    def test_no_byline_sentinel_is_never_looked_up_in_the_identity_map(self, tmp_path, monkeypatch):
        # A map that (incorrectly) carried an empty-string alias must not leak into NO_BYLINE.
        monkeypatch.setattr(resolution_scorer, "_load_identity_map", lambda: {"": "Should Not Apply"})
        ingest = tmp_path / "in.jsonl"
        _write_jsonl(ingest, [_author_record("p1", True, [_signal(None, 0.5, outlet="jpost")])])
        rescore_authors_from_disk(ingest, tmp_path / "out.json")
        board = load_shadow_leaderboard(tmp_path / "out.json")
        assert board[0]["author"] == "(no byline)"


class _FakeResponse:
    def __init__(self, json_data, status_code=200):
        self._json_data = json_data
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("error", request=httpx.Request("GET", "http://ni.test"), response=self)

    def json(self):
        return self._json_data


class TestLoadIdentityMap:
    def test_returns_empty_when_not_configured(self, monkeypatch):
        monkeypatch.setattr(resolution_scorer, "NEWS_INDEXER_URL", None)
        monkeypatch.setattr(resolution_scorer, "NEWS_INDEXER_API_KEY", None)
        assert resolution_scorer._load_identity_map() == {}

    def test_flattens_people_and_aliases_into_a_normalized_map(self, monkeypatch):
        monkeypatch.setattr(resolution_scorer, "NEWS_INDEXER_URL", "http://ni.test")
        monkeypatch.setattr(resolution_scorer, "NEWS_INDEXER_API_KEY", "key")
        people = {
            "people": [
                {
                    "canonical_name": "Itamar Eichner",
                    "aliases": [{"alias": "איתמר אייכנר"}, {"alias": " Itamar Eichner "}],
                },
            ],
        }
        monkeypatch.setattr(resolution_scorer.httpx, "get", lambda *a, **k: _FakeResponse(people))
        identity_map = resolution_scorer._load_identity_map()
        assert identity_map["איתמר אייכנר"] == "Itamar Eichner"
        assert identity_map["Itamar Eichner"] == "Itamar Eichner"

    def test_strips_hebrew_diacritics_so_gershayim_variants_match(self, monkeypatch):
        monkeypatch.setattr(resolution_scorer, "NEWS_INDEXER_URL", "http://ni.test")
        monkeypatch.setattr(resolution_scorer, "NEWS_INDEXER_API_KEY", "key")
        people = {"people": [{"canonical_name": "Shilo Freid", "aliases": [{"alias": "שילה פריד"}]}]}
        monkeypatch.setattr(resolution_scorer.httpx, "get", lambda *a, **k: _FakeResponse(people))
        identity_map = resolution_scorer._load_identity_map()
        # A stray gershayim mark on the raw byline must still hit the same key.
        assert resolution_scorer._normalize_identity("שילֹה פריד") in identity_map

    def test_fails_open_on_http_error_status(self, monkeypatch):
        monkeypatch.setattr(resolution_scorer, "NEWS_INDEXER_URL", "http://ni.test")
        monkeypatch.setattr(resolution_scorer, "NEWS_INDEXER_API_KEY", "key")
        monkeypatch.setattr(resolution_scorer.httpx, "get", lambda *a, **k: _FakeResponse({}, status_code=500))
        assert resolution_scorer._load_identity_map() == {}

    def test_fails_open_on_network_error(self, monkeypatch):
        monkeypatch.setattr(resolution_scorer, "NEWS_INDEXER_URL", "http://ni.test")
        monkeypatch.setattr(resolution_scorer, "NEWS_INDEXER_API_KEY", "key")

        def _raise(*a, **k):
            raise httpx.ConnectError("boom")

        monkeypatch.setattr(resolution_scorer.httpx, "get", _raise)
        assert resolution_scorer._load_identity_map() == {}

    def test_skips_people_with_no_canonical_name_and_empty_aliases(self, monkeypatch):
        monkeypatch.setattr(resolution_scorer, "NEWS_INDEXER_URL", "http://ni.test")
        monkeypatch.setattr(resolution_scorer, "NEWS_INDEXER_API_KEY", "key")
        people = {
            "people": [
                {"canonical_name": None, "aliases": [{"alias": "X"}]},
                {"canonical_name": "Y", "aliases": [{"alias": ""}]},
            ],
        }
        monkeypatch.setattr(resolution_scorer.httpx, "get", lambda *a, **k: _FakeResponse(people))
        assert resolution_scorer._load_identity_map() == {}


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
