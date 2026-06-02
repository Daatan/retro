"""Regression tests for the anti-lookahead scoring guard.

The product's core claim is that it scores only predictions published *before*
an event's outcome. A post-outcome article that reaches the atlas must never
contribute to a source's leaderboard score (which the live Oracle reads as a
credibility weight). These tests lock that invariant at two layers:

1. ``predates_outcome`` — the shared date guard.
2. ``Scorer.run`` — the end-to-end leaderboard path: a post-outcome entry is
   excluded, so a source whose *only* article is post-outcome never appears.
"""

import json

from tm.scorer import Scorer
from tm.utils import predates_outcome


class TestPredatesOutcome:
    def test_before_outcome_is_kept(self):
        assert predates_outcome("2024-12-01", "2024-12-08")

    def test_same_day_is_kept(self):
        assert predates_outcome("2024-12-08", "2024-12-08")

    def test_after_outcome_is_dropped(self):
        assert not predates_outcome("2024-12-20", "2024-12-08")

    def test_datetime_strings_compare_on_date(self):
        assert not predates_outcome("2024-12-09T03:00:00", "2024-12-08")
        assert predates_outcome("2024-12-08T23:59:00", "2024-12-08")

    def test_missing_or_unparseable_is_conservatively_kept(self):
        # We don't silently drop entries we can't evaluate (ingest filters undated).
        assert predates_outcome("", "2024-12-08")
        assert predates_outcome("2024-12-01", "")
        assert predates_outcome("not-a-date", "2024-12-08")


def _write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


class TestScorerExcludesPostOutcome:
    def test_post_outcome_source_is_not_scored(self, tmp_path):
        # Event resolved YES on 2024-12-08.
        _write_json(tmp_path / "events" / "E1.json", {
            "id": "E1", "name": "Test event", "outcome": True,
            "outcome_date": "2024-12-08", "predictive_window_days": 14,
            "category": [],
        })
        for sid in ("s_pre", "s_post"):
            _write_json(tmp_path / "sources" / f"{sid}.json", {"id": sid, "name": sid})

        pred = [{"stance": 0.8, "certainty": 0.7}]
        # Pre-outcome article — should be scored.
        _write_json(tmp_path / "atlas" / "E1" / "s_pre" / "entry_aaaaaaaa.json", {
            "article_date": "2024-12-01", "predictions": pred,
        })
        # Post-outcome article (lookahead leak) — must be excluded.
        _write_json(tmp_path / "atlas" / "E1" / "s_post" / "entry_bbbbbbbb.json", {
            "article_date": "2024-12-20", "predictions": pred,
        })

        Scorer(tmp_path).run()
        board = json.loads((tmp_path / "leaderboard.json").read_text())
        by_id = {row["id"]: row for row in board}

        assert "s_pre" in by_id and by_id["s_pre"]["predictions"] == 1
        # Its only article is post-outcome → 0 scored predictions → dropped entirely.
        assert "s_post" not in by_id
