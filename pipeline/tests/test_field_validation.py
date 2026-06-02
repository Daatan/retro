"""Tests for loud validation of required scoring fields (stance, certainty).

The extractor's Pydantic model requires numeric stance + certainty on every
prediction. The scoring layer reads raw JSON, so a missing/non-numeric value
means upstream corruption or a schema regression. It must be skipped loudly,
never silently scored as a neutral default (stance=0, certainty=0.5) — that
would quietly poison the leaderboard.
"""

import json

from tm.scorer import Scorer
from tm.utils import split_scored_predictions


class TestSplitScoredPredictions:
    def test_valid_predictions_are_usable(self):
        usable, malformed = split_scored_predictions(
            [{"stance": 0.5, "certainty": 0.8}, {"stance": -1.0, "certainty": 0.0}]
        )
        assert len(usable) == 2 and malformed == []

    def test_missing_stance_is_malformed(self):
        usable, malformed = split_scored_predictions([{"certainty": 0.8}])
        assert usable == [] and len(malformed) == 1

    def test_missing_certainty_is_malformed(self):
        usable, malformed = split_scored_predictions([{"stance": 0.5}])
        assert usable == [] and len(malformed) == 1

    def test_non_numeric_is_malformed(self):
        usable, malformed = split_scored_predictions(
            [{"stance": "0.5", "certainty": 0.8}, {"stance": None, "certainty": 0.8}]
        )
        assert usable == [] and len(malformed) == 2

    def test_bool_is_not_a_valid_number(self):
        # bool is an int subclass — must be rejected, not treated as 0/1.
        usable, malformed = split_scored_predictions([{"stance": True, "certainty": 0.8}])
        assert usable == [] and len(malformed) == 1

    def test_partition_keeps_only_valid(self):
        usable, malformed = split_scored_predictions(
            [{"stance": 0.2, "certainty": 0.5}, {"claim": "no fields"}]
        )
        assert usable == [{"stance": 0.2, "certainty": 0.5}] and len(malformed) == 1


def _write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


class TestScorerSkipsMalformed:
    def test_malformed_prediction_not_scored_as_neutral(self, tmp_path, capsys):
        # Event resolved YES. One source has a valid bullish prediction; another
        # has a single prediction missing `stance` (a schema-regression stand-in).
        _write_json(tmp_path / "events" / "E1.json", {
            "id": "E1", "name": "Test", "outcome": True,
            "outcome_date": "2024-12-08", "predictive_window_days": 14, "category": [],
        })
        for sid in ("good", "broken"):
            _write_json(tmp_path / "sources" / f"{sid}.json", {"id": sid, "name": sid})

        _write_json(tmp_path / "atlas" / "E1" / "good" / "entry_aaaaaaaa.json", {
            "article_date": "2024-12-01", "predictions": [{"stance": 0.9, "certainty": 0.8}],
        })
        # Only prediction lacks stance → must be skipped, not scored as stance=0.
        _write_json(tmp_path / "atlas" / "E1" / "broken" / "entry_bbbbbbbb.json", {
            "article_date": "2024-12-01", "predictions": [{"certainty": 0.8, "claim": "x"}],
        })

        Scorer(tmp_path).run()
        board = {row["id"]: row for row in json.loads((tmp_path / "leaderboard.json").read_text())}

        # The good source is scored; the broken source has no usable predictions
        # so it never enters the leaderboard (not scored as a neutral prediction).
        assert "good" in board and board["good"]["predictions"] == 1
        assert "broken" not in board
        # And it was loud.
        out = capsys.readouterr().out
        assert "WARNING" in out and "stance/certainty" in out
