"""Direct unit tests for tm.scorer — the Elo/RS scoring engine.

Existing coverage (test_elo_zero_sum.py, test_field_validation.py,
test_lookahead_guard.py) only incidentally exercises _update_elo and
Scorer.run through specific invariants. This file tests the scoring
primitives (Brier, log score, accuracy, confidence weighting, time decay,
Brier decomposition, calibration bins) directly, plus _update_skill and
edge cases in _update_elo/run not covered elsewhere.
"""

import json
import math

import pytest

from tm.scorer import (
    Scorer,
    stance_to_prob,
    brier_score,
    time_decay_weight,
    brier_decomposition,
    compute_calibration_bins,
)


# ─────────────────────────────────────────────
# Module-level pure functions
# ─────────────────────────────────────────────


class TestStanceToProb:
    def test_neutral_stance_is_half(self):
        assert stance_to_prob(0.0) == 0.5

    def test_full_positive_stance_is_one(self):
        assert stance_to_prob(1.0) == 1.0

    def test_full_negative_stance_is_zero(self):
        assert stance_to_prob(-1.0) == 0.0

    def test_partial_stance(self):
        assert stance_to_prob(0.5) == 0.75


class TestBrierScore:
    def test_perfect_yes_prediction(self):
        assert brier_score(1.0, True) == 0.0

    def test_perfect_no_prediction(self):
        assert brier_score(0.0, False) == 0.0

    def test_worst_possible_prediction(self):
        assert brier_score(1.0, False) == 1.0
        assert brier_score(0.0, True) == 1.0

    def test_neutral_prediction(self):
        assert brier_score(0.5, True) == 0.25
        assert brier_score(0.5, False) == 0.25


class TestTimeDecayWeight:
    def test_missing_article_date_returns_one(self):
        assert time_decay_weight("", "2024-12-08") == 1.0

    def test_missing_outcome_date_returns_one(self):
        assert time_decay_weight("2024-12-01", "") == 1.0

    def test_same_day_is_full_weight(self):
        assert time_decay_weight("2024-12-08", "2024-12-08") == 1.0

    def test_half_life_gives_half_weight(self):
        w = time_decay_weight("2024-11-08", "2024-12-08", half_life_days=30.0)
        assert math.isclose(w, 0.5, rel_tol=1e-6)

    def test_further_before_gives_lower_weight(self):
        w_close = time_decay_weight("2024-12-01", "2024-12-08")
        w_far = time_decay_weight("2024-11-01", "2024-12-08")
        assert w_far < w_close

    def test_article_after_outcome_clamped_to_zero_days(self):
        # days_before is max(0, ...) so a post-outcome article still gets weight 1.0
        w = time_decay_weight("2024-12-20", "2024-12-08")
        assert w == 1.0

    def test_unparseable_dates_return_one(self):
        assert time_decay_weight("not-a-date", "2024-12-08") == 1.0
        assert time_decay_weight("2024-12-01", "not-a-date") == 1.0


class TestBrierDecomposition:
    def test_fewer_than_five_points_returns_none(self):
        assert brier_decomposition([(0.5, 1.0)] * 4) is None

    def test_perfectly_calibrated_predictions(self):
        # 5 predictions, each exactly matching the bin's actual outcome rate.
        pairs = [(0.9, 1.0), (0.9, 1.0), (0.9, 1.0), (0.9, 1.0), (0.9, 0.0)]
        result = brier_decomposition(pairs, n_bins=5)
        assert result is not None
        assert result["n"] == 5
        assert math.isclose(result["o_bar"], 0.8, rel_tol=1e-6)
        # BS = REL - RES + UNC should hold
        assert math.isclose(result["brier"], result["rel"] - result["res"] + result["unc"], abs_tol=1e-4)

    def test_all_same_outcome_collapses_res_and_unc(self):
        pairs = [(0.5, 1.0)] * 5
        result = brier_decomposition(pairs, n_bins=5)
        assert result["res"] == 0.0
        assert result["unc"] == 0.0
        assert result["brier"] == result["rel"]


class TestComputeCalibrationBins:
    def test_fewer_than_ten_points_returns_none(self):
        pairs = [(0.5, 1.0)] * 9
        assert compute_calibration_bins(pairs) is None

    def test_ten_points_returns_bins(self):
        pairs = [(0.05 + 0.1 * i, 1.0 if i % 2 == 0 else 0.0) for i in range(10)]
        result = compute_calibration_bins(pairs, n_bins=10)
        assert result is not None
        assert len(result["labels"]) == 10
        assert sum(result["counts"]) == 10

    def test_empty_bins_default_actual_to_zero(self):
        # All 10 points land in the same bin; other 9 bins are empty.
        pairs = [(0.05, 1.0)] * 10
        result = compute_calibration_bins(pairs, n_bins=10)
        assert result["actual"][0] == 1.0
        assert result["actual"][1] == 0.0
        assert result["counts"][1] == 0


# ─────────────────────────────────────────────
# Scorer instance methods (pure primitives)
# ─────────────────────────────────────────────


@pytest.fixture
def scorer(tmp_path):
    return Scorer(tmp_path)


class TestCalculateBrier(object):
    def test_correct_confident_yes(self, scorer):
        assert scorer.calculate_brier(1.0, True) == 0.0

    def test_wrong_confident_yes(self, scorer):
        assert scorer.calculate_brier(1.0, False) == 1.0


class TestConfidenceWeight:
    def test_zero_certainty_gives_minimum_weight(self, scorer):
        assert scorer.confidence_weight(0.0) == 0.5

    def test_full_certainty_gives_maximum_weight(self, scorer):
        assert scorer.confidence_weight(1.0) == 2.0

    def test_mid_certainty(self, scorer):
        assert scorer.confidence_weight(0.5) == 1.25

    def test_clamps_above_one(self, scorer):
        assert scorer.confidence_weight(5.0) == 2.0

    def test_clamps_below_zero(self, scorer):
        assert scorer.confidence_weight(-5.0) == 0.5


class TestWeightedBrier:
    def test_high_certainty_amplifies_correct_score_to_zero_still(self, scorer):
        # Perfect prediction has brier 0 regardless of the weight multiplier.
        assert scorer.weighted_brier(1.0, 1.0, True) == 0.0

    def test_high_certainty_amplifies_wrong_score(self, scorer):
        low = scorer.weighted_brier(1.0, 0.0, False)   # weight 0.5
        high = scorer.weighted_brier(1.0, 1.0, False)  # weight 2.0
        assert high > low
        assert math.isclose(low, 1.0 * 0.5)
        assert math.isclose(high, 1.0 * 2.0)


class TestCalculateLogScore:
    def test_correct_prediction_is_less_negative(self, scorer):
        correct = scorer.calculate_log_score(0.9, True)
        wrong = scorer.calculate_log_score(0.9, False)
        assert correct > wrong
        assert correct <= 0.0

    def test_extreme_confident_wrong_is_clamped_not_infinite(self, scorer):
        # stance=1.0 -> prob clamped to 0.99, so log(1-0.99) is finite, not -inf.
        score = scorer.calculate_log_score(1.0, False)
        assert math.isfinite(score)
        assert math.isclose(score, math.log(0.01), rel_tol=1e-6)

    def test_extreme_confident_correct_is_clamped(self, scorer):
        score = scorer.calculate_log_score(-1.0, False)
        assert math.isclose(score, math.log(0.99), rel_tol=1e-6)


class TestCalculateAccuracy:
    def test_positive_stance_correct_when_outcome_true(self, scorer):
        assert scorer.calculate_accuracy(0.5, True) == 1

    def test_positive_stance_wrong_when_outcome_false(self, scorer):
        assert scorer.calculate_accuracy(0.5, False) == 0

    def test_negative_stance_correct_when_outcome_false(self, scorer):
        assert scorer.calculate_accuracy(-0.5, False) == 1

    def test_zero_stance_counts_as_wrong_for_true_outcome(self, scorer):
        assert scorer.calculate_accuracy(0.0, True) == 0

    def test_zero_stance_counts_as_wrong_for_false_outcome(self, scorer):
        # stance=0 -> (0 > 0) is False -> matches outcome=False -> "correct"
        assert scorer.calculate_accuracy(0.0, False) == 1


# ─────────────────────────────────────────────
# _update_elo — additional edge cases beyond test_elo_zero_sum.py
# ─────────────────────────────────────────────


class TestUpdateEloEdgeCases:
    def test_all_wrong_is_noop(self, scorer):
        stats = {"a": {"elo": 1200.0}, "b": {"elo": 1200.0}}
        scorer._update_elo(stats, [("a", -0.5), ("b", -0.8)], True)
        assert stats["a"]["elo"] == 1200.0
        assert stats["b"]["elo"] == 1200.0

    def test_single_correct_vs_single_wrong(self, scorer):
        stats = {"a": {"elo": 1200.0}, "b": {"elo": 1200.0}}
        scorer._update_elo(stats, [("a", 0.9), ("b", -0.9)], True, K=32)
        assert stats["a"]["elo"] == 1200.0 + 32 * 1 / 2
        assert stats["b"]["elo"] == 1200.0 - 32 * 1 / 2

    def test_custom_k_scales_movement(self, scorer):
        stats_low = {"a": {"elo": 1200.0}, "b": {"elo": 1200.0}}
        stats_high = {"a": {"elo": 1200.0}, "b": {"elo": 1200.0}}
        scorer._update_elo(stats_low, [("a", 0.9), ("b", -0.9)], True, K=10)
        scorer._update_elo(stats_high, [("a", 0.9), ("b", -0.9)], True, K=100)
        assert (stats_high["a"]["elo"] - 1200.0) > (stats_low["a"]["elo"] - 1200.0)

    def test_backwards_compat_alias(self, scorer):
        stats = {"a": {"elo": 1200.0}, "b": {"elo": 1200.0}}
        scorer.update_elo(stats, [("a", 0.9), ("b", -0.9)], True)
        assert stats["a"]["elo"] > 1200.0


# ─────────────────────────────────────────────
# _update_skill (OpenSkill / PlackettLuce)
# ─────────────────────────────────────────────


class TestUpdateSkill:
    def test_winner_mu_increases_loser_mu_decreases(self, scorer):
        from openskill.models import PlackettLuce

        model = PlackettLuce()
        ratings = {"a": model.rating(), "b": model.rating()}
        scorer._update_skill(model, ratings, [("a", 0.9), ("b", -0.9)], True)
        assert ratings["a"].mu > 25.0
        assert ratings["b"].mu < 25.0

    def test_all_correct_is_noop(self, scorer):
        from openskill.models import PlackettLuce

        model = PlackettLuce()
        r_a, r_b = model.rating(), model.rating()
        ratings = {"a": r_a, "b": r_b}
        scorer._update_skill(model, ratings, [("a", 0.9), ("b", 0.8)], True)
        assert ratings["a"] is r_a
        assert ratings["b"] is r_b

    def test_all_wrong_is_noop(self, scorer):
        from openskill.models import PlackettLuce

        model = PlackettLuce()
        r_a, r_b = model.rating(), model.rating()
        ratings = {"a": r_a, "b": r_b}
        scorer._update_skill(model, ratings, [("a", -0.9), ("b", -0.8)], True)
        assert ratings["a"] is r_a
        assert ratings["b"] is r_b

    def test_multiple_winners_and_losers(self, scorer):
        from openskill.models import PlackettLuce

        model = PlackettLuce()
        ratings = {sid: model.rating() for sid in ("a", "b", "c", "d")}
        preds = [("a", 0.9), ("b", 0.8), ("c", -0.7), ("d", -0.6)]
        scorer._update_skill(model, ratings, preds, True)
        assert ratings["a"].mu > 25.0
        assert ratings["b"].mu > 25.0
        assert ratings["c"].mu < 25.0
        assert ratings["d"].mu < 25.0


# ─────────────────────────────────────────────
# Scorer.run — end-to-end leaderboard construction
# ─────────────────────────────────────────────


def _write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


class TestScorerRun:
    def test_no_sources_produces_empty_leaderboard(self, tmp_path):
        _write_json(tmp_path / "events" / "E1.json", {
            "id": "E1", "name": "Test", "outcome": True,
            "outcome_date": "2024-12-08", "predictive_window_days": 14, "category": [],
        })
        board = Scorer(tmp_path).run()
        assert board == []

    def test_source_with_no_atlas_dir_is_excluded(self, tmp_path):
        _write_json(tmp_path / "events" / "E1.json", {
            "id": "E1", "name": "Test", "outcome": True,
            "outcome_date": "2024-12-08", "predictive_window_days": 14, "category": [],
        })
        _write_json(tmp_path / "sources" / "a.json", {"id": "a", "name": "Source A"})
        board = Scorer(tmp_path).run()
        assert board == []

    def test_two_sources_leaderboard_sorted_by_skill_conservative(self, tmp_path):
        # Event resolved YES. Source "good" is confidently correct, "bad" confidently wrong.
        _write_json(tmp_path / "events" / "E1.json", {
            "id": "E1", "name": "Test event", "outcome": True,
            "outcome_date": "2024-12-08", "predictive_window_days": 14,
            "category": ["politics"],
        })
        for sid in ("good", "bad"):
            _write_json(tmp_path / "sources" / f"{sid}.json", {"id": sid, "name": sid})

        _write_json(tmp_path / "atlas" / "E1" / "good" / "entry_aaaaaaaa.json", {
            "article_date": "2024-12-01",
            "predictions": [{"stance": 0.9, "certainty": 0.9}],
        })
        _write_json(tmp_path / "atlas" / "E1" / "bad" / "entry_bbbbbbbb.json", {
            "article_date": "2024-12-01",
            "predictions": [{"stance": -0.9, "certainty": 0.9}],
        })

        board = Scorer(tmp_path).run()
        by_id = {row["id"]: row for row in board}

        assert by_id["good"]["accuracy"] == 1.0
        assert by_id["bad"]["accuracy"] == 0.0
        assert by_id["good"]["elo"] > 1200.0
        assert by_id["bad"]["elo"] < 1200.0
        # Sorted descending by skill_conservative
        assert board[0]["id"] == "good"
        assert "politics" in by_id["good"]["by_category"]
        assert by_id["good"]["by_category"]["politics"]["accuracy"] == 1.0

    def test_single_predictor_event_gets_no_elo_or_skill_update(self, tmp_path):
        # Only one source predicts on this event -> len(event_predictions) == 1,
        # so run() must not call _update_elo/_update_skill (elo stays default).
        _write_json(tmp_path / "events" / "E1.json", {
            "id": "E1", "name": "Test", "outcome": True,
            "outcome_date": "2024-12-08", "predictive_window_days": 14, "category": [],
        })
        _write_json(tmp_path / "sources" / "solo.json", {"id": "solo", "name": "solo"})
        _write_json(tmp_path / "atlas" / "E1" / "solo" / "entry_aaaaaaaa.json", {
            "article_date": "2024-12-01",
            "predictions": [{"stance": 0.9, "certainty": 0.9}],
        })

        board = Scorer(tmp_path).run()
        assert board[0]["elo"] == 1200.0
        assert board[0]["skill_mu"] == 25.0

    def test_writes_calibration_file_when_enough_predictions(self, tmp_path):
        _write_json(tmp_path / "sources" / "a.json", {"id": "a", "name": "a"})
        for i in range(10):
            eid = f"E{i}"
            _write_json(tmp_path / "events" / f"{eid}.json", {
                "id": eid, "name": eid, "outcome": i % 2 == 0,
                "outcome_date": "2024-12-08", "predictive_window_days": 14, "category": [],
            })
            _write_json(tmp_path / "atlas" / eid / "a" / "entry_aaaaaaaa.json", {
                "article_date": "2024-12-01",
                "predictions": [{"stance": 0.5, "certainty": 0.5}],
            })

        Scorer(tmp_path).run()
        calib = json.loads((tmp_path / "calibration.json").read_text())
        assert calib["n_predictions"] == 10
        assert calib["calibration"] is not None

    def test_non_dir_entries_in_atlas_event_dir_are_skipped(self, tmp_path):
        _write_json(tmp_path / "events" / "E1.json", {
            "id": "E1", "name": "Test", "outcome": True,
            "outcome_date": "2024-12-08", "predictive_window_days": 14, "category": [],
        })
        _write_json(tmp_path / "sources" / "a.json", {"id": "a", "name": "a"})
        _write_json(tmp_path / "atlas" / "E1" / "a" / "entry_aaaaaaaa.json", {
            "article_date": "2024-12-01",
            "predictions": [{"stance": 0.5, "certainty": 0.5}],
        })
        # A stray file (not a directory) directly under the event dir.
        (tmp_path / "atlas" / "E1" / "stray.txt").write_text("noise")

        board = Scorer(tmp_path).run()
        assert board[0]["id"] == "a"

    def test_empty_predictions_list_in_entry_is_skipped(self, tmp_path):
        _write_json(tmp_path / "events" / "E1.json", {
            "id": "E1", "name": "Test", "outcome": True,
            "outcome_date": "2024-12-08", "predictive_window_days": 14, "category": [],
        })
        _write_json(tmp_path / "sources" / "a.json", {"id": "a", "name": "a"})
        _write_json(tmp_path / "atlas" / "E1" / "a" / "entry_aaaaaaaa.json", {
            "article_date": "2024-12-01", "predictions": [],
        })
        board = Scorer(tmp_path).run()
        assert board == []
