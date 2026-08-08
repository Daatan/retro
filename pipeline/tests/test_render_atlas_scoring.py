"""compute_competitive_scores() — sliding-window competitive Brier scoring for the
atlas: only windows with >= min_per_window distinct sources get scored (retro#437).
Event/source ids must come from render_atlas.MVP_EVENTS/SOURCES — the function
iterates those module-level constants, not the keys of the dicts passed in."""

import pytest

from tm.render_atlas import compute_competitive_scores, ScoringConfig, MVP_EVENTS, SOURCES

EID = MVP_EVENTS[0]
SID_A, SID_B, SID_C = SOURCES[0], SOURCES[1], SOURCES[2]


def _cell(*, date, stance):
    return {"article_date": date, "predictions": [{"stance": stance}]}


def _event(outcome=True, outcome_date="2024-06-10"):
    return {"outcome": outcome, "outcome_date": outcome_date}


def test_window_below_min_sources_is_not_scored():
    cells = {(EID, SID_A): [_cell(date="2024-06-01", stance=1.0)]}
    events = {EID: _event()}
    config = ScoringConfig(window_hours=48, min_per_window=2)
    result = compute_competitive_scores(cells, events, config)
    assert result == {}


def test_window_with_enough_sources_scores_all_predictions_in_it():
    cells = {
        (EID, SID_A): [_cell(date="2024-06-01", stance=1.0)],
        (EID, SID_B): [_cell(date="2024-06-01", stance=1.0)],
    }
    events = {EID: _event(outcome=True)}
    config = ScoringConfig(window_hours=48, min_per_window=2)
    result = compute_competitive_scores(cells, events, config)
    assert (EID, SID_A) in result and (EID, SID_B) in result
    # stance=1.0 -> p=1.0, outcome=True -> brier=0.0 (perfect).
    assert result[(EID, SID_A)]["brier"] == pytest.approx(0.0)
    assert result[(EID, SID_A)]["n"] == 1


def test_predictions_after_outcome_date_are_excluded_anti_lookahead():
    cells = {
        (EID, SID_A): [_cell(date="2024-06-01", stance=1.0)],
        (EID, SID_B): [_cell(date="2024-07-01", stance=1.0)],  # after outcome_date
    }
    events = {EID: _event(outcome=True, outcome_date="2024-06-10")}
    config = ScoringConfig(window_hours=48, min_per_window=2)
    result = compute_competitive_scores(cells, events, config)
    # Only sid_a's prediction survives the anti-lookahead filter -> below min_per_window.
    assert result == {}


def test_predictions_outside_window_form_a_separate_group():
    cells = {
        (EID, SID_A): [_cell(date="2024-06-01", stance=1.0)],
        (EID, SID_B): [_cell(date="2024-06-01", stance=1.0)],
        (EID, SID_C): [_cell(date="2024-06-20", stance=-1.0)],  # far outside the 48h window
    }
    events = {EID: _event(outcome=True, outcome_date="2024-06-25")}
    config = ScoringConfig(window_hours=48, min_per_window=2)
    result = compute_competitive_scores(cells, events, config)
    # First window (a, b) has 2 distinct sources -> scored.
    assert (EID, SID_A) in result and (EID, SID_B) in result
    # Second window (c alone) has only 1 distinct source -> not scored.
    assert (EID, SID_C) not in result


def test_skill_score_formula():
    cells = {
        (EID, SID_A): [_cell(date="2024-06-01", stance=0.0)],  # p=0.5
        (EID, SID_B): [_cell(date="2024-06-01", stance=0.0)],
    }
    events = {EID: _event(outcome=True)}
    config = ScoringConfig(window_hours=48, min_per_window=2)
    result = compute_competitive_scores(cells, events, config)
    # brier = (0.5-1.0)^2 = 0.25 -> skill = 1 - 0.25/0.25 = 0.0
    assert result[(EID, SID_A)]["brier"] == pytest.approx(0.25)
    assert result[(EID, SID_A)]["skill"] == pytest.approx(0.0)


def test_unknown_event_is_skipped_without_error():
    result = compute_competitive_scores({}, {}, ScoringConfig())
    assert result == {}


def test_malformed_article_date_is_skipped():
    cells = {
        (EID, SID_A): [{"article_date": "not-a-date", "predictions": [{"stance": 1.0}]}],
        (EID, SID_B): [_cell(date="2024-06-01", stance=1.0)],
    }
    events = {EID: _event()}
    result = compute_competitive_scores(cells, events, ScoringConfig(min_per_window=2))
    assert result == {}
