"""Unit tests for fit_edges.py's pure math helpers (sigmoid, logit, masks,
predict, logloss, brier) and the Series price-lookup class — untested until
now (retro#435). main() itself (full ridge-regression backtest over
graph_pm.json + node_history/) is not covered here: it's an offline research
script, not load-bearing production code, and exercising it needs real
multi-node price history fixtures."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fit_edges  # noqa: E402


# --- sigmoid / logit ---

def test_sigmoid_at_zero_is_half():
    assert fit_edges.sigmoid(0.0) == pytest.approx(0.5)


def test_sigmoid_logit_are_inverses():
    for p in (0.1, 0.4, 0.5, 0.7, 0.95):
        assert fit_edges.sigmoid(fit_edges.logit(p)) == pytest.approx(p, abs=1e-6)


def test_logit_clips_extreme_probabilities():
    # logit clips to [0.001, 0.999] before taking log -> no inf/nan.
    assert np.isfinite(fit_edges.logit(0.0))
    assert np.isfinite(fit_edges.logit(1.0))


def test_sigmoid_clips_extreme_inputs_no_overflow():
    assert fit_edges.sigmoid(1000.0) == pytest.approx(1.0)
    assert fit_edges.sigmoid(-1000.0) == pytest.approx(0.0)


# --- masks ---

def test_masks_single_parent():
    m = fit_edges.masks(1)
    assert m.tolist() == [[0.0], [1.0]]


def test_masks_two_parents_all_combinations():
    m = fit_edges.masks(2)
    assert m.shape == (4, 2)
    assert sorted(m.tolist()) == [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]


# --- predict ---

def test_predict_no_parent_influence_when_weight_zero():
    # w=0 -> every parent-state combo collapses to sigmoid(b), regardless of P.
    P = np.array([[0.1, 0.9], [0.5, 0.5]])
    w = np.array([0.0, 0.0])
    pred = fit_edges.predict(P, w, b=0.0)
    assert pred == pytest.approx([0.5, 0.5])


def test_predict_certain_parent_state_matches_direct_sigmoid():
    # A single parent pinned at prob=1.0 (certainly "on") -> prediction should
    # equal sigmoid(b + w) exactly, since only the all-on state has mass.
    P = np.array([[1.0]])
    w = np.array([0.8])
    b = -0.3
    pred = fit_edges.predict(P, w, b)
    assert pred[0] == pytest.approx(fit_edges.sigmoid(b + w[0]))


def test_predict_output_bounded_zero_one():
    P = np.array([[0.3, 0.7], [0.9, 0.1]])
    w = np.array([2.0, -1.5])
    pred = fit_edges.predict(P, w, b=0.5)
    assert np.all((pred >= 0.0) & (pred <= 1.0))


# --- logloss / brier ---

def test_logloss_perfect_prediction_near_zero():
    pred = np.array([0.999, 0.001])
    actual = np.array([1.0, 0.0])
    assert fit_edges.logloss(pred, actual) < 0.01


def test_logloss_clips_to_avoid_infinite_loss():
    pred = np.array([0.0, 1.0])
    actual = np.array([1.0, 0.0])
    assert np.isfinite(fit_edges.logloss(pred, actual))


def test_brier_matches_direct_formula():
    pred = np.array([0.2, 0.8, 0.5])
    actual = np.array([0.0, 1.0, 1.0])
    expected = float(np.mean((pred - actual) ** 2))
    assert fit_edges.brier(pred, actual) == pytest.approx(expected)


# --- Series ---

def test_series_at_returns_most_recent_value_on_or_before_date(tmp_path, monkeypatch):
    node_history = tmp_path / "node_history"
    node_history.mkdir()
    (node_history / "N1.json").write_text(
        '{"prices": ['
        '{"date": "2024-01-01", "probability": 0.3},'
        '{"date": "2024-01-05", "probability": 0.6},'
        '{"date": "2024-01-10", "probability": 0.9}'
        "]}"
    )
    monkeypatch.setattr(fit_edges, "NODE_HISTORY", node_history)
    s = fit_edges.Series("N1")
    assert s.at("2024-01-05") == pytest.approx(0.6)
    assert s.at("2024-01-07") == pytest.approx(0.6)  # last value on/before the date
    assert s.at("2023-12-31") is None  # before any recorded price


def test_series_clips_extreme_probabilities(tmp_path, monkeypatch):
    node_history = tmp_path / "node_history"
    node_history.mkdir()
    (node_history / "N1.json").write_text(
        '{"prices": [{"date": "2024-01-01", "probability": 1.0},'
        '{"date": "2024-01-02", "probability": 0.0}]}'
    )
    monkeypatch.setattr(fit_edges, "NODE_HISTORY", node_history)
    s = fit_edges.Series("N1")
    assert s.at("2024-01-01") == pytest.approx(0.99)
    assert s.at("2024-01-02") == pytest.approx(0.01)
