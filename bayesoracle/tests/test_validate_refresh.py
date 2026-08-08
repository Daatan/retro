"""Unit tests for validate_refresh.py — the sanity gate pm_analysis_refresh.yml
now runs before committing (retro#430) instead of pushing straight to main
unguarded."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import validate_refresh as vr  # noqa: E402


# --- _check_finite_probability ---

def test_accepts_none():
    vr._check_finite_probability(None, "x")


def test_accepts_in_range_value():
    vr._check_finite_probability(0.42, "x")


def test_rejects_nan():
    with pytest.raises(ValueError, match="NaN/inf"):
        vr._check_finite_probability(float("nan"), "x")


def test_rejects_inf():
    with pytest.raises(ValueError, match="NaN/inf"):
        vr._check_finite_probability(float("inf"), "x")


def test_rejects_out_of_range():
    with pytest.raises(ValueError, match="out of \\[0,1\\]"):
        vr._check_finite_probability(1.5, "x")


def test_rejects_non_numeric():
    with pytest.raises(ValueError, match="not numeric"):
        vr._check_finite_probability("0.5", "x")


# --- validate_edge_weights ---

def test_validate_edge_weights_passes_well_formed_file(tmp_path, monkeypatch):
    weights = tmp_path / "edge_weights.json"
    weights.write_text('[{"source": "A", "target": "B", "pY": 0.1, "pN": 0.9, "implied_p": 0.5}]')
    monkeypatch.setattr(vr, "WEIGHTS_FILE", weights)
    vr.validate_edge_weights()


def test_validate_edge_weights_rejects_empty_list(tmp_path, monkeypatch):
    weights = tmp_path / "edge_weights.json"
    weights.write_text("[]")
    monkeypatch.setattr(vr, "WEIGHTS_FILE", weights)
    with pytest.raises(ValueError, match="empty or not a list"):
        vr.validate_edge_weights()


def test_validate_edge_weights_rejects_nan_probability(tmp_path, monkeypatch):
    weights = tmp_path / "edge_weights.json"
    # json module parses the literal NaN token even though it's not valid JSON.
    weights.write_text('[{"source": "A", "target": "B", "pY": NaN}]')
    monkeypatch.setattr(vr, "WEIGHTS_FILE", weights)
    with pytest.raises(ValueError, match="NaN/inf"):
        vr.validate_edge_weights()


# --- validate_all_json ---

def test_validate_all_json_passes_well_formed_file(tmp_path, monkeypatch):
    all_json = tmp_path / "all.json"
    all_json.write_text('{"NODE_A": []}')
    monkeypatch.setattr(vr, "ALL_JSON", all_json)
    vr.validate_all_json()


def test_validate_all_json_rejects_empty_object(tmp_path, monkeypatch):
    all_json = tmp_path / "all.json"
    all_json.write_text("{}")
    monkeypatch.setattr(vr, "ALL_JSON", all_json)
    with pytest.raises(ValueError, match="empty or not an object"):
        vr.validate_all_json()
