"""Unit tests for compute_edge_probs.py's empirical/LLM blend math — untested
until now (retro#435). This is exactly the code path pm_analysis_refresh.yml's
daily direct-push-to-main cron runs unguarded (retro#430)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import compute_edge_probs as cep  # noqa: E402


# --- _align ---

def test_align_intersects_and_sorts_dates():
    pa = {"2024-01-03": 0.3, "2024-01-01": 0.1, "2024-01-05": 0.9}
    pb = {"2024-01-01": 0.2, "2024-01-03": 0.4}
    xa, xb = cep._align(pa, pb)
    assert xa == [0.1, 0.3]
    assert xb == [0.2, 0.4]


def test_align_no_overlap_returns_empty():
    assert cep._align({"2024-01-01": 0.1}, {"2024-02-01": 0.2}) == ([], [])


# --- corr_estimate ---

def test_corr_estimate_too_few_points_returns_none_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(cep, "HISTORY_DIR", tmp_path)
    (tmp_path / "A.json").write_text(json.dumps({"prices": [
        {"date": "2024-01-01", "probability": 0.5},
    ]}))
    (tmp_path / "B.json").write_text(json.dumps({"prices": [
        {"date": "2024-01-01", "probability": 0.5},
    ]}))
    result = cep.corr_estimate("A", "B")
    assert result["pY_corr"] is None
    assert result["pN_corr"] is None
    assert result["n_corr"] == 1


def test_corr_estimate_missing_node_file_returns_empty_alignment(tmp_path, monkeypatch):
    monkeypatch.setattr(cep, "HISTORY_DIR", tmp_path)
    result = cep.corr_estimate("missing_a", "missing_b")
    assert result["n_corr"] == 0
    assert result["pY_corr"] is None


def test_corr_estimate_bins_populated_with_enough_hi_lo_days(tmp_path, monkeypatch):
    monkeypatch.setattr(cep, "HISTORY_DIR", tmp_path)
    dates = [f"2024-01-{d:02d}" for d in range(1, 25)]
    # First 12 days: parent "clearly YES" (>0.60), child consistently high.
    # Last 12 days: parent "clearly NO" (<0.40), child consistently low.
    a_prices = [0.8] * 12 + [0.2] * 12
    b_prices = [0.9] * 12 + [0.1] * 12
    (tmp_path / "A.json").write_text(json.dumps({
        "prices": [{"date": d, "probability": p} for d, p in zip(dates, a_prices)]
    }))
    (tmp_path / "B.json").write_text(json.dumps({
        "prices": [{"date": d, "probability": p} for d, p in zip(dates, b_prices)]
    }))
    result = cep.corr_estimate("A", "B")
    assert result["n_hi"] == 12 and result["n_lo"] == 12
    assert result["pY_bin"] == pytest.approx(0.9)
    assert result["pN_bin"] == pytest.approx(0.1)


# --- _ci90_data / _ci90_llm ---

def test_ci90_data_widens_with_smaller_n():
    narrow = cep._ci90_data(0.5, std=0.1, n=100)
    wide = cep._ci90_data(0.5, std=0.1, n=10)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


def test_ci90_data_clips_to_valid_probability_range():
    lo, hi = cep._ci90_data(0.98, std=0.5, n=5)
    assert 0.01 <= lo <= hi <= 0.99


def test_ci90_llm_is_fixed_fifteen_points():
    lo, hi = cep._ci90_llm(0.5)
    assert lo == pytest.approx(0.35)
    assert hi == pytest.approx(0.65)


def test_ci90_llm_clips_at_bounds():
    lo, hi = cep._ci90_llm(0.05)
    assert lo == pytest.approx(0.01)


# --- blend ---

def _no_data_corr():
    return {"pY_corr": None, "pN_corr": None, "r2_corr": 0.0, "n_corr": 0,
            "pY_bin": None, "pN_bin": None, "n_hi": 0, "n_lo": 0,
            "std_hi": None, "std_lo": None}


def test_blend_pure_llm_when_no_empirical_data():
    result = cep.blend(0.7, 0.3, _no_data_corr())
    assert result["pY_blend"] == pytest.approx(0.7)
    assert result["pN_blend"] == pytest.approx(0.3)
    assert result["w_emp_Y"] == 0.0 and result["w_emp_N"] == 0.0
    # Falls back to fixed ±15pp LLM-only CI.
    assert result["pY_ci"] == [pytest.approx(0.55), pytest.approx(0.85)]


def test_blend_bin_data_pulls_estimate_toward_empirical_value():
    corr = {**_no_data_corr(), "pY_bin": 0.95, "n_hi": 20, "std_hi": 0.05}
    result = cep.blend(0.5, 0.3, corr)
    # w_llm=1.0, w_emp_Y=20/10=2.0 -> blend dominated by the empirical bin value.
    assert result["pY_blend"] > 0.5
    assert result["w_emp_Y"] == pytest.approx(2.0)


def test_blend_regression_fallback_when_bins_sparse_but_r2_positive():
    corr = {**_no_data_corr(), "pY_corr": 0.8, "r2_corr": 0.5}
    result = cep.blend(0.4, 0.2, corr)
    # w_emp_Y = r2 * 3.0 = 1.5, contributes toward pY_corr=0.8.
    assert result["w_emp_Y"] == pytest.approx(1.5)
    assert result["pY_blend"] > 0.4


def test_blend_output_always_within_valid_probability_bounds():
    corr = {**_no_data_corr(), "pY_bin": 0.99, "n_hi": 50, "std_hi": 0.01,
            "pN_bin": 0.01, "n_lo": 50, "std_lo": 0.01}
    result = cep.blend(0.5, 0.5, corr)
    assert 0.01 <= result["pY_blend"] <= 0.99
    assert 0.01 <= result["pN_blend"] <= 0.99
