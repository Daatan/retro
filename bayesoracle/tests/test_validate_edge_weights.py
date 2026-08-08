"""Unit tests for validate_edge_weights.py (retro#430) — the gate that stops a
NaN/inf regression fallback from compute_edge_probs.py reaching
apply_html_data.py, which would bake an invalid `pY:nan` JS token straight
into bayes.daatan.com's index.html with no warning."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import validate_edge_weights as vew  # noqa: E402


def _edge(**overrides):
    base = {
        "source": "a", "target": "b",
        "pY_blend": 0.6, "pN_blend": 0.4,
        "pY_ci": [0.4, 0.8], "pN_ci": [0.2, 0.6],
    }
    base.update(overrides)
    return base


def test_valid_edges_pass():
    assert vew.validate([_edge()]) == []


def test_empty_edge_list_is_rejected():
    assert vew.validate([]) == ["edge_weights.json is empty"]


def test_nan_probability_is_rejected():
    errors = vew.validate([_edge(pY_blend=float("nan"))])
    assert any("pY_blend" in e and "not a finite number" in e for e in errors)


def test_inf_probability_is_rejected():
    errors = vew.validate([_edge(pN_blend=float("inf"))])
    assert any("pN_blend" in e and "not a finite number" in e for e in errors)


def test_out_of_range_probability_is_rejected():
    errors = vew.validate([_edge(pY_blend=1.5)])
    assert any("pY_blend" in e and "out of [0,1]" in e for e in errors)


def test_malformed_ci_is_rejected():
    errors = vew.validate([_edge(pY_ci=[0.4])])
    assert any("pY_ci" in e and "2-element interval" in e for e in errors)


def test_inverted_ci_is_rejected():
    errors = vew.validate([_edge(pY_ci=[0.8, 0.4])])
    assert any("lower bound" in e and "exceeds upper bound" in e for e in errors)


def test_nan_ci_bound_is_rejected():
    errors = vew.validate([_edge(pN_ci=[float("nan"), 0.6])])
    assert any("pN_ci[0]" in e for e in errors)


def test_multiple_edges_report_all_errors_with_source_target_labels():
    errors = vew.validate([
        _edge(source="x", target="y", pY_blend=float("nan")),
        _edge(source="p", target="q", pN_blend=2.0),
    ])
    assert any(e.startswith("x→y:") for e in errors)
    assert any(e.startswith("p→q:") for e in errors)
    assert len(errors) == 2


def test_is_finite_number_rejects_bool_and_non_numeric():
    assert vew._is_finite_number(True) is False
    assert vew._is_finite_number("0.5") is False
    assert vew._is_finite_number(None) is False
    assert vew._is_finite_number(0.5) is True
    assert vew._is_finite_number(math.nan) is False
