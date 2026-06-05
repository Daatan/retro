"""Unit tests for the BayesOracle core engine."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import core  # noqa: E402

SPECS = Path(__file__).resolve().parents[1]


def _toy(**overrides):
    spec = {
        "name": "toy",
        "nodes": [
            {"id": "R", "layer": 0, "prior": 0.5},
            {"id": "C", "layer": 1, "prior": 0.4},
        ],
        "edges": [{"source": "R", "target": "C", "pYes": 0.8, "pNo": 0.2}],
    }
    spec.update(overrides)
    return spec


# ── validation ────────────────────────────────────────────────────────────────
def test_rejects_cycle():
    spec = _toy(edges=[
        {"source": "R", "target": "C", "pYes": 0.8, "pNo": 0.2},
        {"source": "C", "target": "R", "pYes": 0.8, "pNo": 0.2},
    ])
    with pytest.raises(core.GraphError, match="cycle"):
        core.Graph(spec)


def test_rejects_bad_prior():
    spec = _toy()
    spec["nodes"][0]["prior"] = 1.7
    with pytest.raises(core.GraphError, match="prior"):
        core.Graph(spec)


def test_rejects_unknown_endpoint():
    spec = _toy(edges=[{"source": "GHOST", "target": "C", "pYes": 0.8, "pNo": 0.2}])
    with pytest.raises(core.GraphError, match="not a node"):
        core.Graph(spec)


def test_rejects_bad_conditional():
    spec = _toy(edges=[{"source": "R", "target": "C", "pYes": 1.4, "pNo": 0.2}])
    with pytest.raises(core.GraphError, match="pYes"):
        core.Graph(spec)


# ── inference ─────────────────────────────────────────────────────────────────
def test_baseline_reproduces_priors():
    g = core.Graph(_toy())
    base = g.propagate()
    for nid in g.topo:
        assert base[nid] == pytest.approx(g.prior[nid], abs=1e-4)


@pytest.mark.parametrize("spec_file", ["graph_political.json", "graph_pm.json"])
def test_real_specs_baseline_consistent(spec_file):
    g = core.load_graph(SPECS / spec_file)
    base = g.propagate()
    assert max(abs(base[n] - g.prior[n]) for n in g.topo) < 1e-3


def test_excitatory_edge_raises_child():
    g = core.Graph(_toy())
    up = g.propagate({"R": 0.99})
    assert up["C"] > g.prior["C"]


def test_inhibitory_edge_lowers_child():
    spec = _toy(edges=[{"source": "R", "target": "C", "pYes": 0.2, "pNo": 0.8}])
    g = core.Graph(spec)
    up = g.propagate({"R": 0.99})
    assert up["C"] < g.prior["C"]


def test_observation_locks_node():
    g = core.Graph(_toy())
    out = g.propagate({"C": 0.9})
    assert out["C"] == pytest.approx(0.9, abs=1e-9)


def test_only_logit_difference_matters_not_levels():
    """The engine depends on w = logit(pYes)-logit(pNo), not on the absolute levels.

    0.3/0.1 and 0.9/0.7 have the same logit-difference, so they must give the
    same output. (Making the data *levels* bite is the backtest-fit step, not
    done here — see RETHINK.md §1.1.)
    """
    lo = core.Graph(_toy(edges=[{"source": "R", "target": "C", "pYes": 0.3, "pNo": 0.1}]))
    hi = core.Graph(_toy(edges=[{"source": "R", "target": "C", "pYes": 0.9, "pNo": 0.7}]))
    assert lo.propagate({"R": 0.99})["C"] == pytest.approx(hi.propagate({"R": 0.99})["C"], abs=1e-9)


def test_marginalisation_differs_from_plugin_mean():
    """Exact enumeration (E[sigmoid]) differs from the old sigmoid-of-mean when a
    parent is strictly between 0 and 1 — the real improvement over the old model."""
    g = core.Graph(_toy(edges=[{"source": "R", "target": "C", "pYes": 0.9, "pNo": 0.1}]))
    par = g.parents["C"][0]
    b, w = g._intercept["C"], par["weight"]
    p_r = 0.7
    enumerated = g.propagate({"R": p_r})["C"]
    plugin_mean = core.sigmoid(b + w * p_r)  # old first-order style
    assert abs(enumerated - plugin_mean) > 1e-3


# ── exclusive groups ──────────────────────────────────────────────────────────
def test_exclusive_group_renormalises_when_over_budget():
    spec = {
        "name": "excl",
        "nodes": [
            {"id": "X", "layer": 0, "prior": 0.6},
            {"id": "Y", "layer": 0, "prior": 0.7},
        ],
        "edges": [],
        "exclusive_groups": [{"id": "g", "members": ["X", "Y"], "allow_other": False}],
    }
    g = core.Graph(spec)
    out = g.propagate()
    assert out["X"] + out["Y"] == pytest.approx(1.0, abs=1e-6)


def test_exclusive_group_allows_slack():
    spec = {
        "name": "excl2",
        "nodes": [
            {"id": "X", "layer": 0, "prior": 0.3},
            {"id": "Y", "layer": 0, "prior": 0.4},
        ],
        "edges": [],
        "exclusive_groups": [{"id": "g", "members": ["X", "Y"], "allow_other": True}],
    }
    g = core.Graph(spec)
    out = g.propagate()
    assert out["X"] + out["Y"] == pytest.approx(0.7, abs=1e-6)  # untouched, slack = other


def test_pm_candidate_group_never_exceeds_one():
    g = core.load_graph(SPECS / "graph_pm.json")
    out = g.propagate({"BIBI_OUT": 0.99})
    members = ["BIBI_PM", "BENNETT_PM", "EIZENKOT_PM", "LIEBERMAN_PM", "LAPID_PM"]
    assert sum(out[m] for m in members if m in out) <= 1.0 + 1e-6


# ── topology ──────────────────────────────────────────────────────────────────
def test_topo_orders_parents_before_children():
    g = core.load_graph(SPECS / "graph_political.json")
    pos = {nid: i for i, nid in enumerate(g.topo)}
    for tgt, ps in g.parents.items():
        for par in ps:
            assert pos[par["source"]] < pos[tgt]
