"""
BayesOracle — log-odds DAG propagation for Israeli political scenarios.

The model mirrors the logic in bayesoracle/graph.html: each node has a prior
probability; parent nodes update child nodes via the law of total probability
approximated in log-odds space.

  lo(child) = logit(prior) + Σ (logit(pYes) - logit(pNo)) * (P(parent) - P(parent_prior))

This is a first-order perturbation around the prior.  It is not a full joint
distribution — it's a fast, interpretable approximation that matches the
original graph.html behaviour.
"""
import math
from typing import Optional

# ── Graph definition (matches bayesoracle/graph.html) ─────────────────────────

_NODES: list[dict] = [
    # Layer 0 — Exogenous
    {"id": "ELECTIONS",   "label": "Oct 2026 elections",      "layer": 0, "p": 0.93},
    {"id": "TRUMP",       "label": "Trump backs Bibi",         "layer": 0, "p": 0.85},
    {"id": "BEYACHAD",    "label": "Beyachad stays unified",   "layer": 0, "p": 0.68},
    {"id": "IRAN_REB",    "label": "Iran rebuilds nukes",      "layer": 0, "p": 0.44},
    {"id": "LIKUD_UNITY", "label": "Likud stays unified",      "layer": 0, "p": 0.68},
    # Layer 1 — Pre-election
    {"id": "BIBI_LEAD",  "label": "Likud leads polls",         "layer": 1, "p": 0.44},
    {"id": "CONVICTED",  "label": "Bibi convicted",            "layer": 1, "p": 0.08},
    {"id": "SICK",       "label": "Bibi health crisis",        "layer": 1, "p": 0.13},
    {"id": "DEAD",       "label": "Bibi dies",                 "layer": 1, "p": 0.04},
    # Layer 2 — Key event
    {"id": "A",          "label": "Likud wins most seats",     "layer": 2, "p": 0.46},
    # Layer 3 — Coalition
    {"id": "MANDATE",    "label": "Bibi gets mandate",         "layer": 3, "p": 0.44},
    {"id": "HAREDI_J",   "label": "Haredi parties join",       "layer": 3, "p": 0.74},
    {"id": "FARRIGHT_J", "label": "Ben Gvir / Smotrich join",  "layer": 3, "p": 0.60},
    {"id": "COAL_61",    "label": "Coalition hits 61 seats",   "layer": 3, "p": 0.34},
    # Layer 4 — Government
    {"id": "PM5",    "label": "Bibi: 5th term PM",             "layer": 4, "p": 0.32},
    {"id": "OPP_PM", "label": "Bennett / Lapid PM",            "layer": 4, "p": 0.38},
    # Layer 5 — Outcomes
    {"id": "PARDON",     "label": "Pardon granted",            "layer": 5, "p": 0.14},
    {"id": "JUDICIAL",   "label": "Judicial overhaul resumes", "layer": 5, "p": 0.25},
    {"id": "HAREDI_LAW", "label": "Haredi draft exemption law","layer": 5, "p": 0.28},
    {"id": "SAUDI",      "label": "Saudi normalization deal",  "layer": 5, "p": 0.17},
    {"id": "IRAN2",      "label": "2nd Iran operation",        "layer": 5, "p": 0.19},
]

# (source, target, pYes, pNo)
_EDGES: list[tuple[str, str, float, float]] = [
    ("TRUMP",       "BIBI_LEAD",   0.56, 0.30),
    ("BEYACHAD",    "BIBI_LEAD",   0.36, 0.58),
    ("LIKUD_UNITY", "BIBI_LEAD",   0.54, 0.28),
    ("SICK",        "DEAD",        0.28, 0.03),
    ("ELECTIONS",   "A",           0.46, 0.01),
    ("TRUMP",       "A",           0.54, 0.32),
    ("BEYACHAD",    "A",           0.34, 0.62),
    ("BIBI_LEAD",   "A",           0.78, 0.20),
    ("CONVICTED",   "A",           0.16, 0.50),
    ("LIKUD_UNITY", "A",           0.54, 0.28),
    ("SICK",        "A",           0.30, 0.48),
    ("DEAD",        "A",           0.05, 0.48),
    ("A",           "MANDATE",     0.96, 0.05),
    ("A",           "HAREDI_J",    0.78, 0.56),
    ("A",           "FARRIGHT_J",  0.64, 0.38),
    ("A",           "OPP_PM",      0.06, 0.58),
    ("MANDATE",     "COAL_61",     0.54, 0.07),
    ("HAREDI_J",    "COAL_61",     0.62, 0.22),
    ("FARRIGHT_J",  "COAL_61",     0.56, 0.19),
    ("BEYACHAD",    "OPP_PM",      0.48, 0.22),
    ("COAL_61",     "PM5",         0.93, 0.04),
    ("PM5",         "OPP_PM",      0.02, 0.56),
    ("PM5",         "PARDON",      0.38, 0.04),
    ("TRUMP",       "PARDON",      0.22, 0.07),
    ("PM5",         "JUDICIAL",    0.80, 0.04),
    ("PM5",         "HAREDI_LAW",  0.58, 0.11),
    ("HAREDI_J",    "HAREDI_LAW",  0.46, 0.14),
    ("PM5",         "SAUDI",       0.04, 0.24),
    ("OPP_PM",      "SAUDI",       0.36, 0.05),
    ("TRUMP",       "SAUDI",       0.24, 0.08),
    ("PM5",         "IRAN2",       0.36, 0.09),
    ("IRAN_REB",    "IRAN2",       0.52, 0.05),
    ("TRUMP",       "IRAN2",       0.32, 0.09),
]

# Pre-computed topology (sorted by layer ascending)
_TOPO: list[str] = [n["id"] for n in sorted(_NODES, key=lambda n: n["layer"])]
_PRIOR: dict[str, float] = {n["id"]: n["p"] for n in _NODES}

# Parent index: node_id → list of {source, pYes, pNo}
_PARENTS: dict[str, list[dict]] = {}
for src, tgt, pY, pN in _EDGES:
    _PARENTS.setdefault(tgt, []).append({"source": src, "pYes": pY, "pNo": pN})


def _logit(p: float) -> float:
    p = max(0.001, min(0.999, p))
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def compute_nodes(observations: Optional[dict[str, float]] = None) -> list[dict]:
    """
    Compute BayesOracle node probabilities.

    observations: optional dict of node_id → overridden probability (locked value).
    Returns list of {id, label, layer, prior, p, delta} sorted by layer then id.
    """
    current: dict[str, float] = dict(_PRIOR)

    # Apply observations (locked values)
    locked: set[str] = set()
    if observations:
        for node_id, p in observations.items():
            if node_id in current:
                current[node_id] = max(0.0, min(1.0, p))
                locked.add(node_id)

    # Propagate in topological order
    for node_id in _TOPO:
        if node_id in locked:
            continue
        parents = _PARENTS.get(node_id)
        if not parents:
            continue
        lo = _logit(_PRIOR[node_id])
        for parent in parents:
            lo += (_logit(parent["pYes"]) - _logit(parent["pNo"])) * (
                current[parent["source"]] - _PRIOR[parent["source"]]
            )
        current[node_id] = _sigmoid(lo)

    node_map = {n["id"]: n for n in _NODES}
    result = []
    for node_id in _TOPO:
        n = node_map[node_id]
        p = round(current[node_id], 4)
        result.append({
            "id": node_id,
            "label": n["label"],
            "layer": n["layer"],
            "prior": n["p"],
            "p": p,
            "delta": round(p - n["p"], 4),
            "locked": node_id in locked,
        })
    return result
