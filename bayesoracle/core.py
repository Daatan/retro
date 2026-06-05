"""
BayesOracle core engine — one inference implementation, many graph specs.

This replaces the three divergent propagation rules that used to live in
``graph.html``, ``pm_analysis/index.html`` and ``api/.../bayesoracle.py``.
A graph is a JSON document (see ``graph_political.json`` / ``graph_pm.json``);
this module loads it, validates it, and propagates probabilities.

Inference model — fitted-intercept logistic CPT
-----------------------------------------------
Each node with parents has a conditional probability table parameterised as a
logistic function of its parents' binary states::

    logit P(child=1 | parents) = b + Σ_i  w_i · x_i        x_i ∈ {0, 1}

with per-parent weight ``w_i = logit(pYes_i) − logit(pNo_i)``.  A positive
weight is excitatory, a negative weight inhibitory — both representable, unlike
noisy-OR.  The intercept ``b`` is *fitted* (1-D bisection) so that when every
parent sits at its prior the node's marginal equals the node's own prior.

The child marginal is computed by **exact enumeration** over parent states
(fan-in is small), assuming parents are independent within the enumeration::

    P(child=1) = Σ_combo  P(combo) · sigmoid(b + Σ_i w_i x_i^combo)

What this buys over the old first-order log-odds perturbation
(``sigmoid(logit(prior) + Σ w_i (P_parent − prior_parent))``):

  * it **marginalises over parent uncertainty** (expectation of the sigmoid)
    instead of plugging the mean parent state into one sigmoid — the two differ
    by Jensen's inequality whenever a parent sits strictly between 0 and 1;
  * mutually-exclusive outcomes are handled by simplex normalisation (below),
    not by directed-edge hacks;
  * the graph is loaded once, validated, and shared by every caller.

Note on what this does **not** change: the result is a function only of the
``w_i`` (the logit *differences*) and the priors — the absolute conditional
levels ``pYes``/``pNo`` enter solely through ``w_i``.  Making the data levels
bite requires fitting ``w_i`` to history (see backtest.py / RETHINK.md §1.1),
which is a separate, not-yet-done step.  Both this and the old model reproduce
the priors at baseline; that property is not new.

Secondary edges enter the same sum with their weight scaled by
``SECONDARY_WEIGHT`` (default 0.5), generalising the old pm_analysis behaviour.

Mutual-exclusivity groups renormalise member marginals to the probability
simplex (``Σ ≤ 1``), replacing hacks like the directed ``PM5 → OPP_PM`` edge.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Optional

SECONDARY_WEIGHT = 0.5
_EPS = 1e-9
_PROB_FLOOR, _PROB_CEIL = 0.001, 0.999


# ── math helpers ──────────────────────────────────────────────────────────────
def logit(p: float) -> float:
    p = min(_PROB_CEIL, max(_PROB_FLOOR, p))
    return math.log(p / (1.0 - p))


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


# ── graph model ───────────────────────────────────────────────────────────────
class GraphError(ValueError):
    """Raised when a graph spec is structurally invalid."""


class Graph:
    """A validated BayesOracle graph ready for propagation."""

    def __init__(self, spec: dict):
        self.name: str = spec.get("name", "bayesoracle")
        self.nodes: list[dict] = spec["nodes"]
        self.edges: list[dict] = spec.get("edges", [])
        self.exclusive_groups: list[dict] = spec.get("exclusive_groups", [])

        self.prior: dict[str, float] = {n["id"]: float(n["prior"]) for n in self.nodes}
        self.node_by_id: dict[str, dict] = {n["id"]: n for n in self.nodes}

        # parents[target] -> list of {source, weight, type}
        self.parents: dict[str, list[dict]] = {}
        for e in self.edges:
            w = logit(float(e["pYes"])) - logit(float(e["pNo"]))
            if e.get("type") == "secondary":
                w *= SECONDARY_WEIGHT
            self.parents.setdefault(e["target"], []).append(
                {"source": e["source"], "weight": w,
                 "pYes": float(e["pYes"]), "pNo": float(e["pNo"]),
                 "type": e.get("type", "primary")}
            )

        validate_graph(self)
        self.topo: list[str] = topological_order(self)
        self._intercept: dict[str, float] = {n: self._fit_intercept(n) for n in self.topo}

        # For inline normalisation: map each group to the topo index of its last
        # member, so we can renormalise before any downstream child is computed.
        pos = {nid: i for i, nid in enumerate(self.topo)}
        self._group_after: dict[str, list[dict]] = {}
        for grp in self.exclusive_groups:
            members = [m for m in grp["members"] if m in pos]
            if members:
                last = max(members, key=lambda m: pos[m])
                self._group_after.setdefault(last, []).append(grp)

    # ---- intercept fitting -----------------------------------------------------
    def _baseline_marginal(self, node_id: str, b: float) -> float:
        """Child marginal with parents fixed at their priors, given intercept b."""
        ps = self.parents.get(node_id, [])
        return self._enumerate(ps, b, {p["source"]: self.prior[p["source"]] for p in ps})

    def _fit_intercept(self, node_id: str) -> float:
        """Solve for b so baseline marginal == node prior (bisection)."""
        ps = self.parents.get(node_id)
        if not ps:
            return 0.0  # roots: marginal is the prior itself, intercept unused
        target = min(_PROB_CEIL, max(_PROB_FLOOR, self.prior[node_id]))
        lo, hi = -40.0, 40.0
        for _ in range(100):
            mid = (lo + hi) / 2.0
            if self._baseline_marginal(node_id, mid) < target:
                lo = mid
            else:
                hi = mid
            if hi - lo < 1e-10:
                break
        return (lo + hi) / 2.0

    # ---- enumeration -----------------------------------------------------------
    @staticmethod
    def _enumerate(parents: list[dict], b: float, p_source: dict[str, float]) -> float:
        """Exact marginal P(child=1) by enumerating parent states (independent)."""
        if not parents:
            return sigmoid(b)
        k = len(parents)
        total = 0.0
        for mask in range(1 << k):
            p_combo = 1.0
            z = b
            for i, par in enumerate(parents):
                on = (mask >> i) & 1
                pi = p_source[par["source"]]
                p_combo *= pi if on else (1.0 - pi)
                if on:
                    z += par["weight"]
            if p_combo > _EPS:
                total += p_combo * sigmoid(z)
        return total

    # ---- propagation -----------------------------------------------------------
    def propagate(self, observations: Optional[dict[str, float]] = None) -> dict[str, float]:
        """Return {node_id: probability} given optional locked observations."""
        current = dict(self.prior)
        locked: set[str] = set()
        if observations:
            for nid, p in observations.items():
                if nid in current:
                    current[nid] = min(1.0, max(0.0, float(p)))
                    locked.add(nid)

        for nid in self.topo:
            if nid not in locked:
                ps = self.parents.get(nid)
                if ps:  # roots without observation stay at prior
                    current[nid] = self._enumerate(ps, self._intercept[nid], current)
            # Renormalise any exclusive group whose last member is now computed,
            # so downstream children read the simplex-corrected value.
            for grp in self._group_after.get(nid, []):
                _normalise_group(current, grp, locked)
        return current

    def reconcile(self, values: Optional[dict[str, float]] = None) -> dict[str, float]:
        """
        Predict each node *purely from its parents* (no self-prior anchoring) via
        linear law-of-total-probability — primary edges at weight 1, secondary at
        0.5 — then renormalise exclusive groups.

        This is the divergence/diagnostic model used by pm_analysis: comparing a
        node's parent-implied probability to its own market price surfaces where
        the DAG and the market disagree.  Unlike ``propagate`` it does **not** fit
        an intercept to the node's own prior, so the prior/edge mismatch is *kept*,
        not absorbed.  Nodes without parents keep their prior.
        """
        current = dict(self.prior)
        if values:
            for nid, v in values.items():
                if nid in current:
                    current[nid] = min(1.0, max(0.0, float(v)))
        for nid in self.topo:
            ps = self.parents.get(nid)
            if not ps:
                continue
            wsum = tw = 0.0
            for par in ps:
                w = 0.5 if par["type"] == "secondary" else 1.0
                psrc = current[par["source"]]
                wsum += w * (par["pYes"] * psrc + par["pNo"] * (1.0 - psrc))
                tw += w
            current[nid] = wsum / tw if tw else current[nid]
            for grp in self._group_after.get(nid, []):
                _normalise_group(current, grp, set())
        return current

    # ---- presentation ----------------------------------------------------------
    def compute_nodes(self, observations: Optional[dict[str, float]] = None) -> list[dict]:
        """API-facing view: list of node dicts sorted by layer then id."""
        current = self.propagate(observations)
        locked = set(observations or {})
        out = []
        for nid in sorted(self.node_by_id, key=lambda i: (self.node_by_id[i].get("layer", 0), i)):
            n = self.node_by_id[nid]
            p = round(current[nid], 4)
            out.append({
                "id": nid,
                "label": n.get("label", nid),
                "layer": n.get("layer", 0),
                "prior": round(self.prior[nid], 4),
                "p": p,
                "delta": round(p - self.prior[nid], 4),
                "locked": nid in locked,
            })
        return out


def _normalise_group(current: dict[str, float], grp: dict, locked: set[str]) -> None:
    """Renormalise mutually-exclusive members onto the simplex (Σ ≤ 1)."""
    members = [m for m in grp["members"] if m in current]
    free = [m for m in members if m not in locked]
    locked_mass = sum(current[m] for m in members if m in locked)
    budget = max(0.0, 1.0 - locked_mass)
    s = sum(current[m] for m in free)
    if s <= _EPS:
        return
    if grp.get("allow_other", True):
        # only shrink if over budget; leftover budget is the implicit "other" mass
        if s > budget:
            for m in free:
                current[m] *= budget / s
    else:
        # force the free members to fill the whole budget exactly
        for m in free:
            current[m] *= budget / s


# ── validation & topology (module-level so they can be unit-tested directly) ───
def validate_graph(g: Graph) -> None:
    ids = set(g.node_by_id)
    if len(ids) != len(g.nodes):
        raise GraphError("duplicate node ids")
    for n in g.nodes:
        p = n.get("prior")
        if not isinstance(p, (int, float)) or not (0.0 <= p <= 1.0):
            raise GraphError(f"node {n.get('id')!r} prior must be in [0,1], got {p!r}")
    for e in g.edges:
        for end in ("source", "target"):
            if e.get(end) not in ids:
                raise GraphError(f"edge {end} {e.get(end)!r} is not a node")
        for k in ("pYes", "pNo"):
            v = e.get(k)
            if not isinstance(v, (int, float)) or not (0.0 <= v <= 1.0):
                raise GraphError(f"edge {e['source']}→{e['target']} {k} must be in [0,1], got {v!r}")
    for tgt, ps in g.parents.items():
        if len(ps) > 16:
            raise GraphError(f"node {tgt!r} fan-in {len(ps)} exceeds enumeration cap (16)")
    for grp in g.exclusive_groups:
        for m in grp.get("members", []):
            if m not in ids:
                raise GraphError(f"exclusive group {grp.get('id')!r} member {m!r} is not a node")
    topological_order(g)  # raises on cycle


def topological_order(g: Graph) -> list[str]:
    """Kahn's algorithm; raises GraphError on a cycle."""
    indeg = {nid: 0 for nid in g.node_by_id}
    children: dict[str, list[str]] = {nid: [] for nid in g.node_by_id}
    for tgt, ps in g.parents.items():
        for par in ps:
            indeg[tgt] += 1
            children[par["source"]].append(tgt)
    # stable order: by layer then id among ready nodes
    layer = lambda i: (g.node_by_id[i].get("layer", 0), i)
    ready = sorted([n for n in indeg if indeg[n] == 0], key=layer)
    order: list[str] = []
    while ready:
        nid = ready.pop(0)
        order.append(nid)
        for c in children[nid]:
            indeg[c] -= 1
            if indeg[c] == 0:
                ready.append(c)
        ready.sort(key=layer)
    if len(order) != len(indeg):
        cyc = [n for n in indeg if n not in set(order)]
        raise GraphError(f"graph has a cycle involving: {sorted(cyc)}")
    return order


def load_graph(path: str | Path) -> Graph:
    with open(path, "r", encoding="utf-8") as fh:
        return Graph(json.load(fh))
