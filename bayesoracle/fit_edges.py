#!/usr/bin/env python3
"""
Can fitting the edge weights to price history beat the LLM-derived weights
out-of-sample?  This script answers that *before* any engine/viewer plumbing.

Setup (teacher-forced, per-node CPT — the engine's enumeration model):

    pred_C(t) = E_{s ~ Π Bernoulli(p_i(t))} [ σ(b_C + Σ_i w_i s_i) ]

where p_i(t) are the *actual* historical parent prices.  We compare three
weight settings, each evaluated out-of-sample on a held-out time window:

  * baseline  — w frozen at the LLM/correlation values, intercept b fit on train
  * global-α  — one shared scalar α (w_i -> α·w_i) + per-node b  (smallest model)
  * per-node  — full {w_i, b} per child, ridge-shrunk toward the LLM weights

Metrics on the test window:
  * level Brier vs persistence (child frozen at split) and random-walk (t-1)
  * first-difference MSE (predict ΔP_child) vs a zero-change baseline — this is
    what separates learned conditional structure from mere co-trending.

Honest expectation (see RETHINK.md / backtest.py): daily PM series barely move,
so the exploitable signal is thin; a clean null is a valid, reportable outcome.

Run:  api/.venv/bin/python bayesoracle/fit_edges.py
"""
from __future__ import annotations

import json
from bisect import bisect_right
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).parent
NODE_HISTORY = HERE / "node_history"
SECONDARY_W = 0.5
import os
RIDGE = float(os.environ.get("RIDGE", "2.0"))   # shrink per-node w toward LLM w
CLIP = 1e-4


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def logit(p):
    p = np.clip(p, 0.001, 0.999)
    return np.log(p / (1 - p))


class Series:
    def __init__(self, node_id):
        prices = json.loads((NODE_HISTORY / f"{node_id}.json").read_text())["prices"]
        prices = sorted(prices, key=lambda p: p["date"])
        self.dates = [p["date"] for p in prices]
        self.vals = np.array([min(0.99, max(0.01, float(p["probability"]))) for p in prices])

    def at(self, date):
        i = bisect_right(self.dates, date) - 1
        return float(self.vals[i]) if i >= 0 else None


def masks(k):
    """All 2^k binary parent-state rows as a (2^k, k) array."""
    return np.array([[(m >> i) & 1 for i in range(k)] for m in range(1 << k)], dtype=float)


def predict(P, w, b):
    """P: (T,k) parent probs; w: (k,); b: scalar -> (T,) E[sigmoid] over parent states."""
    T, k = P.shape
    S = masks(k)                                  # (M,k)
    sig = sigmoid(b + S @ w)                       # (M,)
    # combo prob: Π p_i^{s} (1-p_i)^{1-s}  -> (T,M)
    # log for stability
    lp = P[:, None, :] * S[None, :, :] + (1 - P[:, None, :]) * (1 - S[None, :, :])
    combo = np.prod(lp, axis=2)                     # (T,M)
    return combo @ sig                             # (T,)


def logloss(pred, actual):
    pred = np.clip(pred, CLIP, 1 - CLIP)
    return -np.mean(actual * np.log(pred) + (1 - actual) * np.log(1 - pred))


def brier(pred, actual):
    return float(np.mean((pred - actual) ** 2))


def main():
    spec = json.loads((HERE / "graph_pm.json").read_text())
    series = {n["id"]: Series(n["id"]) for n in spec["nodes"]}
    parents = {}
    for e in spec["edges"]:
        w = logit(e["pYes"]) - logit(e["pNo"])
        if e.get("type") == "secondary":
            w *= SECONDARY_W
        parents.setdefault(e["target"], []).append((e["source"], float(w)))

    all_dates = sorted({d for s in series.values() for d in s.dates})
    need = 0.9 * len(series)
    hi = [d for d in all_dates if sum(s.at(d) is not None for s in series.values()) >= need]
    split = hi[int(len(hi) * 0.6)]
    train_d = [d for d in hi if d <= split]
    test_d = [d for d in hi if d > split]

    # Per-child design matrices (teacher-forced on actual parent prices).
    children = [c for c in parents]
    data = {}
    for c in children:
        ps = parents[c]
        w_llm = np.array([w for _, w in ps])

        def rows(dates):
            X, y = [], []
            for d in dates:
                yc = series[c].at(d)
                pr = [series[s].at(d) for s, _ in ps]
                if yc is None or any(v is None for v in pr):
                    continue
                X.append(pr); y.append(yc)
            return np.array(X), np.array(y)

        Xtr, ytr = rows(train_d)
        Xte, yte = rows(test_d)
        if len(ytr) < 20 or len(yte) < 5:
            continue
        data[c] = dict(w_llm=w_llm, Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte)

    # ---- baseline: freeze w=w_llm, fit b per child on train -------------------
    def fit_b(d):
        res = minimize(lambda b: logloss(predict(d["Xtr"], d["w_llm"], b[0]), d["ytr"]),
                       x0=[logit(d["ytr"].mean())], method="Nelder-Mead")
        return res.x[0]

    for c, d in data.items():
        d["b_base"] = fit_b(d)

    # ---- global-α: one shared α (w=α·w_llm) + per-child b, joint train fit -----
    cs = list(data)

    def global_obj(theta):
        a = theta[0]
        tot = 0.0
        for i, c in enumerate(cs):
            d = data[c]
            tot += logloss(predict(d["Xtr"], a * d["w_llm"], theta[1 + i]), d["ytr"])
        return tot
    x0 = [1.0] + [data[c]["b_base"] for c in cs]
    g = minimize(global_obj, x0, method="L-BFGS-B")
    alpha = g.x[0]
    for i, c in enumerate(cs):
        data[c]["b_glob"] = g.x[1 + i]

    # ---- per-node: full {w,b} per child, ridge toward w_llm -------------------
    for c, d in data.items():
        k = len(d["w_llm"])

        def obj(th):
            w, b = th[:k], th[k]
            return logloss(predict(d["Xtr"], w, b), d["ytr"]) + RIDGE * np.mean((w - d["w_llm"]) ** 2)
        r = minimize(obj, np.r_[d["w_llm"], d["b_base"]], method="L-BFGS-B")
        d["w_node"], d["b_node"] = r.x[:k], r.x[k]

    # ---- evaluate OOS ---------------------------------------------------------
    def agg_level(key_w, key_b):
        sq, n = 0.0, 0
        per = {}
        for c, d in data.items():
            w = d["w_llm"] * (alpha if key_w == "glob" else 1.0) if key_w in ("base", "glob") else d["w_node"]
            b = d[{"base": "b_base", "glob": "b_glob", "node": "b_node"}[key_w]]
            pred = predict(d["Xte"], w, b)
            e = np.sum((pred - d["yte"]) ** 2)
            sq += e; n += len(d["yte"]); per[c] = brier(pred, d["yte"])
        return sq / n, per

    def baseline_persist():
        sq, n = 0.0, 0
        for c, d in data.items():
            frozen = d["ytr"][-1]
            sq += np.sum((frozen - d["yte"]) ** 2); n += len(d["yte"])
        return sq / n

    def first_diff(key_w):
        """MSE of predicted ΔP_child vs actual, test window; vs zero-change."""
        sqm, sqz, n = 0.0, 0.0, 0
        for c, d in data.items():
            if len(d["yte"]) < 2:
                continue
            w = d["w_llm"] * (alpha if key_w == "glob" else 1.0) if key_w != "node" else d["w_node"]
            b = d[{"base": "b_base", "glob": "b_glob", "node": "b_node"}[key_w]]
            pred = predict(d["Xte"], w, b)
            dp_m = np.diff(pred); dp_a = np.diff(d["yte"])
            sqm += np.sum((dp_m - dp_a) ** 2); sqz += np.sum(dp_a ** 2); n += len(dp_a)
        return sqm / n, sqz / n

    lvl_base, per_base = agg_level("base", "b_base")
    lvl_glob, _ = agg_level("glob", "b_glob")
    lvl_node, per_node = agg_level("node", "b_node")
    persist = baseline_persist()
    fd_base = first_diff("base")
    fd_glob = first_diff("glob")
    fd_node = first_diff("node")

    print(f"Fit edge weights to history — train ≤ {split}  ({len(train_d)}d), "
          f"test > {split} ({len(test_d)}d), {len(data)} children\n")
    print("OUT-OF-SAMPLE level Brier (lower better):")
    print(f"  persistence (freeze at split) : {persist:.5f}")
    print(f"  LLM weights   (fit b)         : {lvl_base:.5f}")
    print(f"  global-α      (α={alpha:.3f})        : {lvl_glob:.5f}")
    print(f"  per-node fit  (ridge {RIDGE})       : {lvl_node:.5f}")
    print(f"  → per-node skill vs LLM weights : {(lvl_base-lvl_node)/lvl_base:+.1%}")
    print(f"  → per-node skill vs persistence : {(persist-lvl_node)/persist:+.1%}\n")
    print("OUT-OF-SAMPLE first-difference MSE (predict ΔP_child; lower better):")
    print(f"  zero-change baseline          : {fd_base[1]:.6f}")
    print(f"  LLM weights                   : {fd_base[0]:.6f}")
    print(f"  global-α                      : {fd_glob[0]:.6f}")
    print(f"  per-node fit                  : {fd_node[0]:.6f}")
    better = "YES" if fd_node[0] < fd_base[1] else "NO"
    print(f"  → does any fit beat zero-change on ΔP? {better}\n")
    print("Per-child level Brier (LLM vs per-node):")
    for c in sorted(per_base, key=lambda x: per_node[x] - per_base[x]):
        flag = "✓" if per_node[c] < per_base[c] else "✗"
        print(f"  {c:14} LLM {per_base[c]:.5f}  node {per_node[c]:.5f}  {flag}")


if __name__ == "__main__":
    main()
