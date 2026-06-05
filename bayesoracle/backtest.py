#!/usr/bin/env python3
"""
Backtest the PM-backed DAG against realised Polymarket prices.

The honest test (RETHINK.md §1.4): fit the graph's intercepts once at a start
date T0 (priors = each node's price at T0), then for every later date T drive the
*root* markets to their actual price at T, propagate, and predict each non-root
("child") market's price at T.  Score those predictions against the realised
child price and compare to two baselines:

  * persistence  — predict child price at T0 (the frozen start value)
  * random-walk  — predict child price at T-1 (yesterday)

If the DAG cannot beat random-walk, the edge structure is not yet adding value.
Metric: Brier score (lower is better), averaged over all (child, date) pairs.

Run:  python bayesoracle/backtest.py
"""
from __future__ import annotations

import json
from bisect import bisect_right
from pathlib import Path

import core

HERE = Path(__file__).parent
NODE_HISTORY = HERE / "node_history"


class Series:
    """As-of price lookup for one node (last price on or before a date)."""

    def __init__(self, node_id: str):
        prices = json.loads((NODE_HISTORY / f"{node_id}.json").read_text())["prices"]
        prices = sorted(prices, key=lambda p: p["date"])
        self.dates = [p["date"] for p in prices]
        self.vals = [min(0.99, max(0.01, float(p["probability"]))) for p in prices]

    def at(self, date: str) -> float | None:
        i = bisect_right(self.dates, date) - 1
        return self.vals[i] if i >= 0 else None

    def prev(self, date: str) -> float | None:
        i = bisect_right(self.dates, date) - 1
        return self.vals[i - 1] if i >= 1 else None


def brier(pred: float, actual: float) -> float:
    return (pred - actual) ** 2


def run(coverage: float = 0.9, min_gap_days: int = 14) -> dict:
    spec = json.loads((HERE / "graph_pm.json").read_text())
    series = {n["id"]: Series(n["id"]) for n in spec["nodes"]}

    # Global date axis = union of all observation dates.  Start the backtest at
    # the first date where >= `coverage` of nodes already have a price, so the
    # persistence baseline is defined for (almost) every node.
    all_dates = sorted({d for s in series.values() for d in s.dates})
    need = coverage * len(series)
    t0 = next((d for d in all_dates
               if sum(s.at(d) is not None for s in series.values()) >= need),
              all_dates[len(all_dates) // 2])

    # Build the graph with priors = price at T0 (intercepts fit to that state).
    spec_t0 = json.loads(json.dumps(spec))
    for n in spec_t0["nodes"]:
        v = series[n["id"]].at(t0)
        if v is not None:
            n["prior"] = round(v, 4)
    g = core.Graph(spec_t0)

    roots = [nid for nid in g.node_by_id if nid not in g.parents]
    children = [nid for nid in g.node_by_id if nid in g.parents]

    eval_dates = [d for d in all_dates if d > t0]
    eval_dates = eval_dates[min_gap_days::min_gap_days]  # thin to ~biweekly

    tot = {"model": 0.0, "persist": 0.0, "rw": 0.0}
    n_pairs = 0
    per_node: dict[str, dict] = {c: {"model": 0.0, "persist": 0.0, "n": 0} for c in children}

    for T in eval_dates:
        obs = {r: series[r].at(T) for r in roots if series[r].at(T) is not None}
        pred = g.propagate(obs)
        for c in children:
            actual = series[c].at(T)
            if actual is None:
                continue
            persist = series[c].at(t0)
            rw = series[c].prev(T)
            if persist is None or rw is None:
                continue
            tot["model"] += brier(pred[c], actual)
            tot["persist"] += brier(persist, actual)
            tot["rw"] += brier(rw, actual)
            per_node[c]["model"] += brier(pred[c], actual)
            per_node[c]["persist"] += brier(persist, actual)
            per_node[c]["n"] += 1
            n_pairs += 1

    return {
        "t0": t0, "end": all_dates[-1], "n_eval_dates": len(eval_dates),
        "n_pairs": n_pairs, "roots": roots, "children": children,
        "brier": {k: round(v / n_pairs, 4) for k, v in tot.items()} if n_pairs else {},
        "per_node": {
            c: {"model": round(d["model"] / d["n"], 4),
                "persist": round(d["persist"] / d["n"], 4), "n": d["n"]}
            for c, d in per_node.items() if d["n"]
        },
    }


def main() -> None:
    r = run()
    print(f"Backtest  T0={r['t0']}  →  {r['end']}   "
          f"({r['n_eval_dates']} dates, {r['n_pairs']} child·date pairs)")
    print(f"  roots ({len(r['roots'])}): {', '.join(r['roots'])}")
    print()
    b = r["brier"]
    print("  Aggregate Brier (lower = better):")
    print(f"    DAG model    : {b['model']}")
    print(f"    persistence  : {b['persist']}   (freeze child at T0 — the bar to beat)")
    print(f"    random-walk  : {b['rw']}   (yesterday's price — reference only)")
    skill = (b["persist"] - b["model"]) / b["persist"] if b.get("persist") else 0
    print(f"    skill vs persistence : {skill:+.1%}")
    print()
    print("  Per-child (model vs persistence Brier):")
    for c, d in sorted(r["per_node"].items(), key=lambda kv: kv[1]["model"] - kv[1]["persist"]):
        flag = "  ✓ model better" if d["model"] < d["persist"] else "  ✗ worse"
        print(f"    {c:14} model {d['model']:.4f}  persist {d['persist']:.4f}  (n={d['n']}){flag}")


if __name__ == "__main__":
    main()
