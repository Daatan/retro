#!/usr/bin/env python3
"""
Compute empirical edge probabilities from price history, then blend with LLM estimates.

For each edge (A→B):
  - Load aligned daily price history for A and B
  - Bin approach (preferred): split days where parent > HI_THRESH vs < LO_THRESH
      pY_bin = mean(child | parent > 0.60)  — requires ≥ MIN_BIN days in bin
      pN_bin = mean(child | parent < 0.40)
      w_emp  = n_bin / 10  (more data → higher weight)
      SE     = std(child_prices_in_bin) / sqrt(n_bin) × 1.5  (×1.5 for autocorrelation)
  - Regression fallback (when bins are sparse):
      pY_corr = clip(α+β, 0.01, 0.99)  [predicted P_B when P_A=1]
      pN_corr = clip(α,   0.01, 0.99)  [predicted P_B when P_A=0]
      w_emp   = R² × 3.0

Blend:
  w_llm  = 1.0 (baseline)
  pY_blend = (w_llm·pY_llm + w_emp_Y·pY_emp) / (w_llm + w_emp_Y)
  Note: Y and N can have different weights because bin counts can differ.

90% CI on blended estimate:
  margin = 1.645 × SE × 1.5   (when data-backed)
  margin = 0.15                (pure LLM, no data)

Writes corr + blend + CI fields back to bayesoracle/edge_weights.json.

Usage:
    cd /home/mark/projects/retro
    source pipeline/.venv/bin/activate
    python bayesoracle/compute_edge_probs.py
"""

import json
from pathlib import Path

import numpy as np
from scipy import stats

HISTORY_DIR  = Path(__file__).parent / "node_history"
WEIGHTS_FILE = Path(__file__).parent / "edge_weights.json"

HI_THRESH = 0.60  # parent "clearly YES" threshold
LO_THRESH = 0.40  # parent "clearly NO" threshold
MIN_BIN   = 8     # minimum days in a bin to trust it


# ─── Price history ────────────────────────────────────────────────────────────

def _load_prices(node_id: str) -> dict[str, float]:
    path = HISTORY_DIR / f"{node_id}.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return {pt["date"]: pt["probability"] for pt in data["prices"]}


def _align(pa: dict[str, float], pb: dict[str, float]) -> tuple[list[float], list[float]]:
    dates = sorted(pa.keys() & pb.keys())
    return [pa[d] for d in dates], [pb[d] for d in dates]


# ─── Correlation estimate ─────────────────────────────────────────────────────

def corr_estimate(src: str, tgt: str) -> dict:
    pa = _load_prices(src)
    pb = _load_prices(tgt)
    xa, xb = _align(pa, pb)
    n = len(xa)

    if n < 20:
        return {"pY_corr": None, "pN_corr": None, "r2_corr": 0.0, "n_corr": n,
                "pY_bin": None, "pN_bin": None, "n_hi": 0, "n_lo": 0,
                "std_hi": None, "std_lo": None}

    xa_arr = np.array(xa)
    xb_arr = np.array(xb)
    slope, intercept, r, _, _ = stats.linregress(xa_arr, xb_arr)
    r2 = r ** 2

    pY_reg = float(np.clip(intercept + slope, 0.01, 0.99))
    pN_reg = float(np.clip(intercept,         0.01, 0.99))

    hi_mask = xa_arr > HI_THRESH
    lo_mask = xa_arr < LO_THRESH
    n_hi = int(hi_mask.sum())
    n_lo = int(lo_mask.sum())

    pY_bin = float(xb_arr[hi_mask].mean()) if n_hi >= MIN_BIN else None
    pN_bin = float(xb_arr[lo_mask].mean()) if n_lo >= MIN_BIN else None
    std_hi = float(xb_arr[hi_mask].std())  if n_hi >= MIN_BIN else None
    std_lo = float(xb_arr[lo_mask].std())  if n_lo >= MIN_BIN else None

    return {
        "pY_corr": round(pY_reg, 3),
        "pN_corr": round(pN_reg, 3),
        "r2_corr": round(r2, 4),
        "n_corr":  n,
        "pY_bin":  round(pY_bin, 3) if pY_bin is not None else None,
        "pN_bin":  round(pN_bin, 3) if pN_bin is not None else None,
        "n_hi":    n_hi,
        "n_lo":    n_lo,
        "std_hi":  round(std_hi, 3) if std_hi is not None else None,
        "std_lo":  round(std_lo, 3) if std_lo is not None else None,
    }


# ─── CI helper ───────────────────────────────────────────────────────────────

def _ci90_data(p: float, std: float, n: int) -> tuple[float, float]:
    """90% CI from empirical SE, inflated ×1.5 for daily autocorrelation."""
    se = std / np.sqrt(n)
    margin = 1.645 * se * 1.5
    return (round(float(np.clip(p - margin, 0.01, 0.99)), 3),
            round(float(np.clip(p + margin, 0.01, 0.99)), 3))


def _ci90_llm(p: float) -> tuple[float, float]:
    """Fixed ±15pp uncertainty for pure LLM estimates."""
    return (round(float(np.clip(p - 0.15, 0.01, 0.99)), 3),
            round(float(np.clip(p + 0.15, 0.01, 0.99)), 3))


# ─── Blend ───────────────────────────────────────────────────────────────────

def blend(pY_llm: float, pN_llm: float, corr: dict) -> dict:
    w_llm = 1.0
    r2    = corr.get("r2_corr") or 0.0
    n_hi  = corr.get("n_hi") or 0
    n_lo  = corr.get("n_lo") or 0

    # ── pY ──
    if corr.get("pY_bin") is not None:
        pY_emp, w_emp_Y = corr["pY_bin"], n_hi / 10.0
        pY_ci = _ci90_data(
            (w_llm * pY_llm + w_emp_Y * pY_emp) / (w_llm + w_emp_Y),
            corr["std_hi"], n_hi,
        )
    elif corr.get("pY_corr") is not None and r2 > 0:
        pY_emp, w_emp_Y = corr["pY_corr"], r2 * 3.0
        pY_ci = None  # regression CI not reliable; fall back below
    else:
        pY_emp, w_emp_Y = None, 0.0
        pY_ci = None

    pY_b = float(np.clip(
        (w_llm * pY_llm + w_emp_Y * pY_emp) / (w_llm + w_emp_Y)
        if pY_emp is not None else pY_llm,
        0.01, 0.99,
    ))
    if pY_ci is None:
        pY_ci = _ci90_llm(pY_b)

    # ── pN ──
    if corr.get("pN_bin") is not None:
        pN_emp, w_emp_N = corr["pN_bin"], n_lo / 10.0
        pN_ci = _ci90_data(
            (w_llm * pN_llm + w_emp_N * pN_emp) / (w_llm + w_emp_N),
            corr["std_lo"], n_lo,
        )
    elif corr.get("pN_corr") is not None and r2 > 0:
        pN_emp, w_emp_N = corr["pN_corr"], r2 * 3.0
        pN_ci = None
    else:
        pN_emp, w_emp_N = None, 0.0
        pN_ci = None

    pN_b = float(np.clip(
        (w_llm * pN_llm + w_emp_N * pN_emp) / (w_llm + w_emp_N)
        if pN_emp is not None else pN_llm,
        0.01, 0.99,
    ))
    if pN_ci is None:
        pN_ci = _ci90_llm(pN_b)

    return {
        "pY_blend": round(pY_b, 3),
        "pN_blend": round(pN_b, 3),
        "pY_ci":    list(pY_ci),
        "pN_ci":    list(pN_ci),
        "w_llm":    round(w_llm,  3),
        "w_emp_Y":  round(w_emp_Y, 3),
        "w_emp_N":  round(w_emp_N, 3),
    }


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    edges = json.loads(WEIGHTS_FILE.read_text())
    results = []

    DROP = {"pY_corr","pN_corr","r2_corr","n_corr","pY_bin","pN_bin","n_hi","n_lo",
            "std_hi","std_lo","pY_resolved","pN_resolved","n_pairs","n_src","n_tgt",
            "pY_blend","pN_blend","pY_ci","pN_ci","w_llm","w_corr","w_emp_Y","w_emp_N","w_res"}

    for edge in edges:
        src, tgt = edge["source"], edge["target"]
        print(f"{src} → {tgt}")

        corr = corr_estimate(src, tgt)
        bl   = blend(edge["pY"], edge["pN"], corr)

        src_Y = "bin" if corr.get("pY_bin") is not None else "reg" if corr.get("pY_corr") else "llm"
        src_N = "bin" if corr.get("pN_bin") is not None else "reg" if corr.get("pN_corr") else "llm"
        print(f"  pY: {bl['pY_blend']} [{bl['pY_ci'][0]}–{bl['pY_ci'][1]}]  src={src_Y}  w_emp={bl['w_emp_Y']:.2f}")
        print(f"  pN: {bl['pN_blend']} [{bl['pN_ci'][0]}–{bl['pN_ci'][1]}]  src={src_N}  w_emp={bl['w_emp_N']:.2f}")
        print(f"  R²={corr['r2_corr']:.3f}  n={corr['n_corr']}  n_hi={corr['n_hi']}  n_lo={corr['n_lo']}")
        print()

        clean = {k: v for k, v in edge.items() if k not in DROP}
        results.append({**clean, **corr, **bl})

    WEIGHTS_FILE.write_text(json.dumps(results, indent=2))
    print(f"Saved {len(results)} edges to {WEIGHTS_FILE}")

    bin_edges = [r for r in results if r.get("pY_bin") is not None or r.get("pN_bin") is not None]
    print(f"\nEdges with at least one bin estimate: {len(bin_edges)}/28")
    both_bins = [r for r in results if r.get("pY_bin") is not None and r.get("pN_bin") is not None]
    print(f"Edges with both bins: {len(both_bins)}/28")


if __name__ == "__main__":
    main()
