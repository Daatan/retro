#!/usr/bin/env python3
"""
Read edge_weights.json and print updated JS snippets for pm_analysis/index.html:
  1. NODES pY/pN values (primary edges only)
  2. EXTRA_EDGES pY/pN values (secondary edges only)
  3. Full EDGE_META JS object

Usage:
    cd /home/mark/projects/retro
    python bayesoracle/gen_html_data.py
"""
import json
from pathlib import Path

WEIGHTS_FILE = Path(__file__).parent / "edge_weights.json"

# Primary edge: target node's `primary` field = source
PRIMARY = {
    "ELECTIONS":    "CEASEFIRE_X",
    "BIBI_OUT":     "ELECTIONS",
    "BIBI_PM":      "BIBI_OUT",
    "BENNETT_PM":   "BIBI_OUT",
    "EIZENKOT_PM":  "BIBI_OUT",
    "LIEBERMAN_PM": "BIBI_OUT",
    "LAPID_PM":     "BIBI_OUT",
    "ANNEX_WB":     "BIBI_PM",
    "ANNEX_GAZA":   "BIBI_PM",
    "US_GAZA":      "BIBI_PM",
    "TURKEY_CLASH": "BIBI_PM",
    "SAUDI_NORM":   "BIBI_PM",
    "SAUDI_ACCORDS":"BIBI_PM",
    "LEBANON_NORM": "BIBI_PM",
    "SYRIA_NORM":   "BIBI_PM",
    "STRIKE_5":     "IRAN_NUKE",
    "PAHLAVI":      "IRAN_REGIME",
    "US_DECLARE":   "US_WAR_IRAN",
}

# Invert to (src, tgt) → "primary" for lookup
PRIMARY_PAIRS = {(v, k) for k, v in PRIMARY.items()}

edges = json.loads(WEIGHTS_FILE.read_text())
by_pair = {(e["source"], e["target"]): e for e in edges}


def pct(v):
    return f"{round(v*100)}%"


# ── 1. NODES pY/pN ────────────────────────────────────────────────────────────
print("=" * 60)
print("NODES pY/pN  (update each matching node in NODES array)")
print("=" * 60)
for tgt, src in sorted(PRIMARY.items()):
    e = by_pair.get((src, tgt))
    if e:
        print(f"  {tgt:<15}  pY:{e['pY_blend']}  pN:{e['pN_blend']}")
    else:
        print(f"  {tgt:<15}  *** edge not found ***")

# ── 2. EXTRA_EDGES pY/pN ─────────────────────────────────────────────────────
SECONDARY = [
    ("IRAN_DEAL",   "ELECTIONS"),
    ("TRUMP_IL",    "ELECTIONS"),
    ("CEASEFIRE_X", "BIBI_OUT"),
    ("TRUMP_IL",    "BIBI_OUT"),
    ("TRUMP_IL",    "BIBI_PM"),
    ("TRUMP_IL",    "STRIKE_5"),
    ("IRAN_DEAL",   "STRIKE_5"),
    ("BIBI_PM",     "STRIKE_5"),
    ("US_WAR_IRAN", "PAHLAVI"),
    ("TRUMP_IL",    "US_DECLARE"),
]

print()
print("=" * 60)
print("EXTRA_EDGES pY/pN  (update each matching entry)")
print("=" * 60)
for src, tgt in SECONDARY:
    e = by_pair.get((src, tgt))
    if e:
        print(f"  {src}→{tgt:<15}  pY:{e['pY_blend']}  pN:{e['pN_blend']}")
    else:
        print(f"  {src}→{tgt:<15}  *** not found ***")

# ── 3. EDGE_META JS object ───────────────────────────────────────────────────
print()
print("=" * 60)
print("EDGE_META  (full replacement block)")
print("=" * 60)
print("  const EDGE_META = {")
for e in edges:
    src, tgt = e["source"], e["target"]
    pY_c = e.get("pY_bin") if e.get("pY_bin") is not None else e.get("pY_corr")
    pN_c = e.get("pN_bin") if e.get("pN_bin") is not None else e.get("pN_corr")
    r2   = e.get("r2_corr", 0)
    n    = e.get("n_corr", 0)
    wy   = e.get("w_emp_Y", 0)
    wn   = e.get("w_emp_N", 0)
    yci  = e.get("pY_ci", [None, None])
    nci  = e.get("pN_ci", [None, None])
    src_Y = "bin" if e.get("pY_bin") is not None else "reg"
    src_N = "bin" if e.get("pN_bin") is not None else "reg"
    print(f"    '{src}→{tgt}': "
          f"{{pY_llm:{e['pY']}, pN_llm:{e['pN']}, "
          f"pY_c:{pY_c}, pN_c:{pN_c}, "
          f"r2:{r2}, n:{n}, "
          f"wy:{wy}, wn:{wn}, "
          f"src_Y:'{src_Y}', src_N:'{src_N}', "
          f"pY_ci:{yci}, pN_ci:{nci}}},")
print("  };")
