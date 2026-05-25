#!/usr/bin/env python3
"""
Patch bayesoracle/pm_analysis/index.html with fresh data:
  1. Current PM prices  (from gamma-api.polymarket.com)
  2. pY/pN blend values (from edge_weights.json)
  3. Full EDGE_META block replacement
  4. Date strings in poll-status and staleness warning
Also rebuilds bayesoracle/node_history/all.json.

Run after fetch_node_history.py + compute_edge_probs.py.

Usage:
    cd /home/mark/projects/retro
    source pipeline/.venv/bin/activate
    python bayesoracle/apply_html_data.py
"""

import json
import re
import time
from datetime import date
from pathlib import Path

import httpx

WEIGHTS_FILE = Path(__file__).parent / "edge_weights.json"
HTML_FILE    = Path(__file__).parent / "pm_analysis" / "index.html"
HISTORY_DIR  = Path(__file__).parent / "node_history"
ALL_JSON     = HISTORY_DIR / "all.json"
GAMMA_URL    = "https://gamma-api.polymarket.com/markets/{pmId}"

# Primary edge: child → parent
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

PM_IDS = {
    "IRAN_DEAL": "665325", "CEASEFIRE_X": "1090489", "TRUMP_IL": "665260",
    "US_WAR_IRAN": "665374", "IRAN_REGIME": "663583", "IRAN_NUKE": "665521",
    "ELECTIONS": "1280208", "BIBI_OUT": "567688", "BIBI_PM": "682705",
    "BENNETT_PM": "682706", "EIZENKOT_PM": "682708", "LIEBERMAN_PM": "682710",
    "LAPID_PM": "682707", "ANNEX_WB": "665515", "ANNEX_GAZA": "636921",
    "US_GAZA": "665485", "TURKEY_CLASH": "665484", "SAUDI_NORM": "665218",
    "SAUDI_ACCORDS": "665474", "LEBANON_NORM": "665414", "SYRIA_NORM": "677273",
    "STRIKE_5": "678748", "PAHLAVI": "1115288", "US_DECLARE": "1170144",
}


def fetch_pm_prices(client: httpx.Client) -> dict[str, float]:
    prices = {}
    for node_id, pm_id in PM_IDS.items():
        try:
            r = client.get(GAMMA_URL.format(pmId=pm_id), timeout=10)
            r.raise_for_status()
            raw = r.json().get("outcomePrices", "[]")
            pts = json.loads(raw) if isinstance(raw, str) else raw
            if pts:
                prices[node_id] = round(float(pts[0]), 4)
            time.sleep(0.2)
        except Exception as e:
            print(f"  ✗ {node_id}: {e}")
    return prices


def edge_meta_block(edges: list[dict]) -> str:
    lines = ["  const EDGE_META = {"]
    for e in edges:
        src, tgt = e["source"], e["target"]
        pY_c = e.get("pY_bin") if e.get("pY_bin") is not None else e.get("pY_corr")
        pN_c = e.get("pN_bin") if e.get("pN_bin") is not None else e.get("pN_corr")
        src_Y = "bin" if e.get("pY_bin") is not None else "reg"
        src_N = "bin" if e.get("pN_bin") is not None else "reg"
        lines.append(
            f"    '{src}→{tgt}': "
            f"{{pY_llm:{e['pY']}, pN_llm:{e['pN']}, "
            f"pY_c:{pY_c}, pN_c:{pN_c}, "
            f"r2:{e['r2_corr']}, n:{e['n_corr']}, "
            f"wy:{e['w_emp_Y']}, wn:{e['w_emp_N']}, "
            f"src_Y:'{src_Y}', src_N:'{src_N}', "
            f"pY_ci:{e['pY_ci']}, pN_ci:{e['pN_ci']}}},"
        )
    lines.append("  };")
    return "\n".join(lines)


def rebuild_all_json() -> None:
    all_data = {}
    for f in sorted(HISTORY_DIR.glob("*.json")):
        if f.name == "all.json":
            continue
        data = json.loads(f.read_text())
        all_data[f.stem] = {pt["date"]: pt["probability"] for pt in data["prices"]}
    ALL_JSON.write_text(json.dumps(all_data, separators=(",", ":")))
    print(f"  Rebuilt all.json: {ALL_JSON.stat().st_size // 1024}KB, {len(all_data)} nodes")


def main() -> None:
    edges    = json.loads(WEIGHTS_FILE.read_text())
    by_pair  = {(e["source"], e["target"]): e for e in edges}
    html     = HTML_FILE.read_text()
    changes  = 0

    # ── 1. Fetch live PM prices ───────────────────────────────────────────────
    print("Fetching PM prices…")
    client = httpx.Client(headers={"User-Agent": "BayesOracle/1.0"})
    pm_prices = fetch_pm_prices(client)
    client.close()
    print(f"  Got {len(pm_prices)}/{len(PM_IDS)} prices")

    # ── 2. Patch pm:X.XXX for each node ──────────────────────────────────────
    for node_id, p in pm_prices.items():
        pm_id = PM_IDS[node_id]
        new_html, n = re.subn(
            r"(pmId:'" + pm_id + r"',\s+pm:)[\d.]+",
            lambda m, p=p: m.group(1) + str(p),
            html,
        )
        html = new_html
        changes += n

    # ── 3. Patch pY/pN in NODES (primary edges) ───────────────────────────────
    for tgt, src in PRIMARY.items():
        e = by_pair.get((src, tgt))
        if not e:
            continue
        pY, pN = e["pY_blend"], e["pN_blend"]
        new_html, n = re.subn(
            r"(id:'" + re.escape(tgt) + r"'[^}]*?pY:)[\d.]+([,\s]+pN:)[\d.]+",
            lambda m, pY=pY, pN=pN: m.group(1) + str(pY) + m.group(2) + str(pN),
            html, flags=re.DOTALL,
        )
        html = new_html
        changes += n

    # ── 4. Patch pY/pN in EXTRA_EDGES (secondary edges) ──────────────────────
    for src, tgt in SECONDARY:
        e = by_pair.get((src, tgt))
        if not e:
            continue
        pY, pN = e["pY_blend"], e["pN_blend"]
        new_html, n = re.subn(
            r"(source:'" + re.escape(src) + r"',\s+target:'" + re.escape(tgt) + r"',\s+pY:)[\d.]+(\s*,\s*pN:)[\d.]+",
            lambda m, pY=pY, pN=pN: m.group(1) + str(pY) + m.group(2) + str(pN),
            html,
        )
        html = new_html
        changes += n

    # ── 5. Replace EDGE_META block ────────────────────────────────────────────
    new_meta = edge_meta_block(edges)
    html, n = re.subn(
        r"  const EDGE_META = \{.*?\};",
        new_meta,
        html, flags=re.DOTALL,
    )
    changes += n

    # ── 6. Update date strings ────────────────────────────────────────────────
    d = date.today()
    today_full  = f"{d.strftime('%B')} {d.day}, {d.year}"   # e.g. "May 25, 2026"
    today_mon_y = f"{d.strftime('%B')} {d.year}"             # e.g. "May 2026"
    html, n1 = re.subn(r"(Prices baked in from )[^<\"]+", r"\g<1>" + today_full, html)
    html, n2 = re.subn(r"(⚠ Conditionals: LLM-estimated, )[A-Za-z]+ \d{4}",
                       r"\g<1>" + today_mon_y, html)
    changes += n1 + n2

    HTML_FILE.write_text(html)
    print(f"  Patched index.html ({changes} substitutions)")

    # ── 7. Rebuild all.json ───────────────────────────────────────────────────
    rebuild_all_json()


if __name__ == "__main__":
    main()
