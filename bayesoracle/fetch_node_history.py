#!/usr/bin/env python3
"""
Fetch daily CLOB price history for all 23 DAG nodes.

For each node:
  1. Hit Gamma API by pmId → get clobTokenIds[0] (YES token)
  2. Fetch CLOB price history (daily, max interval)
  3. Save to bayesoracle/node_history/{NODE_ID}.json

Resumable — skips nodes that already have a history file.

Usage:
    cd /home/mark/projects/retro
    source pipeline/.venv/bin/activate
    python bayesoracle/fetch_node_history.py
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

GAMMA_URL = "https://gamma-api.polymarket.com/markets/{pmId}"
CLOB_URL  = "https://clob.polymarket.com/prices-history"
OUT_DIR   = Path(__file__).parent / "node_history"
OUT_DIR.mkdir(exist_ok=True)

# pmId for each DAG node (matches calibrate_edges.py NODES + pm_analysis/index.html)
NODES: dict[str, str] = {
    "IRAN_DEAL":     "665325",
    "CEASEFIRE_X":   "1090489",
    "TRUMP_IL":      "665260",
    "US_WAR_IRAN":   "665374",
    "IRAN_REGIME":   "663583",
    "IRAN_NUKE":     "665521",
    "ELECTIONS":     "1280208",
    "BIBI_OUT":      "567688",
    "BIBI_PM":       "682705",
    "BENNETT_PM":    "682706",
    "EIZENKOT_PM":   "682708",
    "LIEBERMAN_PM":  "682710",
    "LAPID_PM":      "682707",
    "ANNEX_WB":      "665515",
    "ANNEX_GAZA":    "636921",
    "US_GAZA":       "665485",
    "TURKEY_CLASH":  "665484",
    "SAUDI_NORM":    "665218",
    "SAUDI_ACCORDS": "665474",
    "LEBANON_NORM":  "665414",
    "SYRIA_NORM":    "677273",
    "STRIKE_5":      "678748",
    "PAHLAVI":       "1115288",
    "US_DECLARE":    "1170144",
}


def _get_clob_token(client: httpx.Client, pm_id: str) -> str | None:
    r = client.get(GAMMA_URL.format(pmId=pm_id), timeout=10)
    r.raise_for_status()
    raw = r.json().get("clobTokenIds", "[]")
    tokens = json.loads(raw) if isinstance(raw, str) else raw
    return tokens[0] if tokens else None


def _fetch_history(client: httpx.Client, token: str) -> list[dict]:
    r = client.get(CLOB_URL, params={"market": token, "interval": "max", "fidelity": 1440}, timeout=15)
    r.raise_for_status()
    raw = r.json().get("history", [])
    seen, daily = set(), []
    for pt in sorted(raw, key=lambda x: x["t"]):
        date = datetime.fromtimestamp(pt["t"], tz=timezone.utc).strftime("%Y-%m-%d")
        if date not in seen:
            seen.add(date)
            daily.append({"date": date, "probability": round(float(pt["p"]), 4)})
    return daily


def main() -> None:
    client = httpx.Client(headers={"User-Agent": "BayesOracle/1.0"})
    total = len(NODES)

    for i, (node_id, pm_id) in enumerate(NODES.items(), 1):
        out_path = OUT_DIR / f"{node_id}.json"
        print(f"[{i}/{total}] {node_id} (pmId={pm_id})", end=" ", flush=True)
        try:
            # Reuse cached token to save a Gamma API call
            existing_prices: dict[str, float] = {}
            token: str | None = None
            if out_path.exists():
                saved = json.loads(out_path.read_text())
                existing_prices = {pt["date"]: pt["probability"] for pt in saved["prices"]}
                token = saved.get("clob_token_yes")

            if not token:
                token = _get_clob_token(client, pm_id)
                time.sleep(0.3)
            if not token:
                print("✗ no CLOB token")
                continue

            new_pts = _fetch_history(client, token)
            new_prices = {pt["date"]: pt["probability"] for pt in new_pts}

            # Merge: new data overwrites old for same date (handles corrections)
            merged = {**existing_prices, **new_prices}
            all_prices = [{"date": d, "probability": p} for d, p in sorted(merged.items())]

            out_path.write_text(json.dumps({
                "node_id": node_id,
                "pm_id": pm_id,
                "clob_token_yes": token,
                "prices": all_prices,
            }, indent=2))
            added = len(all_prices) - len(existing_prices)
            print(f"→ {len(all_prices)} points (+{added} new)")
        except Exception as exc:
            print(f"✗ {exc}")

        time.sleep(0.5)

    client.close()
    done = sum(1 for n in NODES if (OUT_DIR / f"{n}.json").exists())
    print(f"\nDone: {done}/{total} nodes have history")


if __name__ == "__main__":
    main()
