"""Read-only scorecard over the Polymarket paper bot's ledger (retro#620).

The bot (``polymarket_paper/paper.py``, a systemd oneshot on the same box)
appends two JSONL files under ``settings.resolved_polymarket_paper_ledger_dir``;
this module turns them into the report ``GET /pm/paper`` serves. Nothing here
writes, and nothing here talks to Polymarket — every number comes from the
ledger, so the report is reproducible from the files alone.

Two views, on purpose (most of the Israeli-election universe ends 2026-10-27,
but the seat buckets end 11-30 and the PM market resolves on swearing-in):

* **mark-to-market** — every open paper position valued at the latest
  snapshot price. Available for everything, every day.
* **resolution** — Brier of the Oracle vs the market on markets Gamma reports
  as resolved, both scored from the *same* snapshot (the last one carrying an
  Oracle number), so the comparison is fair: the market price is the one the
  Oracle was disagreeing with at the time.

``oracle_yes`` is the raw Oracle number. ``oracle_yes_coherent`` is that number
normalised over a mutually exclusive partition (retro#831), present only on rows
the bot could normalise, so its Brier covers a subset of markets. The report
scores raw and market on that same subset for a like-for-like comparison.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

SNAPSHOTS = "snapshots.jsonl"
TRADES = "trades.jsonl"


def _read(path: Path) -> Iterator[dict]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except ValueError:
                    continue  # a torn last line must not take the report down


def _brier(p: float, outcome: str) -> float:
    y = 1.0 if outcome == "YES" else 0.0
    return (p - y) ** 2


def build_report(ledger_dir: Path) -> dict:
    snapshots = list(_read(ledger_dir / SNAPSHOTS))
    trades = list(_read(ledger_dir / TRADES))

    latest: dict[str, dict] = {}          # market → last snapshot (price series)
    last_oracle: dict[str, dict] = {}     # market → last snapshot with an Oracle number
    n_calls = 0
    for row in snapshots:                 # file order is time order
        mid = row["market_id"]
        latest[mid] = row
        if row.get("oracle_yes") is not None:
            last_oracle[mid] = row
            n_calls += 1
        elif row.get("oracle_reason"):
            n_calls += 1

    markets = []
    brier_o: list[float] = []
    brier_m: list[float] = []
    coherent_scored: list[tuple[float, float, float]] = []  # (coherent, raw, market) Brier
    for mid, row in latest.items():
        o = last_oracle.get(mid)
        outcome = row.get("resolved_outcome")
        entry = {
            "market_id": mid,
            "event_slug": row.get("event_slug"),
            "question": row.get("question"),
            "end_date": row.get("end_date"),
            "closed": bool(row.get("closed")),
            "resolved_outcome": outcome,
            "market_yes": row.get("market_yes"),
            "last_ts": row.get("ts"),
            "oracle_yes": None if o is None else o.get("oracle_yes"),
            "oracle_ts": None if o is None else o.get("ts"),
            "market_yes_at_oracle": None if o is None else o.get("market_yes"),
            "edge": None if o is None else o.get("edge"),
            "oracle_yes_coherent": None if o is None else o.get("oracle_yes_coherent"),
            "edge_coherent": None if o is None else o.get("edge_coherent"),
            "coherent_sum": None if o is None else o.get("coherent_sum"),
            "oracle_confidence": None if o is None else o.get("oracle_confidence"),
            "articles_used": None if o is None else o.get("articles_used"),
            "oracle_reason": row.get("oracle_reason"),
            "n_snapshots": 0,
            "brier_oracle": None,
            "brier_market": None,
            "brier_coherent": None,
        }
        if outcome in ("YES", "NO") and o is not None and o.get("market_yes") is not None:
            entry["brier_oracle"] = round(_brier(float(o["oracle_yes"]), outcome), 4)
            entry["brier_market"] = round(_brier(float(o["market_yes"]), outcome), 4)
            brier_o.append(entry["brier_oracle"])
            brier_m.append(entry["brier_market"])
            if entry["oracle_yes_coherent"] is not None:
                entry["brier_coherent"] = round(_brier(float(entry["oracle_yes_coherent"]), outcome), 4)
                coherent_scored.append((entry["brier_coherent"], entry["brier_oracle"], entry["brier_market"]))
        markets.append(entry)
    counts: dict[str, int] = {}
    for row in snapshots:
        counts[row["market_id"]] = counts.get(row["market_id"], 0) + 1
    for m in markets:
        m["n_snapshots"] = counts.get(m["market_id"], 0)
    markets.sort(key=lambda m: (m["event_slug"] or "", -(m["market_yes"] or 0.0)))

    positions = []
    realized = 0.0
    unrealized = 0.0
    wins = losses = 0
    for t in trades:
        row = latest.get(t["market_id"], {})
        outcome = row.get("resolved_outcome")
        side = t["side"]
        shares = float(t["shares"])
        notional = float(t["notional_usd"])
        pos = {**{k: t.get(k) for k in ("trade_id", "ts", "market_id", "question", "side", "entry_price",
                                         "shares", "notional_usd", "oracle_yes", "oracle_yes_coherent",
                                         "decision_basis", "market_yes", "edge")},
               "status": "open", "resolved_outcome": outcome, "current_side_price": None,
               "value_usd": None, "pnl_usd": None}
        if outcome in ("YES", "NO"):
            won = outcome == side
            value = shares * (1.0 if won else 0.0)
            pos.update(status="won" if won else "lost", current_side_price=1.0 if won else 0.0,
                       value_usd=round(value, 2), pnl_usd=round(value - notional, 2))
            realized += value - notional
            wins += int(won)
            losses += int(not won)
        else:
            yes = row.get("market_yes")
            if yes is not None:
                price = float(yes) if side == "YES" else 1.0 - float(yes)
                value = shares * price
                pos.update(current_side_price=round(price, 4), value_usd=round(value, 2),
                           pnl_usd=round(value - notional, 2))
                unrealized += value - notional
        positions.append(pos)

    ts_all = [r.get("ts") for r in snapshots if r.get("ts")]
    scored = len(brier_o)
    n_coherent = len(coherent_scored)

    def _coherent_mean(i: int) -> Optional[float]:
        return round(sum(c[i] for c in coherent_scored) / n_coherent, 4) if n_coherent else None

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "first_snapshot": min(ts_all) if ts_all else None,
        "last_snapshot": max(ts_all) if ts_all else None,
        "summary": {
            "n_markets": len(markets),
            "n_with_oracle": sum(1 for m in markets if m["oracle_yes"] is not None),
            "n_resolved": sum(1 for m in markets if m["resolved_outcome"] in ("YES", "NO")),
            "n_scored": scored,
            "brier_oracle": round(sum(brier_o) / scored, 4) if scored else None,
            "brier_market": round(sum(brier_m) / scored, 4) if scored else None,
            "n_scored_coherent": n_coherent,
            "brier_coherent": _coherent_mean(0),
            "brier_oracle_on_coherent": _coherent_mean(1),
            "brier_market_on_coherent": _coherent_mean(2),
            "oracle_calls": n_calls,
            "n_positions": len(positions),
            "n_open": sum(1 for p in positions if p["status"] == "open"),
            "n_won": wins,
            "n_lost": losses,
            "notional_usd": round(sum(float(t["notional_usd"]) for t in trades), 2),
            "unrealized_pnl_usd": round(unrealized, 2),
            "realized_pnl_usd": round(realized, 2),
        },
        "markets": markets,
        "positions": positions,
    }


def _f(v: Optional[float], nd: int = 3) -> str:
    return "—" if v is None else f"{v:.{nd}f}"


def render_markdown(report: dict) -> str:
    s = report["summary"]
    out = ["# Polymarket paper bot — scorecard", "",
           f"_Snapshots {report['first_snapshot'] or '—'} → {report['last_snapshot'] or '—'}; "
           f"generated {report['generated_at']}. Paper positions only; no funds, no orders._", "",
           "| | |", "|---|---|",
           f"| Markets tracked | {s['n_markets']} (Oracle number on {s['n_with_oracle']}) |",
           f"| Resolved / scored | {s['n_resolved']} / {s['n_scored']} |",
           f"| **Brier — Oracle** | **{_f(s['brier_oracle'], 4)}** |",
           f"| **Brier — market** | **{_f(s['brier_market'], 4)}** |",
           f"| Brier on normalised partitions (n={s['n_scored_coherent']}): coherent / raw / market | "
           f"{_f(s['brier_coherent'], 4)} / {_f(s['brier_oracle_on_coherent'], 4)} / {_f(s['brier_market_on_coherent'], 4)} |",
           f"| Paper positions | {s['n_positions']} ({s['n_open']} open, {s['n_won']} won, {s['n_lost']} lost) |",
           f"| Notional | ${s['notional_usd']:.0f} |",
           f"| P&L unrealized / realized | ${s['unrealized_pnl_usd']:.2f} / ${s['realized_pnl_usd']:.2f} |",
           f"| Oracle calls | {s['oracle_calls']} |", "",
           "## Markets", "",
           "| Market | Price | Oracle | Coherent | Edge | Outcome | Brier O / M | Articles |",
           "|---|---:|---:|---:|---:|---|---|---:|"]
    for m in report["markets"]:
        out.append(f"| {m['question']} | {_f(m['market_yes'])} | {_f(m['oracle_yes'])} | {_f(m['oracle_yes_coherent'])} | {_f(m['edge'])} | "
                   f"{m['resolved_outcome'] or ('closed' if m['closed'] else 'open')} | "
                   f"{_f(m['brier_oracle'])} / {_f(m['brier_market'])} | {m['articles_used'] if m['articles_used'] is not None else '—'} |")
    out += ["", "## Paper positions", "",
            "| Opened | Market | Side | Entry | Now | P&L | Status |", "|---|---|---|---:|---:|---:|---|"]
    for p in report["positions"]:
        pnl = "—" if p["pnl_usd"] is None else f"${p['pnl_usd']:.2f}"
        out.append(f"| {str(p['ts'])[:10]} | {p['question']} | {p['side']} | {_f(p['entry_price'])} | "
                   f"{_f(p['current_side_price'])} | {pnl} | {p['status']} |")
    return "\n".join(out) + "\n"
