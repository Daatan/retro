"""Read-only scorecard over the Manifold paper bot's ledger (retro#918).

The bot (``manifold_paper/paper.py``, a systemd oneshot on the same box) appends
two JSONL files under ``settings.resolved_manifold_paper_ledger_dir``; this module
turns them into the report ``GET /manifold/paper`` serves. Nothing here writes
and nothing here talks to Manifold — every number comes from the ledger.

Unlike the Polymarket scorecard this one is a *calibration* report, because that
is what the arena is for (retro#918):

* **Brier / log-loss** of the Oracle vs the market, both scored from the *same*
  snapshot: the **first** one carrying an Oracle number, on markets that resolved
  YES or NO. It only counts when it was made at least ``MIN_HORIZON_DAYS`` before
  the market's close and the market price then sat inside the scoring band — a
  forecast made hours before the outcome, against a price already at 0 or 1,
  proves nothing (rule fixed 2026-10-13 while 0 markets were scored, so it is not
  tuned to a result; before that it was the *last* snapshot). Excluded markets are
  counted (``n_excluded_horizon`` / ``n_excluded_band``), never silently dropped.
  MKT and CANCEL resolutions are counted, never scored.
* **Brier difference** (oracle − market) with a deterministic bootstrap 95% CI;
  negative means the Oracle beat the market. The pre-registered bar is
  ``n_scored >= MIN_SCORED`` and the CI's mean within ``PASS_MARGIN`` of zero.
* **Reliability table** — scored Oracle forecasts binned by predicted
  probability, predicted mean vs observed YES frequency.
* **Coverage** — share of tracked markets that got an Oracle number at all
  (the rest are abstentions / insufficient data / price-only).

Manifold ToS: this is evaluation only. The ledger and this report must never be
fed back as training or prompt-tuning data for the production Oracle.
"""

from __future__ import annotations

import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

SNAPSHOTS = "snapshots.jsonl"
TRADES = "trades.jsonl"

MIN_SCORED = 150          # pre-registered in retro#918
PASS_MARGIN = 0.01        # Oracle Brier may exceed the market's by at most this
# Mirrors manifold_paper/paper.py (Params.min_horizon_days, oracle_price_min/max); the two
# packages do not import each other, so keep them in step by hand.
MIN_HORIZON_DAYS = 5.0
SCORE_PRICE_MIN = 0.10
SCORE_PRICE_MAX = 0.90
N_BINS = 5
BOOTSTRAP_ITERS = 2000
_EPS = 1e-4


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


def _y(outcome: str) -> float:
    return 1.0 if outcome == "YES" else 0.0


def _brier(p: float, outcome: str) -> float:
    return (p - _y(outcome)) ** 2


def _logloss(p: float, outcome: str) -> float:
    p = min(max(p, _EPS), 1.0 - _EPS)
    return -math.log(p if outcome == "YES" else 1.0 - p)


def _bootstrap_ci(diffs: list[float]) -> Optional[list[float]]:
    """95% percentile CI of the mean difference. Seeded: the report is reproducible."""
    n = len(diffs)
    if n < 2:
        return None
    rng = random.Random(918)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(BOOTSTRAP_ITERS))
    return [round(means[int(0.025 * BOOTSTRAP_ITERS)], 4), round(means[int(0.975 * BOOTSTRAP_ITERS) - 1], 4)]


def _reliability(pairs: list[tuple[float, str]]) -> list[dict]:
    bins: list[list[tuple[float, str]]] = [[] for _ in range(N_BINS)]
    for p, outcome in pairs:
        bins[min(int(p * N_BINS), N_BINS - 1)].append((p, outcome))
    out = []
    for i, rows in enumerate(bins):
        out.append({
            "bin": f"{i / N_BINS:.1f}-{(i + 1) / N_BINS:.1f}",
            "n": len(rows),
            "mean_predicted": round(sum(p for p, _ in rows) / len(rows), 4) if rows else None,
            "observed_yes": round(sum(_y(o) for _, o in rows) / len(rows), 4) if rows else None,
        })
    return out


def _exclusion(o: dict) -> Optional[str]:
    """Why the first Oracle snapshot cannot be scored, or None when it can.

    ``horizon``: made less than MIN_HORIZON_DAYS before the market's close (or the
    timestamps are unreadable — a forecast of unknown lead time is not evidence).
    ``band``: the market price at that moment was already near 0 or 1."""
    try:
        close = datetime.fromisoformat(str(o.get("end_date")))
        made = datetime.fromisoformat(str(o.get("ts")))
        if (close - made).total_seconds() < MIN_HORIZON_DAYS * 86400:
            return "horizon"
    except (TypeError, ValueError):
        return "horizon"
    pm = float(o["market_yes"])
    if not (SCORE_PRICE_MIN <= pm <= SCORE_PRICE_MAX):
        return "band"
    return None


def build_report(ledger_dir: Path) -> dict:
    snapshots = list(_read(ledger_dir / SNAPSHOTS))
    trades = list(_read(ledger_dir / TRADES))

    latest: dict[str, dict] = {}
    first_oracle: dict[str, dict] = {}
    n_calls = 0
    counts: dict[str, int] = {}
    for row in snapshots:  # file order is time order
        mid = row["market_id"]
        latest[mid] = row
        counts[mid] = counts.get(mid, 0) + 1
        if row.get("oracle_yes") is not None:
            first_oracle.setdefault(mid, row)
            n_calls += 1
        elif row.get("oracle_reason"):
            n_calls += 1

    markets = []
    scored: list[tuple[float, float, str]] = []  # (oracle, market, outcome)
    n_void = 0
    n_excl_horizon = n_excl_band = 0
    for mid, row in latest.items():
        o = first_oracle.get(mid)
        outcome = row.get("resolved_outcome")
        raw = row.get("resolution_raw")
        if raw in ("MKT", "CANCEL"):
            n_void += 1
        entry = {
            "market_id": mid, "market_slug": row.get("market_slug"), "question": row.get("question"),
            "creator": row.get("creator"), "unique_bettors": row.get("unique_bettors"),
            "end_date": row.get("end_date"), "closed": bool(row.get("closed")),
            "resolved_outcome": outcome, "resolution_raw": raw,
            "market_yes": row.get("market_yes"), "last_ts": row.get("ts"),
            "oracle_yes": None if o is None else o.get("oracle_yes"),
            "oracle_ts": None if o is None else o.get("ts"),
            "market_yes_at_oracle": None if o is None else o.get("market_yes"),
            "edge": None if o is None else o.get("edge"),
            "articles_used": None if o is None else o.get("articles_used"),
            "oracle_reason": row.get("oracle_reason"),
            "n_snapshots": counts[mid], "brier_oracle": None, "brier_market": None,
            "excluded": None,
        }
        if outcome in ("YES", "NO") and o is not None and o.get("market_yes") is not None:
            po, pm = float(o["oracle_yes"]), float(o["market_yes"])
            excluded = _exclusion(o)
            if excluded == "horizon":
                n_excl_horizon += 1
            elif excluded == "band":
                n_excl_band += 1
            entry["excluded"] = excluded
            if excluded is None:
                entry["brier_oracle"] = round(_brier(po, outcome), 4)
                entry["brier_market"] = round(_brier(pm, outcome), 4)
                scored.append((po, pm, outcome))
        markets.append(entry)
    markets.sort(key=lambda m: (m["end_date"] or "", m["market_id"]))

    n = len(scored)
    diffs = [_brier(po, oc) - _brier(pm, oc) for po, pm, oc in scored]
    mean_diff = round(sum(diffs) / n, 4) if n else None
    ci = _bootstrap_ci(diffs)

    positions = []
    realized = unrealized = 0.0
    wins = losses = 0
    for t in trades:
        row = latest.get(t["market_id"], {})
        outcome = row.get("resolved_outcome")
        side, shares, notional = t["side"], float(t["shares"]), float(t["notional_mana"])
        pos = {**{k: t.get(k) for k in ("trade_id", "ts", "market_id", "question", "side", "entry_price",
                                         "shares", "notional_mana", "oracle_yes", "market_yes", "edge")},
               "status": "open", "resolved_outcome": outcome, "current_side_price": None,
               "value_mana": None, "pnl_mana": None}
        if outcome in ("YES", "NO"):
            won = outcome == side
            value = shares * (1.0 if won else 0.0)
            pos.update(status="won" if won else "lost", current_side_price=1.0 if won else 0.0,
                       value_mana=round(value, 2), pnl_mana=round(value - notional, 2))
            realized += value - notional
            wins += int(won)
            losses += int(not won)
        elif row.get("resolution_raw") in ("MKT", "CANCEL"):
            pos.update(status="void")
        else:
            yes = row.get("market_yes")
            if yes is not None:
                price = float(yes) if side == "YES" else 1.0 - float(yes)
                value = shares * price
                pos.update(current_side_price=round(price, 4), value_mana=round(value, 2),
                           pnl_mana=round(value - notional, 2))
                unrealized += value - notional
        positions.append(pos)

    ts_all = [r.get("ts") for r in snapshots if r.get("ts")]
    with_oracle = sum(1 for m in markets if m["oracle_yes"] is not None)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "first_snapshot": min(ts_all) if ts_all else None,
        "last_snapshot": max(ts_all) if ts_all else None,
        "summary": {
            "n_markets": len(markets),
            "n_with_oracle": with_oracle,
            "coverage": round(with_oracle / len(markets), 4) if markets else None,
            "n_resolved": sum(1 for m in markets if m["resolved_outcome"] in ("YES", "NO")),
            "n_void": n_void,
            "n_scored": n,
            "n_excluded_horizon": n_excl_horizon,
            "n_excluded_band": n_excl_band,
            "scoring_rule": {"snapshot": "first with an Oracle number",
                             "min_horizon_days": MIN_HORIZON_DAYS,
                             "price_band": [SCORE_PRICE_MIN, SCORE_PRICE_MAX]},
            "brier_oracle": round(sum(_brier(po, oc) for po, _, oc in scored) / n, 4) if n else None,
            "brier_market": round(sum(_brier(pm, oc) for _, pm, oc in scored) / n, 4) if n else None,
            "logloss_oracle": round(sum(_logloss(po, oc) for po, _, oc in scored) / n, 4) if n else None,
            "logloss_market": round(sum(_logloss(pm, oc) for _, pm, oc in scored) / n, 4) if n else None,
            "brier_diff_mean": mean_diff,
            "brier_diff_ci95": ci,
            "criterion": {
                "min_scored": MIN_SCORED, "pass_margin": PASS_MARGIN,
                "enough_data": n >= MIN_SCORED,
                "passes": (n >= MIN_SCORED and mean_diff is not None and mean_diff <= PASS_MARGIN),
            },
            "oracle_calls": n_calls,
            "n_positions": len(positions),
            "n_open": sum(1 for p in positions if p["status"] == "open"),
            "n_won": wins, "n_lost": losses,
            "notional_mana": round(sum(float(t["notional_mana"]) for t in trades), 2),
            "unrealized_pnl_mana": round(unrealized, 2),
            "realized_pnl_mana": round(realized, 2),
        },
        "reliability": _reliability([(po, oc) for po, _, oc in scored]),
        "markets": markets,
        "positions": positions,
    }


def _f(v: Optional[float], nd: int = 3) -> str:
    return "—" if v is None else f"{v:.{nd}f}"


def render_markdown(report: dict) -> str:
    s = report["summary"]
    c = s["criterion"]
    ci = s["brier_diff_ci95"]
    verdict = ("**PASS**" if c["passes"] else "not passing") if c["enough_data"] else f"too early ({s['n_scored']}/{c['min_scored']} scored)"
    out = ["# Manifold paper bot — calibration scorecard", "",
           f"_Snapshots {report['first_snapshot'] or '—'} → {report['last_snapshot'] or '—'}; "
           f"generated {report['generated_at']}. Hypothetical positions only; no account, no bets, play money._", "",
           "| | |", "|---|---|",
           f"| Markets tracked | {s['n_markets']} (Oracle number on {s['n_with_oracle']}, coverage {_f(s['coverage'], 2)}) |",
           f"| Resolved YES/NO · void (MKT/CANCEL) · scored | {s['n_resolved']} · {s['n_void']} · {s['n_scored']} |",
           f"| Resolved but not scorable (< {s['scoring_rule']['min_horizon_days']:g} d lead · price outside "
           f"{s['scoring_rule']['price_band'][0]:g}–{s['scoring_rule']['price_band'][1]:g}) | "
           f"{s['n_excluded_horizon']} · {s['n_excluded_band']} |",
           f"| **Brier — Oracle / market** | **{_f(s['brier_oracle'], 4)} / {_f(s['brier_market'], 4)}** |",
           f"| Log-loss — Oracle / market | {_f(s['logloss_oracle'], 4)} / {_f(s['logloss_market'], 4)} |",
           f"| Brier diff (oracle − market), 95% CI | {_f(s['brier_diff_mean'], 4)} {'—' if ci is None else f'[{ci[0]:+.4f}, {ci[1]:+.4f}]'} |",
           f"| Pre-registered bar (≥{c['min_scored']} scored, diff ≤ +{c['pass_margin']}) | {verdict} |",
           f"| Hypothetical positions | {s['n_positions']} ({s['n_open']} open, {s['n_won']} won, {s['n_lost']} lost) |",
           f"| Notional / P&L unrealized / realized (Mana) | M${s['notional_mana']:.0f} / M${s['unrealized_pnl_mana']:.2f} / M${s['realized_pnl_mana']:.2f} |",
           f"| Oracle calls | {s['oracle_calls']} |", "",
           "## Reliability (scored Oracle forecasts)", "",
           "| Predicted bin | n | Mean predicted | Observed YES |", "|---|---:|---:|---:|"]
    for b in report["reliability"]:
        out.append(f"| {b['bin']} | {b['n']} | {_f(b['mean_predicted'])} | {_f(b['observed_yes'])} |")
    out += ["", "## Markets", "",
            "| Market | Price | Oracle | Edge | Outcome | Brier O / M | Articles |", "|---|---:|---:|---:|---|---|---:|"]
    for m in report["markets"]:
        state = m["resolved_outcome"] or m["resolution_raw"] or ("closed" if m["closed"] else "open")
        out.append(f"| {m['question']} | {_f(m['market_yes'])} | {_f(m['oracle_yes'])} | {_f(m['edge'])} | {state} | "
                   f"{_f(m['brier_oracle'])} / {_f(m['brier_market'])} | "
                   f"{m['articles_used'] if m['articles_used'] is not None else '—'} |")
    out += ["", "## Hypothetical positions", "",
            "| Opened | Market | Side | Entry | Now | P&L (Mana) | Status |", "|---|---|---|---:|---:|---:|---|"]
    for p in report["positions"]:
        pnl = "—" if p["pnl_mana"] is None else f"M${p['pnl_mana']:.2f}"
        out.append(f"| {str(p['ts'])[:10]} | {p['question']} | {p['side']} | {_f(p['entry_price'])} | "
                   f"{_f(p['current_side_price'])} | {pnl} | {p['status']} |")
    return "\n".join(out) + "\n"
