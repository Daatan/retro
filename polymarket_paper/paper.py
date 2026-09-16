#!/usr/bin/env python3
"""Polymarket paper-trading bot — dry run only (retro#620).

Each run: read the universe from Gamma, ask the Oracle for a probability on
every market that is worth asking about, write one snapshot row per market,
and open a notional paper position where the Oracle disagrees with the price
by more than ``EDGE_THRESHOLD``. Positions are held to resolution; nothing is
ever submitted anywhere. The scorecard is ``GET /pm/paper`` on the Oracle API,
which reads the same ledger directory.

Env vars:
    ORACLE_API_KEY          named Oracle relay key (required; the metaculus one
                            is fine — see README for its max_articles cap)
    ORACLE_BASE_URL         default https://oracle.daatan.com
    PAPER_LEDGER_DIR        default ./data
    PAPER_EVENT_SLUGS       comma-separated Gamma event slugs; default = the
                            Israeli election cluster (universe.py)
    MAX_MARKETS_PER_RUN     default 12 — Oracle calls per run (each ~2¢ and
                            up to ~4 min at p99)
    STALE_AFTER_HOURS       default 20 — re-forecast a market only when its
                            last Oracle snapshot is older than this. The unit
                            fires every 6h, so each market is asked ~once a day
                            and the 1h forecast cache never serves a stale hit.
    EDGE_THRESHOLD          default 0.10 — |oracle − market| needed to open a
                            paper position
    TRADE_PRICE_MIN/MAX     default 0.05 / 0.95 — never open a position on a
                            market priced outside this band (long-shot noise)
    ORACLE_PRICE_MIN/MAX    default 0.02 / 0.98 — outside this band the market
                            is snapshotted (price only) but the Oracle is not
                            called; 0.0005-priced no-hopers are not worth 2¢
    NOTIONAL_USD            default 100 — size of every paper position
    DRY_RUN                 "true": log, call nothing, write nothing
"""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from gamma_client import GammaClient, Market
from ledger import Ledger
from oracle_client import OracleClient
from universe import event_slugs_from_env

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("polymarket-paper")

MAX_CRITERIA_CHARS = 1500


@dataclass(frozen=True)
class Params:
    max_markets_per_run: int = 12
    stale_after: timedelta = timedelta(hours=20)
    edge_threshold: float = 0.10
    trade_price_min: float = 0.05
    trade_price_max: float = 0.95
    oracle_price_min: float = 0.02
    oracle_price_max: float = 0.98
    notional_usd: float = 100.0

    def as_dict(self) -> dict:
        return {
            "max_markets_per_run": self.max_markets_per_run,
            "stale_after_hours": self.stale_after.total_seconds() / 3600,
            "edge_threshold": self.edge_threshold,
            "trade_price_band": [self.trade_price_min, self.trade_price_max],
            "oracle_price_band": [self.oracle_price_min, self.oracle_price_max],
            "notional_usd": self.notional_usd,
        }


# ── probability scale ───────────────────────────────────────────────────────
# The Oracle returns a stance mean in [-1, 1]; Polymarket prices are YES
# probabilities in [0, 1]. This is the ONE place the conversion happens
# (retro#620 "be explicit about probability scales at every boundary").

def stance_to_prob(mean: float) -> float:
    if not -1.0 <= mean <= 1.0:
        raise ValueError(f"Oracle mean out of [-1, 1]: {mean!r}")
    return (mean + 1.0) / 2.0


# ── decisions (pure) ────────────────────────────────────────────────────────

def _parse_ts(value: str | None) -> Optional[datetime]:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def wants_oracle_call(m: Market, last: Optional[dict], p: Params, now: datetime) -> bool:
    """Ask the Oracle about this market on this run?"""
    if m.closed or not m.has_price:
        return False
    if not (p.oracle_price_min <= m.yes_price <= p.oracle_price_max):  # type: ignore[operator]
        return False
    if last is None or last.get("oracle_yes") is None and last.get("oracle_reason") is None:
        # never asked (or an earlier row was price-only)
        return True
    asked_at = _parse_ts(last.get("ts"))
    return asked_at is None or now - asked_at >= p.stale_after


def trade_side(oracle_yes: float, market_yes: float, p: Params) -> Optional[str]:
    """"YES" / "NO" when the edge clears the threshold inside the tradeable band."""
    if not (p.trade_price_min <= market_yes <= p.trade_price_max):
        return None
    # Round first: 0.50 − 0.40 is 0.0999… in floating point, and a threshold
    # check that misses "exactly at threshold" is a bug nobody would ever see.
    edge = round(oracle_yes - market_yes, 4)
    if edge >= p.edge_threshold:
        return "YES"
    if edge <= -p.edge_threshold:
        return "NO"
    return None


def trade_row(m: Market, side: str, oracle_yes: float, p: Params, now: datetime, run_id: str) -> dict:
    entry = m.yes_price if side == "YES" else 1.0 - m.yes_price  # type: ignore[operator]
    return {
        "ts": now.isoformat(),
        "run_id": run_id,
        "trade_id": uuid.uuid4().hex[:12],
        "market_id": m.market_id,
        "event_slug": m.event_slug,
        "question": m.question,
        "side": side,
        "entry_price": round(entry, 4),
        "notional_usd": p.notional_usd,
        "shares": round(p.notional_usd / entry, 4),
        "oracle_yes": round(oracle_yes, 4),
        "market_yes": round(m.yes_price, 4),  # type: ignore[arg-type]
        "edge": round(oracle_yes - m.yes_price, 4),  # type: ignore[operator]
        "paper": True,
    }


def snapshot_row(m: Market, oracle: Optional[dict], now: datetime, run_id: str) -> dict:
    row: dict = {
        "ts": now.isoformat(),
        "run_id": run_id,
        "market_id": m.market_id,
        "event_slug": m.event_slug,
        "market_slug": m.market_slug,
        "question": m.question,
        "end_date": m.end_date,
        "closed": m.closed,
        "resolved_outcome": m.resolved_outcome,
        "market_yes": None if m.yes_price is None else round(m.yes_price, 4),
        "volume_usd": round(m.volume_usd, 2),
        "oracle_yes": None,
        "oracle_reason": None,
        "edge": None,
    }
    if oracle is None:
        return row
    if oracle.get("insufficient_data"):
        row["oracle_reason"] = oracle.get("reason") or "insufficient_data"
    else:
        oy = stance_to_prob(float(oracle["mean"]))
        row["oracle_yes"] = round(oy, 4)
        row["edge"] = None if m.yes_price is None else round(oy - m.yes_price, 4)
        row["oracle_ci_yes"] = [round(stance_to_prob(float(oracle["ci_low"])), 4),
                                round(stance_to_prob(float(oracle["ci_high"])), 4)]
    row["oracle_confidence"] = oracle.get("confidence")
    row["oracle_settled"] = bool(oracle.get("settled", False))
    row["articles_used"] = oracle.get("articles_used")
    row["articles_found"] = oracle.get("articles_found")
    row["oracle_provider"] = oracle.get("provider")
    row["fallback_path"] = oracle.get("fallback_path")
    prov = oracle.get("provenance") or {}
    row["oracle_build"] = prov.get("build") or prov.get("git_sha")
    return row


def resolution_criteria(m: Market) -> Optional[str]:
    d = m.description.strip()
    if not d:
        return None
    return d if len(d) <= MAX_CRITERIA_CHARS else d[:MAX_CRITERIA_CHARS].rsplit(" ", 1)[0]


# ── run ─────────────────────────────────────────────────────────────────────

def run(gamma: GammaClient, oracle: OracleClient, ledger: Ledger, slugs: tuple[str, ...],
        p: Params, *, now: Optional[datetime] = None, dry_run: bool = False) -> dict:
    now = now or datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%SZ")
    markets = gamma.universe(slugs)
    latest = ledger.latest_snapshot_by_market()
    open_ids = ledger.open_trade_market_ids()
    log.info("run %s: %d markets from %d events, params=%s", run_id, len(markets), len(slugs), p.as_dict())

    calls = 0
    stats = {"markets": len(markets), "oracle_calls": 0, "oracle_insufficient": 0,
             "trades_opened": 0, "price_only": 0, "skipped_no_price": 0, "errors": 0}
    for m in markets:
        if not m.has_price:
            stats["skipped_no_price"] += 1
            continue
        oracle_resp: Optional[dict] = None
        if calls < p.max_markets_per_run and wants_oracle_call(m, latest.get(m.market_id), p, now):
            calls += 1
            if dry_run:
                log.info("DRY RUN would forecast %s %r", m.market_id, m.question)
            else:
                try:
                    oracle_resp = oracle.forecast(m.question, resolution_criteria(m))
                    stats["oracle_calls"] += 1
                except Exception as exc:  # one bad market must not end the run
                    stats["errors"] += 1
                    log.warning("forecast failed for %s: %s", m.market_id, exc)
        else:
            stats["price_only"] += 1

        row = snapshot_row(m, oracle_resp, now, run_id)
        if row["oracle_reason"]:
            stats["oracle_insufficient"] += 1
        log.info("%s yes=%s oracle=%s edge=%s reason=%s %r", m.market_id, row["market_yes"],
                 row["oracle_yes"], row["edge"], row["oracle_reason"], m.question[:70])
        if dry_run:
            continue
        ledger.append_snapshot(row)

        if row["oracle_yes"] is not None and m.market_id not in open_ids and not m.closed:
            side = trade_side(row["oracle_yes"], m.yes_price, p)  # type: ignore[arg-type]
            if side:
                t = trade_row(m, side, row["oracle_yes"], p, now, run_id)
                ledger.append_trade(t)
                open_ids.add(m.market_id)
                stats["trades_opened"] += 1
                log.info("PAPER TRADE %s %s @%.3f  oracle=%.3f market=%.3f  %r",
                         side, m.market_id, t["entry_price"], t["oracle_yes"], t["market_yes"], m.question[:70])
    log.info("done %s", stats)
    return stats


def _required_env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise SystemExit(f"{name} is required")
    return v


def params_from_env(env: dict = os.environ) -> Params:  # type: ignore[assignment]
    g = env.get
    return Params(
        max_markets_per_run=int(g("MAX_MARKETS_PER_RUN", "12")),
        stale_after=timedelta(hours=float(g("STALE_AFTER_HOURS", "20"))),
        edge_threshold=float(g("EDGE_THRESHOLD", "0.10")),
        trade_price_min=float(g("TRADE_PRICE_MIN", "0.05")),
        trade_price_max=float(g("TRADE_PRICE_MAX", "0.95")),
        oracle_price_min=float(g("ORACLE_PRICE_MIN", "0.02")),
        oracle_price_max=float(g("ORACLE_PRICE_MAX", "0.98")),
        notional_usd=float(g("NOTIONAL_USD", "100")),
    )


def main() -> None:
    dry_run = os.environ.get("DRY_RUN", "").lower() in ("1", "true", "yes")
    api_key = os.environ.get("ORACLE_API_KEY") or ("dry-run" if dry_run else _required_env("ORACLE_API_KEY"))
    base_url = os.environ.get("ORACLE_BASE_URL", "https://oracle.daatan.com")
    ledger = Ledger(Path(os.environ.get("PAPER_LEDGER_DIR", "./data")))
    slugs = event_slugs_from_env(os.environ.get("PAPER_EVENT_SLUGS"))
    with GammaClient() as gamma, OracleClient(api_key, base_url=base_url) as oracle:
        run(gamma, oracle, ledger, slugs, params_from_env(), dry_run=dry_run)


if __name__ == "__main__":
    main()
