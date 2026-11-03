#!/usr/bin/env python3
"""Polymarket paper-trading bot — dry run only (retro#620).

Each run: read the universe from Gamma, ask the Oracle for a probability on
every market that is worth asking about, write one snapshot row per market,
and open a notional paper position where the Oracle disagrees with the price
by more than ``EDGE_THRESHOLD``. Positions are held to resolution; nothing is
ever submitted anywhere. The scorecard is ``GET /pm/paper`` on the Oracle API,
which reads the same ledger directory.

The Oracle scores each bucket of a partition on its own, so the YES numbers of
mutually exclusive buckets can sum well past 1. On those events the bot decides
on the normalised number and keeps the raw one (see ``coherent_yes``, retro#831).

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


def coherent_yes(event_markets: list[Market], oracle_by_market: dict[str, float],
                 p: Params) -> Optional[tuple[dict[str, float], float]]:
    """Sum-normalised YES numbers for one event's partition (retro#831).

    Returns ``({market_id: p/Σp}, Σp)``, or None when the event is not a partition
    whose every bucket has a number, in which case the raw numbers stand. Applies
    only to negRisk events, because only there are the buckets mutually exclusive:

    * resolved NO counts as 0; resolved YES makes every open bucket 0.
    * closed but unresolved: the partition is unknown, so None.
    * outside the Oracle price band the Oracle is never asked, so the market price
      stands in for that bucket. It enters Σp but gets no coherent number of its own.
    * inside the band with no Oracle number yet: None.

    Numbers may be as-of different runs, since siblings are asked at different
    times. Normalising removes the excess over 1 and adds no skill. The Oracle's
    CI is not rescaled: it describes the raw number.
    """
    priced = [m for m in event_markets if m.has_price]
    if len(priced) < 2 or not all(m.neg_risk for m in priced):
        return None
    closed = [m for m in priced if m.closed]
    if any(m.resolved_outcome is None for m in closed):
        return None
    live = [m for m in priced if not m.closed]
    if not live:
        return None
    in_band = [m for m in live if p.oracle_price_min <= m.yes_price <= p.oracle_price_max]  # type: ignore[operator]
    if any(m.resolved_outcome == "YES" for m in closed):
        return {m.market_id: 0.0 for m in in_band}, 1.0
    if not in_band or any(m.market_id not in oracle_by_market for m in in_band):
        return None
    numbers = {m.market_id: oracle_by_market[m.market_id] for m in in_band}
    total = sum(numbers.values()) + sum(m.yes_price for m in live if m.market_id not in numbers)  # type: ignore[misc]
    if total <= 0.0:
        return None
    return {mid: round(v / total, 4) for mid, v in numbers.items()}, round(total, 4)


def trade_row(m: Market, side: str, oracle_yes: float, p: Params, now: datetime, run_id: str,
              coherent: Optional[float] = None) -> dict:
    """``oracle_yes`` is always the raw number; ``edge`` is the one the decision used."""
    entry = m.yes_price if side == "YES" else 1.0 - m.yes_price  # type: ignore[operator]
    decided_on = oracle_yes if coherent is None else coherent
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
        "oracle_yes_coherent": coherent,
        "decision_basis": "raw" if coherent is None else "coherent",
        "market_yes": round(m.yes_price, 4),  # type: ignore[arg-type]
        "edge": round(decided_on - m.yes_price, 4),  # type: ignore[operator]
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
        "oracle_yes_coherent": None,
        "coherent_sum": None,
        "edge_coherent": None,
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
    oracle_by_market = ledger.last_oracle_by_market()
    open_ids = ledger.open_trade_market_ids()
    log.info("run %s: %d markets from %d events, params=%s", run_id, len(markets), len(slugs), p.as_dict())

    stats = {"markets": len(markets), "oracle_calls": 0, "oracle_insufficient": 0,
             "trades_opened": 0, "price_only": 0, "skipped_no_price": 0, "errors": 0,
             "asked": 0, "coherent_events": 0}
    by_event: dict[str, list[Market]] = {}
    for m in markets:
        by_event.setdefault(m.event_slug, []).append(m)
    # One event at a time: its rows and trades need every sibling's answer, but
    # are still written before the next event's Oracle calls, so a crash loses
    # at most one event's worth of them.
    for event_markets in by_event.values():
        rows: list[tuple[Market, dict]] = []
        for m in event_markets:
            if not m.has_price:
                stats["skipped_no_price"] += 1
                continue
            row = _forecast_market(m, oracle, latest.get(m.market_id), p, now, run_id, stats, dry_run)
            if row["oracle_yes"] is not None:
                oracle_by_market[m.market_id] = row["oracle_yes"]
            rows.append((m, row))
        coherent = coherent_yes(event_markets, oracle_by_market, p)
        if coherent is not None:
            stats["coherent_events"] += 1
        for m, row in rows:
            _apply_coherent(row, m, coherent)
            log.info("%s yes=%s oracle=%s coherent=%s edge=%s reason=%s %r", m.market_id, row["market_yes"],
                     row["oracle_yes"], row["oracle_yes_coherent"], row["edge"], row["oracle_reason"], m.question[:70])
            if dry_run:
                continue
            ledger.append_snapshot(row)
            if row["oracle_yes"] is not None and m.market_id not in open_ids and not m.closed:
                decided_on = row["oracle_yes"] if row["oracle_yes_coherent"] is None else row["oracle_yes_coherent"]
                side = trade_side(decided_on, m.yes_price, p)  # type: ignore[arg-type]
                if side:
                    t = trade_row(m, side, row["oracle_yes"], p, now, run_id, coherent=row["oracle_yes_coherent"])
                    ledger.append_trade(t)
                    open_ids.add(m.market_id)
                    stats["trades_opened"] += 1
                    log.info("PAPER TRADE %s %s @%.3f  decided_on=%.3f (%s) market=%.3f  %r", side, m.market_id,
                             t["entry_price"], decided_on, t["decision_basis"], t["market_yes"], m.question[:70])
    log.info("done %s", stats)
    return stats


def _forecast_market(m: Market, oracle: OracleClient, last: Optional[dict], p: Params, now: datetime,
                     run_id: str, stats: dict, dry_run: bool) -> dict:
    """Ask the Oracle if this market is due (within the per-run cap) and build its row."""
    oracle_resp: Optional[dict] = None
    if stats["asked"] < p.max_markets_per_run and wants_oracle_call(m, last, p, now):
        stats["asked"] += 1
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
    return row


def _apply_coherent(row: dict, m: Market, coherent: Optional[tuple[dict[str, float], float]]) -> None:
    if coherent is None or m.market_id not in coherent[0]:
        return
    value = coherent[0][m.market_id]
    row["oracle_yes_coherent"] = value
    row["coherent_sum"] = coherent[1]
    row["edge_coherent"] = None if m.yes_price is None else round(value - m.yes_price, 4)


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
