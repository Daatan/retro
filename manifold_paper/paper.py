#!/usr/bin/env python3
"""Manifold paper-trading bot — read-only evaluation (retro#918).

Each run: discover open binary markets on Manifold (plus re-check every market
already in the ledger until it resolves), ask the Oracle for a probability on
every market worth asking about, write one snapshot row per market, and open a
hypothetical Mana position where the Oracle disagrees with the price by more
than ``EDGE_THRESHOLD``. Nothing is ever sent to Manifold: no account, no API
key, GET-only client. The scorecard is ``GET /manifold/paper`` on the Oracle
API, which reads the same ledger directory.

Manifold ToS: evaluation only — outcomes recorded here must not be used as
training / few-shot / prompt-tuning data for the production Oracle.

Env vars:
    ORACLE_API_KEY          named Oracle relay key (required; `manifold-paper`
                            entry in ORACLE_API_KEYS, falling back to metaculus)
    ORACLE_BASE_URL         default https://oracle.daatan.com
    PAPER_LEDGER_DIR        default ./data
    MAX_MARKETS_PER_RUN     default 12 — Oracle calls per run
    STALE_AFTER_HOURS       default 20 — re-forecast a market only when its last
                            Oracle snapshot is older than this
    EDGE_THRESHOLD          default 0.10 — |oracle − market| to open a position
    TRADE_PRICE_MIN/MAX     default 0.05 / 0.95
    ORACLE_PRICE_MIN/MAX    default 0.02 / 0.98
    NOTIONAL_MANA           default 100 — size of every hypothetical position
    MIN_BETTORS             default 15   — universe: unique bettors
    MIN_VOLUME              default 500  — universe: Mana volume
    MIN_HOURS_TO_CLOSE      default 12   — universe: closes no sooner than this
    MAX_DAYS_TO_CLOSE       default 30   — universe: closes no later than this
    DRY_RUN                 "true": log, call no Oracle, write nothing
"""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from ledger import Ledger
from manifold_client import ManifoldClient, Market
from oracle_client import OracleClient
from universe import UniverseParams, eligible

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("manifold-paper")

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
    notional_mana: float = 100.0

    def as_dict(self) -> dict:
        return {
            "max_markets_per_run": self.max_markets_per_run,
            "stale_after_hours": self.stale_after.total_seconds() / 3600,
            "edge_threshold": self.edge_threshold,
            "trade_price_band": [self.trade_price_min, self.trade_price_max],
            "oracle_price_band": [self.oracle_price_min, self.oracle_price_max],
            "notional_mana": self.notional_mana,
        }


# ── probability scale ───────────────────────────────────────────────────────
# The Oracle returns a stance mean in [-1, 1]; Manifold `probability` is a YES
# probability in [0, 1]. This is the ONE place the conversion happens
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
        "market_slug": m.market_slug,
        "question": m.question,
        "side": side,
        "entry_price": round(entry, 4),
        "notional_mana": p.notional_mana,
        "shares": round(p.notional_mana / entry, 4),
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
        "market_slug": m.market_slug,
        "question": m.question,
        "end_date": m.end_date,
        "closed": m.closed,
        "resolved_outcome": m.resolved_outcome,
        "resolution_raw": m.resolution_raw,
        "market_yes": None if m.yes_price is None else round(m.yes_price, 4),
        "volume_mana": round(m.volume, 2),
        "unique_bettors": m.unique_bettors,
        "creator": m.creator,
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


# ── universe ────────────────────────────────────────────────────────────────

def gather_universe(client: ManifoldClient, ledger: Ledger, up: UniverseParams, now: datetime,
                    *, search_pages: int = 3) -> list[Market]:
    """Tracked-and-unresolved markets (fetched by id, so a market that dropped out
    of the "open" search still gets its resolution recorded) plus newly eligible
    candidates from search. Ordered: soonest close first, so the per-run Oracle
    cap goes to the markets that will resolve inside the runway."""
    tracked: dict[str, Market] = {}
    for mid, row in ledger.latest_snapshot_by_market().items():
        if row.get("resolved_outcome") is None and row.get("resolution_raw") is None:
            try:
                m = client.market(mid)
            except Exception as exc:  # a flaky id must not end the run
                log.warning("re-check failed for %s: %s", mid, exc)
                continue
            if m is not None:
                tracked[mid] = m
    candidates = {m.market_id: m for m in client.search_open_binary(pages=search_pages)
                  if m.market_id not in tracked and eligible(m, up, now)}
    markets = list(tracked.values()) + list(candidates.values())
    # description is not in search rows; fetch it once for a new market only
    # when the Oracle is about to be asked (see run()).
    markets.sort(key=lambda m: m.end_date or "9999")
    return markets


# ── run ─────────────────────────────────────────────────────────────────────

def run(client: ManifoldClient, oracle: OracleClient, ledger: Ledger, p: Params, up: UniverseParams,
        *, now: Optional[datetime] = None, dry_run: bool = False) -> dict:
    now = now or datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%SZ")
    markets = gather_universe(client, ledger, up, now)
    latest = ledger.latest_snapshot_by_market()
    open_ids = ledger.open_trade_market_ids()
    log.info("run %s: %d markets, params=%s", run_id, len(markets), p.as_dict())

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
                    if not m.description:  # search rows carry no description
                        full = client.market(m.market_id)
                        if full is not None:
                            m = full
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
        notional_mana=float(g("NOTIONAL_MANA", "100")),
    )


def universe_params_from_env(env: dict = os.environ) -> UniverseParams:  # type: ignore[assignment]
    g = env.get
    return UniverseParams(
        min_bettors=int(g("MIN_BETTORS", "15")),
        min_volume=float(g("MIN_VOLUME", "500")),
        min_hours_to_close=float(g("MIN_HOURS_TO_CLOSE", "12")),
        max_days_to_close=float(g("MAX_DAYS_TO_CLOSE", "30")),
    )


def main() -> None:
    dry_run = os.environ.get("DRY_RUN", "").lower() in ("1", "true", "yes")
    api_key = os.environ.get("ORACLE_API_KEY") or ("dry-run" if dry_run else _required_env("ORACLE_API_KEY"))
    base_url = os.environ.get("ORACLE_BASE_URL", "https://oracle.daatan.com")
    ledger = Ledger(Path(os.environ.get("PAPER_LEDGER_DIR", "./data")))
    with ManifoldClient() as client, OracleClient(api_key, base_url=base_url) as oracle:
        run(client, oracle, ledger, params_from_env(), universe_params_from_env(), dry_run=dry_run)


if __name__ == "__main__":
    main()
