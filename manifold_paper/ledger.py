"""Append-only JSONL ledger for the paper bot.

Two files in one directory (``PAPER_LEDGER_DIR``, default ``./data``; on the
oracle box ``/home/ubuntu/truthmachine/data/polymarket_paper``):

  snapshots.jsonl  one row per (run, market): market price, Oracle probability,
                   edge, Oracle telemetry. Written for every market in the
                   universe on every run, Oracle call or not — the price series
                   is what the mark-to-market view is built from.
  trades.jsonl     one row per paper position opened. Never edited; a position
                   is "closed" by the market resolving, which the scorecard
                   reads off the latest snapshot.

Plain JSONL rather than the API's diskcache-deduped ledgers because exactly one
writer exists (a systemd oneshot; systemd never overlaps activations of the same
unit), so there is no cross-process race to guard against.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

SNAPSHOTS = "snapshots.jsonl"
TRADES = "trades.jsonl"


class Ledger:
    def __init__(self, directory: Path):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

    # ── writes ──────────────────────────────────────────────────────────────
    def append_snapshot(self, row: dict) -> None:
        self._append(SNAPSHOTS, row)

    def append_trade(self, row: dict) -> None:
        self._append(TRADES, row)

    def _append(self, name: str, row: dict) -> None:
        with (self.dir / name).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    # ── reads ───────────────────────────────────────────────────────────────
    def snapshots(self) -> Iterator[dict]:
        yield from self._read(SNAPSHOTS)

    def trades(self) -> Iterator[dict]:
        yield from self._read(TRADES)

    def _read(self, name: str) -> Iterator[dict]:
        path = self.dir / name
        if not path.exists():
            return
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def latest_snapshot_by_market(self) -> dict[str, dict]:
        """market_id → most recent snapshot row (file order == time order)."""
        latest: dict[str, dict] = {}
        for row in self.snapshots():
            latest[row["market_id"]] = row
        return latest

    def open_trade_market_ids(self) -> set[str]:
        return {t["market_id"] for t in self.trades()}
