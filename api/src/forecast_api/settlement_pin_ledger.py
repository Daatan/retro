"""Settlement-pin ledger — retro#361 Phase 1.

Records, at ingest time (``POST /leaderboard/ingest``), every settlement
pin's snapshot (what the pin said — direction, mean, CI, suppression state)
alongside the eventual resolved outcome, so "pins contradicted by
resolution" becomes queryable. See ``docs/ORACLE_VARIABLES.md`` §9 and
retro#361 for the originating quality-loop proposal.

Same diskcache-backed, cross-worker-safe shape as ``resolution_feedback.py``
(retro#434 / PR#450): one append-only JSONL file plus a ``diskcache.Cache``
dedup directory next to it, shared by every gunicorn worker on the box.
``diskcache.Cache.add()`` is atomic across processes (a SQLite transaction),
so two workers racing on the same prediction_id can't both write a row.

Deliberately NOT built here: the classification heuristic (extraction error
/ question semantics / genuine miss / pin-correct) the originating issue
also proposes. That requires its own review of real contradicted pins and is
tracked as separate follow-up work — this module only records the raw
pin-vs-outcome pairing and reports which pins disagree.

Only settled pins are recorded — an ingest payload with no
``settlement_snapshot``, or one whose snapshot has ``settled=False`` (the
pool never pinned), has nothing to post-mortem and is silently skipped.
"""
import asyncio
import json
import logging
from pathlib import Path

import diskcache

from .models import IngestResolutionRequest, SettlementPinLedgerEntry

logger = logging.getLogger(__name__)

_stores: dict[str, diskcache.Cache] = {}
_loaded_paths: set[str] = set()
_lock = asyncio.Lock()


def _dedup_dir(path: Path) -> Path:
    return path.parent / f".{path.stem}_dedup_cache"


def _get_store(path: Path) -> diskcache.Cache:
    key = str(path)
    store = _stores.get(key)
    if store is None:
        store = diskcache.Cache(str(_dedup_dir(path)))
        _stores[key] = store
    return store


def _load_ids_from_disk(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            ids.add(json.loads(line)["prediction_id"])
        except (json.JSONDecodeError, KeyError):
            logger.warning("Skipping malformed settlement-pin-ledger line %d in %s", lineno, path)
    return ids


def _load_records_from_disk(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records: list[dict] = []
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            logger.warning("Skipping malformed settlement-pin-ledger line %d in %s", lineno, path)
    return records


def _append_line_to_disk(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(line + "\n")


async def _ensure_loaded(path: Path) -> None:
    """Backfill the dedup store from the JSONL file — needed once per store
    directory to cover prediction_ids recorded before this store existed (or
    by an older deploy). ``store.add`` is idempotent, so redundant backfills
    across racing workers are harmless."""
    key = str(path)
    if key in _loaded_paths:
        return
    async with _lock:
        if key in _loaded_paths:
            return
        store = _get_store(path)
        ids = await asyncio.to_thread(_load_ids_from_disk, path)

        def _backfill() -> None:
            for prediction_id in ids:
                store.add(prediction_id, True)

        await asyncio.to_thread(_backfill)
        _loaded_paths.add(key)
        logger.info("Settlement-pin ledger backfilled: %d pins already on record", len(ids))


async def record_settlement_pin(path: Path, req: IngestResolutionRequest) -> bool:
    """Record ``req.settlement_snapshot`` vs ``req.outcome``, if there is one
    to record. Returns True iff a ledger row was written — False when the
    payload carries no snapshot, the snapshot was never settled (no pin to
    post-mortem), or ``prediction_id`` is already on record (idempotent,
    same atomic-across-workers guard as
    ``resolution_feedback.ingest_resolution``).
    """
    snapshot = req.settlement_snapshot
    if snapshot is None or not snapshot.settled:
        return False

    await _ensure_loaded(path)
    store = _get_store(path)

    added = await asyncio.to_thread(store.add, req.prediction_id, True)
    if not added:
        return False

    pin_direction = "yes" if snapshot.mean >= 0 else "no"
    outcome_direction = "yes" if req.outcome else "no"
    entry = SettlementPinLedgerEntry(
        prediction_id=req.prediction_id,
        outcome=req.outcome,
        resolved_at=req.resolved_at,
        pin_mean=snapshot.mean,
        pin_ci_low=snapshot.ci_low,
        pin_ci_high=snapshot.ci_high,
        pin_direction=pin_direction,
        settled_sources=snapshot.settled_sources,
        settlement_suppressed=snapshot.settlement_suppressed,
        settlement_suppression_reason=snapshot.settlement_suppression_reason,
        contradicted=pin_direction != outcome_direction,
    )
    await asyncio.to_thread(_append_line_to_disk, path, entry.model_dump_json())

    if entry.contradicted:
        logger.warning(
            "Settlement pin contradicted by resolution: prediction_id=%s pin_direction=%s outcome=%s",
            req.prediction_id, pin_direction, outcome_direction,
        )
    return True


async def load_ledger(path: Path) -> list[SettlementPinLedgerEntry]:
    """All recorded settlement-pin entries, freshly read from disk — the
    report always reflects what's actually on disk rather than the
    in-process dedup store, which only tracks ids, not the full rows."""
    records = await asyncio.to_thread(_load_records_from_disk, path)
    entries: list[SettlementPinLedgerEntry] = []
    for lineno, record in enumerate(records, start=1):
        try:
            entries.append(SettlementPinLedgerEntry(**record))
        except Exception:
            logger.warning("Skipping unparsable settlement-pin-ledger record %d in %s", lineno, path)
    return entries


async def contradicted_pins(path: Path) -> list[SettlementPinLedgerEntry]:
    """The report retro#361 Phase 1 asks for: settlement pins whose declared
    direction disagrees with the eventual resolution."""
    entries = await load_ledger(path)
    return [e for e in entries if e.contradicted]


def recorded_count(path: Path) -> int:
    return len(_get_store(path))
