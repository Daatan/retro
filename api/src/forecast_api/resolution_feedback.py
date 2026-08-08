"""
Resolution feedback ingest — credibility feedback loop, step 1
(docs/ORACLE_VARIABLES.md "Open, in suggested order" — resolution-outcome
feedback loop: daatan Prediction resolves -> which sources contributed ->
retro scoring).

Storage only. Accumulates one JSONL line per resolved forecast, pushed by
daatan via POST /leaderboard/ingest. Nothing here computes a score or
touches leaderboard.json / get_credibility_weight() — that's a separate,
not-yet-built step, deliberately kept out of this one so ingestion can ship
and start accumulating real data before the scoring design is settled.

Idempotent on prediction_id: dedup state lives in a diskcache.Cache directory
next to the JSONL file, shared by all gunicorn workers on the box (same fix
as ForecastCache's migration for retro#405 — see cache.py's docstring). The
old design used a per-process in-memory set, which let a fire-and-forget
retry that landed on the *other* worker slip past the "already ingested"
check and double-count a resolution (retro#434). diskcache.Cache.add() is
atomic across processes (backed by a SQLite transaction), so two workers
racing on the same prediction_id can't both win.
"""
import asyncio
import json
import logging
from pathlib import Path

import diskcache

from .models import IngestResolutionRequest, IngestResolutionResponse

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
            logger.warning("Skipping malformed resolution-feedback line %d in %s", lineno, path)
    return ids


def _append_line_to_disk(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(line + "\n")


async def _ensure_loaded(path: Path) -> None:
    """Backfill the dedup store from the JSONL file — needed once per store
    directory to cover prediction_ids ingested before this store existed
    (or by an older deploy). ``store.add`` is idempotent, so redundant
    backfills across racing workers are harmless."""
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
        logger.info("Resolution feedback store backfilled: %d predictions already ingested", len(ids))


async def ingest_resolution(path: Path, req: IngestResolutionRequest) -> IngestResolutionResponse:
    await _ensure_loaded(path)
    store = _get_store(path)

    added = await asyncio.to_thread(store.add, req.prediction_id, True)
    if not added:
        return IngestResolutionResponse(already_ingested=True, sources_recorded=0)

    record = {
        "prediction_id": req.prediction_id,
        "outcome": req.outcome,
        "resolved_at": req.resolved_at,
        "sources": [s.model_dump() for s in req.sources],
        "author_signals": [s.model_dump() for s in req.author_signals],
    }
    await asyncio.to_thread(_append_line_to_disk, path, json.dumps(record))

    return IngestResolutionResponse(
        already_ingested=False,
        sources_recorded=len(req.sources),
        author_signals_recorded=len(req.author_signals),
    )


def ingested_count(path: Path) -> int:
    return len(_get_store(path))
