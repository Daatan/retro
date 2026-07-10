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

Idempotent on prediction_id: an in-memory set (loaded from disk at first
use, same lazy-cache shape as leaderboard.py's _cache) guards against
double-counting a fire-and-forget retry from the daatan side.
"""
import asyncio
import json
import logging
from pathlib import Path

from .models import IngestResolutionRequest, IngestResolutionResponse

logger = logging.getLogger(__name__)

_ingested_ids: set[str] = set()
_lock = asyncio.Lock()
_loaded = False


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
    global _loaded
    if _loaded:
        return
    async with _lock:
        if _loaded:
            return
        loaded = await asyncio.to_thread(_load_ids_from_disk, path)
        _ingested_ids.update(loaded)
        _loaded = True
        logger.info("Resolution feedback store loaded: %d predictions already ingested", len(_ingested_ids))


async def ingest_resolution(path: Path, req: IngestResolutionRequest) -> IngestResolutionResponse:
    await _ensure_loaded(path)

    async with _lock:
        if req.prediction_id in _ingested_ids:
            return IngestResolutionResponse(already_ingested=True, sources_recorded=0)

        record = {
            "prediction_id": req.prediction_id,
            "outcome": req.outcome,
            "resolved_at": req.resolved_at,
            "sources": [s.model_dump() for s in req.sources],
        }
        await asyncio.to_thread(_append_line_to_disk, path, json.dumps(record))
        _ingested_ids.add(req.prediction_id)

    return IngestResolutionResponse(already_ingested=False, sources_recorded=len(req.sources))


def ingested_count() -> int:
    return len(_ingested_ids)
