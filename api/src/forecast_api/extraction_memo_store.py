"""Extraction memo — reuse the extractor's raw output for an identical call (retro#895).

Measured 09-26→10-02 (retro#849 comment): 28% of extractor calls repeat a (url, question)
pair already extracted in the previous 24 h, 36% within the 7-day window — nearly all from
the bayesoracle series cron and the paper bot (retro#620), whose daily re-asks always miss the
1-hour whole-response ``forecast_cache``, plus the ``retry_relaxed_search`` shadow re-run
(retro#621), which re-extracts the primary pass's articles. ~$1.9/day at a 24 h TTL.

The key is sha256 over the extractor schema hash, the model id and the FULL text the model is
sent (``PROMPT_PREFIX`` + the rendered suffix with article text, question, event description /
resolution criteria, deadline, article date, short-form / language / conditional tails). Any
change to any of those misses naturally — no separate version constant to forget to bump —
the same way ``settlement_verdict_store`` keys on its built prompt.

The value is the model's RAW ``ExtractionOutput`` JSON, stored before any post-processing.
Callers run the full ``enforce_*`` chain, Jev shadow/class and every audit on a hit exactly as
on a miss, so a fix to post-processing (retro#879/#881/#885 changed it without touching the
prompt) applies to memoised extractions immediately. Haiku at temperature 0 is not perfectly
deterministic (retro#532), so a hit freezes one roll per exact input for the TTL.

Same diskcache shape as ``subject_card_store`` / ``settlement_verdict_store``: one directory
under ``data_dir``, shared by every gunicorn worker, surviving reloads and deploys. Every
operation fails open: a full disk or a store bug degrades to "call the model", never breaks
extraction. Deleting the directory on the box is the manual invalidation lever.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
from pathlib import Path
from typing import Optional

import diskcache

logger = logging.getLogger(__name__)

_stores: dict[str, diskcache.Cache] = {}

# An entry is the raw ExtractionOutput JSON, typically 1-8 KB. ~650 extractor calls/day at a
# 24 h TTL is a few MB; the bound (LRU eviction) leaves room for a 7-day TTL and stops a
# runaway caller filling the data disk.
_SIZE_LIMIT_BYTES = 256 * 1024 * 1024


def _get_store(path: Path) -> diskcache.Cache:
    key = str(path)
    store = _stores.get(key)
    if store is None:
        store = diskcache.Cache(key, size_limit=_SIZE_LIMIT_BYTES)
        _stores[key] = store
    return store


class ExtractionMemoStore:
    """Implements ``tm.extractor.ExtractionMemo`` over a diskcache directory."""

    def __init__(self, path: Path, *, ttl_seconds: float, schema_hash: str) -> None:
        self.path = path
        self.ttl_seconds = ttl_seconds
        self.schema_hash = schema_hash

    def key(self, *, model: str, prompt: str) -> str:
        payload = "\x1f".join(("extract_memo/v1", self.schema_hash, model, prompt))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    async def get(self, key: str) -> Optional[str]:
        try:
            entry = await asyncio.to_thread(_get_store(self.path).get, key)
        except Exception:  # noqa: BLE001 - fail open, never break extraction
            logger.warning("event=extract_memo_error op=get", exc_info=True)
            return None
        if entry is not None and not isinstance(entry, str):
            logger.warning("event=extract_memo_error op=get reason=malformed_entry")
            return None
        return entry

    async def put(self, key: str, value: str) -> None:
        try:
            await asyncio.to_thread(
                _get_store(self.path).set, key, value, expire=self.ttl_seconds,
            )
        except Exception:  # noqa: BLE001 - fail open, never break extraction
            logger.warning("event=extract_memo_error op=put", exc_info=True)
