"""Event decomposition store — decide each question's WHO/WHAT/SCOPE once (retro#758).

retro#697 measured that emitting the extractor's own WHO/WHAT/SCOPE decomposition
as *output* moves no pins (``docs/SETTLED_DECISION_AB.md``). retro#758 tried the
opposite direction — the same decomposition as *input*, appended to the RELATED
EVENT block the extractor already sees — and got the weak rater's best result on
the deadline/denial sentinel case (11/15 -> 15/15, Nova Lite) with no prompt edit
and no schema change: the decomposition is data appended to ``event_description``,
not a change to ``PROMPT_PREFIX``/``PROMPT_SUFFIX``.

The decomposition depends only on the question and its resolution criteria, not
on any one article, so it is computed once per question and reused across every
article in a ``/forecast`` batch — same shape as ``settlement_verdict_store.py``
(retro#532): a ``diskcache.Cache`` directory under ``data_dir``, shared by every
gunicorn worker, surviving reloads and deploys. Every operation fails open (log
and carry on) — a full disk or a cache bug must degrade to "no decomposition
appended", the pre-#758 behaviour, never break extraction.
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

# Entries are one short line (~150 bytes); this bounds the store at roughly
# 400k questions with LRU eviction, far beyond what /forecast will ever see.
_SIZE_LIMIT_BYTES = 64 * 1024 * 1024


def _get_store(path: Path) -> diskcache.Cache:
    key = str(path)
    store = _stores.get(key)
    if store is None:
        store = diskcache.Cache(key, size_limit=_SIZE_LIMIT_BYTES)
        _stores[key] = store
    return store


def decomposition_key(question: str, resolution_criteria: Optional[str], *, model: str) -> str:
    """One question's identity. Hashing the exact inputs (rather than a separate
    question id) means an edited question or resolution_criteria naturally
    invalidates the cache — no version constant to remember to bump."""
    payload = "\x1f".join((model, question, resolution_criteria or ""))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


async def get_decomposition(path: Path, key: str) -> Optional[str]:
    try:
        entry = await asyncio.to_thread(_get_store(path).get, key)
    except Exception:  # noqa: BLE001 - fail open, never break extraction
        logger.warning("event=event_decomposition_store_error op=get", exc_info=True)
        return None
    if entry is not None and not isinstance(entry, str):
        logger.warning("event=event_decomposition_store_error op=get reason=malformed_entry")
        return None
    return entry


async def put_decomposition(path: Path, key: str, decomposition: str) -> None:
    try:
        await asyncio.to_thread(_get_store(path).set, key, decomposition)
    except Exception:  # noqa: BLE001 - fail open, never break extraction
        logger.warning("event=event_decomposition_store_error op=put", exc_info=True)
