"""Per-row Jev pass-1 value, persisted so a pooled estimate can be recomputed with it (retro#955).

Jev's pass-1 ``max_noul`` (retro#850) was only ever written to ``oracle_log.txt``; the evidence
pool rows carry the Nova ``relevance_score`` and nothing else, so "what would the pooled
estimate be if relevance were Jev's number" (retro#852) needed a log-parsing backfill. This
store keeps ``(url, q8) -> max_noul`` where ``q8`` is the ``question_fingerprint`` of the claim
text, the same key ``event=jev_gate`` already logs.

Used only by the shadow comparison in ``run_pool_aggregate`` (``pool_shadow_a_enabled``);
nothing live reads it. Same diskcache shape as ``extraction_memo_store``: one directory under
``data_dir``, shared by every gunicorn worker. Writes never expire (a value is a property of
the article text and the question, not of time); the size bound is LRU. Every operation fails
open — a store problem degrades to "no Jev value", never breaks extraction or a recompute.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import diskcache

logger = logging.getLogger(__name__)

_stores: dict[str, diskcache.Cache] = {}

# A value is ~100 bytes of key + a float. ~650 extractor calls/day -> a few MB per year.
_SIZE_LIMIT_BYTES = 64 * 1024 * 1024


def _get_store(path: Path) -> diskcache.Cache:
    key = str(path)
    store = _stores.get(key)
    if store is None:
        store = diskcache.Cache(key, size_limit=_SIZE_LIMIT_BYTES)
        _stores[key] = store
    return store


def _key(url: str, q8: str) -> str:
    return f"jev1\x1f{q8}\x1f{url}"


def put_max_noul(path: Path, url: Optional[str], q8: str, max_noul: Optional[float]) -> None:
    """Record one pass-1 verdict. A no-op without a url, a question fingerprint or a value."""
    if not url or not q8 or max_noul is None:
        return
    try:
        _get_store(path).set(_key(url, q8), float(max_noul))
    except Exception:  # noqa: BLE001 - fail open
        logger.warning("event=jev_value_store_error op=put", exc_info=True)


def get_max_noul(path: Path, url: Optional[str], q8: str) -> Optional[float]:
    if not url or not q8:
        return None
    try:
        v = _get_store(path).get(_key(url, q8))
    except Exception:  # noqa: BLE001 - fail open
        logger.warning("event=jev_value_store_error op=get", exc_info=True)
        return None
    return float(v) if isinstance(v, (int, float)) else None
