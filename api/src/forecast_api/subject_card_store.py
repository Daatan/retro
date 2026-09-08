"""Subject card store — derive each question's subject card once (retro#805).

Same shape as ``event_decomposition_store`` (retro#758) and ``settlement_verdict_store``
(retro#532): a ``diskcache.Cache`` directory under ``data_dir``, shared by every gunicorn
worker, surviving reloads and deploys. The card depends only on the question, its
resolution criteria, the model and the prompt version, so one derivation serves every
article in every ``/forecast`` batch for that question. Every operation fails open (log
and carry on): a full disk or a cache bug must degrade to "no subject card → gate
skipped", never break extraction.

Entries are the card's JSON; an entry that no longer parses (schema change) reads as a
miss, so the card is re-derived rather than trusted.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
from pathlib import Path
from typing import Optional

import diskcache

from .subject_card import SUBJECT_CARD_PROMPT_VERSION, SubjectCard

logger = logging.getLogger(__name__)

_stores: dict[str, diskcache.Cache] = {}

# A card is ~1 KB; this bounds the store at ~64k questions with LRU eviction.
_SIZE_LIMIT_BYTES = 64 * 1024 * 1024


def _get_store(path: Path) -> diskcache.Cache:
    key = str(path)
    store = _stores.get(key)
    if store is None:
        store = diskcache.Cache(key, size_limit=_SIZE_LIMIT_BYTES)
        _stores[key] = store
    return store


def subject_card_key(question: str, resolution_criteria: Optional[str], *, model: str) -> str:
    """Hash the exact inputs — an edited question or criteria naturally invalidates the
    entry; the prompt version is in the key so a prompt change does too."""
    payload = "\x1f".join((SUBJECT_CARD_PROMPT_VERSION, model, question, resolution_criteria or ""))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


async def get_subject_card(path: Path, key: str) -> Optional[SubjectCard]:
    try:
        entry = await asyncio.to_thread(_get_store(path).get, key)
    except Exception:  # noqa: BLE001 - fail open, never break extraction
        logger.warning("event=subject_card_store_error op=get", exc_info=True)
        return None
    if entry is None:
        return None
    try:
        return SubjectCard.model_validate_json(entry)
    except Exception:  # noqa: BLE001 - a stale/malformed entry is a miss
        logger.warning("event=subject_card_store_error op=get reason=malformed_entry")
        return None


async def put_subject_card(path: Path, key: str, card: SubjectCard) -> None:
    try:
        await asyncio.to_thread(_get_store(path).set, key, card.model_dump_json())
    except Exception:  # noqa: BLE001 - fail open, never break extraction
        logger.warning("event=subject_card_store_error op=put", exc_info=True)
