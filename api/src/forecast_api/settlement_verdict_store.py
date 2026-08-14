"""Settlement verdict store — decide each match-gate verdict once (retro#532).

The match gate's verdict (``settlement_verifier.py``) is an LLM judgment, and
repeating that judgment is not idempotent even at ``temperature=0``: measured
over every gated question to 2026-08-14, 6 of the 13 questions the gate ever
saw more than once returned BOTH verdicts on an unchanged vote-set. Against
daatan's one-way ``settled`` latch that re-roll is a ratchet — a question the
gate vetoes on 31 of 32 rolls still pins permanently on its one lucky YES
(the Netanyahu row: 1 True / 31 False, latched).

So the verdict is decided once per input and remembered. The key covers the
exact prompt the model would see (question, direction and votes, prefix text
included — so editing the prompt invalidates naturally) plus the model, the
sample count, and the settlement config fingerprint. A cached YES and a
cached NO are equally durable: determinism is the property being bought, and
a veto that could be out-rolled by recomputing until it flips would be no
latch at all. A hit means the gate does not re-ask until the vote-set, the
config, the model or the prompt changes — which is exactly retro#532's
option A ("a veto is sticky") implemented through option C (the cache).

Two things are deliberately never cached (the gate enforces this; the store
just holds what it is given):

- **Errored verdicts.** Fail-open (``settles=True, errored=True``) exists so
  an LLM outage never suppresses a pin; caching one would turn a transient
  timeout into a permanent pin-keeper.
- **Undecided rolls** — a sample set where any call errored or the decided
  samples tied. The decision procedure didn't complete; the next recompute
  rolls fresh.

Same diskcache shape as ``settlement_pin_ledger.py``'s dedup store
(retro#434 pattern): one ``diskcache.Cache`` directory under ``data_dir``,
shared by every gunicorn worker (SQLite transactions make writes atomic
across processes), surviving reloads and deploys — a verdict store that
reset on every deploy would re-roll exactly when recomputes are busiest.
Every operation fails open (log and carry on): a full disk must degrade to
the pre-store behaviour, never break pinning.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import diskcache

logger = logging.getLogger(__name__)

_stores: dict[str, diskcache.Cache] = {}

# Verdict entries are ~300 bytes; this bounds the store at roughly 200k
# verdicts with LRU eviction, far beyond the ~33 pins production has ever
# published. A bound exists so a runaway caller can't fill the data disk.
_SIZE_LIMIT_BYTES = 64 * 1024 * 1024


def _get_store(path: Path) -> diskcache.Cache:
    key = str(path)
    store = _stores.get(key)
    if store is None:
        store = diskcache.Cache(key, size_limit=_SIZE_LIMIT_BYTES)
        _stores[key] = store
    return store


def verdict_key(prompt: str, *, model: str, samples: int, config_fingerprint: str) -> str:
    """One decision's identity. Hashing the built prompt (rather than its
    parts) means anything that changes what the model is asked — the
    question, the direction, a vote's claim/quote/outlet/date, the prefix
    text itself — changes the key, with no separate prompt-version constant
    to forget to bump."""
    payload = "\x1f".join((model, str(samples), config_fingerprint, prompt))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


async def get_verdict(path: Path, key: str) -> Optional[dict]:
    try:
        entry = await asyncio.to_thread(_get_store(path).get, key)
    except Exception:  # noqa: BLE001 - fail open, never break pinning
        logger.warning("event=settlement_verdict_store_error op=get", exc_info=True)
        return None
    if entry is not None and not isinstance(entry, dict):
        logger.warning("event=settlement_verdict_store_error op=get reason=malformed_entry")
        return None
    return entry


async def put_verdict(
    path: Path, key: str, *, settles: bool, reason: str, model: str, agree: int, samples: int,
) -> None:
    entry = {
        "settles": settles,
        "reason": reason,
        "model": model,
        "agree": agree,
        "samples": samples,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        await asyncio.to_thread(_get_store(path).set, key, entry)
    except Exception:  # noqa: BLE001 - fail open, never break pinning
        logger.warning("event=settlement_verdict_store_error op=put", exc_info=True)
