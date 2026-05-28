"""
SimHash-based near-duplicate detection for vault2 articles.

Two articles are considered near-duplicates when their SimHash Hamming distance
is ≤ HAMMING_THRESHOLD.  At threshold=8 (out of 64 bits) this catches syndicated
rewrites that share ≥ ~85% of word 3-grams while ignoring articles that merely
cover the same topic.

The index is stored in vault2/simhash_index.json as {simhash_hex: article_hash}.
New articles are checked before LLM extraction — if a near-duplicate exists,
the new article's hash is mapped to the existing article's hash so the
extraction result is reused.

No external dependencies beyond Python stdlib and numpy (already in pipeline).
"""
import hashlib
import json
import logging
import re
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

HAMMING_THRESHOLD = 8   # bits different — tune upward if too many false positives
_BITS = 64


def _shingles(text: str, n: int = 3) -> list[str]:
    """Word n-grams from normalised text."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]


def simhash(text: str) -> int:
    """Compute a 64-bit SimHash fingerprint of text."""
    v = np.zeros(_BITS, dtype=np.int64)
    for shingle in _shingles(text):
        h = int(hashlib.md5(shingle.encode()).hexdigest(), 16)
        for bit in range(_BITS):
            v[bit] += 1 if (h >> bit) & 1 else -1
    result = 0
    for bit in range(_BITS):
        if v[bit] > 0:
            result |= 1 << bit
    return result


def hamming(a: int, b: int) -> int:
    """Number of differing bits between two 64-bit integers."""
    return bin(a ^ b).count("1")


class SimhashIndex:
    """
    Lightweight on-disk index: simhash_hex → article_hash.

    Usage:
        idx = SimhashIndex(vault_dir)
        canonical = idx.find_near_duplicate(text)   # None or existing art_hash
        idx.add(art_hash, text)                      # register new article
        idx.save()                                   # flush to disk
    """

    def __init__(self, vault_dir: Path):
        self._path = vault_dir / "simhash_index.json"
        self._index: dict[str, str] = {}  # simhash_hex → article_hash
        self._dirty = False
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                self._index = json.loads(self._path.read_text())
                logger.debug("SimhashIndex: loaded %d entries", len(self._index))
            except Exception as exc:
                logger.warning("SimhashIndex: could not load index: %s — starting empty", exc)

    def save(self) -> None:
        if not self._dirty:
            return
        self._path.write_text(json.dumps(self._index))
        self._dirty = False

    def find_near_duplicate(self, text: str) -> str | None:
        """
        Return the canonical article_hash if a near-duplicate already exists,
        otherwise None.
        """
        if not self._index:
            return None
        h = simhash(text)
        for stored_hex, art_hash in self._index.items():
            stored = int(stored_hex, 16)
            if hamming(h, stored) <= HAMMING_THRESHOLD:
                return art_hash
        return None

    def add(self, art_hash: str, text: str) -> None:
        """Register a new article in the index."""
        h = simhash(text)
        self._index[format(h, "016x")] = art_hash
        self._dirty = True
