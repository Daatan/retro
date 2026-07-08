"""
In-process TTL cache for ForecastResponse.

Scope: the Oracle API runs with a single uvicorn worker on one EC2 instance, so
a process-local dict is a correct cache. If we ever go multi-worker or
multi-instance, swap the backend to Redis/diskcache; the public surface here
(``get`` / ``set`` / ``stats`` / ``clear``) is intentionally small so that
migration is mechanical.

Design choices:
  * Key = sha256(normalized_question | max_articles). Normalization is
    ``question.strip().casefold()``, so trivial whitespace/casing variants
    collapse to the same entry.
  * TTL check happens on read — we do not have a background sweeper. With
    ``cache_max_entries`` entries max the memory footprint is bounded even if
    nothing is ever read.
  * Lazy LRU eviction: when we exceed ``max_entries`` on insert we drop the
    oldest entry by insertion order (``OrderedDict.popitem(last=False)``).
  * ``placeholder=True`` responses are **not cached**. They represent "no
    articles found" situations and caching them would turn a transient
    upstream failure into a 1-hour outage for that question.
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock
from typing import Optional

from .models import ForecastResponse


@dataclass
class _Entry:
    response: ForecastResponse
    expires_at: float


@dataclass
class _SearchEntry:
    results: list
    expires_at: float


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    stores: int = 0
    evictions: int = 0
    size: int = 0

    def as_dict(self) -> dict:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "stores": self.stores,
            "evictions": self.evictions,
            "size": self.size,
        }


class ForecastCache:
    """Bounded TTL cache for ``ForecastResponse`` objects."""

    def __init__(self, *, ttl_seconds: int, max_entries: int) -> None:
        self._ttl = max(0, ttl_seconds)
        self._max = max(1, max_entries)
        self._data: "OrderedDict[str, _Entry]" = OrderedDict()
        self._lock = Lock()
        self._stats = CacheStats()

    @property
    def enabled(self) -> bool:
        return self._ttl > 0

    @staticmethod
    def make_key(
        question: str,
        max_articles: Optional[int],
        articles_hash: Optional[str] = None,
        claim_meta: Optional[str] = None,
    ) -> str:
        """``claim_meta`` folds the request's temporal metadata (claim_direction +
        claim_deadline) into the key — the settlement direction guard makes the
        answer depend on it, so a metadata-less response must not be served to a
        metadata-bearing request (or vice versa)."""
        normalized = question.strip().casefold()
        payload = f"{normalized}|{max_articles or ''}|{articles_hash or ''}|{claim_meta or ''}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def get(self, key: str) -> Optional[ForecastResponse]:
        if not self.enabled:
            return None
        now = time.time()
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                self._stats.misses += 1
                return None
            if entry.expires_at <= now:
                # Expired — drop and miss.
                del self._data[key]
                self._stats.misses += 1
                self._stats.size = len(self._data)
                return None
            # Refresh LRU position on hit.
            self._data.move_to_end(key)
            self._stats.hits += 1
            return entry.response

    def set(self, key: str, response: ForecastResponse) -> None:
        if not self.enabled:
            return
        # Never cache placeholder/empty responses; see module docstring.
        if response.placeholder:
            return
        now = time.time()
        with self._lock:
            self._data[key] = _Entry(response=response, expires_at=now + self._ttl)
            self._data.move_to_end(key)
            self._stats.stores += 1
            while len(self._data) > self._max:
                self._data.popitem(last=False)
                self._stats.evictions += 1
            self._stats.size = len(self._data)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self._stats.size = 0

    def stats(self) -> CacheStats:
        with self._lock:
            # Return a snapshot so callers can't mutate internal state.
            return CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                stores=self._stats.stores,
                evictions=self._stats.evictions,
                size=len(self._data),
            )


class SearchCache:
    """Bounded TTL cache for raw search results (list[SearchResult]).

    Keyed on sha256(normalized_question | limit). Separate from ForecastCache
    so the same article set can be reused across multiple forecast calls for
    the same question even after the 1-hour forecast TTL expires. Default TTL
    is 4 hours — news search results are stable within that window.

    Empty result lists are never cached; a failed search should retry rather
    than returning a stale empty list for hours.
    """

    def __init__(self, *, ttl_seconds: int, max_entries: int) -> None:
        self._ttl = max(0, ttl_seconds)
        self._max = max(1, max_entries)
        self._data: "OrderedDict[str, _SearchEntry]" = OrderedDict()
        self._lock = Lock()

    @property
    def enabled(self) -> bool:
        return self._ttl > 0

    @staticmethod
    def make_key(question: str, limit: int) -> str:
        normalized = question.strip().casefold()
        return hashlib.sha256(f"{normalized}|{limit}".encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[list]:
        if not self.enabled:
            return None
        now = time.time()
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            if entry.expires_at <= now:
                del self._data[key]
                return None
            self._data.move_to_end(key)
            return entry.results

    def set(self, key: str, results: list) -> None:
        if not self.enabled or not results:
            return
        now = time.time()
        with self._lock:
            self._data[key] = _SearchEntry(results=results, expires_at=now + self._ttl)
            self._data.move_to_end(key)
            while len(self._data) > self._max:
                self._data.popitem(last=False)


def build_cache_from_settings() -> ForecastCache:
    """Construct the process-wide forecast cache from :mod:`.config` settings."""
    from .config import settings
    return ForecastCache(
        ttl_seconds=settings.cache_ttl_seconds,
        max_entries=settings.cache_max_entries,
    )


def build_search_cache_from_settings() -> SearchCache:
    """Construct the process-wide search cache from :mod:`.config` settings."""
    from .config import settings
    return SearchCache(
        ttl_seconds=settings.search_cache_ttl_seconds,
        max_entries=settings.search_cache_max_entries,
    )


# Process-wide singletons. Imported by forecaster and exposed via /health.
forecast_cache: ForecastCache = build_cache_from_settings()
search_cache: SearchCache = build_search_cache_from_settings()
