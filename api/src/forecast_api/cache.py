"""
Disk-backed TTL cache for ForecastResponse, shared across workers.

The Oracle API runs ``gunicorn --workers 2`` (``infra/oracle-api.service``). The
first backend here was a process-local dict whose docstring called the design
"correct" on a single-worker premise the deployment had already broken
(retro#405): each worker had its own entries and counters, repeat-question hits
were roughly halved, ``_inflight`` could not coalesce across processes, and a
SIGHUP reload emptied everything — which also capped daatan#1262's re-ask (the
120 s retry found the completed run only when it landed on the worker that held
it). The backend is now ``diskcache``: one SQLite-backed directory shared by
both workers on the box, surviving reloads, no new infrastructure — exactly the
migration the old module prescribed for itself. Redis becomes the answer only
if the API ever goes multi-instance.

What is and is not shared now:
  * **Entries and hit/miss/store counters: shared and cumulative.** They live in
    ``settings.cache_dir`` (under ``/tmp`` — the unit sets no ``PrivateTmp``, so
    this survives SIGHUP reloads and full service restarts, clearing only on
    reboot). ``/health`` no longer round-robins between two disjoint counter sets.
  * **``_inflight`` (forecaster.py): still per-worker.** Two identical
    simultaneous requests on different workers still both run the pipeline; the
    loser's result now at least lands in the shared cache for the next caller.
  * **``stats().evictions`` is always 0.** diskcache culls internally by byte
    volume (``cache_size_limit_mb``, least-recently-used) and does not expose a
    cull count. The old entry-count bound (``max_entries``) went with it.

Design choices kept from the first backend:
  * Key = sha256(normalized_question | max_articles | articles_hash |
    claim_meta). Normalization is ``question.strip().casefold()``.
  * ``placeholder=True`` responses are **not** cached — caching a "no articles
    found" response would turn a transient upstream failure into a 1-hour
    outage for that question.
  * ``cache_ttl_seconds=0`` disables the cache entirely (no directory is even
    opened).

New failure mode, handled: values pickle through diskcache, and a deploy can
change ``ForecastResponse``'s shape so an entry written by the previous code
no longer unpickles. An unreadable entry is deleted and treated as a miss,
never an error — entries only live an hour.

``test_cache.py`` pins the deployed worker count against this docstring so the
two cannot silently diverge again (that drift is how retro#405 happened).
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock
from typing import Optional

import diskcache

from .models import ForecastResponse


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
    """TTL cache for ``ForecastResponse`` objects, shared across workers via diskcache."""

    def __init__(self, *, ttl_seconds: int, directory: str, size_limit_bytes: int = 128 * 1024 * 1024) -> None:
        self._ttl = max(0, ttl_seconds)
        self._cache: Optional[diskcache.Cache] = None
        self._meta: Optional[diskcache.Cache] = None
        if self._ttl > 0:
            self._cache = diskcache.Cache(
                directory,
                statistics=1,
                eviction_policy="least-recently-used",
                size_limit=size_limit_bytes,
            )
            # The stores counter lives in a sidecar cache (statistics disabled):
            # reading it from the main cache would count a hit/miss of its own
            # every time /health asks for stats.
            self._meta = diskcache.Cache(f"{directory}/meta", statistics=0)

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
        if self._cache is None:
            return None
        try:
            return self._cache.get(key, default=None)
        except Exception:
            # Unreadable pickle from an older deploy's model shape — miss, not error.
            try:
                self._cache.delete(key)
            except Exception:
                pass
            return None

    def set(self, key: str, response: ForecastResponse) -> None:
        if self._cache is None or self._meta is None:
            return
        # Never cache placeholder/empty responses; see module docstring.
        if response.placeholder:
            return
        self._cache.set(key, response, expire=self._ttl)
        self._meta.incr("stores", default=0)

    def clear(self) -> None:
        """Drop all entries AND reset every counter — hits/misses/stores start over."""
        if self._cache is None or self._meta is None:
            return
        self._cache.clear()
        self._cache.stats(reset=True)
        self._meta.clear()

    def stats(self) -> CacheStats:
        if self._cache is None or self._meta is None:
            return CacheStats()
        # Cull expired rows first so `size` counts live entries, not corpses —
        # diskcache's len() would otherwise include expired-but-unculled rows.
        self._cache.expire()
        hits, misses = self._cache.stats()
        return CacheStats(
            hits=int(hits),
            misses=int(misses),
            stores=int(self._meta.get("stores", default=0) or 0),
            evictions=0,
            size=len(self._cache),
        )


class SearchCache:
    """Bounded TTL cache for raw search results (list[SearchResult]).

    Keyed on sha256(normalized_question | limit). Separate from ForecastCache
    so the same article set can be reused across multiple forecast calls for
    the same question even after the 1-hour forecast TTL expires. Default TTL
    is 4 hours — news search results are stable within that window.

    Still process-local (per-worker), unlike ForecastCache: a duplicated search
    costs one provider round-trip, not a paid LLM extraction, so the shared
    backend wasn't worth the pickle churn here. Revisit if provider spend says
    otherwise.

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
    """Construct the shared (cross-worker) forecast cache from :mod:`.config` settings."""
    from .config import settings
    return ForecastCache(
        ttl_seconds=settings.cache_ttl_seconds,
        directory=settings.cache_dir,
        size_limit_bytes=settings.cache_size_limit_mb * 1024 * 1024,
    )


def build_search_cache_from_settings() -> SearchCache:
    """Construct the process-wide search cache from :mod:`.config` settings."""
    from .config import settings
    return SearchCache(
        ttl_seconds=settings.search_cache_ttl_seconds,
        max_entries=settings.search_cache_max_entries,
    )


# Singletons. forecast_cache is one shared store: each worker process builds its
# own diskcache handle, but they all point at settings.cache_dir. Imported by
# forecaster and exposed via /health.
forecast_cache: ForecastCache = build_cache_from_settings()
search_cache: SearchCache = build_search_cache_from_settings()
