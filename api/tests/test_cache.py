"""Unit tests for :mod:`forecast_api.cache`."""

from __future__ import annotations

import pathlib
import re
import time

from forecast_api import cache as cache_module
from forecast_api.cache import ForecastCache
from forecast_api.models import ForecastResponse, SourceSignal


def _resp(*, placeholder: bool = False, question: str = "q", articles: int = 3) -> ForecastResponse:
    return ForecastResponse(
        question=question,
        mean=0.25,
        std=0.1,
        ci_low=0.15,
        ci_high=0.35,
        articles_used=articles,
        sources=[
            SourceSignal(
                source_id="reuters",
                source_name="Reuters",
                url="https://reuters.com/a",
                stance=0.5,
                certainty=0.9,
                credibility_weight=1.2,
                claims=["Something will happen."],
            )
        ],
        placeholder=placeholder,
    )


def _cache(tmp_path, *, ttl_seconds: int = 60) -> ForecastCache:
    return ForecastCache(ttl_seconds=ttl_seconds, directory=str(tmp_path / "cache"))


class TestKeyDerivation:
    def test_same_question_different_casing_and_whitespace_collapse(self):
        k1 = ForecastCache.make_key(" Will X happen? ", 5)
        k2 = ForecastCache.make_key("will x happen?", 5)
        assert k1 == k2

    def test_different_max_articles_produces_different_key(self):
        assert ForecastCache.make_key("q", 5) != ForecastCache.make_key("q", 10)

    def test_none_vs_unset_max_articles_is_the_same_key(self):
        assert ForecastCache.make_key("q", None) == ForecastCache.make_key("q", None)


class TestHitMissPersistence:
    def test_store_then_get_returns_equal_response(self, tmp_path):
        # Equality, not identity: the response pickles through the disk backend.
        cache = _cache(tmp_path)
        key = ForecastCache.make_key("q", 5)
        response = _resp()
        cache.set(key, response)
        assert cache.get(key) == response

    def test_miss_when_empty(self, tmp_path):
        cache = _cache(tmp_path)
        assert cache.get("missing") is None

    def test_stats_counters_track_operations(self, tmp_path):
        cache = _cache(tmp_path)
        key = ForecastCache.make_key("q", 5)
        cache.get(key)  # miss
        cache.set(key, _resp())  # store
        cache.get(key)  # hit
        stats = cache.stats()
        assert stats.hits == 1
        assert stats.misses == 1
        assert stats.stores == 1
        assert stats.size == 1


class TestSharedAcrossWorkers:
    """The point of retro#405: two gunicorn workers must see one cache, and the
    counters must survive a SIGHUP reload instead of resetting per deploy."""

    def test_second_instance_sees_first_instances_entry(self, tmp_path):
        # Two ForecastCache objects on one directory = two workers on one box.
        worker_a = _cache(tmp_path)
        worker_b = _cache(tmp_path)
        key = ForecastCache.make_key("q", 5)
        response = _resp()
        worker_a.set(key, response)
        assert worker_b.get(key) == response

    def test_stats_are_shared_and_survive_reconstruction(self, tmp_path):
        worker_a = _cache(tmp_path)
        worker_b = _cache(tmp_path)
        key = ForecastCache.make_key("q", 5)
        worker_a.get(key)  # miss, counted once for the whole box
        worker_a.set(key, _resp())
        assert worker_b.get(key) is not None  # hit from the other worker

        # A SIGHUP reload constructs fresh objects over the same directory —
        # under the dict backend this zeroed every counter and dropped every entry.
        reloaded = _cache(tmp_path)
        assert reloaded.get(key) is not None
        stats = reloaded.stats()
        assert stats.hits >= 2  # worker_b's hit + reloaded's own
        assert stats.misses == 1
        assert stats.stores == 1
        assert stats.size == 1


class TestTTL:
    def test_expired_entry_is_dropped_on_read(self, tmp_path):
        cache = _cache(tmp_path, ttl_seconds=1)
        key = ForecastCache.make_key("q", 5)
        cache.set(key, _resp())
        time.sleep(1.05)
        assert cache.get(key) is None
        assert cache.stats().size == 0

    def test_ttl_zero_disables_cache(self, tmp_path):
        cache = _cache(tmp_path, ttl_seconds=0)
        assert cache.enabled is False
        cache.set("k", _resp())
        assert cache.get("k") is None
        assert cache.stats().stores == 0
        # Disabled means disabled: not even the cache directory is created.
        assert not (tmp_path / "cache").exists()


class TestPlaceholderRule:
    def test_placeholder_response_is_not_stored(self, tmp_path):
        """A placeholder (no articles) must not poison the cache for an hour."""
        cache = _cache(tmp_path)
        cache.set("k", _resp(placeholder=True))
        assert cache.get("k") is None
        assert cache.stats().stores == 0


class TestClear:
    def test_clear_drops_all_entries_and_resets_counters(self, tmp_path):
        cache = _cache(tmp_path)
        cache.set("a", _resp(question="a"))
        cache.set("b", _resp(question="b"))
        cache.get("a")
        cache.clear()
        stats = cache.stats()
        assert stats.size == 0
        assert stats.hits == 0
        assert stats.stores == 0
        assert cache.get("a") is None


class TestWorkerCountMatchesTheDocstring:
    """The cache design references the deployed worker count: the docstring
    explains what is shared between the 2 workers and what stays per-process
    (``_inflight``). That premise went stale silently once before — the old
    process-local dict documented a single worker while the box ran two
    (retro#405). Pin doc against deployment so they cannot diverge again."""

    @staticmethod
    def _deployed_worker_count() -> int:
        # Read from the unit file that actually starts the API, not from a constant
        # we would have to remember to update — the point is to catch a change made
        # in infra/ by someone not looking at this module.
        unit = (
            pathlib.Path(__file__).resolve().parents[2] / "infra" / "oracle-api.service"
        )
        text = unit.read_text()
        m = re.search(r"--workers\s+(\d+)", text)
        assert m, f"no --workers flag found in {unit}"
        return int(m.group(1))

    def test_the_docstring_states_the_real_worker_count(self):
        workers = self._deployed_worker_count()
        doc = cache_module.__doc__ or ""
        assert f"--workers {workers}" in doc, (
            f"infra/oracle-api.service runs --workers {workers}; the cache module "
            "docstring must state that count and its consequences. If the count "
            "changed, re-read the docstring's shared/per-worker breakdown — "
            "_inflight coalescing and the stats semantics are described against "
            "the current deployment."
        )

    def test_a_worker_count_change_should_force_a_reread(self):
        # Deliberately asserts the CURRENT state rather than a range: at one
        # worker the shared backend is merely unnecessary, at three+ the
        # _inflight duplicate-pipeline note understates the cost. Either way,
        # whoever changes the count should re-read cache.py's docstring.
        assert self._deployed_worker_count() == 2
