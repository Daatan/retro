"""Event decomposition store (retro#758) — cache identity and fail-open contract.

Same coverage shape as the settlement_verdict_store it mirrors: a round trip
survives, differing inputs get different keys, and a broken store degrades to
"cache miss" rather than raising.
"""
from __future__ import annotations

from pathlib import Path

from forecast_api.event_decomposition_store import (
    decomposition_key,
    get_decomposition,
    put_decomposition,
)


class TestDecompositionKey:
    def test_same_inputs_same_key(self):
        k1 = decomposition_key("Will X happen?", "criteria", model="m")
        k2 = decomposition_key("Will X happen?", "criteria", model="m")
        assert k1 == k2

    def test_different_question_different_key(self):
        k1 = decomposition_key("Will X happen?", None, model="m")
        k2 = decomposition_key("Will Y happen?", None, model="m")
        assert k1 != k2

    def test_different_resolution_criteria_different_key(self):
        k1 = decomposition_key("Will X happen?", "criteria A", model="m")
        k2 = decomposition_key("Will X happen?", "criteria B", model="m")
        assert k1 != k2

    def test_different_model_different_key(self):
        k1 = decomposition_key("Will X happen?", None, model="model-a")
        k2 = decomposition_key("Will X happen?", None, model="model-b")
        assert k1 != k2

    def test_none_and_empty_criteria_are_the_same_key(self):
        assert decomposition_key("Q", None, model="m") == decomposition_key("Q", "", model="m")


class TestRoundTrip:
    async def test_put_then_get(self, tmp_path):
        path = tmp_path / "decomp_cache"
        key = decomposition_key("Will X happen?", None, model="m")
        await put_decomposition(path, key, "WHO: X. WHAT: Y. SCOPE: Z.")
        assert await get_decomposition(path, key) == "WHO: X. WHAT: Y. SCOPE: Z."

    async def test_miss_returns_none(self, tmp_path):
        path = tmp_path / "decomp_cache"
        assert await get_decomposition(path, "nonexistent-key") is None


class TestFailsOpen:
    async def test_a_broken_path_degrades_to_a_miss_not_a_raise(self):
        # A regular file where diskcache expects a directory it can create/open
        # under — this must not propagate.
        bad_path = Path("/dev/null/not-a-real-dir")
        result = await get_decomposition(bad_path, "any-key")
        assert result is None
        # put must also swallow the error rather than raise.
        await put_decomposition(bad_path, "any-key", "WHO: X. WHAT: Y. SCOPE: Z.")
