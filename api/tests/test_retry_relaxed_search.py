"""retro#621 fallback ladder rung 1 — retry once with a wider article limit
when the primary /forecast pass returns insufficient_data.

Shadow-gated exactly like premise_verifier/precursor_match/settled_grounding
(retro#575/#608/#609): retry_relaxed_search_enabled runs the retry and logs
what it would have produced; retry_relaxed_search_enforce additionally lets a
recovered result replace the empty response the caller sees. Both default
False. LLM/network are never touched here — _run_forecast_inner itself is
monkeypatched (same technique as test_run_forecast_guards.py), so these tests
exercise only run_forecast's/​_maybe_retry_relaxed_search's orchestration.
"""

from forecast_api import forecaster
from forecast_api.auth import ApiKeyClient
from forecast_api.config import ApiSettings
from forecast_api.config import settings as api_settings
from forecast_api.models import ArticleInput, ForecastRequest


def _insufficient(reason="no_decisive_signal"):
    return forecaster._empty_response("q", reason=reason)


def _usable(*, mean=0.3):
    return forecaster._empty_response("q").model_copy(
        update={"insufficient_data": False, "placeholder": False, "mean": mean, "articles_used": 3}
    )


class TestRetryRelaxedSearchShippedDefaults:
    """Mirrors TestPrecursorMatchShippedDefaults — a silent flip to on is
    exactly the change that should never pass review unnoticed."""

    def test_enabled_ships_disabled(self):
        assert ApiSettings.model_fields["retry_relaxed_search_enabled"].default is False

    def test_enforce_ships_disabled(self):
        assert ApiSettings.model_fields["retry_relaxed_search_enforce"].default is False


class TestRetryDisabled:
    async def test_flag_off_returns_primary_unchanged_and_calls_inner_once(self, monkeypatch):
        calls = []

        async def inner(req, cache_key, limit, total_start):
            calls.append(limit)
            return _insufficient()

        monkeypatch.setattr(forecaster, "_run_forecast_inner", inner)
        monkeypatch.setattr(api_settings, "retry_relaxed_search_enabled", False)

        resp = await forecaster.run_forecast(ForecastRequest(question="retry-off — unique question"))

        assert len(calls) == 1
        assert resp.insufficient_data is True
        assert resp.fallback_path == "primary"


class TestRetryShadowOnly:
    async def test_enabled_without_enforce_still_returns_primary(self, monkeypatch):
        calls = []

        async def inner(req, cache_key, limit, total_start):
            calls.append(limit)
            # First (primary) call insufficient; retry call recovers.
            return _insufficient() if len(calls) == 1 else _usable()

        monkeypatch.setattr(forecaster, "_run_forecast_inner", inner)
        monkeypatch.setattr(api_settings, "retry_relaxed_search_enabled", True)
        monkeypatch.setattr(api_settings, "retry_relaxed_search_enforce", False)

        resp = await forecaster.run_forecast(ForecastRequest(question="retry-shadow — unique question"))

        # Retry did run (for the shadow log) but its recovery must not surface.
        assert len(calls) == 2
        assert calls[1] > calls[0]
        assert resp.insufficient_data is True
        assert resp.fallback_path == "primary"


class TestRetryEnforced:
    async def test_recovered_retry_replaces_the_empty_response(self, monkeypatch):
        calls = []

        async def inner(req, cache_key, limit, total_start):
            calls.append(limit)
            return _insufficient() if len(calls) == 1 else _usable(mean=0.42)

        monkeypatch.setattr(forecaster, "_run_forecast_inner", inner)
        monkeypatch.setattr(api_settings, "retry_relaxed_search_enabled", True)
        monkeypatch.setattr(api_settings, "retry_relaxed_search_enforce", True)

        resp = await forecaster.run_forecast(ForecastRequest(question="retry-enforced — unique question"))

        assert len(calls) == 2
        assert resp.insufficient_data is False
        assert resp.mean == 0.42
        assert resp.fallback_path == "retry-relaxed"

    async def test_still_insufficient_retry_falls_back_to_primary(self, monkeypatch):
        calls = []

        async def inner(req, cache_key, limit, total_start):
            calls.append(limit)
            return _insufficient()

        monkeypatch.setattr(forecaster, "_run_forecast_inner", inner)
        monkeypatch.setattr(api_settings, "retry_relaxed_search_enabled", True)
        monkeypatch.setattr(api_settings, "retry_relaxed_search_enforce", True)

        resp = await forecaster.run_forecast(ForecastRequest(question="retry-still-empty — unique question"))

        assert len(calls) == 2
        assert resp.insufficient_data is True
        assert resp.fallback_path == "primary"


class TestRetryNotAttempted:
    async def test_successful_primary_never_retries(self, monkeypatch):
        calls = []

        async def inner(req, cache_key, limit, total_start):
            calls.append(limit)
            return _usable()

        monkeypatch.setattr(forecaster, "_run_forecast_inner", inner)
        monkeypatch.setattr(api_settings, "retry_relaxed_search_enabled", True)
        monkeypatch.setattr(api_settings, "retry_relaxed_search_enforce", True)

        resp = await forecaster.run_forecast(ForecastRequest(question="retry-not-needed — unique question"))

        assert len(calls) == 1
        assert resp.fallback_path == "primary"

    async def test_caller_supplied_articles_never_retries(self, monkeypatch):
        calls = []

        async def inner(req, cache_key, limit, total_start):
            calls.append(limit)
            return _insufficient()

        monkeypatch.setattr(forecaster, "_run_forecast_inner", inner)
        monkeypatch.setattr(api_settings, "retry_relaxed_search_enabled", True)
        monkeypatch.setattr(api_settings, "retry_relaxed_search_enforce", True)

        resp = await forecaster.run_forecast(ForecastRequest(
            question="retry-supplied-articles — unique question",
            articles=[ArticleInput(url="https://example.com/a", title="t", snippet="s")],
        ))

        # A caller who supplied articles skipped search entirely — a wider
        # limit on a search that never ran can't recover anything.
        assert len(calls) == 1
        assert resp.fallback_path == "primary"

    async def test_at_per_key_limit_cap_never_retries(self, monkeypatch):
        calls = []

        async def inner(req, cache_key, limit, total_start):
            calls.append(limit)
            return _insufficient()

        monkeypatch.setattr(forecaster, "_run_forecast_inner", inner)
        monkeypatch.setattr(api_settings, "retry_relaxed_search_enabled", True)
        monkeypatch.setattr(api_settings, "retry_relaxed_search_enforce", True)

        # Caller already asked for the client's max — a 2x-relaxed retry has
        # nowhere to go.
        resp = await forecaster.run_forecast(
            ForecastRequest(question="retry-at-cap — unique question", max_articles=5),
            client=ApiKeyClient(name="capped", max_articles=5),
        )

        assert len(calls) == 1
        assert resp.fallback_path == "primary"
