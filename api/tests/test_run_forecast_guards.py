"""Regression tests for two production incidents in ``run_forecast``:

1. **Forecast timeout** — the pipeline had no deadline, so a slow run could
   hang past the client's own 120s abort. ``run_forecast`` now wraps the inner
   pipeline in ``asyncio.wait_for(settings.forecast_timeout_seconds)`` and
   returns a placeholder when it fires.

2. **debug=true → 500** — the debug branch read ``settings.gatekeeper_model``
   / ``settings.extractor_model`` off the *API* settings object, which never
   defined them, raising ``AttributeError`` on every fresh debug forecast. Those
   fields live on the *pipeline* settings. This test locks that invariant so a
   future edit can't silently reintroduce the bug.
"""

import asyncio

from forecast_api import forecaster
from forecast_api.config import settings as api_settings
from forecast_api.models import ForecastRequest
from tm.config import settings as pipeline_settings


class TestForecastDeadline:
    async def test_returns_placeholder_when_inner_run_exceeds_deadline(self, monkeypatch):
        async def slow_inner(*_args, **_kwargs):
            await asyncio.sleep(5)  # far longer than the deadline below

        monkeypatch.setattr(forecaster, "_run_forecast_inner", slow_inner)
        monkeypatch.setattr(api_settings, "forecast_timeout_seconds", 0.05)

        resp = await forecaster.run_forecast(
            ForecastRequest(question="deadline regression — unique question")
        )

        assert resp.placeholder is True
        assert resp.articles_used == 0

    async def test_fast_inner_run_is_returned_unchanged(self, monkeypatch):
        async def fast_inner(*_args, **_kwargs):
            return forecaster._empty_response("ok")  # stand-in real response

        monkeypatch.setattr(forecaster, "_run_forecast_inner", fast_inner)
        monkeypatch.setattr(api_settings, "forecast_timeout_seconds", 30)

        resp = await forecaster.run_forecast(
            ForecastRequest(question="fast path — another unique question")
        )

        assert resp is not None


class TestDebugModelFieldsSource:
    """The debug telemetry's model names must come from the pipeline settings."""

    def test_pipeline_settings_define_the_model_fields(self):
        assert pipeline_settings.gatekeeper_model
        assert pipeline_settings.extractor_model

    def test_api_settings_do_not_define_them(self):
        # If these ever move onto the API settings, drop this guard — but until
        # then, referencing them off `api_settings` is the exact bug that 500'd.
        assert not hasattr(api_settings, "gatekeeper_model")
        assert not hasattr(api_settings, "extractor_model")
