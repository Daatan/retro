"""An empty forecast must say *why* — not just return a bland 0.0.

These lock the observability contract: every insufficient-data response carries
``insufficient_data=True``, a ``reason``, and the per-article ``outcome_counts``
so callers can tell "search returned junk" from "extractor is erroring" from
"timed out".
"""

import asyncio

import pytest

from forecast_api import forecaster
from forecast_api.forecaster import _empty_response, _reason_from_outcomes, run_forecast
from forecast_api.models import ForecastRequest


class TestReasonFromOutcomes:
    def test_off_topic_majority(self):
        assert _reason_from_outcomes({"gate_rejected": 6}) == "all_articles_off_topic"

    def test_errors_majority(self):
        assert _reason_from_outcomes({"gate_error": 3, "gate_rejected": 1}) == "extraction_errors"
        assert _reason_from_outcomes({"extract_error": 2, "unhandled_error": 1, "ok": 0}) == "extraction_errors"

    def test_fetch_failures_majority(self):
        assert _reason_from_outcomes({"empty_text": 5, "gate_rejected": 1}) == "all_fetches_failed"

    def test_empty_histogram(self):
        assert _reason_from_outcomes({}) == "no_usable_predictions"


class TestEmptyResponseFields:
    def test_carries_reason_and_flags(self):
        r = _empty_response("q?", reason="all_articles_off_topic", articles_found=6,
                            outcome_counts={"gate_rejected": 6})
        assert r.insufficient_data is True
        assert r.placeholder is True
        assert r.reason == "all_articles_off_topic"
        assert r.articles_found == 6
        assert r.outcome_counts == {"gate_rejected": 6}

    def test_defaults_are_safe(self):
        r = _empty_response("q?")
        assert r.insufficient_data is True and r.reason is None and r.outcome_counts == {}


class TestNoSearchResultsReason:
    def test_empty_search_sets_reason(self, monkeypatch):
        # Search returns nothing; distillation no-ops (no network).
        monkeypatch.setattr(forecaster, "search_articles", lambda q, limit: [])

        async def _no_distill(question):
            return question
        monkeypatch.setattr(forecaster, "_distill_query", _no_distill)

        req = ForecastRequest(question="Totally unique probe question 9f3a2b — will X?")
        resp = asyncio.run(run_forecast(req))

        assert resp.insufficient_data is True
        assert resp.reason == "no_search_results"
        assert resp.articles_used == 0
