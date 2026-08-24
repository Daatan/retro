"""Tests for the confidence bucket exposed on plain HTTP responses (retro#618).

Bucket boundaries themselves are pinned by test_mcp_server.py::TestConfidenceBucket
(F16, retro#365) — these tests cover the two things retro#618 actually changed:
aggregation.confidence_bucket's new insufficient_data=None short-circuit, and
that ForecastResponse/PoolAggregateResponse now serialize a "confidence" field
at all (previously only mcp_server._confidence computed this, for MCP tool
callers only — a plain /forecast or /pool/aggregate caller got nothing).
"""

from __future__ import annotations

from forecast_api.aggregation import confidence_bucket
from forecast_api.models import ForecastResponse, PoolAggregateResponse


class TestConfidenceBucketInsufficientData:
    def test_insufficient_data_is_none_regardless_of_ci(self):
        # Same ci_low/ci_high _build_insufficient_data_response uses (-0.2, 0.2,
        # 0 articles) — width alone would land "medium", but there's no real
        # estimate here to be confident about.
        assert confidence_bucket(
            settled=False, ci_low=-0.2, ci_high=0.2, articles_used=0,
            insufficient_data=True,
        ) is None

    def test_insufficient_data_overrides_settled(self):
        assert confidence_bucket(
            settled=True, ci_low=-0.9, ci_high=0.9, articles_used=1,
            insufficient_data=True,
        ) is None

    def test_sufficient_data_still_buckets_normally(self):
        assert confidence_bucket(
            settled=False, ci_low=-0.29, ci_high=0.29, articles_used=9,
        ) == "high"


class TestForecastResponseConfidenceField:
    def _resp(self, **kw) -> ForecastResponse:
        defaults = dict(
            question="Will X happen?", mean=0.0, std=0.1,
            ci_low=-0.29, ci_high=0.29, articles_used=9, sources=[],
        )
        defaults.update(kw)
        return ForecastResponse(**defaults)

    def test_confidence_appears_in_serialized_response(self):
        resp = self._resp()
        assert resp.confidence == "high"
        assert resp.model_dump()["confidence"] == "high"

    def test_thin_single_source_pool_is_low_confidence(self):
        # The retro#618 incident's shape: one source, near-full-range CI.
        resp = self._resp(ci_low=-0.74, ci_high=0.98, articles_used=1)
        assert resp.confidence == "low"

    def test_insufficient_data_response_has_null_confidence(self):
        resp = self._resp(
            ci_low=-0.2, ci_high=0.2, articles_used=0,
            insufficient_data=True, reason="no_sources",
        )
        assert resp.confidence is None
        assert resp.model_dump()["confidence"] is None


class TestPoolAggregateResponseConfidenceField:
    def test_confidence_appears_in_serialized_response(self):
        resp = PoolAggregateResponse(
            mean=0.0, std=0.1, ci_low=-0.29, ci_high=0.29, articles_used=9,
        )
        assert resp.confidence == "high"
        assert resp.model_dump()["confidence"] == "high"

    def test_insufficient_data_response_has_null_confidence(self):
        resp = PoolAggregateResponse(
            mean=0.0, std=0.0, ci_low=0.0, ci_high=0.0, articles_used=0,
            insufficient_data=True, reason="no_sources",
        )
        assert resp.confidence is None
