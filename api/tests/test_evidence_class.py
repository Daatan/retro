"""S2 shadow-classification tests (retro docs/ORACLE_VARIABLES.md §5).

evidence_class is EXPERIMENTAL and shadow-only this round: the extractor may
classify it, but nothing in the pooling/weighting math reads it yet. These
tests pin the pydantic contract (only the five listed values are accepted)
and prove the shadow property (identical claims produce an identical pooled
result whether or not evidence_class is populated).
"""

import pytest
from pydantic import ValidationError

from forecast_api import forecaster
from forecast_api.models import ArticleInput, ForecastRequest
from tm.models import ExtractionOutput, GatekeeperOutput, PredictionExtraction


_TITLES = [
    "Washington weighs jet deal as sanctions thaw gathers pace",
    "Analysts split on defense-pact timeline after summit",
]


def _article(i: int) -> ArticleInput:
    return ArticleInput(
        url=f"https://source-{i}.example.com/story-{i}",
        title=_TITLES[i - 1],
        snippet=f"A snippet with enough length to pass the fallback minimum, variant {i}.",
        source=f"source-{i}",
        published_date="2026-07-01",
        text="A long enough prefetched article body about the event in question." * 3,
    )


def _gate_ok():
    return GatekeeperOutput(
        is_prediction=True, reason="on topic", prediction_count_estimate=1,
        relevance_score=1.0,
    ), {"total_tokens": 10}


def _prediction(evidence_class=None) -> PredictionExtraction:
    return PredictionExtraction(
        quote="quote", claim="claim", stance=0.4, certainty=0.6,
        evidence_class=evidence_class,
    )


def _patch_pipeline(monkeypatch, extractions_by_source):
    async def fake_gate(**_kwargs):
        return _gate_ok()

    async def fake_extract(**kwargs):
        preds = extractions_by_source[kwargs["source_name"]]
        return ExtractionOutput(predictions=preds), {"total_tokens": 20}

    monkeypatch.setattr(forecaster, "check_is_prediction", fake_gate)
    monkeypatch.setattr(forecaster, "extract_predictions", fake_extract)


class TestPydanticContract:
    def test_accepts_each_of_the_five_documented_values(self):
        for value in ["reported_fact", "cited_probability", "cited_share", "reporting", "opinion"]:
            p = _prediction(evidence_class=value)
            assert p.evidence_class == value

    def test_defaults_to_none(self):
        assert PredictionExtraction(quote="q", claim="c", stance=0.0, certainty=0.5).evidence_class is None

    def test_rejects_a_value_outside_the_five(self):
        with pytest.raises(ValidationError):
            _prediction(evidence_class="strong_opinion")


class TestShadowOnly:
    async def test_evidence_class_does_not_change_the_pooled_result(self, monkeypatch):
        """Same stance/certainty, only evidence_class differs — the pooled
        mean, CI, and per-source weight must be byte-identical either way."""
        _patch_pipeline(monkeypatch, {
            "source-1": [_prediction(evidence_class=None)],
            "source-2": [_prediction(evidence_class="reporting")],
        })
        resp_without = await forecaster.run_forecast(ForecastRequest(
            question="evidence class shadow — without A",
            articles=[_article(1), _article(2)],
        ))

        _patch_pipeline(monkeypatch, {
            "source-1": [_prediction(evidence_class="cited_probability")],
            "source-2": [_prediction(evidence_class="opinion")],
        })
        resp_with = await forecaster.run_forecast(ForecastRequest(
            question="evidence class shadow — with B",
            articles=[_article(1), _article(2)],
        ))

        assert resp_with.mean == pytest.approx(resp_without.mean)
        assert resp_with.ci_low == pytest.approx(resp_without.ci_low)
        assert resp_with.ci_high == pytest.approx(resp_without.ci_high)
        for s_with, s_without in zip(resp_with.sources, resp_without.sources):
            assert s_with.stance == pytest.approx(s_without.stance)
            assert s_with.certainty == pytest.approx(s_without.certainty)
            assert s_with.credibility_weight == pytest.approx(s_without.credibility_weight)

    async def test_evidence_class_is_not_surfaced_on_the_response(self, monkeypatch):
        """Foundation round doesn't change the API contract — SourceSignal
        (and therefore the /forecast response) has no evidence_class field."""
        _patch_pipeline(monkeypatch, {
            "source-1": [_prediction(evidence_class="reported_fact")],
        })
        resp = await forecaster.run_forecast(ForecastRequest(
            question="evidence class shadow — not on response",
            articles=[_article(1)],
        ))
        assert not hasattr(resp.sources[0], "evidence_class")
