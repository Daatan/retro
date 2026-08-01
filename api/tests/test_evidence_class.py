"""S2 weight-cutover tests (retro docs/ORACLE_VARIABLES.md §5).

evidence_class now drives the cross-article `weight` term via
evidence_class_weight() (see test_aggregation.py's TestEvidenceClassWeight and
TestFranceWorldCupRegression for the pure-function coverage). These tests pin
the pydantic contract (only the five listed values are accepted) and prove
evidence_class is load-bearing end-to-end through run_forecast — no longer the
shadow-only property this file locked in before the cutover.
"""

import pytest
from pydantic import ValidationError

from forecast_api import forecaster
from forecast_api.config import settings as api_settings
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


def _prediction(evidence_class=None, stance=0.4, certainty=0.6) -> PredictionExtraction:
    # A cited_probability claim has to name an allowlisted source or
    # enforce_anchor_provenance (F4, retro#369) demotes it to `reporting` before
    # any of these gates run. These tests are about the class weight table, not
    # about provenance — test_anchor_provenance.py owns that — so the fixture
    # names one. "Kalshi" is on cited_probability_source_allowlist.
    quote = "Kalshi traders put it at 20%." if evidence_class == "cited_probability" else "quote"
    return PredictionExtraction(
        quote=quote, claim="claim", stance=stance, certainty=certainty,
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


class TestWeightCutover:
    async def test_evidence_class_changes_the_pooled_weight(self, monkeypatch):
        """Two sources disagree in stance (+0.9 vs -0.9), same certainty, same
        credibility (both unknown test domains → neutral 1.0), same recency and
        relevance (same published_date, gatekeeper mock relevance_score=1.0).
        Only evidence_class differs, so whichever source is classified
        cited_probability (4.0×) must dominate the pool — proving evidence_class
        is now load-bearing, not shadow. Swapping which source carries the
        cited_probability label flips the pooled direction, ruling out a
        source-ordering artifact."""
        _patch_pipeline(monkeypatch, {
            "source-1": [_prediction(evidence_class="cited_probability", stance=0.9)],
            "source-2": [_prediction(evidence_class="opinion", stance=-0.9)],
        })
        resp_a = await forecaster.run_forecast(ForecastRequest(
            question="evidence class weight cutover — source-1 dominant",
            articles=[_article(1), _article(2)],
        ))
        assert resp_a.mean > 0

        _patch_pipeline(monkeypatch, {
            "source-1": [_prediction(evidence_class="opinion", stance=0.9)],
            "source-2": [_prediction(evidence_class="cited_probability", stance=-0.9)],
        })
        resp_b = await forecaster.run_forecast(ForecastRequest(
            question="evidence class weight cutover — source-2 dominant",
            articles=[_article(1), _article(2)],
        ))
        assert resp_b.mean < 0

    async def test_evidence_class_is_surfaced_on_the_response(self, monkeypatch):
        """evidence_class IS exposed on SourceSignal (see
        TestEvidenceClassExposure below) — the credibility feedback loop
        (docs/ORACLE_VARIABLES.md §9) needs the actual label, not just its
        resolved weight, to exclude opinion-class articles from the
        resolution-outcome signal: evidence_weight alone can't distinguish an
        opinion-class article from a low-certainty unclassified one."""
        _patch_pipeline(monkeypatch, {
            "source-1": [_prediction(evidence_class="reported_fact")],
        })
        resp = await forecaster.run_forecast(ForecastRequest(
            question="evidence class weight cutover — on response",
            articles=[_article(1)],
        ))
        assert resp.sources[0].evidence_class == "reported_fact"


class TestEvidenceWeightExposure:
    """evidence_weight (the resolved per-source weighting factor) is exposed
    on SourceSignal so a caller can later recompute a pool of already-extracted
    sources without re-deriving class_weight lookups outside retro — see
    docs/ORACLE_VARIABLES.md's recompute-over-pool item."""

    async def test_classified_source_exposes_its_class_weight(self, monkeypatch):
        _patch_pipeline(monkeypatch, {
            "source-1": [_prediction(evidence_class="cited_probability", certainty=0.5)],
        })
        resp = await forecaster.run_forecast(ForecastRequest(
            question="evidence weight exposure — classified",
            articles=[_article(1)],
        ))
        assert resp.sources[0].evidence_weight == pytest.approx(
            api_settings.evidence_class_weight["cited_probability"],
        )

    async def test_unclassified_source_exposes_its_capped_certainty(self, monkeypatch):
        """R3/F10: an unclassified claim still falls back to its own certainty,
        capped at the weakest class's weight so silence about the evidence type
        can never out-weigh an answer (certainty 0.73 → 0.25, not 0.73)."""
        _patch_pipeline(monkeypatch, {
            "source-1": [_prediction(evidence_class=None, certainty=0.73)],
        })
        resp = await forecaster.run_forecast(ForecastRequest(
            question="evidence weight exposure — unclassified",
            articles=[_article(1)],
        ))
        assert resp.sources[0].evidence_weight == pytest.approx(
            api_settings.evidence_class_weight_unclassified_cap,
        )

    async def test_hedged_unclassified_source_stays_below_the_cap(self, monkeypatch):
        """The cap is a ceiling, not a floor — a barely-certain unclassified claim
        keeps its own lower weight rather than being promoted to 0.25."""
        _patch_pipeline(monkeypatch, {
            "source-1": [_prediction(evidence_class=None, certainty=0.08)],
        })
        resp = await forecaster.run_forecast(ForecastRequest(
            question="evidence weight exposure — hedged unclassified",
            articles=[_article(1)],
        ))
        assert resp.sources[0].evidence_weight == pytest.approx(0.08)


class TestEvidenceClassExposure:
    """evidence_class itself (the label, not just its resolved weight) is
    exposed on SourceSignal — added for the credibility feedback loop
    (docs/ORACLE_VARIABLES.md §9), which needs to exclude opinion-class
    articles from the resolution-outcome signal and can't do that from
    evidence_weight alone. evidence_class is a per-claim field, so an
    article with several claims exposes the most common non-null label."""

    async def test_single_claim_exposes_its_class(self, monkeypatch):
        _patch_pipeline(monkeypatch, {
            "source-1": [_prediction(evidence_class="opinion")],
        })
        resp = await forecaster.run_forecast(ForecastRequest(
            question="evidence class exposure — single claim",
            articles=[_article(1)],
        ))
        assert resp.sources[0].evidence_class == "opinion"

    async def test_unclassified_single_claim_exposes_none(self, monkeypatch):
        _patch_pipeline(monkeypatch, {
            "source-1": [_prediction(evidence_class=None)],
        })
        resp = await forecaster.run_forecast(ForecastRequest(
            question="evidence class exposure — unclassified",
            articles=[_article(1)],
        ))
        assert resp.sources[0].evidence_class is None

    async def test_multiple_claims_exposes_the_most_common_class(self, monkeypatch):
        _patch_pipeline(monkeypatch, {
            "source-1": [
                _prediction(evidence_class="reported_fact"),
                _prediction(evidence_class="reported_fact"),
                _prediction(evidence_class="opinion"),
            ],
        })
        resp = await forecaster.run_forecast(ForecastRequest(
            question="evidence class exposure — majority label",
            articles=[_article(1)],
        ))
        assert resp.sources[0].evidence_class == "reported_fact"

    async def test_multiple_claims_ignores_unclassified_ones_when_picking_the_label(self, monkeypatch):
        """A mix of one classified and one unclassified claim on the same
        article must expose the classified label, not None — an unclassified
        claim shouldn't be able to blank out real signal from a sibling claim."""
        _patch_pipeline(monkeypatch, {
            "source-1": [
                _prediction(evidence_class=None),
                _prediction(evidence_class="cited_share"),
            ],
        })
        resp = await forecaster.run_forecast(ForecastRequest(
            question="evidence class exposure — mixed classified/unclassified",
            articles=[_article(1)],
        ))
        assert resp.sources[0].evidence_class == "cited_share"
