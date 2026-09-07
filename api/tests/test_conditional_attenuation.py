"""retro#568 (Phase 4, docs/CONDITIONAL_CAPTURE.md) — shadow-first attenuation of
conditional claims in `reduce_article()`.

These exercise the flag-gated behaviour at the `reduce_article()` level (the
per-coefficient math itself is covered in test_aggregation.py's
TestConditionalAttenuationCoefficient / TestClaimWeightedStance):
  * enabled=False computes nothing, regardless of enforce.
  * enabled=True/enforce=False shadow-computes without moving the live value.
  * enabled=True/enforce=True makes the shadow value the live one.
  * an article with no conditional claims leaves the shadow field None even
    with enabled=True — this is the steady state for the large majority of
    real articles and is worth pinning explicitly.
  * the shadow log fires only when a conditional claim is present.
"""
from __future__ import annotations

import logging
from typing import Any

from forecast_api import forecaster
from forecast_api.config import settings
from forecast_api.models import ArticleInput, ClaimDetail, ForecastRequest
from tm.models import ExtractionOutput, GatekeeperOutput, PredictionExtraction


def _reduce(claims, **overrides):
    kwargs = dict(
        settlement_min_stance=settings.settlement_min_claim_stance,
        settlement_min_certainty=settings.settlement_min_claim_certainty,
        class_weights=settings.evidence_class_weight,
        class_weight_default=settings.evidence_class_weight_default,
        class_weight_unclassified_cap=settings.evidence_class_weight_unclassified_cap,
    )
    kwargs.update(overrides)
    return forecaster.reduce_article(claims, **kwargs)


def _plain_claim(**overrides):
    # Deliberately a different stance from _conditional_claim()'s default — if both
    # claims agreed, attenuating one's weight wouldn't move the weighted mean at all
    # (a weighted mean of equal values is that value regardless of the weights).
    data = dict(claim="X will happen.", stance=0.2, certainty=0.9)
    data.update(overrides)
    return ClaimDetail(**data)


def _conditional_claim(**overrides):
    data = dict(
        claim="X will happen if Y does.",
        stance=0.8,
        certainty=0.9,
        is_conditional=True,
        strength="possible",  # coefficient 0.5, per the ordinal table
    )
    data.update(overrides)
    return ClaimDetail(**data)


class TestReduceArticleConditionalShadow:
    def test_disabled_computes_nothing(self):
        claims = [_conditional_claim(), _plain_claim()]
        reduction = _reduce(claims, conditional_attenuation_enabled=False, conditional_attenuation_enforce=False)
        assert reduction.stance_conditional_shadow is None
        # enforce alone, without enabled, must not compute or move anything either.
        reduction = _reduce(claims, conditional_attenuation_enabled=False, conditional_attenuation_enforce=True)
        assert reduction.stance_conditional_shadow is None

    def test_no_conditional_claims_leaves_shadow_none(self):
        claims = [_plain_claim(), _plain_claim(stance=-0.2, certainty=0.4)]
        reduction = _reduce(claims, conditional_attenuation_enabled=True, conditional_attenuation_enforce=False)
        assert reduction.stance_conditional_shadow is None

    def test_shadow_computed_but_live_value_unchanged_when_not_enforced(self):
        claims = [_conditional_claim(), _plain_claim()]
        baseline = _reduce(claims, conditional_attenuation_enabled=False, conditional_attenuation_enforce=False)
        shadowed = _reduce(claims, conditional_attenuation_enabled=True, conditional_attenuation_enforce=False)
        assert shadowed.stance == baseline.stance
        assert shadowed.stance_conditional_shadow is not None
        assert shadowed.stance_conditional_shadow != shadowed.stance

    def test_enforce_makes_shadow_value_live(self):
        claims = [_conditional_claim(), _plain_claim()]
        reduction = _reduce(claims, conditional_attenuation_enabled=True, conditional_attenuation_enforce=True)
        assert reduction.stance_conditional_shadow is not None
        assert reduction.stance == reduction.stance_conditional_shadow

    def test_fact_signal_coefficients_recomputed_over_its_own_subset(self):
        # fact_claims is a strict subset of scored: the third claim carries no
        # fact_signal at all, and must not participate in either fact_signal
        # computation (nor need a coefficient computed for it).
        claims = [
            _conditional_claim(fact_signal=0.6),
            _plain_claim(stance=-0.2, fact_signal=-0.2),
            _plain_claim(claim="No fact bearing.", stance=0.1, fact_signal=None),
        ]
        baseline = _reduce(claims, conditional_attenuation_enabled=False, conditional_attenuation_enforce=False)
        shadowed = _reduce(claims, conditional_attenuation_enabled=True, conditional_attenuation_enforce=False)
        assert shadowed.fact_signal == baseline.fact_signal
        assert shadowed.fact_signal_conditional_shadow is not None
        assert shadowed.fact_signal_conditional_shadow != shadowed.fact_signal

    def test_dominant_claim_selection_unaffected_by_attenuation(self):
        # The conditional claim has the larger |fact_signal| — dominant selection
        # must still pick it regardless of enabled/enforce, since it keys off the
        # raw fact_signal, not the attenuated one.
        claims = [
            _conditional_claim(fact_signal=0.9, event_actors="conditional-actor"),
            _plain_claim(fact_signal=0.1, event_actors="plain-actor"),
        ]
        reduction = _reduce(claims, conditional_attenuation_enabled=True, conditional_attenuation_enforce=True)
        assert reduction.event_actors == "conditional-actor"


_BODY = (
    "Fixture article body for the conditional-attenuation shadow-log suite. The "
    "gatekeeper and extractor are stubbed, so no model ever reads this text; it "
    "exists to clear the pipeline's non-empty-body checks. "
) * 3


def _prediction(**over: Any) -> PredictionExtraction:
    return PredictionExtraction(**{
        "quote": "Fixture quote.",
        "claim": "Fixture claim.",
        "stance": 0.4,
        "certainty": 0.6,
        **over,
    })


def _stub_pipeline(monkeypatch, claims: list[PredictionExtraction]) -> None:
    async def fake_gate(**kwargs):
        return (
            GatekeeperOutput(
                is_prediction=True,
                reason="fixture gate",
                prediction_count_estimate=len(claims),
                relevance_score=1.0,
            ),
            {"total_tokens": 0},
        )

    async def fake_extract(**kwargs):
        return (ExtractionOutput(predictions=list(claims)), {"total_tokens": 0})

    monkeypatch.setattr(forecaster, "check_is_prediction", fake_gate)
    monkeypatch.setattr(forecaster, "extract_predictions", fake_extract)
    monkeypatch.setattr(forecaster, "get_credibility_weight", lambda sid: 1.0)


async def _run(monkeypatch, claims: list[PredictionExtraction], question: str, url: str):
    """Run the real /forecast pipeline over one stubbed article."""
    _stub_pipeline(monkeypatch, claims)
    resp = await forecaster.run_forecast(ForecastRequest(
        question=question,
        articles=[ArticleInput(
            url=url,
            title="Fixture dispatch",
            snippet="Fixture snippet, long enough to be usable by the pipeline.",
            source="fixture",
            published_date="2026-09-01",
            text=_BODY,
        )],
    ))
    assert resp.sources, "fixture article should have produced a source signal"
    return resp.sources[0]


class TestConditionalAttenuationShadowLog:
    """Exercises the real call site in forecaster.py (not reduce_article() directly),
    since the shadow log lives there, not inside the pure reduction function.
    `settings.conditional_attenuation_enabled` ships True, so no monkeypatching of
    settings is needed to observe it.
    """

    async def test_log_fires_when_a_conditional_claim_is_present(self, monkeypatch, caplog):
        claims = [
            _prediction(claim="X will happen if Y does.", stance=0.8, certainty=0.9,
                        is_conditional=True, strength="possible"),
            _prediction(claim="Unrelated colour.", stance=0.1, certainty=0.3),
        ]
        with caplog.at_level(logging.INFO, logger="forecast_api.forecaster"):
            await _run(monkeypatch, claims, "[C568-log-present] Will the event occur?",
                       "https://fixture.example.test/c568-present")

        assert "event=conditional_attenuation_shadow" in caplog.text
        assert "n_conditional=1" in caplog.text

    async def test_log_does_not_fire_with_no_conditional_claims(self, monkeypatch, caplog):
        claims = [
            _prediction(claim="A.", stance=0.5, certainty=0.6),
            _prediction(claim="B.", stance=-0.2, certainty=0.4),
        ]
        with caplog.at_level(logging.INFO, logger="forecast_api.forecaster"):
            await _run(monkeypatch, claims, "[C568-log-absent] Will the event occur?",
                       "https://fixture.example.test/c568-absent")

        assert "event=conditional_attenuation_shadow" not in caplog.text
