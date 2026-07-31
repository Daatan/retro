"""Settlement-hardening tests (the 2026-07-08 F-35 / McConnell false-pin class).

Three code-enforced rules on top of the base settlement override:

1. **Claim gates** — a ``settled`` claim counts toward the pin only at
   settlement grade (|stance| and certainty ≥ 0.9, the extractor's own stated
   rule). Hedged half-settlements are demoted to ordinary evidence.
2. **No override flip** — settled claims skip ``resolve_stance_certainty``, so
   a cited retrospective number can't reverse an accomplished fact's direction.
3. **Direction guard** — with the caller's temporal metadata present, an early
   settlement may only assert that the event OCCURRED (arrival → YES,
   survival → NO) until the claim deadline passes. Fail-open without metadata.
"""

import pytest

from forecast_api import forecaster
from forecast_api.aggregation import settlement_direction_allowed, settlement_grade
from forecast_api.config import settings as api_settings
from forecast_api.models import ArticleInput, ForecastRequest
from tm.models import ExtractionOutput, GatekeeperOutput, PredictionExtraction


_TITLES = [
    "Washington weighs jet deal as sanctions thaw gathers pace",
    "Analysts split on defense-pact timeline after summit",
    "Historic program exit still shapes procurement debate today",
    "Officials tour production line amid renewed negotiations",
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
        is_prediction=True, reason="on topic", prediction_count_estimate=2,
        relevance_score=1.0,
    ), {"total_tokens": 10}


def _prediction(
    stance: float,
    certainty: float,
    settled: bool | None = None,
    quantitative_estimate: float | None = None,
    evidence_class: str | None = None,
) -> PredictionExtraction:
    # Positive settled claims carry an event_date on/before the article date
    # (2026-07-01): enforce_settlement_event_date demotes undated positive
    # settlements, and these tests exercise the gates DOWNSTREAM of that guard.
    # Negative settlements carry none by definition (the event never occurred).
    # test_settlement_event_date.py (pipeline) covers the guard itself.
    return PredictionExtraction(
        quote="quote", claim="claim", stance=stance, certainty=certainty,
        settled=settled, quantitative_estimate=quantitative_estimate,
        evidence_class=evidence_class,
        event_date="2026-06-30" if settled and stance > 0 else None,
    )


def _patch_pipeline(monkeypatch, extractions_by_source: dict[str, list[PredictionExtraction]]):
    async def fake_gate(**_kwargs):
        return _gate_ok()

    async def fake_extract(**kwargs):
        preds = extractions_by_source[kwargs["source_name"]]
        return ExtractionOutput(predictions=preds), {"total_tokens": 20}

    monkeypatch.setattr(forecaster, "check_is_prediction", fake_gate)
    monkeypatch.setattr(forecaster, "extract_predictions", fake_extract)
    monkeypatch.setattr(api_settings, "cache_ttl_seconds", 0)


class TestClaimGates:
    async def test_hedged_settled_claims_do_not_pin(self, monkeypatch):
        # The F-35 regression: two sources whose "settled" claims are hedged
        # background history (stance −0.8 / certainty ≤ 0.7) against genuinely
        # positive live coverage. Pre-hardening this pinned the estimate to 3%.
        _patch_pipeline(monkeypatch, {
            "source-1": [_prediction(-0.8, 0.6, settled=True), _prediction(0.3, 0.7)],
            "source-2": [_prediction(-0.5, 0.7, settled=True), _prediction(0.25, 0.65)],
            "source-3": [_prediction(0.4, 0.6)],
        })
        resp = await forecaster.run_forecast(ForecastRequest(
            question="settlement hardening — hedged claims A",
            articles=[_article(1), _article(2), _article(3)],
        ))

        assert resp.settled is False
        assert resp.mean != pytest.approx(-api_settings.settlement_stance)
        # Demoted claims still vote as ordinary evidence — no source reports settled.
        assert all(not s.settled for s in resp.sources)

    async def test_settlement_grade_claims_still_pin(self, monkeypatch):
        _patch_pipeline(monkeypatch, {
            "source-1": [_prediction(1.0, 0.95, settled=True)],
            "source-2": [_prediction(0.9, 0.9, settled=True)],  # exactly at both gates
        })
        resp = await forecaster.run_forecast(ForecastRequest(
            question="settlement hardening — grade claims B",
            articles=[_article(1), _article(2)],
        ))

        assert resp.settled is True
        assert resp.mean == pytest.approx(api_settings.settlement_stance)


class TestNoOverrideFlip:
    async def test_cited_retro_odds_cannot_flip_a_settlement(self, monkeypatch):
        # Post-event coverage cites the pre-event odds ("models gave him 22%").
        # resolve_stance_certainty would rewrite stance to 2×0.22−1 = −0.56 and
        # flip the settlement direction; settled claims must keep ±1.0.
        _patch_pipeline(monkeypatch, {
            "source-1": [_prediction(1.0, 0.95, settled=True, quantitative_estimate=0.22)],
            "source-2": [_prediction(0.95, 0.9, settled=True)],
        })
        resp = await forecaster.run_forecast(ForecastRequest(
            question="settlement hardening — retro odds C",
            articles=[_article(1), _article(2)],
        ))

        assert resp.settled is True
        assert resp.mean == pytest.approx(api_settings.settlement_stance)  # +, not −


class TestDirectionGuard:
    async def test_early_negative_settlement_on_arrival_claim_is_suppressed(self, monkeypatch):
        _patch_pipeline(monkeypatch, {
            "source-1": [_prediction(-1.0, 0.95, settled=True)],
            "source-2": [_prediction(-0.95, 0.9, settled=True)],
        })
        resp = await forecaster.run_forecast(ForecastRequest(
            question="settlement hardening — direction guard D",
            articles=[_article(1), _article(2)],
            claim_direction="arrival",
            claim_deadline="2099-12-31",
        ))

        assert resp.settled is False
        assert resp.mean != pytest.approx(-api_settings.settlement_stance)

    async def test_negative_settlement_allowed_after_deadline(self, monkeypatch):
        _patch_pipeline(monkeypatch, {
            "source-1": [_prediction(-1.0, 0.95, settled=True)],
            "source-2": [_prediction(-0.95, 0.9, settled=True)],
        })
        resp = await forecaster.run_forecast(ForecastRequest(
            question="settlement hardening — direction guard E",
            articles=[_article(1), _article(2)],
            claim_direction="arrival",
            # Just past _article()'s fixed 2026-07-01 published_date, not years
            # past it — this test is about settlement_direction_allowed, not
            # the stale_undated_foreclosure staleness guard (retro#295); an
            # arbitrarily-old deadline would trip that unrelated check too.
            claim_deadline="2026-06-20",
        ))

        assert resp.settled is True
        assert resp.mean == pytest.approx(-api_settings.settlement_stance)

    async def test_no_metadata_an_undated_negative_no_longer_pins(self, monkeypatch):
        # Pre-revalidation this was the fail-open case: two undated negatives
        # pinned 3% with no claim metadata at all — exactly the F-35/Netanyahu
        # background-history class from the 2026-07-16 audit. The per-vote rule
        # is deliberately fail-closed on the ANCHOR: an undated non-occurrence
        # vote has nothing tying it to this claim's window.
        _patch_pipeline(monkeypatch, {
            "source-1": [_prediction(-1.0, 0.95, settled=True)],
            "source-2": [_prediction(-0.95, 0.9, settled=True)],
        })
        resp = await forecaster.run_forecast(ForecastRequest(
            question="settlement hardening — direction guard F",
            articles=[_article(1), _article(2)],
        ))

        assert resp.settled is False
        assert resp.mean < 0  # the stances still vote as ordinary evidence

    async def test_no_metadata_a_dated_foreclosure_still_pins(self, monkeypatch):
        # Fail-open on the CHECKS, closed only on the anchor: callers that
        # don't classify claims (MCP, test console) still get negative pins
        # when the foreclosing event is dated.
        def neg(stance: float, certainty: float) -> PredictionExtraction:
            return PredictionExtraction(
                quote="q", claim="c", stance=stance, certainty=certainty,
                settled=True, event_date="2026-06-15",
            )
        _patch_pipeline(monkeypatch, {
            "source-1": [neg(-1.0, 0.95)],
            "source-2": [neg(-0.95, 0.9)],
        })
        resp = await forecaster.run_forecast(ForecastRequest(
            question="settlement hardening — direction guard G",
            articles=[_article(1), _article(2)],
        ))

        assert resp.settled is True
        assert resp.mean == pytest.approx(-api_settings.settlement_stance)


class TestBelowGradeRealignment:
    async def test_below_grade_settled_claim_is_still_realigned(self, monkeypatch):
        # A settled=true claim that fails the settlement_grade gate (hedged,
        # low certainty) is demoted to ordinary evidence — but pre-fix it kept
        # its raw, unrealigned stance/certainty even when it cited an explicit
        # quantitative_estimate, while still earning quantitative_anchor_multiplier's
        # weight premium via that same estimate. That let a misaligned stance
        # get amplified rather than corrected — exactly what
        # resolve_stance_certainty exists to prevent (see its docstring).
        # Since retro#362 the realignment requires evidence_class ==
        # "cited_probability" — the premium it defends against is that class's.
        _patch_pipeline(monkeypatch, {
            "source-1": [_prediction(
                -0.7, 0.5, settled=True, quantitative_estimate=0.7,
                evidence_class="cited_probability",
            )],
            "source-2": [_prediction(0.0, 0.5)],
        })
        resp = await forecaster.run_forecast(ForecastRequest(
            question="settlement hardening — below-grade realignment G",
            articles=[_article(1), _article(2)],
        ))

        # Demoted: doesn't clear settlement_grade even after realignment.
        assert not resp.sources[0].settled
        # Realigned to the cited number (2*0.7-1 = 0.4, certainty forced to
        # 0.9), not left at the raw, extractor-asserted (-0.7, 0.5).
        assert resp.sources[0].stance == pytest.approx(0.4)
        assert resp.sources[0].certainty == pytest.approx(0.9)


class TestPureHelpers:
    def test_settlement_grade_gates(self):
        kw = {"min_stance": 0.9, "min_certainty": 0.9}
        assert settlement_grade(1.0, 0.95, **kw)
        assert settlement_grade(-0.9, 0.9, **kw)  # boundary inclusive
        assert not settlement_grade(-0.8, 0.95, **kw)  # F-35 stance
        assert not settlement_grade(1.0, 0.52, **kw)  # F-35 certainty
        assert not settlement_grade(0.0, 1.0, **kw)

    def test_direction_allowed_matrix(self):
        # Early: only "the event occurred" is coherent.
        assert settlement_direction_allowed(1.0, "arrival", "2099-01-01", today="2026-07-09")
        assert not settlement_direction_allowed(-1.0, "arrival", "2099-01-01", today="2026-07-09")
        assert settlement_direction_allowed(-1.0, "survival", "2099-01-01", today="2026-07-09")
        assert not settlement_direction_allowed(1.0, "survival", "2099-01-01", today="2026-07-09")
        # Deadline passed (or today): both directions coherent.
        assert settlement_direction_allowed(-1.0, "arrival", "2026-01-01", today="2026-07-09")
        assert settlement_direction_allowed(-1.0, "arrival", "2026-07-09", today="2026-07-09")
        # Fail-open without metadata.
        assert settlement_direction_allowed(-1.0, None, None)
        assert settlement_direction_allowed(-1.0, "arrival", "not-a-date")
        assert settlement_direction_allowed(-1.0, "none", "2099-01-01")
