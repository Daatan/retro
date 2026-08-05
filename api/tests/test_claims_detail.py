"""F1/F15 — per-claim persistence (retro#364).

Two things are under test, and they are deliberately different in kind.

**The forecast side** must actually keep the claims. A multi-claim article is
collapsed into five article-level scalars computed over five different claim
subsets, and before this change the claims that produced them were discarded
at the wire. The tests below pin that ``claims_detail`` carries every claim,
in order, with the values the fusion *actually consumed* — which is what makes
the article-level scalars re-derivable from it rather than a second, parallel
truth. A claims list that cannot reproduce the article's own vote would
reintroduce the defect it is meant to fix.

**The pool side** must change nothing. ``PoolSourceInput`` grew identity and
per-claim fields; ``run_pool_aggregate`` reads none of them. Per R8 (retro#370)
this PR is additive persistence only, so the published estimate has to be
bit-identical whether or not a caller sends the new fields — asserted here on
the whole response object, not on a summary of it.

The pipeline is stubbed the same way the R8 matrix stubs it (gatekeeper,
extractor and the leaderboard lookup only), so the enforcer chain, claim
fusion and pooling all run as production code.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import pytest

from forecast_api import forecaster
from forecast_api.aggregation import claim_weighted_stance, resolve_stance_certainty
from forecast_api.models import (
    ArticleInput,
    ClaimDetail,
    ForecastRequest,
    PoolAggregateRequest,
    PoolSourceInput,
)
from tm.models import ExtractionOutput, GatekeeperOutput, PredictionExtraction


_BODY = (
    "Fixture article body for the per-claim persistence suite. The gatekeeper "
    "and extractor are stubbed, so no model ever reads this text; it exists to "
    "clear the pipeline's non-empty-body checks. "
) * 3


def _claim(**over: Any) -> PredictionExtraction:
    return PredictionExtraction(**{
        "quote": "Fixture quote.",
        "claim": "Fixture claim.",
        "stance": 0.4,
        "certainty": 0.6,
        **over,
    })


def _patch(monkeypatch, claims: list[PredictionExtraction], *, relevance: float = 1.0) -> None:
    async def fake_gate(**kwargs):
        return (
            GatekeeperOutput(
                is_prediction=True,
                reason="fixture gate",
                prediction_count_estimate=len(claims),
                relevance_score=relevance,
            ),
            {"total_tokens": 0},
        )

    async def fake_extract(**kwargs):
        return (ExtractionOutput(predictions=list(claims)), {"total_tokens": 0})

    monkeypatch.setattr(forecaster, "check_is_prediction", fake_gate)
    monkeypatch.setattr(forecaster, "extract_predictions", fake_extract)
    monkeypatch.setattr(forecaster, "get_credibility_weight", lambda sid: 1.0)


async def _one_source(monkeypatch, claims: list[PredictionExtraction], question: str):
    """Run the real pipeline over a single stubbed article, return its SourceSignal."""
    _patch(monkeypatch, claims)
    resp = await forecaster.run_forecast(ForecastRequest(
        question=question,
        articles=[ArticleInput(
            url="https://fixture.example.test/a1",
            title="Vellum sonata dispatch",
            snippet="Fixture snippet, long enough to be usable by the pipeline.",
            source="fixture",
            published_date="2026-07-28",
            text=_BODY,
        )],
    ))
    assert resp.sources, "fixture article should have produced a source signal"
    return resp.sources[0]


class TestClaimsArePersisted:
    async def test_one_entry_per_claim_in_order(self, monkeypatch):
        s = await _one_source(monkeypatch, [
            _claim(claim="First claim.", stance=0.8),
            _claim(claim="Second claim.", stance=-0.2),
            _claim(claim="Third claim.", stance=0.1),
        ], "[F1-order] Will the event occur?")

        assert s.claims_detail is not None
        assert [c.claim for c in s.claims_detail] == [
            "First claim.", "Second claim.", "Third claim.",
        ]
        assert [c.stance for c in s.claims_detail] == [0.8, -0.2, 0.1]

    async def test_claim_with_empty_summary_is_kept(self, monkeypatch):
        """``claims`` drops falsy summaries; ``claims_detail`` must not.

        That claim still carried certainty weight in the article's stance, so
        omitting it would make the reduction unreproducible from what we kept.
        """
        s = await _one_source(monkeypatch, [
            _claim(claim="Real claim.", stance=0.9, certainty=0.9),
            _claim(claim="", stance=-0.9, certainty=0.9),
        ], "[F1-empty] Will the event occur?")

        assert s.claims == ["Real claim."]
        assert len(s.claims_detail) == 2
        assert s.claims_detail[1].claim == ""
        assert s.claims_detail[1].stance == -0.9

    async def test_quote_is_kept_with_the_claim(self, monkeypatch):
        s = await _one_source(monkeypatch, [
            _claim(quote="The minister said the vote would be held in March."),
        ], "[F1-quote] Will the event occur?")

        assert s.claims_detail[0].quote == "The minister said the vote would be held in March."


class TestArticleScalarsAreDerivable:
    async def test_stance_reduces_from_claims_detail(self, monkeypatch):
        claims = [
            _claim(claim="A.", stance=0.9, certainty=0.9),
            _claim(claim="B.", stance=-0.3, certainty=0.2),
            _claim(claim="C.", stance=0.1, certainty=0.5),
        ]
        s = await _one_source(monkeypatch, claims, "[F1-derive] Will the event occur?")

        derived = claim_weighted_stance(
            [c.stance for c in s.claims_detail],
            [c.certainty for c in s.claims_detail],
            [c.specificity for c in s.claims_detail],
        )
        assert round(derived, 3) == s.stance

    async def test_fact_signal_reduces_from_claims_detail(self, monkeypatch):
        """The fact lane is the claim-weighted mean over the fact-bearing
        claims of the same scored set (no settlement here, so that is all of
        them). Opinion claims carry no fact_signal and must not participate."""
        claims = [
            _claim(claim="A.", stance=0.5, certainty=0.8, fact_signal=0.25, is_occurrence=True),
            _claim(claim="B.", stance=0.4, certainty=0.4, fact_signal=-0.5, is_occurrence=True),
            _claim(claim="C.", stance=0.2, certainty=0.6, evidence_class="opinion"),
        ]
        s = await _one_source(monkeypatch, claims, "[F1-fact] Will the event occur?")

        fact_claims = [c for c in s.claims_detail if c.fact_signal is not None]
        assert len(fact_claims) == 2, "the opinion claim must not enter the fact lane"
        derived = claim_weighted_stance(
            [c.fact_signal for c in fact_claims],
            [c.certainty for c in fact_claims],
            [c.specificity for c in fact_claims],
        )
        assert round(derived, 3) == s.fact_signal

    async def test_values_are_post_resolution_not_raw_extractor_output(self, monkeypatch):
        """A cited probability realigns the claim's stance/certainty before
        fusion. ``claims_detail`` must record the realigned pair — persisting
        the extractor's raw numbers would leave a claims list that cannot
        reproduce the article's own stance."""
        raw_stance, raw_certainty, quant = 0.2, 0.3, 0.85
        expected_stance, expected_certainty = resolve_stance_certainty(
            raw_stance, raw_certainty, quant, evidence_class="cited_probability",
        )
        assert (expected_stance, expected_certainty) != (raw_stance, raw_certainty), \
            "fixture must actually trigger realignment, else this test proves nothing"

        s = await _one_source(monkeypatch, [
            # The quote must name an allowlisted source, or enforce_anchor_provenance
            # (F4, retro#369) demotes the class and the realignment never fires.
            _claim(quote="Polymarket prices this at 85%.",
                   stance=raw_stance, certainty=raw_certainty,
                   quantitative_estimate=quant, evidence_class="cited_probability"),
        ], "[F1-resolved] Will the event occur?")

        assert s.claims_detail[0].stance == pytest.approx(expected_stance)
        assert s.claims_detail[0].certainty == pytest.approx(expected_certainty)
        assert s.claims_detail[0].quantitative_estimate == quant

    async def test_persisted_class_is_the_enforced_one(self, monkeypatch):
        """An unattributed cited_probability is demoted by enforce_anchor_provenance
        (F4, retro#369). The persisted class must be the demoted one — the class
        the weighting actually used — not the extractor's original label."""
        s = await _one_source(monkeypatch, [
            _claim(quote="A market prices this at 85%.", stance=0.2, certainty=0.3,
                   quantitative_estimate=0.85, evidence_class="cited_probability"),
        ], "[F1-provenance] Will the event occur?")

        assert s.claims_detail[0].evidence_class == "reporting"
        assert s.claims_detail[0].quantitative_estimate == 0.85, \
            "the figure itself is kept — only its class was demoted"

    async def test_precursor_cap_is_reflected_in_the_persisted_claim(self, monkeypatch):
        """enforce_precursor_cap (F9, retro#367) clamps |fact_signal| when the
        claim is only a precursor. It runs before fusion, so the persisted
        value is the clamped one — the number the estimator would see."""
        s = await _one_source(monkeypatch, [
            _claim(fact_signal=0.9, is_occurrence=False),
        ], "[F1-cap] Will the event occur?")

        assert s.claims_detail[0].fact_signal == pytest.approx(0.3)
        assert s.claims_detail[0].is_occurrence is False


class TestFacetsAreClaimLevel:
    async def test_each_claim_keeps_its_own_facets(self, monkeypatch):
        """The article-level facets ride from the single dominant claim. That
        is why an over-cap interested-party claim diluted by in-contract
        siblings is invisible above this layer (retro#378) — here it is not."""
        s = await _one_source(monkeypatch, [
            _claim(claim="Independently reported.", fact_signal=0.15,
                   is_occurrence=True, verified=True, event_actors="Alpha"),
            _claim(claim="Claimed by a belligerent.", fact_signal=-0.6,
                   is_occurrence=True, verified=False, event_actors="Beta"),
        ], "[F1-facets] Will the event occur?")

        # Dominant claim = the second (larger |fact_signal|), so the article says:
        assert s.verified is False
        assert s.event_actors == "Beta"
        # ...while the claim layer keeps both, un-collapsed:
        assert [c.verified for c in s.claims_detail] == [True, False]
        assert [c.event_actors for c in s.claims_detail] == ["Alpha", "Beta"]


class TestTheReductionReplaysFromPersistedClaims:
    """Item 3 of F1: the scalars are DERIVED from the per-claim layer.

    The tests above check individual scalars by re-deriving them by hand. These
    check the stronger property that makes derivability structural rather than
    incidental: ``reduce_article()`` — the exact function the pipeline uses —
    replayed over nothing but the persisted ``claims_detail``, reproduces
    *every* field of the signal it produced.

    That is the contract a stored pool row has to satisfy for retroactive
    backtesting, R1 fitting and F3 attribution to be possible at all. It fails
    the moment a scalar starts reading something the claim layer does not keep,
    or ``build_claims_detail`` drops a field the reduction consumes — neither
    of which any per-scalar test would catch.
    """

    @staticmethod
    def _replay(claims_detail):
        from forecast_api.config import settings

        return forecaster.reduce_article(
            claims_detail,
            settlement_min_stance=settings.settlement_min_claim_stance,
            settlement_min_certainty=settings.settlement_min_claim_certainty,
            class_weights=settings.evidence_class_weight,
            class_weight_default=settings.evidence_class_weight_default,
            class_weight_unclassified_cap=settings.evidence_class_weight_unclassified_cap,
        )

    @staticmethod
    def _assert_replays(signal, replayed) -> None:
        assert round(replayed.stance, 3) == signal.stance
        assert round(replayed.certainty, 3) == signal.certainty
        assert round(replayed.evidence_weight, 3) == signal.evidence_weight
        assert replayed.evidence_class == signal.evidence_class
        assert (replayed.settled or None) == signal.settled
        assert replayed.settlement_event_date == signal.settlement_event_date
        assert replayed.quantitative_estimate == signal.quantitative_estimate
        assert replayed.claims == signal.claims
        assert (
            round(replayed.fact_signal, 3) if replayed.fact_signal is not None else None
        ) == signal.fact_signal
        assert replayed.event_actors == signal.event_actors
        assert replayed.event_target == signal.event_target
        assert replayed.is_occurrence == signal.is_occurrence
        assert replayed.verified == signal.verified

    async def test_every_scalar_replays_on_the_ordinary_path(self, monkeypatch):
        """Mixed classes, mixed fact-bearing, one anchor — the five reductions
        run over four different subsets here, so a replay that agrees on all of
        them is not agreeing by coincidence."""
        s = await _one_source(monkeypatch, [
            _claim(claim="Reported fact.", stance=0.7, certainty=0.8,
                   evidence_class="reported_fact", fact_signal=0.2, is_occurrence=True,
                   verified=True, event_actors="Alpha", event_target="Beta"),
            _claim(claim="Columnist's view.", stance=-0.4, certainty=0.3,
                   evidence_class="opinion"),
            _claim(quote="Polymarket prices this at 85%.", claim="Cited market price.",
                   stance=0.2, certainty=0.3, quantitative_estimate=0.85,
                   evidence_class="cited_probability", fact_signal=0.55,
                   is_occurrence=True, verified=False, event_actors="Gamma"),
            _claim(claim="", stance=0.1, certainty=0.5),
        ], "[F1-replay] Will the event occur?")

        assert len(s.claims_detail) == 4
        self._assert_replays(s, self._replay(s.claims_detail))

    async def test_every_scalar_replays_on_the_settlement_path(self, monkeypatch):
        """The settlement subset is where the reductions diverge most: stance
        comes from the settled claim alone while certainty and evidence_weight
        still average the colour quotes it displaced. The replay has to
        reconstruct that split from ``settled`` + the gates, not be told it."""
        s = await _one_source(monkeypatch, [
            _claim(claim="The result was declared.", stance=1.0, certainty=0.95,
                   settled=True, event_date="2026-07-20",
                   evidence_class="reported_fact", fact_signal=0.9, is_occurrence=True,
                   verified=True),
            _claim(claim="Colour quote from the losing camp.", stance=-0.6,
                   certainty=0.4, evidence_class="opinion"),
        ], "[F1-replay-settled] Will the event occur?")

        assert s.settled is True, "fixture must clear the settlement gates"
        assert s.stance == 1.0, "settlement claim replaces the claim set for stance"
        assert s.certainty < 0.95, "certainty still averages the displaced claim"
        self._assert_replays(s, self._replay(s.claims_detail))

    async def test_a_demoted_settlement_replays_as_ordinary_evidence(self, monkeypatch):
        """A ``settled`` claim below the gates votes as ordinary evidence. The
        persisted claim keeps ``settled=True``, so the replay must re-apply
        settlement_grade() rather than trust the flag — otherwise a stored row
        would reconstruct a pin the pipeline never published."""
        s = await _one_source(monkeypatch, [
            _claim(claim="Hedged 'it is over' claim.", stance=0.5, certainty=0.5,
                   settled=True, event_date="2026-07-20"),
            _claim(claim="Ordinary reporting.", stance=0.2, certainty=0.6),
        ], "[F1-replay-demoted] Will the event occur?")

        assert s.settled is None, "below-gate settlement must not pin"
        assert s.claims_detail[0].settled is True, "the demotion stays visible per-claim"
        self._assert_replays(s, self._replay(s.claims_detail))


#: Article age for fixtures that must stay settlement-grade, kept RELATIVE to today.
#:
#: `settlement_quality_floor` (0.20) gates the pin on the winning direction's combined
#: WEIGHT, and weight carries a recency term — so a hardcoded date silently ages a pin
#: fixture under the floor with no code change of any kind. This suite pinned
#: "2026-07-20" and went red on 2026-08-05, the day those rows turned 16 days old:
#: combined weight 0.209 -> 0.189, straight through the 0.20 floor overnight.
#:
#: Three days keeps recency ~0.74 and the combined weight ~0.68 — far enough clear that
#: no plausible floor retune re-arms the bomb. Fixtures that deliberately test an OLD row
#: still pass an absolute date explicitly.
_RECENT = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")


class TestPoolWireIsAdditive:
    """R8: additive persistence only — the estimate must not move."""

    @staticmethod
    def _bare(**over) -> dict:
        return {
            "stance": 0.5,
            "certainty": 0.6,
            "credibility_weight": 1.0,
            "relevance_score": 0.9,
            "evidence_weight": 0.6,
            "published_date": _RECENT,
            "settled": False,
            **over,
        }

    @staticmethod
    def _enrichment(index: int) -> dict:
        return {
            "url": f"https://outlet{index}.example.test/story",
            "source_id": f"outlet{index}.example.test",
            "outlet": f"Outlet {index}",
            "evidence_class": "reported_fact",
            "fact_signal": -0.4,
            "event_actors": "Alpha",
            "event_target": "Beta",
            "is_occurrence": False,
            "verified": False,
            "claims_detail": [
                {
                    "claim": f"Claim {index}.",
                    "quote": f"Quote {index}.",
                    "stance": 0.5,
                    "certainty": 0.6,
                    "evidence_class": "reported_fact",
                    "fact_signal": -0.4,
                    "is_occurrence": False,
                    "verified": False,
                },
            ],
        }

    async def test_estimate_is_bit_identical_with_and_without_the_new_fields(self):
        specs = [
            self._bare(stance=0.7, credibility_weight=1.4),
            self._bare(stance=-0.2, relevance_score=0.6),
            self._bare(stance=0.35, published_date="2026-06-01", evidence_weight=0.9),
        ]
        bare = await forecaster.run_pool_aggregate(PoolAggregateRequest(
            sources=[PoolSourceInput(**s) for s in specs],
            claim_deadline="2026-12-31",
            claim_direction="arrival",
        ))
        enriched = await forecaster.run_pool_aggregate(PoolAggregateRequest(
            sources=[
                PoolSourceInput(**{**s, **self._enrichment(i)})
                for i, s in enumerate(specs)
            ],
            claim_deadline="2026-12-31",
            claim_direction="arrival",
        ))
        assert enriched.model_dump() == bare.model_dump()

    async def test_settlement_path_is_bit_identical_too(self):
        """The pin is the highest-blast-radius branch in the estimator; a
        widened contract must not perturb it either."""
        specs = [
            self._bare(stance=0.95, certainty=0.95, settled=True,
                       settlement_event_date="2026-07-19"),
            self._bare(stance=0.96, certainty=0.95, settled=True,
                       settlement_event_date="2026-07-18"),
        ]
        bare = await forecaster.run_pool_aggregate(PoolAggregateRequest(
            sources=[PoolSourceInput(**s) for s in specs],
            claim_deadline="2026-12-31",
            claim_direction="arrival",
            claim_created_at="2026-07-01",
        ))
        enriched = await forecaster.run_pool_aggregate(PoolAggregateRequest(
            sources=[
                PoolSourceInput(**{**s, **self._enrichment(i)})
                for i, s in enumerate(specs)
            ],
            claim_deadline="2026-12-31",
            claim_direction="arrival",
            claim_created_at="2026-07-01",
        ))
        assert bare.settled is True, "fixture must exercise the settlement pin"
        assert enriched.model_dump() == bare.model_dump()

    def test_legacy_eight_field_row_still_validates(self):
        """Callers that never send the new fields must keep working — they are
        Optional with a None default, not merely 'usually absent'."""
        row = PoolSourceInput(**self._bare())
        assert row.claims_detail is None
        assert row.url is None
        assert row.evidence_class is None

    def test_claims_detail_round_trips_through_the_request_model(self):
        req = PoolAggregateRequest(sources=[
            PoolSourceInput(**{**self._bare(), **self._enrichment(0)}),
        ])
        detail = req.sources[0].claims_detail
        assert isinstance(detail[0], ClaimDetail)
        assert detail[0].claim == "Claim 0."
        assert detail[0].fact_signal == -0.4
        assert detail[0].verified is False
        # Unset per-claim fields default to None rather than being dropped.
        assert detail[0].specificity is None
        assert detail[0].event_date is None
