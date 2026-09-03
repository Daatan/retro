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
from pydantic import ValidationError

from forecast_api.models import ClaimDetail

from forecast_api import forecaster
from forecast_api.aggregation import claim_weighted_stance, resolve_stance_certainty
from forecast_api.models import (
    ArticleInput,
    ClaimDetail,
    ForecastRequest,
    PoolAggregateRequest,
    PoolSourceInput,
    SourceSignal,
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


class TestFacetRidesTheDominantClaim:
    """retro#485: `facet` (announcement/denial/neither) is wired end-to-end —
    schema-only since #483, now actually populated by reduce_article() and
    threaded onto SourceSignal, same dominant-claim rule as the other facets."""

    async def test_article_level_facet_is_the_dominant_claims(self, monkeypatch):
        s = await _one_source(monkeypatch, [
            _claim(claim="A minor precursor.", fact_signal=0.15,
                   is_occurrence=False, facet="neither"),
            _claim(claim="The event was announced.", fact_signal=0.6,
                   is_occurrence=True, facet="announcement"),
        ], "[F1-485-facet] Will the event occur?")

        assert s.facet == "announcement"
        assert [c.facet for c in s.claims_detail] == ["neither", "announcement"]

    async def test_facet_is_absent_when_no_claim_scored_a_fact_signal(self, monkeypatch):
        s = await _one_source(monkeypatch, [
            _claim(claim="Nothing bears on the event.",
                   fact_signal_absent_reason="no_fact_found"),
        ], "[F1-485-no-facet] Will the event occur?")

        assert s.fact_signal is None
        assert s.facet is None

    async def test_r8_estimate_unchanged_by_facet(self, monkeypatch):
        """R8: facet is pure shadow metadata (retro#485) — nothing in
        aggregation reads it, so the published estimate must be bit-identical
        regardless of which facet (or none) rides the dominant claim."""
        async def _forecast(question: str, **claim_kwargs):
            _patch(monkeypatch, [_claim(
                claim="The event was announced.", fact_signal=0.6,
                is_occurrence=True, **claim_kwargs,
            )])
            return await forecaster.run_forecast(ForecastRequest(
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

        resp_a = await _forecast("[F1-485-r8-a] Will the event occur?", facet="announcement")
        resp_b = await _forecast("[F1-485-r8-b] Will the event occur?", facet="denial")

        assert resp_a.mean == resp_b.mean
        assert resp_a.ci_low == resp_b.ci_low
        assert resp_a.ci_high == resp_b.ci_high
        assert resp_a.settled == resp_b.settled
        assert resp_a.sources[0].facet == "announcement"
        assert resp_b.sources[0].facet == "denial"


class TestFactSignalNullIsDistinguishable:
    """retro#471: fact_signal's null must not conflate 'no relevant fact',
    'a contrary fact too weak to anchor a value', and 'opinion' — a consumer
    reading claims_detail must be able to tell them apart per claim."""

    async def test_no_fact_found_is_distinguishable_from_opinion(self, monkeypatch):
        s = await _one_source(monkeypatch, [
            _claim(claim="Nothing bears on the event.",
                   fact_signal_absent_reason="no_fact_found"),
            _claim(claim="Pure opinion column.",
                   evidence_class="opinion", fact_signal_absent_reason="opinion"),
        ], "[F1-tristate] Will the event occur?")

        reasons = [c.fact_signal_absent_reason for c in s.claims_detail]
        assert reasons == ["no_fact_found", "opinion"]
        assert all(c.fact_signal is None for c in s.claims_detail)

    async def test_contrary_below_anchor_is_its_own_state(self, monkeypatch):
        s = await _one_source(monkeypatch, [
            _claim(claim="A weak, ambiguous contrary fact.",
                   fact_signal_absent_reason="contrary_below_anchor"),
        ], "[F1-tristate-contrary] Will the event occur?")

        assert s.claims_detail[0].fact_signal is None
        assert s.claims_detail[0].fact_signal_absent_reason == "contrary_below_anchor"

    async def test_reason_is_absent_when_fact_signal_is_present(self, monkeypatch):
        """The reason only exists to explain a null — a scored claim has none."""
        s = await _one_source(monkeypatch, [
            _claim(claim="A graded fact.", fact_signal=-0.4, is_occurrence=True),
        ], "[F1-tristate-scored] Will the event occur?")

        assert s.claims_detail[0].fact_signal == -0.4
        assert s.claims_detail[0].fact_signal_absent_reason is None

    async def test_article_level_reason_with_no_ambiguity(self, monkeypatch):
        """retro#481: the article-level reduction mirrors a single claim's
        reason directly when there is nothing to disambiguate."""
        s = await _one_source(monkeypatch, [
            _claim(claim="Nothing bears on the event.",
                   fact_signal_absent_reason="no_fact_found"),
        ], "[F1-481-single] Will the event occur?")

        assert s.fact_signal is None
        assert s.fact_signal_absent_reason == "no_fact_found"

    async def test_article_level_reason_picks_the_most_common(self, monkeypatch):
        """retro#481: with several fact_signal-null claims disagreeing on
        why, the article-level reason is the majority vote (same Counter
        tie-break as `evidence_class`'s `representative_class`)."""
        s = await _one_source(monkeypatch, [
            _claim(claim="Nothing bears on the event, take one.",
                   fact_signal_absent_reason="no_fact_found"),
            _claim(claim="Nothing bears on the event, take two.",
                   fact_signal_absent_reason="no_fact_found"),
            _claim(claim="Pure opinion column.",
                   evidence_class="opinion", fact_signal_absent_reason="opinion"),
        ], "[F1-481-majority] Will the event occur?")

        assert s.fact_signal is None
        assert s.fact_signal_absent_reason == "no_fact_found"

    async def test_article_level_reason_is_absent_when_fact_signal_is_present(self, monkeypatch):
        """retro#481: a scored article has a real fact_signal, so the
        article-level absent-reason is None, mirroring the per-claim rule."""
        s = await _one_source(monkeypatch, [
            _claim(claim="A graded fact.", fact_signal=-0.4, is_occurrence=True),
        ], "[F1-481-scored] Will the event occur?")

        assert s.fact_signal == -0.4
        assert s.fact_signal_absent_reason is None

    async def test_r8_estimate_unchanged_by_fact_signal_absent_reason(self, monkeypatch):
        """R8: fact_signal_absent_reason is pure shadow metadata (retro#481)
        — nothing in aggregation reads it, so the published estimate must be
        bit-identical regardless of which reason (or none) is carried."""
        async def _forecast(question: str, **claim_kwargs):
            _patch(monkeypatch, [_claim(claim="Nothing bears on the event.", **claim_kwargs)])
            return await forecaster.run_forecast(ForecastRequest(
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

        # Distinct questions — forecast_cache keys on (question, articles_hash),
        # not on the stubbed claim fields, so a shared question would make the
        # second call return the first call's cached response.
        resp_a = await _forecast("[F1-481-r8-a] Will the event occur?", fact_signal_absent_reason="no_fact_found")
        resp_b = await _forecast("[F1-481-r8-b] Will the event occur?", fact_signal_absent_reason="opinion")

        assert resp_a.mean == resp_b.mean
        assert resp_a.ci_low == resp_b.ci_low
        assert resp_a.ci_high == resp_b.ci_high
        assert resp_a.settled == resp_b.settled
        assert resp_a.sources[0].fact_signal_absent_reason == "no_fact_found"
        assert resp_b.sources[0].fact_signal_absent_reason == "opinion"


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
        assert replayed.fact_signal_absent_reason == signal.fact_signal_absent_reason
        assert replayed.facet == signal.facet
        assert replayed.reader_confidence_level == signal.reader_confidence_level
        assert replayed.reader_confidence_traps == signal.reader_confidence_traps

    async def test_every_scalar_replays_on_the_ordinary_path(self, monkeypatch):
        """Mixed classes, mixed fact-bearing, one anchor — the five reductions
        run over four different subsets here, so a replay that agrees on all of
        them is not agreeing by coincidence."""
        s = await _one_source(monkeypatch, [
            _claim(claim="Reported fact.", stance=0.7, certainty=0.8,
                   evidence_class="reported_fact", fact_signal=0.2, is_occurrence=True,
                   verified=True, event_actors="Alpha", event_target="Beta",
                   reader_confidence={"level": "high"}),
            _claim(claim="Columnist's view.", stance=-0.4, certainty=0.3,
                   evidence_class="opinion",
                   reader_confidence={"level": "low", "trap": "tone_vs_content"}),
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
            # Unique per index (retro#780): distinct actors/target so the
            # rows never collide on `event_key` and trigger the echo
            # collapse -- this class tests additivity of STORAGE fields,
            # not the collapse itself (covered in test_aggregation.py's
            # TestEventKeyDependenceCollapse), and a same-event collision
            # here is a false positive for that.
            "event_actors": f"Alpha{index}",
            "event_target": f"Beta{index}",
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


class TestConditionalFields:
    """Phase 1 capture (v1.1): conditional claims are pre-resolution shadow fields.

    All 9 conditional fields are Optional and default to None. They are recorded
    PRE-RESOLUTION (before enforce_* chain), asymmetric with other ClaimDetail fields.
    """

    def test_conditional_fields_are_optional_and_default_to_none(self):
        """Claims without conditionals have all 9 fields null."""
        claim = ClaimDetail(
            claim="Regular claim.",
            quote="A quote.",
            stance=0.5,
            certainty=0.6,
        )
        assert claim.is_conditional is None
        assert claim.antecedent_text is None
        assert claim.antecedent_text_en is None
        assert claim.antecedent_polarity is None
        assert claim.relation is None
        assert claim.strength is None
        assert claim.stated_probability is None
        assert claim.is_counterfactual is None
        assert claim.speaker is None

    def test_conditional_fields_can_be_populated(self):
        """Conditional claims carry all fields populated."""
        claim = ClaimDetail(
            claim="Likud gains 15 seats.",
            quote="If the ceasefire holds, Likud is expected to gain 15 seats.",
            stance=0.4,
            certainty=0.6,
            is_conditional=True,
            antecedent_text="if the ceasefire holds",
            antecedent_text_en="the ceasefire holds",
            antecedent_polarity=True,
            relation="raises",
            strength=None,
            stated_probability=None,
            is_counterfactual=False,
            speaker="Analysts",
        )
        assert claim.is_conditional is True
        assert claim.antecedent_text == "if the ceasefire holds"
        assert claim.antecedent_text_en == "the ceasefire holds"
        assert claim.antecedent_polarity is True
        assert claim.relation == "raises"
        assert claim.speaker == "Analysts"

    def test_build_claims_detail_carries_the_conditional_fields(self):
        """retro#566: the projection onto the wire must copy all 9 fields.

        The three tests above exercise only the ClaimDetail model; the
        projection in ``forecaster.build_claims_detail`` was never covered, and
        it silently dropped every conditional field from 2026-08-09 until this
        test existed — 12 days of prod rows with the keys present and all null.
        """
        p = PredictionExtraction(
            claim="Likud gains 15 seats.",
            quote="If the ceasefire holds, Likud is expected to gain 15 seats.",
            stance=0.4,
            certainty=0.6,
            is_conditional=True,
            antecedent_text="if the ceasefire holds",
            antecedent_text_en="the ceasefire holds",
            antecedent_polarity=True,
            relation="raises",
            strength="likely",
            stated_probability=0.7,
            is_counterfactual=False,
            speaker="Analysts",
        )
        (d,) = forecaster.build_claims_detail([p])
        assert d.is_conditional is True
        assert d.antecedent_text == "if the ceasefire holds"
        assert d.antecedent_text_en == "the ceasefire holds"
        assert d.antecedent_polarity is True
        assert d.relation == "raises"
        assert d.strength == "likely"
        assert d.stated_probability == 0.7
        assert d.is_counterfactual is False
        assert d.speaker == "Analysts"

    def test_conditional_fields_round_trip_through_json(self):
        """ClaimDetail with conditionals survives JSON serialization."""
        claim = ClaimDetail(
            claim="Likud gains 15 seats.",
            quote="If the ceasefire holds, Likud is expected to gain 15 seats.",
            stance=0.4,
            certainty=0.6,
            is_conditional=True,
            antecedent_text="if the ceasefire holds",
            antecedent_text_en="the ceasefire holds",
            antecedent_polarity=True,
            relation="raises",
            speaker="Analysts",
        )
        # Serialize to JSON dict (via Pydantic model_dump)
        dumped = claim.model_dump()
        # Deserialize back
        restored = ClaimDetail(**dumped)
        assert restored.is_conditional is True
        assert restored.antecedent_text == "if the ceasefire holds"
        assert restored.relation == "raises"

    def test_conditional_fields_in_claims_detail_list(self):
        """Multiple claims with mixed conditional/non-conditional fields."""
        claims = [
            ClaimDetail(claim="A.", quote="Q1.", stance=0.5, certainty=0.6),
            ClaimDetail(
                claim="B.", quote="If X then B.", stance=0.4, certainty=0.5,
                is_conditional=True, antecedent_text="if X", antecedent_text_en="X",
                relation="raises",
            ),
            ClaimDetail(claim="C.", quote="Q3.", stance=-0.2, certainty=0.7),
        ]
        assert claims[0].is_conditional is None
        assert claims[1].is_conditional is True
        assert claims[2].is_conditional is None

    async def test_settlement_gate_unchanged_with_conditional_fields(self, monkeypatch):
        """CRITICAL (§3.0): Settlement-match gate verdicts unchanged when conditional
        fields are populated. The gate reads only 4 fields (claim, quote, event_date, settled)
        which are all outside the new conditional field set.

        This test runs the settlement gate on the exact same source twice: once with
        conditional fields populated, once without. Verdicts must be identical.
        """
        from forecast_api.forecaster import _settlement_votes

        # Settlement claim: should pin when sufficiently confident
        settlement_claim = {
            "claim": "The election was held.",
            "quote": "The election took place on Tuesday.",
            "stance": 1.0,
            "certainty": 0.95,
            "settled": True,
            "event_date": "2026-07-20",
        }
        # Dummy colour claim
        colour_claim = {
            "claim": "Candidate X will win.",
            "quote": "X leads in polls.",
            "stance": 0.3,
            "certainty": 0.5,
            "settled": False,
        }

        # Run gate WITHOUT conditional fields
        claims_bare = [
            ClaimDetail(**settlement_claim),
            ClaimDetail(**colour_claim),
        ]
        verdicts_bare = _settlement_votes(outlet="AP", claims_detail=claims_bare)

        # Run gate WITH conditional fields populated
        settlement_with_conditionals = {
            **settlement_claim,
            "is_conditional": False,
            "antecedent_text": None,
            "antecedent_text_en": None,
            "antecedent_polarity": None,
            "relation": None,
            "strength": None,
            "stated_probability": None,
            "is_counterfactual": None,
            "speaker": None,
        }
        claims_enriched = [
            ClaimDetail(**settlement_with_conditionals),
            ClaimDetail(**colour_claim),
        ]
        verdicts_enriched = _settlement_votes(outlet="AP", claims_detail=claims_enriched)

        # Verdicts must be identical
        assert verdicts_bare == verdicts_enriched, \
            "Settlement gate verdicts changed when conditional fields were populated"


class TestClaimStrengthCertaintyAlias:
    """`ClaimDetail.certainty` and `.claim_strength` are one number under two
    names for one schema cycle (Oracle 1.5 Phase 1, retro#680).

    daatan persists `claims_detail` rows (daatan#1235), so rows written before
    the rename carry only `certainty` and rows written after may carry only
    `claim_strength`. Both must load, and both must come back out carrying both
    names — otherwise a consumer reading the name this deploy didn't write sees
    a missing required field on a row that is perfectly well-formed.
    """

    def test_a_pre_rename_row_loads_and_gains_the_new_name(self):
        c = ClaimDetail.model_validate({"claim": "c", "stance": 0.4, "certainty": 0.82})
        assert (c.certainty, c.claim_strength) == (0.82, 0.82)

    def test_a_post_rename_row_loads_and_keeps_the_alias(self):
        c = ClaimDetail.model_validate({"claim": "c", "stance": 0.4, "claim_strength": 0.82})
        assert (c.certainty, c.claim_strength) == (0.82, 0.82)

    def test_a_row_carrying_both_names_is_accepted_when_they_agree(self):
        c = ClaimDetail.model_validate(
            {"claim": "c", "stance": 0.4, "certainty": 0.82, "claim_strength": 0.82}
        )
        assert c.claim_strength == 0.82

    def test_a_row_whose_two_names_disagree_is_rejected_loudly(self):
        """Silently picking a winner would publish one of two numbers that both
        claim to be the same field — the failure mode this alias exists to avoid."""
        with pytest.raises(ValidationError, match="must match"):
            ClaimDetail.model_validate(
                {"claim": "c", "stance": 0.4, "certainty": 0.82, "claim_strength": 0.10}
            )

    def test_both_names_are_emitted_on_the_wire(self):
        dumped = ClaimDetail.model_validate(
            {"claim": "c", "stance": 0.4, "claim_strength": 0.82}
        ).model_dump()
        assert dumped["certainty"] == dumped["claim_strength"] == 0.82


class TestReaderConfidenceThreading:
    """`reader_confidence` {level, trap} through the claim layer and its
    article-level rollup (Oracle 1.5 Phase 1, retro#681).

    The rollup is deliberately shaped like none of the other five reductions,
    and that is what these pin. A mean would average away the one claim the
    reader wobbled on — which is the only claim the field exists to surface —
    and a most-common vote over traps would throw away the second of two
    genuinely different difficulties. Both would look correct in review and
    make the harvested data useless for the Phase 4 consumer.
    """

    @staticmethod
    def _rc(level, trap=None):
        return {"level": level, "trap": trap}

    @staticmethod
    def _reduce(claims):
        from forecast_api.config import settings

        return forecaster.reduce_article(
            claims,
            settlement_min_stance=settings.settlement_min_claim_stance,
            settlement_min_certainty=settings.settlement_min_claim_certainty,
            class_weights=settings.evidence_class_weight,
            class_weight_default=settings.evidence_class_weight_default,
            class_weight_unclassified_cap=settings.evidence_class_weight_unclassified_cap,
        )

    @staticmethod
    def _detail(level=None, trap=None, **over):
        payload = {"claim": "c", "stance": 0.4, "claim_strength": 0.6, **over}
        if level is not None:
            payload["reader_confidence"] = {"level": level, "trap": trap}
        return ClaimDetail.model_validate(payload)

    def test_the_claim_layer_carries_it_verbatim(self):
        """daatan stores claims_detail as it arrives, so the shape the model
        answered in is the shape that has to survive the projection."""
        [detail] = forecaster.build_claims_detail([
            _claim(reader_confidence={"level": "low", "trap": "negation"})
        ])
        assert detail.reader_confidence is not None
        assert detail.reader_confidence.level == "low"
        assert detail.reader_confidence.trap == "negation"

    def test_a_claim_without_it_projects_to_none(self):
        [detail] = forecaster.build_claims_detail([_claim()])
        assert detail.reader_confidence is None

    def test_the_article_takes_the_WORST_level_not_a_mean(self):
        """An article is only as readable as its least readable claim. Two
        confident claims must not average a `low` one out of existence."""
        reduced = self._reduce([
            self._detail("high"), self._detail("high"), self._detail("low")
        ])
        assert reduced.reader_confidence_level == "low"

    def test_medium_beats_high_but_loses_to_low(self):
        assert self._reduce(
            [self._detail("high"), self._detail("medium")]
        ).reader_confidence_level == "medium"
        assert self._reduce(
            [self._detail("medium"), self._detail("low")]
        ).reader_confidence_level == "low"

    def test_traps_are_collected_not_voted(self):
        """Two claims tripping two different traps is two facts about this
        article. `evidence_class`'s most-common vote would keep one."""
        reduced = self._reduce([
            self._detail("medium", "negation"),
            self._detail("medium", "numeric_comparison"),
        ])
        assert reduced.reader_confidence_traps == ["negation", "numeric_comparison"]

    def test_repeated_traps_are_deduped_in_first_seen_order(self):
        reduced = self._reduce([
            self._detail("medium", "tone_vs_content"),
            self._detail("low", "negation"),
            self._detail("high", "tone_vs_content"),
        ])
        assert reduced.reader_confidence_traps == ["tone_vs_content", "negation"]

    def test_a_level_with_no_trap_contributes_a_level_and_no_trap(self):
        reduced = self._reduce([self._detail("high"), self._detail("low")])
        assert reduced.reader_confidence_level == "low"
        assert reduced.reader_confidence_traps is None

    def test_an_article_where_nothing_answered_rolls_up_to_none(self):
        """Every row extracted before v5. Neither field may invent a value."""
        reduced = self._reduce([self._detail(), self._detail()])
        assert reduced.reader_confidence_level is None
        assert reduced.reader_confidence_traps is None

    def test_the_rollup_spans_claims_the_settlement_subset_excluded(self):
        """`stance` reduces over the settlement-grade claims when an article
        has any; this must not. A claim that failed the settlement bar is
        still a claim the reader had to read, and hiding its `low` behind a
        confident settlement claim is precisely the miss Phase 4 needs."""
        reduced = self._reduce([
            self._detail("high", stance=0.98, claim_strength=0.98, settled=True),
            self._detail("low", "inference_needed", stance=0.2, claim_strength=0.3),
        ])
        assert reduced.settled is True
        assert reduced.reader_confidence_level == "low"
        assert reduced.reader_confidence_traps == ["inference_needed"]

    async def test_it_reaches_the_wire_on_a_real_forecast(self, monkeypatch):
        """End to end through the stubbed pipeline: the extractor answers, and
        both the per-claim field and the article rollup come out on the
        SourceSignal the caller persists."""
        source = await _one_source(monkeypatch, [
            _claim(claim="A", reader_confidence={"level": "high"}),
            _claim(claim="B", reader_confidence={"level": "low", "trap": "negation"}),
        ], "[P1-rc] Will the event occur?")

        assert source.reader_confidence_level == "low"
        assert source.reader_confidence_traps == ["negation"]
        assert [c.reader_confidence.level for c in source.claims_detail] == ["high", "low"]


def test_the_provenance_schema_version_is_pinned():
    """The four wiring tests that used to hard-code "1.1" now derive from this
    constant, so nothing else fails when it moves. This is the one place a bump
    has to be deliberate — schema_version is what daatan gates on."""
    from forecast_api.models import PROVENANCE_SCHEMA_VERSION

    assert PROVENANCE_SCHEMA_VERSION == "1.6"


class TestClaimDecompositionThreading:
    """`claim_actor` / `claim_predicate` / `claim_scope` through the live path
    (Oracle 1.5 Phase 1, retro#697).

    Unlike every other shadow field on SourceSignal, these are not properties of
    the source at all — they decompose the QUESTION's event, which `PROMPT_PREFIX`
    § MATCH THE EVENT has required the model to do since v1 without ever giving it
    a field to answer in. So the value of an end-to-end test here is the same as
    for `consensus_view` and larger: nothing reads them, so nothing would notice
    them arriving None, and the projection is the step that silently drops a field
    (retro#566).
    """

    async def _decomposed(self, monkeypatch, question, **fields):
        async def fake_extract(**kwargs):
            return (
                ExtractionOutput(predictions=[_claim(claim="A")], **fields),
                {"total_tokens": 0},
            )

        _patch(monkeypatch, [_claim(claim="A")])
        monkeypatch.setattr(forecaster, "extract_predictions", fake_extract)
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
        return resp.sources[0]

    async def test_all_three_reach_the_source_signal(self, monkeypatch):
        source = await self._decomposed(
            monkeypatch, "[P1-cd] Will the event occur?",
            claim_actor={"name": "Party Y", "type": "party"},
            claim_predicate="withdraws from the parliamentary race",
            claim_scope="at least one party, before the election",
        )
        assert source.claim_actor is not None
        assert source.claim_actor.name == "Party Y"
        assert source.claim_actor.type == "party"
        assert source.claim_predicate == "withdraws from the parliamentary race"
        assert source.claim_scope == "at least one party, before the election"

    async def test_they_do_not_displace_their_neighbours(self, monkeypatch):
        """Six article/question-level fields now ride the same outcome, adjacent
        to each other, so a positional slip would still populate every one of them
        and still look right in review — the failure `consensus_view` was given
        the same test for. Values that cannot be mistaken for one another are what
        makes it detectable: the two strings are distinct, and neither is a
        `consensus_view` member.
        """
        source = await self._decomposed(
            monkeypatch, "[P1-cd-order] Will the event occur?",
            author_lean=0.8,
            author_lean_certainty=0.45,
            consensus_view="expects_no",
            claim_actor={"name": "Airline A", "type": "company"},
            claim_predicate="operates daily departures from Hub H",
            claim_scope="more than 250 a day, by the deadline",
        )
        assert source.author_lean == 0.8
        assert source.author_lean_certainty == 0.45
        assert source.consensus_view == "expects_no"
        assert source.claim_actor.name == "Airline A"
        assert source.claim_predicate == "operates daily departures from Hub H"
        assert source.claim_scope == "more than 250 a day, by the deadline"

    async def test_an_unanswered_decomposition_rolls_up_to_none(self, monkeypatch):
        """The prompt asks for all three on every call, so None here means the
        model did not answer — which is the number the fill rate measures. A
        default would make the field read as filled on exactly the calls where it
        was not.
        """
        source = await _one_source(
            monkeypatch, [_claim(claim="A")], "[P1-cd-null] Will the event occur?"
        )
        assert source.claim_actor is None
        assert source.claim_predicate is None
        assert source.claim_scope is None

    async def test_a_malformed_actor_does_not_drop_the_article(self, monkeypatch):
        """The drop guard's whole purpose, tested through the live path rather
        than on the model in isolation: `complete_structured` runs instructor with
        `max_retries=1`, so a bare string where the object belongs would raise out
        of ExtractionOutput and cost a real article. The claim must still arrive;
        only the malformed field is nulled.
        """
        async def fake_extract(**kwargs):
            return (
                ExtractionOutput.model_validate({
                    "predictions": [_claim(claim="A").model_dump()],
                    "claim_actor": "Party Y",
                    "claim_predicate": "withdraws from the race",
                }),
                {"total_tokens": 0},
            )

        _patch(monkeypatch, [_claim(claim="A")])
        monkeypatch.setattr(forecaster, "extract_predictions", fake_extract)
        resp = await forecaster.run_forecast(ForecastRequest(
            question="[P1-cd-malformed] Will the event occur?",
            articles=[ArticleInput(
                url="https://fixture.example.test/a1",
                title="Vellum sonata dispatch",
                snippet="Fixture snippet, long enough to be usable by the pipeline.",
                source="fixture",
                published_date="2026-07-28",
                text=_BODY,
            )],
        ))
        assert len(resp.sources) == 1                       # the article survived
        assert resp.sources[0].claims_detail                # and so did its claim
        assert resp.sources[0].claim_actor is None          # only the bad field went
        assert resp.sources[0].claim_predicate == "withdraws from the race"

    def test_they_are_question_level_and_get_no_claim_layer_copy(self):
        """The asymmetry against `tone`/`voice`, pinned so it is not read as an
        oversight. Those are per-claim with an article-level rollup because they
        are properties of a quote. These three describe the RELATED EVENT, which
        is identical for every claim in the article and every article in the
        forecast — a per-claim copy would be N identical strings billed on every
        claim, and `settlement_semantic.ClaimSubject`, the consumer, wants exactly
        one per question.
        """
        assert "claim_actor" in SourceSignal.model_fields
        assert "claim_predicate" in SourceSignal.model_fields
        assert "claim_scope" in SourceSignal.model_fields
        for field in ("claim_actor", "claim_predicate", "claim_scope"):
            assert field not in ClaimDetail.model_fields


class TestReportKindAndConsensusViewThreading:
    """`report_kind` (per claim) and `consensus_view` (per article) through the
    live path (Oracle 1.5 Phase 1, retro#686 — unparked from #673).

    Both are shadow: populated, read by nothing. Which is exactly
    why they get an end-to-end test rather than a projection test. A shadow
    field nothing reads has no consumer to notice it arriving as None, so the
    only thing standing between "harvested" and "silently 0% filled for a month"
    is a test that runs the real pipeline and looks at what comes out the far
    end.
    """

    async def test_report_kind_reaches_the_claim_layer_per_claim(self, monkeypatch):
        """daatan persists claims_detail verbatim (daatan#1235), so the value
        the model answered has to survive the projection unchanged — and stay
        attached to ITS claim. An article mixing a level report and a change
        report is the normal case, not an edge case."""
        source = await _one_source(monkeypatch, [
            _claim(claim="The rate stands at 8.75%", report_kind="level"),
            _claim(claim="The bank cut by 25bp", report_kind="change"),
        ], "[P1-rk] Will the event occur?")

        assert [c.report_kind for c in source.claims_detail] == ["level", "change"]

    async def test_a_claim_without_report_kind_projects_to_none(self, monkeypatch):
        """Every claim extracted before v8, and every claim where the model
        judged that neither kind fits — the prompt asks it to omit rather than
        guess, so None has to mean "no answer", not "defaulted to level"."""
        source = await _one_source(
            monkeypatch, [_claim(claim="A")], "[P1-rk-null] Will the event occur?"
        )
        assert source.claims_detail[0].report_kind is None

    async def test_consensus_view_reaches_the_source_signal(self, monkeypatch):
        """The field is article-level, so it rides on the SourceSignal beside
        `author_lean` rather than on any claim. This is the test that fails if
        `_process_article`'s tuple widens without its unpack site following —
        the failure mode that makes an added field look shipped and arrive
        None."""
        async def fake_extract(**kwargs):
            return (
                ExtractionOutput(predictions=[_claim(claim="A")], consensus_view="divided"),
                {"total_tokens": 0},
            )

        _patch(monkeypatch, [_claim(claim="A")])
        monkeypatch.setattr(forecaster, "extract_predictions", fake_extract)

        resp = await forecaster.run_forecast(ForecastRequest(
            question="[P1-cv] Will the event occur?",
            articles=[ArticleInput(
                url="https://fixture.example.test/a1",
                title="Vellum sonata dispatch",
                snippet="Fixture snippet, long enough to be usable by the pipeline.",
                source="fixture",
                published_date="2026-07-28",
                text=_BODY,
            )],
        ))
        assert resp.sources[0].consensus_view == "divided"

    async def test_consensus_view_does_not_displace_author_lean(self, monkeypatch):
        """Both are article-level, both ride the same widened tuple, and they
        are adjacent in it — so a positional unpack that slipped by one would
        still populate every field and still look right in review. Values that
        cannot be mistaken for each other are what makes that detectable: an
        author who expects yes while reporting that everyone else expects no.
        """
        async def fake_extract(**kwargs):
            return (
                ExtractionOutput(
                    predictions=[_claim(claim="A")],
                    author_lean=0.8,
                    author_lean_certainty=0.45,
                    consensus_view="expects_no",
                ),
                {"total_tokens": 0},
            )

        _patch(monkeypatch, [_claim(claim="A")])
        monkeypatch.setattr(forecaster, "extract_predictions", fake_extract)

        resp = await forecaster.run_forecast(ForecastRequest(
            question="[P1-cv-order] Will the event occur?",
            articles=[ArticleInput(
                url="https://fixture.example.test/a1",
                title="Vellum sonata dispatch",
                snippet="Fixture snippet, long enough to be usable by the pipeline.",
                source="fixture",
                published_date="2026-07-28",
                text=_BODY,
            )],
        ))
        source = resp.sources[0]
        assert source.author_lean == 0.8
        assert source.author_lean_certainty == 0.45
        assert source.consensus_view == "expects_no"

    async def test_an_article_that_says_nothing_rolls_up_to_none(self, monkeypatch):
        """Most articles never report what anyone else expects. Null is the
        ordinary answer, and a default would make the field read as filled on
        exactly the articles carrying no consensus at all."""
        source = await _one_source(
            monkeypatch, [_claim(claim="A")], "[P1-cv-null] Will the event occur?"
        )
        assert source.consensus_view is None

    def test_report_kind_gets_no_article_rollup_on_purpose(self):
        """`reader_confidence` rolls up to the SourceSignal (worst level, union
        of traps) because its Phase 4 consumer gates a whole article. This one
        does not, and the asymmetry is deliberate rather than an oversight:
        E3b's recency integrator reads report_kind per CLAIM — a level report
        resets the decay, a change report enters it — so an article-level
        summary would answer a question nothing asks and would have to invent a
        reduction (a vote? a worst-case?) with no consumer to justify it.

        Pinned because the obvious review note is "why isn't this rolled up
        like the last one".
        """
        from forecast_api.models import SourceSignal

        assert "report_kind" in ClaimDetail.model_fields
        assert "report_kind" not in SourceSignal.model_fields
        assert "consensus_view" in SourceSignal.model_fields
        assert "consensus_view" not in ClaimDetail.model_fields


class TestQuantityThreading:
    """`quantity` through the live path (Oracle 1.5 Phase 1, retro#683).

    Shadow, like every field above it: populated, read by nothing. So it gets an
    end-to-end test for the reason retro#686's do — a shadow field has no
    consumer to notice it arriving as None, and the only thing standing between
    "harvested" and "silently 0% filled for a month" is a test that runs the real
    pipeline and looks at what comes out the far end. retro#566 is the precedent:
    nine conditional fields elicited, answered, and dropped by the projection.
    """

    async def test_it_reaches_the_claim_layer_verbatim(self, monkeypatch):
        source = await _one_source(monkeypatch, [
            _claim(claim="Throughput was 1.4 million containers",
                   quantity={"value": 1400000, "unit": "containers", "comparator": "=",
                             "as_of": "2026-06-30"}),
            _claim(claim="Volume stayed below 2 million containers",
                   quantity={"value": 2000000, "unit": "containers", "comparator": "<"}),
        ], "[P1-q] Will the event occur?")

        first, second = source.claims_detail
        assert (first.quantity.value, first.quantity.unit) == (1400000, "containers")
        assert (first.quantity.comparator, first.quantity.as_of) == ("=", "2026-06-30")
        assert (second.quantity.comparator, second.quantity.value_hi) == ("<", None)

    async def test_a_range_keeps_both_bounds(self, monkeypatch):
        """`value_hi` is the one field that is meaningless alone, so a projection
        carrying `value` and dropping it would look populated and compare wrong."""
        source = await _one_source(monkeypatch, [
            _claim(claim="Between 1.8 and 2.2 million containers",
                   quantity={"value": 1800000, "unit": "containers",
                             "comparator": "between", "value_hi": 2200000}),
        ], "[P1-q-range] Will the event occur?")

        q = source.claims_detail[0].quantity
        assert (q.value, q.value_hi, q.comparator) == (1800000, 2200000, "between")

    async def test_a_claim_without_a_figure_projects_to_none(self, monkeypatch):
        """Most claims. The prompt asks the model to omit rather than invent a
        number, so None has to mean "the quote states no figure" — not zero."""
        source = await _one_source(
            monkeypatch, [_claim(claim="A")], "[P1-q-null] Will the event occur?"
        )
        assert source.claims_detail[0].quantity is None

    async def test_it_does_not_displace_quantitative_estimate(self, monkeypatch):
        """The two are near neighbours in both models and the prompt spends a
        paragraph keeping them apart. retro#362's bug was a share written into
        `quantitative_estimate` and rewritten to stance = 2*qe-1 in code, so a
        projection that crossed the two would reintroduce it on the wire."""
        source = await _one_source(monkeypatch, [
            _claim(claim="A poll-aggregator model gives 22%", quantitative_estimate=0.22),
            _claim(claim="The party polls at 28% of the vote",
                   quantity={"value": 28, "unit": "percent", "comparator": "="}),
        ], "[P1-q-qe] Will the event occur?")

        cited, share = source.claims_detail
        assert cited.quantitative_estimate == 0.22 and cited.quantity is None
        assert share.quantitative_estimate is None and share.quantity.value == 28

    async def test_a_malformed_quantity_costs_the_field_and_not_the_article(self, monkeypatch):
        """The boundary the drop-guard exists for, seen from the far end. The
        extractor's `_drop_malformed_quantity` nulls a half-answer rather than
        letting it raise out of ExtractionOutput — and what that buys is this: the
        claim, its stance and every other field still reach the caller."""
        source = await _one_source(monkeypatch, [
            _claim(claim="Thirty six seats", stance=0.7, quantity={"value": 36}),
        ], "[P1-q-bad] Will the event occur?")

        claim = source.claims_detail[0]
        assert claim.quantity is None
        assert claim.stance == 0.7 and claim.claim == "Thirty six seats"


class TestToneAndVoiceThreading:
    """`tone` and `voice` through the live path (Oracle 1.5 Phase 1, retro#684).

    Same reason as `quantity` above: shadow fields have no consumer to notice
    them arriving as None, so the only thing between "harvested" and "silently
    0% filled for a month" is a test that runs the real pipeline and reads what
    comes out the far end (retro#566).

    Both roll up to the SourceSignal as the DOMINANT claim's, unlike
    `report_kind` — the asymmetry is pinned below.
    """

    async def test_tone_reaches_the_claim_layer_per_claim(self, monkeypatch):
        source = await _one_source(monkeypatch, [
            _claim(claim="A catastrophic breach", tone="alarm"),
            _claim(claim="A welcome recovery", tone="approve"),
            _claim(claim="The rate stands at 8.75%", tone="neutral"),
        ], "[P1-tone] Will the event occur?")

        assert [c.tone for c in source.claims_detail] == ["alarm", "approve", "neutral"]

    async def test_voice_reaches_the_claim_layer_with_its_name(self, monkeypatch):
        """`attributed_to` is the half that makes a wire collapsible — it is the
        name the reception matrix keys its column on — so a projection carrying
        `kind` and dropping the name would look populated and count thirty
        correlated outlets as thirty sources anyway."""
        source = await _one_source(monkeypatch, [
            _claim(claim="Reuters reported the review reopened",
                   voice={"kind": "wire", "attributed_to": "Reuters"}),
            _claim(claim="The reporter's own read", voice={"kind": "byline"}),
        ], "[P1-voice] Will the event occur?")

        wire, byline = source.claims_detail
        assert (wire.voice.kind, wire.voice.attributed_to) == ("wire", "Reuters")
        assert byline.voice.kind == "byline" and byline.voice.attributed_to is None

    async def test_a_claim_the_model_did_not_answer_projects_to_none(self, monkeypatch):
        """The prompt asks for `tone` on every prediction, `neutral` included, so
        None here means the model did not answer — not that the quote was
        even-handed. That is the distinction the fill rate measures."""
        source = await _one_source(
            monkeypatch, [_claim(claim="A")], "[P1-tv-null] Will the event occur?"
        )
        assert source.claims_detail[0].tone is None
        assert source.claims_detail[0].voice is None

    async def test_both_ride_the_dominant_claim_to_the_article_level(self, monkeypatch):
        """The issue's threading rule: article-level `tone`/`voice` are the
        dominant claim's — the same claim `facet` and `verified` ride from, picked
        by max |fact_signal|. Pinned with the dominant claim NOT first in the
        list, so an implementation that took `claims_detail[0]` would fail here.
        """
        source = await _one_source(monkeypatch, [
            _claim(claim="A minor precursor", stance=0.2, fact_signal=0.2,
                   tone="neutral", voice={"kind": "byline"}),
            _claim(claim="The decisive report", stance=0.9, fact_signal=0.9,
                   tone="alarm", voice={"kind": "wire", "attributed_to": "Reuters"}),
        ], "[P1-tv-dom] Will the event occur?")

        assert source.tone == "alarm"
        assert (source.voice.kind, source.voice.attributed_to) == ("wire", "Reuters")
        assert [c.tone for c in source.claims_detail] == ["neutral", "alarm"], (
            "the per-claim layer keeps every claim's own answer, not the dominant one's"
        )

    async def test_the_article_rollup_is_none_when_no_claim_carries_a_fact(self, monkeypatch):
        """`tone`/`voice` ride the fact lane, so they inherit its emptiness rule:
        no `fact_signal` anywhere means no dominant claim to ride, exactly as
        `facet` and `verified` already behave. The per-claim values survive."""
        source = await _one_source(monkeypatch, [
            _claim(claim="Pure opinion", tone="alarm", voice={"kind": "byline"},
                   fact_signal_absent_reason="opinion"),
        ], "[P1-tv-nofact] Will the event occur?")

        assert source.fact_signal is None
        assert source.tone is None and source.voice is None
        assert source.claims_detail[0].tone == "alarm"

    async def test_a_malformed_voice_costs_the_field_and_not_the_article(self, monkeypatch):
        """The drop-guard boundary from the far end: `_drop_malformed_voice`
        nulls a half-answer rather than raising out of ExtractionOutput, and what
        that buys is the claim, its stance and every other field still arriving.
        """
        source = await _one_source(monkeypatch, [
            _claim(claim="Reuters said so", stance=0.7,
                   voice={"attributed_to": "Reuters"}, tone="bullish"),
        ], "[P1-tv-bad] Will the event occur?")

        claim = source.claims_detail[0]
        assert claim.voice is None and claim.tone is None
        assert claim.stance == 0.7 and claim.claim == "Reuters said so"

    def test_they_roll_up_where_report_kind_does_not(self):
        """The asymmetry against the test above, made explicit because the obvious
        review note is "why does THIS one roll up when the last one didn't".
        `report_kind`'s consumer (E3b's recency integrator) reads per claim, so an
        article-level summary would have to invent a reduction with nothing asking
        for it. These two have an article-level consumer already named: Phase 3
        S2's reception matrix counts sources per ARTICLE, and S4 projects the
        article's register out at rating time.
        """
        from forecast_api.models import SourceSignal

        for field in ("tone", "voice"):
            assert field in ClaimDetail.model_fields
            assert field in SourceSignal.model_fields
        assert "report_kind" not in SourceSignal.model_fields


class TestGroundsWithdrawn:
    """`grounds` is no longer elicited (retro#774) — pin the withdrawal, not the threading.

    What used to live here proved the field reached the claim layer and rolled up
    as the dominant claim's. That contract is deliberately gone: the field left
    `PredictionExtraction`, so it left the JSON schema instructor serialises into
    every call, so the model is not asked. These pin the two things a reader will
    actually want to know afterwards — nothing new carries it, and the rows that
    already do still parse.

    Why it went: asking for `grounds` alongside `evidence_class` took the share of
    pool claims arriving unclassified from 0.32% (35/10,945) to 4.26% (10/235) on
    prod, comparing rows that carry an `extractor_prompt_version`. An unclassified
    claim is capped at 0.25, 4x below `reported_fact`. A field read by nothing was
    buying that with live weighting quality.
    """

    async def test_nothing_extracted_now_carries_grounds(self, monkeypatch):
        """Even when the payload offers one. The field is gone from the extraction
        model, so a volunteered `grounds` is dropped on the way in rather than
        projected — which is the whole point of removing it from the schema."""
        source = await _one_source(monkeypatch, [
            _claim(claim="The ministry pledged to act", stance=0.7, fact_signal=0.7,
                   grounds={"kind": "authority_asserted", "basis": "the 12 March statement"}),
        ], "[P1-grounds-withdrawn] Will the event occur?")

        assert source.claims_detail[0].grounds is None
        assert source.grounds is None, "no dominant-claim rollup either"
        assert source.claims_detail[0].stance == 0.7, "the claim is untouched"

    def test_the_wire_field_survives_so_stored_rows_still_parse(self):
        """Rows written 2026-08-31 -> 09-02 carry `grounds`. Withdrawing the
        question must not make that history unreadable, so `ClaimDetail` keeps the
        field and keeps validating it."""
        from forecast_api.models import ClaimDetail

        stored = ClaimDetail.model_validate({
            "claim": "The ministry pledged to act",
            "stance": 0.7,
            "certainty": 0.65,
            "claim_strength": 0.65,
            "grounds": {"kind": "authority_asserted", "basis": "the 12 March statement"},
        })
        assert stored.grounds is not None
        assert (stored.grounds.kind, stored.grounds.basis) == (
            "authority_asserted", "the 12 March statement")
