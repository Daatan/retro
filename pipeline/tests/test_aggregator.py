"""aggregate_predictions() — cell-level certainty×specificity-weighted mean/median.
Pure math, no LLM/network (unlike aggregate_article_predictions, mocked elsewhere)."""

import pytest

from tm.aggregator import aggregate_article_predictions, aggregate_predictions
from tm.models import PredictionExtraction


def _pred(stance, certainty, specificity=None, **overrides):
    overrides.setdefault("quote", "q")
    overrides.setdefault("claim", "c")
    return PredictionExtraction(
        stance=stance, certainty=certainty, specificity=specificity, **overrides,
    )


def test_empty_predictions_raises():
    with pytest.raises(ValueError):
        aggregate_predictions([])


def test_single_prediction_passes_through():
    sig = aggregate_predictions([_pred(0.6, 0.8, specificity=0.5)])
    assert sig.claim_count == 1
    assert sig.stance == pytest.approx(0.6)
    assert sig.certainty == pytest.approx(0.8)
    assert sig.specificity == pytest.approx(0.5)
    assert sig.quotes == ["q"] and sig.claims == ["c"]


def test_weighted_mean_favors_higher_certainty_specificity():
    # p1: weight 0.9*1.0=0.9, stance=1.0. p2: weight 0.1*1.0=0.1, stance=-1.0.
    sig = aggregate_predictions([
        _pred(1.0, 0.9, specificity=1.0),
        _pred(-1.0, 0.1, specificity=1.0),
    ])
    expected = (1.0 * 0.9 + -1.0 * 0.1) / (0.9 + 0.1)
    assert sig.stance == pytest.approx(expected)
    assert sig.stance > 0  # dominated by the higher-weight prediction


def test_missing_specificity_defaults_weight_to_certainty():
    # No specificity given -> weight = certainty * 1.0.
    sig = aggregate_predictions([_pred(0.4, 0.5), _pred(0.8, 0.5)])
    assert sig.stance == pytest.approx((0.4 + 0.8) / 2)
    assert sig.specificity is None


def test_zero_total_weight_falls_back_to_plain_mean():
    # certainty=0 for both -> weight sum is 0 -> _weighted_mean falls back to mean().
    sig = aggregate_predictions([_pred(1.0, 0.0), _pred(-1.0, 0.0)])
    assert sig.stance == pytest.approx(0.0)
    assert sig.certainty == pytest.approx(0.0)


def test_optional_field_none_when_all_predictions_lack_it():
    sig = aggregate_predictions([_pred(0.2, 0.5), _pred(0.3, 0.5)])
    assert sig.sentiment is None
    assert sig.hedge_ratio is None
    assert sig.magnitude is None
    assert sig.source_authority is None


def test_optional_field_skips_predictions_missing_it():
    sig = aggregate_predictions([
        _pred(0.2, 0.5, sentiment=0.8),
        _pred(0.3, 0.5, sentiment=None),
    ])
    # Only the first prediction contributes -> weighted mean of a single value is itself.
    assert sig.sentiment == pytest.approx(0.8)


def test_time_horizon_majority_vote():
    sig = aggregate_predictions([
        _pred(0.1, 0.5, time_horizon="short"),
        _pred(0.2, 0.5, time_horizon="short"),
        _pred(0.3, 0.5, time_horizon="long"),
    ])
    assert sig.time_horizon == "short"


def test_time_horizon_none_when_all_missing():
    sig = aggregate_predictions([_pred(0.1, 0.5), _pred(0.2, 0.5)])
    assert sig.time_horizon is None


def test_prediction_type_majority_vote():
    sig = aggregate_predictions([
        _pred(0.1, 0.5, prediction_type="binary"),
        _pred(0.2, 0.5, prediction_type="continuous"),
        _pred(0.3, 0.5, prediction_type="binary"),
    ])
    assert sig.prediction_type == "binary"


def test_time_horizon_days_weighted_median():
    # Weights: 0.9, 0.05, 0.05 (certainty, no specificity) -> cumulative hits 50% at the
    # first (highest-weight) value once sorted by value, i.e. the dominant contributor wins.
    sig = aggregate_predictions([
        _pred(0.1, 0.9, time_horizon_days=30),
        _pred(0.1, 0.05, time_horizon_days=60),
        _pred(0.1, 0.05, time_horizon_days=90),
    ])
    assert sig.time_horizon_days == 30


def test_time_horizon_days_none_when_all_missing():
    sig = aggregate_predictions([_pred(0.1, 0.5), _pred(0.2, 0.5)])
    assert sig.time_horizon_days is None


def test_time_horizon_days_zero_weight_falls_back_to_plain_median():
    sig = aggregate_predictions([
        _pred(0.1, 0.0, time_horizon_days=10),
        _pred(0.1, 0.0, time_horizon_days=20),
        _pred(0.1, 0.0, time_horizon_days=30),
    ])
    assert sig.time_horizon_days == 20


def test_claim_count_and_quotes_claims_preserve_order():
    sig = aggregate_predictions([
        _pred(0.1, 0.5, quote="q1", claim="c1"),
        _pred(0.2, 0.5, quote="q2", claim="c2"),
    ])
    assert sig.claim_count == 2
    assert sig.quotes == ["q1", "q2"]
    assert sig.claims == ["c1", "c2"]


class TestReaderConfidenceSurvivesTheArticleCollapse:
    """`aggregate_article_predictions` asks the LLM for a unified signal, and
    AGGREGATOR_PROMPT does not ask for `reader_confidence` (retro#681). The
    field is carried across in code instead — otherwise the batch lane would
    null it on precisely the articles that tripped aggregation, i.e. the ones
    whose claims disagree most and whose reader is likeliest to have struggled.
    """

    @staticmethod
    async def _collapse(monkeypatch, predictions):
        from tm import aggregator

        async def fake_complete_structured(*args, **kwargs):
            return _pred(0.1, 0.5, claim="unified"), {}

        monkeypatch.setattr(aggregator, "complete_structured", fake_complete_structured)
        return await aggregate_article_predictions(predictions, "E", "S", "2026-08-28")

    async def test_the_least_confident_reading_is_the_one_carried(self, monkeypatch):
        out = await self._collapse(monkeypatch, [
            _pred(0.8, 0.9, reader_confidence={"level": "high"}),
            _pred(-0.3, 0.6, reader_confidence={"level": "low", "trap": "conflicting_signals"}),
            _pred(0.2, 0.7, reader_confidence={"level": "medium", "trap": "negation"}),
        ])
        assert out.reader_confidence.level == "low"
        assert out.reader_confidence.trap == "conflicting_signals"

    async def test_the_level_and_its_own_trap_stay_paired(self, monkeypatch):
        """Taking the worst level from one claim and a trap from another would
        describe a reading no claim actually produced."""
        out = await self._collapse(monkeypatch, [
            _pred(0.8, 0.9, reader_confidence={"level": "low"}),
            _pred(-0.3, 0.6, reader_confidence={"level": "high", "trap": "negation"}),
        ])
        assert out.reader_confidence.level == "low"
        assert out.reader_confidence.trap is None

    async def test_nothing_answered_stays_none(self, monkeypatch):
        out = await self._collapse(monkeypatch, [_pred(0.8, 0.9), _pred(-0.3, 0.6)])
        assert out.reader_confidence is None


class TestTheLeadClaimsReadingSurvivesTheArticleCollapse:
    """retro#721 — the same failure as `reader_confidence` above, on eleven more
    fields. AGGREGATOR_PROMPT's output template names twelve fields; the schema
    instructor serialises into the call names ~35; the template wins and the rest
    come back null. Measured deterministic, 5/5 identical runs, on ~25% of batch
    extractions. Carried in code, from the one claim the aggregate followed.

    `_collapse` returns a BARE prediction on purpose — that is exactly what the
    real model does, so these tests fail without the carry.
    """

    @staticmethod
    async def _collapse(monkeypatch, predictions, agg_stance=0.7):
        from tm import aggregator

        async def fake_complete_structured(*args, **kwargs):
            return _pred(agg_stance, 0.5, claim="unified"), {}

        monkeypatch.setattr(aggregator, "complete_structured", fake_complete_structured)
        return await aggregate_article_predictions(predictions, "E", "S", "2026-08-28")

    @staticmethod
    def _rich(stance, **overrides):
        overrides.setdefault("settled", True)
        overrides.setdefault("event_date", "2026-03-15")
        overrides.setdefault("evidence_class", "reported_fact")
        overrides.setdefault("fact_signal", 0.7)
        overrides.setdefault("facet", "announcement")
        overrides.setdefault("report_kind", "change")
        overrides.setdefault("event_actors", "the central bank")
        overrides.setdefault("event_target", "benchmark rate")
        overrides.setdefault("is_occurrence", True)
        overrides.setdefault("quantitative_estimate", 0.45)
        overrides.setdefault("speaker", "Rossi")
        return _pred(stance, 0.8, **overrides)

    async def test_the_fields_the_llm_drops_are_restored(self, monkeypatch):
        out = await self._collapse(monkeypatch, [
            self._rich(0.8), self._rich(-0.7, settled=False, report_kind="level"),
        ])
        assert out.settled is True
        assert out.event_date == "2026-03-15"
        assert out.evidence_class == "reported_fact"
        assert out.fact_signal == 0.7
        assert out.facet == "announcement"
        assert out.report_kind == "change"
        assert out.event_actors == "the central bank"
        assert out.event_target == "benchmark rate"
        assert out.is_occurrence is True
        assert out.quantitative_estimate == 0.45

    async def test_the_nearest_same_sign_claim_is_the_one_carried(self, monkeypatch):
        """Two positive claims: the aggregate follows the nearer one."""
        out = await self._collapse(monkeypatch, [
            self._rich(0.2, event_date="2026-01-01"),
            self._rich(0.75, event_date="2026-09-09"),
            self._rich(-0.6, event_date="2026-12-31"),
        ], agg_stance=0.7)
        assert out.event_date == "2026-09-09"

    async def test_a_settlement_is_not_inherited_across_a_sign_flip(self, monkeypatch):
        """The guard that matters. Aggregation fires precisely on articles whose
        claims disagree, so a settling claim and the aggregate routinely point
        opposite ways. `settled` gates an ENFORCING resolution path — inheriting
        it from a claim the aggregate rejected would manufacture an outcome the
        article does not carry."""
        out = await self._collapse(monkeypatch, [
            self._rich(0.9, settled=True, event_date="2026-03-15"),
            self._rich(-0.2, settled=False, event_date=None),
        ], agg_stance=-0.6)
        assert out.settled is False
        assert out.event_date is None

    async def test_no_same_sign_claim_leaves_the_fields_null(self, monkeypatch):
        """If nothing in the article points the way the aggregate does, no claim's
        evidence describes it — null is the honest answer, not a nearest guess."""
        out = await self._collapse(monkeypatch, [
            self._rich(0.8), self._rich(0.4),
        ], agg_stance=-0.5)
        assert out.settled is None
        assert out.event_date is None
        assert out.report_kind is None

    async def test_a_zero_stance_aggregate_carries_nothing(self, monkeypatch):
        out = await self._collapse(monkeypatch, [self._rich(0.8)], agg_stance=0.0)
        assert out.settled is None

    async def test_speaker_is_deliberately_dropped(self, monkeypatch):
        """A collapsed article signal has no single speaker; taking the lead
        claim's would attribute the whole article to one quoted person."""
        out = await self._collapse(monkeypatch, [self._rich(0.8), self._rich(0.6)])
        assert out.speaker is None

    async def test_the_llm_keeps_the_fields_it_actually_synthesises(self, monkeypatch):
        """The carry must not overwrite the collapse's own output."""
        out = await self._collapse(monkeypatch, [
            self._rich(0.8), self._rich(0.6),
        ], agg_stance=0.7)
        assert out.stance == 0.7
        assert out.claim == "unified"

    async def test_conditional_fields_stay_with_their_own_antecedent(self, monkeypatch):
        """`is_conditional` resolved separately from `antecedent_text` would yield
        a conditional claim with no antecedent — a reading no claim produced."""
        out = await self._collapse(monkeypatch, [
            self._rich(0.8, is_conditional=True, antecedent_text="if the vote passes"),
            self._rich(0.1, is_conditional=False),
        ], agg_stance=0.75)
        assert out.is_conditional is True
        assert out.antecedent_text == "if the vote passes"

    def test_every_field_is_either_synthesised_or_carried(self):
        """The partition must stay total. A field added to PredictionExtraction
        and listed nowhere is carried by default — which is the safe direction —
        but one added to _LLM_SYNTHESISED_FIELDS without being added to
        AGGREGATOR_PROMPT's template would be silently nulled again. This asserts
        the two sets only ever name fields that exist."""
        from tm.aggregator import _LLM_SYNTHESISED_FIELDS, _SPECIALLY_HANDLED_FIELDS

        known = set(PredictionExtraction.model_fields)
        assert _LLM_SYNTHESISED_FIELDS <= known, (
            f"names no longer on the model: {_LLM_SYNTHESISED_FIELDS - known}"
        )
        assert _SPECIALLY_HANDLED_FIELDS <= known, (
            f"names no longer on the model: {_SPECIALLY_HANDLED_FIELDS - known}"
        )
        assert not (_LLM_SYNTHESISED_FIELDS & _SPECIALLY_HANDLED_FIELDS)
