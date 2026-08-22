"""Tests for forecast_api.antecedent (retro#573, pool-split Option 1).

Pure-function tests against source_matches_antecedent()/filter_pool_by_antecedent()
directly — no aggregation involved. Integration with run_pool_aggregate() is
covered separately in test_pool_aggregate.py::TestAntecedentPoolSplit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from forecast_api.antecedent import (
    DEFAULT_ANTECEDENT_JACCARD_THRESHOLD,
    filter_pool_by_antecedent,
    source_matches_antecedent,
)
from forecast_api.clustering import shingles


def _claim(**over) -> dict:
    return {"claim": "consequent happens", "stance": 0.5, "certainty": 0.6, **over}


def _shingles(text: str) -> frozenset:
    return shingles(text, 2)


@dataclass
class _Source:
    claims_detail: Optional[list] = field(default=None)


class TestSourceMatchesAntecedent:
    def test_no_claims_detail_is_kept(self):
        assert source_matches_antecedent(None, _shingles("the ceasefire holds"), True, 0.25) is True
        assert source_matches_antecedent([], _shingles("the ceasefire holds"), True, 0.25) is True

    def test_unconditional_claim_is_always_kept(self):
        claims = [_claim(is_conditional=False)]
        assert source_matches_antecedent(claims, _shingles("anything at all"), True, 0.25) is True

    def test_claim_missing_is_conditional_field_is_kept(self):
        claims = [_claim()]  # is_conditional absent entirely (legacy row)
        assert source_matches_antecedent(claims, _shingles("anything at all"), True, 0.25) is True

    def test_matching_antecedent_is_kept(self):
        claims = [_claim(
            is_conditional=True, antecedent_text_en="the ceasefire holds", antecedent_polarity=True,
        )]
        assert source_matches_antecedent(claims, _shingles("the ceasefire holds"), True, 0.25) is True

    def test_mismatched_antecedent_is_dropped(self):
        claims = [_claim(
            is_conditional=True, antecedent_text_en="China invades Taiwan", antecedent_polarity=True,
        )]
        assert source_matches_antecedent(claims, _shingles("the ceasefire holds"), True, 0.25) is False

    def test_polarity_mismatch_is_dropped(self):
        # Same antecedent text, opposite polarity: "X happens" != "X does not happen".
        claims = [_claim(
            is_conditional=True, antecedent_text_en="the ceasefire holds", antecedent_polarity=False,
        )]
        assert source_matches_antecedent(claims, _shingles("the ceasefire holds"), True, 0.25) is False

    def test_polarity_none_treated_as_affirmative(self):
        claims = [_claim(
            is_conditional=True, antecedent_text_en="the ceasefire holds", antecedent_polarity=None,
        )]
        assert source_matches_antecedent(claims, _shingles("the ceasefire holds"), True, 0.25) is True

    def test_negated_query_matches_negated_claim(self):
        claims = [_claim(
            is_conditional=True, antecedent_text_en="the ceasefire holds", antecedent_polarity=False,
        )]
        assert source_matches_antecedent(claims, _shingles("the ceasefire holds"), False, 0.25) is True

    def test_conditional_with_no_antecedent_text_is_dropped(self):
        claims = [_claim(is_conditional=True, antecedent_text_en=None)]
        assert source_matches_antecedent(claims, _shingles("the ceasefire holds"), True, 0.25) is False

    def test_mixed_claims_one_matching_is_kept(self):
        claims = [
            _claim(is_conditional=True, antecedent_text_en="China invades Taiwan", antecedent_polarity=True),
            _claim(is_conditional=True, antecedent_text_en="the ceasefire holds", antecedent_polarity=True),
        ]
        assert source_matches_antecedent(claims, _shingles("the ceasefire holds"), True, 0.25) is True

    def test_pydantic_claimdetail_objects_work_same_as_dicts(self):
        from forecast_api.models import ClaimDetail

        claims = [ClaimDetail(
            claim="x", stance=0.5, certainty=0.6,
            is_conditional=True, antecedent_text_en="the ceasefire holds", antecedent_polarity=True,
        )]
        assert source_matches_antecedent(claims, _shingles("the ceasefire holds"), True, 0.25) is True


class TestFilterPoolByAntecedent:
    def test_none_query_is_a_no_op(self):
        sources = [_Source(claims_detail=[_claim(is_conditional=True, antecedent_text_en="X")]) for _ in range(3)]
        assert filter_pool_by_antecedent(sources, None) == sources

    def test_empty_string_query_is_a_no_op(self):
        sources = [_Source() for _ in range(2)]
        assert filter_pool_by_antecedent(sources, "") == sources

    def test_filters_out_non_matching_sources(self):
        matching = _Source(claims_detail=[_claim(
            is_conditional=True, antecedent_text_en="the ceasefire holds", antecedent_polarity=True,
        )])
        other = _Source(claims_detail=[_claim(
            is_conditional=True, antecedent_text_en="China invades Taiwan", antecedent_polarity=True,
        )])
        unconditional = _Source(claims_detail=[_claim(is_conditional=False)])
        result = filter_pool_by_antecedent([matching, other, unconditional], "the ceasefire holds")
        assert result == [matching, unconditional]

    def test_default_threshold_is_exported_and_used(self):
        # Sanity: the module constant matches what filter_pool_by_antecedent defaults to.
        assert DEFAULT_ANTECEDENT_JACCARD_THRESHOLD == 0.25
