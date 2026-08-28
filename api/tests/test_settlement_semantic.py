"""Unit tests for the retro#691 candidate settlement gates.

These gates are analysis-only — nothing calls them in the pipeline — but they are
the thing a future enforcement PR would wire in, so their contract is pinned here
rather than left to the backtest script's aggregate numbers.
"""
from __future__ import annotations

import pytest

from forecast_api.settlement_semantic import (
    IRREVERSIBLE,
    POINT_IN_TIME,
    ClaimSubject,
    SettlementCandidate,
    apply_gates,
    claim_subject_from_question,
    gate_announcement_facet,
    gate_occurrence_consistency,
    gate_point_in_time,
    gate_predicate_echo,
    pin_survives,
)

PM_ON_A_DATE = "Benjamin Netanyahu will be the Prime Minister of Israel on December 31, 2026."
WINS_ELECTION = (
    "Benjamin Netanyahu will win the 2026 Israeli general election and be appointed Prime Minister."
)


def _cand(**kw) -> SettlementCandidate:
    base = dict(claim="", stance=1.0, certainty=0.95, outlet="example.com")
    base.update(kw)
    return SettlementCandidate(**base)


class TestClaimSubjectProxy:
    def test_state_on_a_date_is_point_in_time(self):
        assert claim_subject_from_question(PM_ON_A_DATE).outcome_kind == POINT_IN_TIME

    def test_remain_phrasing_is_point_in_time(self):
        q = "Andy Burnham will remain Prime Minister of the United Kingdom at least until January 2028."
        assert claim_subject_from_question(q).outcome_kind == POINT_IN_TIME

    def test_a_result_is_irreversible(self):
        assert claim_subject_from_question(WINS_ELECTION).outcome_kind == IRREVERSIBLE

    def test_actor_words_are_not_event_terms(self):
        # Otherwise any article merely mentioning the actor satisfies "names the event".
        subject = claim_subject_from_question(WINS_ELECTION)
        assert "netanyahu" not in subject.event_terms
        assert "election" in subject.event_terms

    def test_proxy_is_flagged(self):
        assert claim_subject_from_question(WINS_ELECTION).proxy is True


class TestPredicateEcho:
    def test_actor_present_but_no_event_term_is_caught(self):
        # The live retro#691 case: the heaviest settlement vote in the pool.
        subject = claim_subject_from_question(WINS_ELECTION)
        cand = _cand(claim="Yair Netanyahu left Florida and returned to Israel",
                     event_actors="Yair Netanyahu", event_target="Israel")
        assert gate_predicate_echo(subject, cand) == "settlement_event_mismatch"

    def test_unrelated_actor_is_caught(self):
        subject = claim_subject_from_question(WINS_ELECTION)
        cand = _cand(claim="Yoav Gallant was dismissed as defence minister",
                     event_actors="Yoav Gallant", event_target="defence ministry")
        assert gate_predicate_echo(subject, cand) == "settlement_actor_mismatch"

    def test_the_real_event_passes(self):
        subject = claim_subject_from_question(WINS_ELECTION)
        cand = _cand(claim="Netanyahu won the 2026 general election and was appointed Prime Minister",
                     event_actors="Benjamin Netanyahu", event_target="Prime Minister")
        assert gate_predicate_echo(subject, cand) is None

    def test_fails_open_without_a_claim_subject(self):
        cand = _cand(claim="anything at all")
        assert gate_predicate_echo(ClaimSubject(), cand) is None

    def test_fails_open_without_candidate_text(self):
        subject = claim_subject_from_question(WINS_ELECTION)
        assert gate_predicate_echo(subject, _cand(claim="")) is None


class TestPointInTime:
    def test_mid_window_fact_cannot_settle_a_state_claim(self):
        subject = claim_subject_from_question(PM_ON_A_DATE)
        cand = _cand(event_date="2026-08-05")
        assert gate_point_in_time(subject, cand, deadline=__import__("datetime").date(2026, 12, 31)) \
            == "settlement_before_evaluation_date"

    def test_fact_on_the_evaluation_date_stands(self):
        subject = claim_subject_from_question(PM_ON_A_DATE)
        cand = _cand(event_date="2026-12-31")
        assert gate_point_in_time(subject, cand, deadline=__import__("datetime").date(2026, 12, 31)) is None

    def test_irreversible_claims_are_out_of_scope(self):
        # A result genuinely IS settled mid-window; that is predicate_echo's job.
        subject = claim_subject_from_question(WINS_ELECTION)
        cand = _cand(event_date="2026-08-05")
        assert gate_point_in_time(subject, cand, deadline=__import__("datetime").date(2026, 12, 31)) is None

    def test_fails_open_without_a_deadline(self):
        subject = claim_subject_from_question(PM_ON_A_DATE)
        assert gate_point_in_time(subject, _cand(event_date="2026-08-05"), deadline=None) is None


class TestCheapGates:
    @pytest.mark.parametrize("facet", ["announcement", "denial"])
    def test_announcement_shapes_are_caught(self, facet):
        assert gate_announcement_facet(ClaimSubject(), _cand(facet=facet)) == "settled_on_announcement"

    def test_other_facets_pass(self):
        assert gate_announcement_facet(ClaimSubject(), _cand(facet="neither")) is None

    def test_settled_but_not_occurrence_is_contradictory(self):
        assert gate_occurrence_consistency(ClaimSubject(), _cand(is_occurrence=False)) \
            == "settled_but_not_occurrence"

    def test_unjudged_occurrence_fails_open(self):
        assert gate_occurrence_consistency(ClaimSubject(), _cand(is_occurrence=None)) is None


class TestPinArithmetic:
    def test_gates_demote_rather_than_delete(self):
        subject = claim_subject_from_question(WINS_ELECTION)
        cands = [
            _cand(claim="Yair Netanyahu returned to Israel", event_actors="Yair Netanyahu",
                  event_target="Israel", outlet="toi.com"),
            _cand(claim="Netanyahu won the 2026 general election", event_actors="Benjamin Netanyahu",
                  event_target="Prime Minister", outlet="jpost.com"),
        ]
        out = apply_gates(subject, cands, gates=["predicate_echo"], deadline="2026-12-31")
        assert len(out.kept) == 1 and len(out.demoted) == 1
        assert out.demoted[0][1] == "settlement_event_mismatch"

    def test_pin_counts_distinct_outlets_not_claims(self):
        subject = claim_subject_from_question(WINS_ELECTION)
        same_outlet = [
            _cand(claim="Netanyahu won the 2026 general election", event_actors="Benjamin Netanyahu",
                  event_target="Prime Minister", outlet="jpost.com"),
            _cand(claim="Netanyahu appointed Prime Minister after the election",
                  event_actors="Benjamin Netanyahu", event_target="Prime Minister", outlet="jpost.com"),
        ]
        out = apply_gates(subject, same_outlet, gates=["predicate_echo"], deadline="2026-12-31")
        assert len(out.kept) == 2
        assert pin_survives(out, min_sources=2) is False
