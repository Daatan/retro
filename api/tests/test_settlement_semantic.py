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
    claim_subject_from_fields,
    claim_subject_from_question,
    gate_announcement_facet,
    gate_occurrence_consistency,
    gate_facet_missing,
    gate_point_in_time,
    gate_predicate_echo,
    pin_survives,
)
from forecast_api.settlement_semantic import ALL_GATES

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


class TestClaimSubjectFromFields:
    """The real subject, built from the extractor's own decomposition (retro#697).

    What the proxy above has been standing in for. These pin the constructor's
    shape and the ONE property the proxy structurally cannot have — a tight event
    term set — not the gate's accuracy, which cannot be re-measured until a pool
    extracted by prompt v11 exists.
    """

    ACTOR = {"name": "Benjamin Netanyahu", "type": "person"}

    def test_it_is_not_flagged_as_a_proxy(self):
        """The flag is the whole point: it is what lets a scorer tell a real
        number from a lower-bound one. Nothing else in the module sets it False."""
        subject = claim_subject_from_fields(
            self.ACTOR, "wins the general election", "the 2026 Israeli general election",
        )
        assert subject.proxy is False

    def test_the_actor_is_the_elicited_name(self):
        subject = claim_subject_from_fields(
            self.ACTOR, "wins the general election", "the 2026 Israeli general election",
        )
        assert subject.actors == ("Benjamin Netanyahu",)

    def test_actor_words_are_not_event_terms(self):
        """Same exclusion the proxy makes, same reason: otherwise an article that
        merely mentions the actor satisfies "names the event"."""
        subject = claim_subject_from_fields(
            self.ACTOR, "Netanyahu wins the general election", "the 2026 election",
        )
        assert "netanyahu" not in subject.event_terms
        assert "election" in subject.event_terms

    def test_the_term_set_is_tighter_than_the_proxys(self):
        """The measured mechanism behind `gate_predicate_echo`'s 0.46 recall.

        The proxy takes every content word in the question, so a long question
        yields a wide bag — mean 5.9 terms and up to 25 across the 172 prod
        questions that have an evidence pool, with 17% over eight. The gate
        demands only ONE of them echo, so the wider the bag the easier it is to
        satisfy and the less the gate can ever fire. A decomposition names the
        action and the scope and nothing else.
        """
        question = (
            "A single large political list will not be formed, including at least four of "
            "the following six politicians: Benny Gantz, Yoaz Hendel, Zeev Elkin and Gideon "
            "Saar, before the final deadline for submitting candidate lists for the general "
            "elections in Israel."
        )
        proxy = claim_subject_from_question(question)
        real = claim_subject_from_fields(
            {"name": "a single large political list", "type": "party"},
            "is formed",
            "at least four of six named politicians, before the candidate-list deadline",
            question=question,
        )
        assert len(proxy.event_terms) > 15          # the whole question, minus one name
        assert len(real.event_terms) < len(proxy.event_terms) / 2

    def test_outcome_kind_still_comes_from_the_question(self):
        """Deliberately not a fourth elicited field: the point-in-time test is a
        lexical property of the question that `_ON_DATE`/`_REMAIN` decide
        deterministically, and eliciting it would buy an LLM judgement where a
        regex is already correct."""
        assert claim_subject_from_fields(
            self.ACTOR, "is the Prime Minister", "Israel", question=PM_ON_A_DATE,
        ).outcome_kind == POINT_IN_TIME
        assert claim_subject_from_fields(
            self.ACTOR, "wins the election", "Israel", question=WINS_ELECTION,
        ).outcome_kind == IRREVERSIBLE

    def test_an_omitted_question_defaults_to_irreversible(self):
        """Same answer an unmatched question gives, so a caller that has no
        question text is not silently handed a different regime."""
        assert claim_subject_from_fields(
            self.ACTOR, "wins the election", "Israel",
        ).outcome_kind == IRREVERSIBLE

    @pytest.mark.parametrize("actor, predicate, scope", [
        (None, "wins the election", "Israel"),
        ("Benjamin Netanyahu", "wins the election", "Israel"),   # a bare string, not the object
        ({"type": "person"}, "wins the election", "Israel"),     # no name
        ({"name": "  "}, "wins the election", "Israel"),         # blank name
        ({"name": "Benjamin Netanyahu", "type": "person"}, None, None),
        ({"name": "Benjamin Netanyahu", "type": "person"}, "", "  "),
    ])
    def test_an_incomplete_decomposition_returns_none(self, actor, predicate, scope):
        """None, so the caller falls back to the proxy. Building a subject out of
        two of three fields would produce a `proxy=False` operand that is not
        actually a decomposition — worse than the lower bound it replaced, and
        silently so."""
        assert claim_subject_from_fields(actor, predicate, scope) is None

    def test_a_predicate_of_only_stopwords_returns_none(self):
        """`_content_terms` drops short and non-discriminating words, so a
        predicate can survive the emptiness checks and still yield no terms.
        `gate_predicate_echo` fails open on an empty term set, which would make
        this indistinguishable from having no decomposition at all."""
        assert claim_subject_from_fields(self.ACTOR, "will be", "the") is None

    def test_the_real_subject_drives_the_gate(self):
        """End to end: the constructor's output is a `ClaimSubject`, so it goes
        into the gate unchanged. "Yair Netanyahu left Florida" echoes the actor
        stem and no event term — the gate's own worked example, now reachable
        without the proxy."""
        subject = claim_subject_from_fields(
            self.ACTOR, "wins the general election", "the 2026 Israeli general election",
        )
        assert gate_predicate_echo(
            subject, _cand(claim="Yair Netanyahu left Florida", event_actors="Yair Netanyahu"),
        ) == "settlement_event_mismatch"
        assert gate_predicate_echo(
            subject,
            _cand(claim="Netanyahu wins the general election", event_actors="Benjamin Netanyahu"),
        ) is None


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


class TestFacetMissing:
    """The replacement for the refuted announcement gate (retro#691).

    Measured on the 387-pair labelled set, post-facet-rollout rows only: 80% of
    claims with no facet are ADJACENT (82 of 102) against a 56% base rate —
    0.80 precision, and it costs 2 of 26 defensible pins.
    """

    def test_fires_when_no_facet_was_elicited(self):
        subject = claim_subject_from_question(PM_ON_A_DATE)
        cand = _cand(claim="The Knesset approved a coalition government.", facet=None)
        assert gate_facet_missing(subject, cand) == "settled_without_facet"

    def test_fires_on_empty_string_too(self):
        subject = claim_subject_from_question(PM_ON_A_DATE)
        cand = _cand(claim="The Knesset approved a coalition government.", facet="")
        assert gate_facet_missing(subject, cand) == "settled_without_facet"

    def test_leaves_announcement_alone(self):
        """The whole point of the inversion: `announcement` sits BELOW the base
        adjacency rate (45% vs 56%), so demoting it costs more good pins than it
        saves. Firing here would reintroduce gate_announcement_facet's 129 false
        positives and its 13-of-26 true-pin loss."""
        subject = claim_subject_from_question(PM_ON_A_DATE)
        cand = _cand(claim="The US imposed 50% tariffs on Canadian goods.", facet="announcement")
        assert gate_facet_missing(subject, cand) is None

    def test_leaves_any_populated_facet_alone(self):
        subject = claim_subject_from_question(PM_ON_A_DATE)
        for facet in ("announcement", "denial", "neither", "occurrence"):
            assert gate_facet_missing(subject, _cand(claim="X happened.", facet=facet)) is None, facet

    def test_registered_in_all_gates(self):
        assert ALL_GATES["facet_missing"] is gate_facet_missing


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
