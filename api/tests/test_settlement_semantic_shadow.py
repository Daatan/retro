"""retro#691 — the deterministic semantic gates, shadow-wired into the forecaster.

Shadow means shadow: these gates log and do nothing. The tests that matter most
here are therefore the negative ones — that nothing they do can reach a
published number, and that the widened ``SettlementVote`` did not disturb the
verifier prompt (which would silently invalidate every cached verdict).
"""
import logging

import pytest

from forecast_api import forecaster
from forecast_api.config import settings as api_settings
from forecast_api.settlement_verifier import SettlementVote, build_prompt

# An IRREVERSIBLE (ARRIVAL) question is the default fixture: a result can be
# legitimately settled mid-window, so point_in_time stays out of the way and the
# other gates are what the assertions are actually testing. The point-in-time
# shape gets its own test below.
QUESTION = "Benjamin Netanyahu will win the 2026 Israeli general election and be appointed Prime Minister."
ON_A_DATE = "Benjamin Netanyahu will be the Prime Minister of Israel on December 31, 2026."
DEADLINE = "2026-12-31"


def _vote(claim: str, *, outlet: str, facet: str | None = "announcement",
          is_occurrence: bool | None = True, event_date: str | None = "2026-08-20",
          **kw) -> SettlementVote:
    return SettlementVote(
        outlet=outlet, claim=claim, quote=f'"{claim}"', event_date=event_date,
        stance=1.0, claim_strength=0.95, facet=facet, is_occurrence=is_occurrence,
        evidence_class="reported_fact", **kw,
    )


def _shadow(caplog, votes, *, deadline: str | None = DEADLINE, question: str = QUESTION):
    with caplog.at_level(logging.WARNING, logger=forecaster.logger.name):
        forecaster._log_semantic_gate_shadow(question, votes, claim_deadline=deadline)
    return [r.getMessage() for r in caplog.records
            if "event=settlement_semantic_gates" in r.getMessage()]


class TestShadowLogging:
    def test_a_clean_vote_set_is_logged_as_surviving(self, caplog):
        votes = [
            _vote("Netanyahu won the election.", outlet="a.example"),
            _vote("Netanyahu won the election.", outlet="b.example"),
        ]
        lines = _shadow(caplog, votes)
        assert len(lines) == 1
        assert "would_block=False" in lines[0]
        assert "votes=2" in lines[0]

    def test_a_demoted_vote_set_is_logged_as_blocking(self, caplog):
        """`is_occurrence=False` on every vote: occurrence_consistency demotes
        both, no outlets survive, the pin would not have reached min_sources."""
        votes = [
            _vote("Tensions escalated ahead of the vote.", outlet="a.example", is_occurrence=False),
            _vote("Tensions escalated ahead of the vote.", outlet="b.example", is_occurrence=False),
        ]
        lines = _shadow(caplog, votes)
        assert len(lines) == 1
        assert "would_block=True" in lines[0]
        assert "demoted=2" in lines[0]
        assert "settled_but_not_occurrence" in lines[0]

    def test_the_log_carries_the_question_hash_the_verifier_logs(self, caplog):
        """The two lines are paired by this hash — without it the shadow data
        cannot be joined to the verdict it is meant to be compared against."""
        lines = _shadow(caplog, [_vote("X happened.", outlet="a.example")])
        assert f"question={forecaster._question_hash(QUESTION)}" in lines[0]

    def test_one_surviving_outlet_still_blocks_at_min_sources_two(self, caplog):
        votes = [
            _vote("Netanyahu won the election.", outlet="a.example"),
            _vote("A precursor step occurred.", outlet="b.example", is_occurrence=False),
        ]
        lines = _shadow(caplog, votes)
        assert "would_block=True" in lines[0]
        assert "outlets_left=1" in lines[0], "one clean outlet is short of min_sources=2"

    def test_a_point_in_time_question_demotes_an_event_before_its_deadline(self, caplog):
        """The retro#691 shape: a state evaluated ON a date cannot be settled by
        anything that happened before it — the state can still change."""
        votes = [
            _vote("Netanyahu was sworn in as Prime Minister.", outlet="a.example",
                  event_date="2026-08-20"),
            _vote("Netanyahu was sworn in as Prime Minister.", outlet="b.example",
                  event_date="2026-08-20"),
        ]
        lines = _shadow(caplog, votes, question=ON_A_DATE)
        assert "would_block=True" in lines[0]
        assert "settlement_before_evaluation_date" in lines[0]

    def test_the_same_votes_survive_on_an_irreversible_question(self, caplog):
        """Same facts, same dates — only the question shape differs. Guards
        against point_in_time widening into ARRIVAL questions, where a mid-window
        result is legitimately settled."""
        votes = [
            _vote("Netanyahu won the election.", outlet="a.example", event_date="2026-08-20"),
            _vote("Netanyahu won the election.", outlet="b.example", event_date="2026-08-20"),
        ]
        assert "would_block=False" in _shadow(caplog, votes, question=QUESTION)[0]


class TestKillSwitchAndConfig:
    def test_disabled_emits_nothing(self, caplog, monkeypatch):
        monkeypatch.setattr(api_settings, "settlement_semantic_gates_enabled", False)
        assert _shadow(caplog, [_vote("X happened.", outlet="a.example")]) == []

    def test_unknown_gate_names_are_ignored_not_fatal(self, caplog, monkeypatch):
        monkeypatch.setattr(api_settings, "settlement_semantic_gates",
                            "occurrence_consistency, not_a_real_gate")
        lines = _shadow(caplog, [_vote("X.", outlet="a.example", is_occurrence=False)])
        assert len(lines) == 1
        assert "gates=occurrence_consistency" in lines[0]

    def test_an_all_unknown_gate_list_emits_nothing(self, caplog, monkeypatch):
        monkeypatch.setattr(api_settings, "settlement_semantic_gates", "nonsense,also_nonsense")
        assert _shadow(caplog, [_vote("X.", outlet="a.example")]) == []

    def test_the_refuted_announcement_gate_is_not_in_the_default_set(self):
        """It destroys 13 of 26 defensible pins (retro#691). Adding it back to
        the default is the one change here that would do real damage, so it is
        asserted rather than left to review."""
        assert "announcement_facet" not in api_settings.settlement_semantic_gates

    def test_predicate_echo_is_not_in_the_default_set(self):
        """Left out until retro#697 supplies a real claim dyad — today it runs
        on a regex proxy and overlaps facet_missing heavily."""
        assert "predicate_echo" not in api_settings.settlement_semantic_gates


class TestNeverBreaksAForecast:
    def test_a_raising_gate_is_swallowed(self, caplog, monkeypatch):
        def boom(*_a, **_kw):
            raise RuntimeError("gate exploded")
        monkeypatch.setattr(forecaster, "claim_subject_from_question", boom)
        with caplog.at_level(logging.ERROR, logger=forecaster.logger.name):
            forecaster._log_semantic_gate_shadow(
                QUESTION, [_vote("X.", outlet="a.example")], claim_deadline=DEADLINE,
            )
        assert any("settlement_semantic_gates outcome=error" in r.getMessage()
                   for r in caplog.records)

    def test_no_votes_is_not_an_error(self, caplog):
        lines = _shadow(caplog, [])
        assert len(lines) == 1
        assert "votes=0" in lines[0]

    def test_a_missing_deadline_is_tolerated(self, caplog):
        lines = _shadow(caplog, [_vote("X.", outlet="a.example")], deadline=None)
        assert len(lines) == 1


class TestVerifierPromptUnchanged:
    """The widened SettlementVote must not reach the verifier prompt.

    ``verdict_key`` is keyed on the built prompt, so a single extra rendered
    field would invalidate every cached verdict at once and silently re-roll
    decisions the cache exists to make sticky (retro#532).
    """

    def test_shadow_fields_do_not_appear_in_the_prompt(self):
        vote = _vote("The step has been taken.", outlet="outlet.example",
                     event_actors="ZZACTORSZZ", event_target="ZZTARGETZZ")
        prompt = build_prompt(QUESTION, [vote])
        for leak in ("ZZACTORSZZ", "ZZTARGETZZ", "announcement", "reported_fact", "0.95"):
            assert leak not in prompt, leak

    def test_prompt_is_identical_with_and_without_the_shadow_fields(self):
        bare = SettlementVote(outlet="outlet.example", claim="The step has been taken.",
                              quote='"The step has been taken."', event_date="2026-08-20")
        rich = _vote("The step has been taken.", outlet="outlet.example",
                     event_actors="Someone", event_target="Something")
        assert build_prompt(QUESTION, [bare]) == build_prompt(QUESTION, [rich])
