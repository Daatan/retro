"""Deterministic winner-entity check for versus/sports questions — retro#401.

The 2026-07-15 England-Argentina incident (retro#360): four articles plainly
reporting "Argentina beat England" were extracted as stance +1.0 for "England
will win", and the false-unanimous settlement pinned the estimate at 97% —
Brier 0.94, one of the two worst misses in the resolved corpus. retro#313
shipped exactly the facets a deterministic check needs — `event_actors`,
`event_target`, `is_occurrence` — but nothing downstream ever read them.
`enforce_winner_entity_consistency` is that check: for a question that names
two contesting actors, does the dominant fact's actor->target dyad actually
agree with the stance sign it carries?

Conservative by design: a caught claim is NEUTRALISED (stance zeroed, settled
stripped), never sign-flipped — the dyad match is confident evidence the
existing sign is wrong, not evidence of what the correct value should be.
"""
import pytest

from tm.extractor import enforce_winner_entity_consistency, _match_versus_question
from tm.models import PredictionExtraction

INCIDENT_QUESTION = (
    "England will win their FIFA World Cup 2026 semi-final match against "
    "Argentina on July 15"
)


def pred(
    stance: float,
    event_actors: str | None,
    event_target: str | None,
    *,
    is_occurrence: bool | None = True,
    settled: bool | None = None,
    certainty: float = 0.9,
    verified: bool | None = True,
):
    return PredictionExtraction(
        quote="q", claim="c", stance=stance, certainty=certainty, settled=settled,
        fact_signal=stance, event_actors=event_actors, event_target=event_target,
        is_occurrence=is_occurrence, verified=verified,
    )


# ── question parsing ────────────────────────────────────────────────────────

@pytest.mark.parametrize("question,expected", [
    (INCIDENT_QUESTION, ("England", "Argentina")),
    ("Will Liverpool beat Chelsea in their Premier League match?", ("Liverpool", "Chelsea")),
    ("Will the Lakers defeat the Celtics in Game 7?", ("Lakers", "Celtics")),
    ("Will Real Madrid win against Barcelona in El Clasico?", ("Real Madrid", "Barcelona")),
    ("[B20] England will win their World Cup semi-final against Argentina", ("England", "Argentina")),
])
def test_versus_questions_parse_subject_and_rival(question, expected):
    assert _match_versus_question(question) == expected


@pytest.mark.parametrize("question", [
    "Will inflation fall below 3 percent this year?",
    "Netanyahu will win the 2026 Israeli election",
    "Will England win their World Cup semi-final on 2026-07-15",  # no rival named
    "Argentina vs England: who advances?",  # no verb, ambiguous subject — not parsed
    "Will Israel beat inflation this year?",  # rival isn't a named entity
    "Will the grantor issue the licence by the deadline?",
    "Will the named team win their semi-final?",  # B19's deliberately-generic fixture question
])
def test_non_versus_or_ambiguous_questions_do_not_parse(question):
    assert _match_versus_question(question) is None


# ── the incident itself ─────────────────────────────────────────────────────

def test_the_incident_shape_is_neutralised():
    """Argentina (the rival) beating England (the subject) extracted as a
    positive, settled stance FOR England is exactly retro#360's failure."""
    [out] = enforce_winner_entity_consistency(
        [pred(1.0, "Argentina", "England", settled=True)], INCIDENT_QUESTION,
    )
    assert out.stance == 0.0
    assert out.settled is False


def test_the_correctly_signed_vote_is_untouched():
    """aa.com.tr's row in the incident: Argentina beat England, extracted
    (correctly) as a negative stance for England winning. Nothing to fix."""
    [out] = enforce_winner_entity_consistency(
        [pred(-1.0, "Argentina", "England", settled=True)], INCIDENT_QUESTION,
    )
    assert out.stance == -1.0
    assert out.settled is True


def test_symmetric_case_subject_beats_rival_with_negative_stance():
    """The mirror image: England (subject) beating Argentina (rival) extracted
    as a negative stance for England winning is the same bug in reverse."""
    [out] = enforce_winner_entity_consistency(
        [pred(-0.8, "England", "Argentina")], INCIDENT_QUESTION,
    )
    assert out.stance == 0.0


def test_subject_beats_rival_with_positive_stance_is_correct_and_untouched():
    [out] = enforce_winner_entity_consistency(
        [pred(0.9, "England", "Argentina", settled=True)], INCIDENT_QUESTION,
    )
    assert out.stance == 0.9
    assert out.settled is True


# ── fail-open conditions ────────────────────────────────────────────────────

def test_non_versus_question_is_a_no_op():
    preds = [pred(1.0, "Argentina", "England", settled=True)]
    out = enforce_winner_entity_consistency(preds, "Will inflation fall below 3 percent?")
    assert out[0].stance == 1.0
    assert out[0].settled is True


def test_a_precursor_is_left_alone():
    """is_occurrence=false: the fact isn't the event itself, so it can't tell
    us who won it. enforce_precursor_cap already governs its magnitude."""
    [out] = enforce_winner_entity_consistency(
        [pred(1.0, "Argentina", "England", is_occurrence=False)], INCIDENT_QUESTION,
    )
    assert out.stance == 1.0


def test_an_unjudged_occurrence_flag_fails_open():
    """is_occurrence=None: the extractor declined to judge whether this fact
    IS the event. Never invent a judgement the model didn't make."""
    [out] = enforce_winner_entity_consistency(
        [pred(1.0, "Argentina", "England", is_occurrence=None)], INCIDENT_QUESTION,
    )
    assert out.stance == 1.0


def test_a_claim_with_no_dyad_is_untouched():
    [out] = enforce_winner_entity_consistency(
        [pred(1.0, None, None)], INCIDENT_QUESTION,
    )
    assert out.stance == 1.0


@pytest.mark.parametrize("actors,target", [
    (None, "England"),
    ("Argentina", None),
])
def test_a_partial_dyad_is_untouched(actors, target):
    [out] = enforce_winner_entity_consistency(
        [pred(1.0, actors, target)], INCIDENT_QUESTION,
    )
    assert out.stance == 1.0


def test_a_third_party_dyad_is_untouched():
    """Neither actor nor target names the subject or rival — a fact about a
    different pair entirely (the DYAD rule's own territory, retro#313).
    This function has nothing to say about it."""
    [out] = enforce_winner_entity_consistency(
        [pred(1.0, "Brazil", "Germany")], INCIDENT_QUESTION,
    )
    assert out.stance == 1.0


def test_a_dyad_naming_both_subject_and_rival_as_actor_is_untouched():
    """event_actors mentioning both ('Argentina and England players clashed')
    is not confidently 'the rival acting' — ambiguous, so left alone."""
    [out] = enforce_winner_entity_consistency(
        [pred(1.0, "Argentina and England", "the pitch")], INCIDENT_QUESTION,
    )
    assert out.stance == 1.0


def test_an_empty_list_is_a_no_op():
    assert enforce_winner_entity_consistency([], INCIDENT_QUESTION) == []


def test_a_neutralised_claim_keeps_every_other_field():
    p = pred(1.0, "Argentina", "England", settled=True, certainty=0.85, verified=False)
    [out] = enforce_winner_entity_consistency([p], INCIDENT_QUESTION)
    assert out.stance == 0.0
    assert out.settled is False
    assert out.certainty == 0.85
    assert out.verified is False
    assert (out.event_actors, out.event_target) == ("Argentina", "England")
    assert out.fact_signal == 1.0  # shadow lane untouched; nothing reads it yet


def test_an_unsettled_wrong_signed_claim_is_still_neutralised():
    """The incident's catastrophic failure was the settlement pin, but an
    unsettled wrongly-signed claim still skews the pool as ordinary evidence —
    the check isn't scoped to settled claims only."""
    [out] = enforce_winner_entity_consistency(
        [pred(0.7, "Argentina", "England", settled=None)], INCIDENT_QUESTION,
    )
    assert out.stance == 0.0
    assert out.settled is False


def test_multiple_predictions_only_the_conflicting_one_moves():
    correct = pred(-1.0, "Argentina", "England", settled=True)
    conflicting = pred(1.0, "Argentina", "England", settled=True)
    unrelated = pred(0.4, None, None)
    out = enforce_winner_entity_consistency(
        [correct, conflicting, unrelated], INCIDENT_QUESTION,
    )
    assert out[0].stance == -1.0 and out[0].settled is True
    assert out[1].stance == 0.0 and out[1].settled is False
    assert out[2].stance == 0.4
