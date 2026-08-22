"""Log-only: does a strong-stance claim about ONE named actor land on a fact
dyad that never names them (retro#545, slice ii)?

The issue's wrong-entity examples are single-named-actor claims scored against
an article about someone else entirely — a Yoaz Hendel claim scored against an
Almog Cohen article (-0.851 @ 0.875), and separately against an Oren Smadja
article in the same pool. Unlike the versus/sports shape
(`enforce_winner_entity_consistency`, retro#401), there's no rival to compare
against — just a named actor the article's fact dyad never mentions.

Coverage and precision on this shape are unmeasured (event_actors/event_target
populate on only ~38% of the strong-stance band, prod audit 2026-08-22), so
this only logs — it never mutates stance/certainty/settled.
"""
from tm.extractor import (
    _ENTITY_DYAD_AUDIT_CERTAINTY_GATE,
    _ENTITY_DYAD_AUDIT_STANCE_GATE,
    _extract_named_entities,
    audit_named_entity_dyad_mismatch,
)
from tm.models import PredictionExtraction


def pred(
    stance: float = 0.9,
    certainty: float = 0.9,
    *,
    event_actors: str | None = "Almog Cohen",
    event_target: str | None = "the Knesset",
    claim: str = "Yoaz Hendel will run in the 26th Knesset",
):
    return PredictionExtraction(
        quote="q", claim=claim, stance=stance, certainty=certainty,
        event_actors=event_actors, event_target=event_target,
    )


QUESTION = "Will Yoaz Hendel run in the 26th Knesset elections?"


# ── the mismatch itself ────────────────────────────────────────────────────

def test_the_hendel_cohen_shape_logs_a_warning(caplog):
    """Question names Yoaz Hendel; the dyad names Almog Cohen instead."""
    preds = [pred()]
    with caplog.at_level("WARNING", logger="tm.extractor"):
        audit_named_entity_dyad_mismatch(preds, QUESTION)
    assert any(
        "event=entity_dyad_mismatch" in r.message for r in caplog.records
    )


def test_matching_actor_does_not_log(caplog):
    preds = [pred(event_actors="Yoaz Hendel", event_target="the Knesset")]
    with caplog.at_level("WARNING", logger="tm.extractor"):
        audit_named_entity_dyad_mismatch(preds, QUESTION)
    assert not any(
        "event=entity_dyad_mismatch" in r.message for r in caplog.records
    )


def test_matching_target_does_not_log(caplog):
    """A hit on EITHER side of the dyad is enough to clear the row."""
    preds = [pred(event_actors="the Knesset speaker", event_target="Yoaz Hendel")]
    with caplog.at_level("WARNING", logger="tm.extractor"):
        audit_named_entity_dyad_mismatch(preds, QUESTION)
    assert not any(
        "event=entity_dyad_mismatch" in r.message for r in caplog.records
    )


# ── fail-open branches ──────────────────────────────────────────────────────

def test_no_entity_in_question_is_a_no_op(caplog):
    preds = [pred()]
    with caplog.at_level("WARNING", logger="tm.extractor"):
        audit_named_entity_dyad_mismatch(preds, "will it happen soon")
    assert not any(
        "event=entity_dyad_mismatch" in r.message for r in caplog.records
    )


def test_missing_event_actors_fails_open(caplog):
    preds = [pred(event_actors=None)]
    with caplog.at_level("WARNING", logger="tm.extractor"):
        audit_named_entity_dyad_mismatch(preds, QUESTION)
    assert not any(
        "event=entity_dyad_mismatch" in r.message for r in caplog.records
    )


def test_missing_event_target_fails_open(caplog):
    preds = [pred(event_target=None)]
    with caplog.at_level("WARNING", logger="tm.extractor"):
        audit_named_entity_dyad_mismatch(preds, QUESTION)
    assert not any(
        "event=entity_dyad_mismatch" in r.message for r in caplog.records
    )


def test_below_stance_gate_fails_open(caplog):
    preds = [pred(stance=_ENTITY_DYAD_AUDIT_STANCE_GATE - 0.1)]
    with caplog.at_level("WARNING", logger="tm.extractor"):
        audit_named_entity_dyad_mismatch(preds, QUESTION)
    assert not any(
        "event=entity_dyad_mismatch" in r.message for r in caplog.records
    )


def test_below_certainty_gate_fails_open(caplog):
    preds = [pred(certainty=_ENTITY_DYAD_AUDIT_CERTAINTY_GATE - 0.1)]
    with caplog.at_level("WARNING", logger="tm.extractor"):
        audit_named_entity_dyad_mismatch(preds, QUESTION)
    assert not any(
        "event=entity_dyad_mismatch" in r.message for r in caplog.records
    )


# ── never mutates ───────────────────────────────────────────────────────────

def test_never_mutates_the_prediction():
    preds = [pred()]
    out = audit_named_entity_dyad_mismatch(preds, QUESTION)
    assert out is preds
    assert out[0].stance == 0.9
    assert out[0].certainty == 0.9
    assert out[0].settled is None
    assert out[0].event_actors == "Almog Cohen"
    assert out[0].event_target == "the Knesset"


# ── the entity extractor itself ─────────────────────────────────────────────

def test_extract_named_entities_finds_the_subject():
    assert "Yoaz Hendel" in _extract_named_entities(QUESTION)


def test_extract_named_entities_drops_leading_stopwords():
    entities = _extract_named_entities("Will England win the match")
    assert "Will" not in entities
    assert "England" in entities


def test_extract_named_entities_dedupes_case_insensitively():
    entities = _extract_named_entities("Yoaz Hendel met Yoaz Hendel again")
    assert len(entities) == 1
