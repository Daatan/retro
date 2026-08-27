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
    _extract_actor_shaped_entities,
    _extract_named_entities,
    _mentions_entity_stem,
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


# ── stem-matched false positives (retro#545 slice ii, 2026-08-24 review) ───

def test_institutional_alias_actor_does_not_log(caplog):
    """"Donald Trump" is the subject; "Trump administration" names him via a
    prefix of his surname, not the full phrase — must no longer fire."""
    preds = [pred(event_actors="Trump administration", event_target="the tariff order")]
    with caplog.at_level("WARNING", logger="tm.extractor"):
        audit_named_entity_dyad_mismatch(preds, "Will Donald Trump sign the order?")
    assert not any(
        "event=entity_dyad_mismatch " in r.message for r in caplog.records
    )


def test_institutional_alias_target_does_not_log(caplog):
    """Same shape, mismatch on event_target instead of event_actors — proves
    both dyad fields go through the looser stem match."""
    preds = [pred(event_actors="the White House", event_target="Trump administration")]
    with caplog.at_level("WARNING", logger="tm.extractor"):
        audit_named_entity_dyad_mismatch(preds, "Will Donald Trump sign the order?")
    assert not any(
        "event=entity_dyad_mismatch " in r.message for r in caplog.records
    )


def test_adjectival_form_does_not_log(caplog):
    """"Israel" as a prefix of "Israeli" inside "Israeli government" — the
    word-boundary regex in _mentions_entity fails here; the stem match doesn't."""
    preds = [pred(event_actors="Israeli government", event_target="the ceasefire")]
    with caplog.at_level("WARNING", logger="tm.extractor"):
        audit_named_entity_dyad_mismatch(preds, "Will Israel agree to a ceasefire?")
    assert not any(
        "event=entity_dyad_mismatch " in r.message for r in caplog.records
    )


def test_adjectival_form_iran_does_not_log(caplog):
    preds = [pred(event_actors="Iranian-backed militias", event_target="the base")]
    with caplog.at_level("WARNING", logger="tm.extractor"):
        audit_named_entity_dyad_mismatch(preds, "Will Iran retaliate against the base?")
    assert not any(
        "event=entity_dyad_mismatch " in r.message for r in caplog.records
    )


def test_generic_org_suffix_still_logs(caplog):
    """A genuinely different party sharing the word "Party" must still fire —
    the generic-anchor exclusion exists specifically to prevent this loose
    match, so this pins that as deliberate."""
    preds = [pred(event_actors="the Republican Party's leadership", event_target="the bill")]
    with caplog.at_level("WARNING", logger="tm.extractor"):
        audit_named_entity_dyad_mismatch(preds, "Will the Democratic Party pass the bill?")
    assert any(
        "event=entity_dyad_mismatch " in r.message for r in caplog.records
    )


def test_short_acronym_subject_still_logs(caplog):
    """Below the 4-char anchor floor: "US" must not loose-match "USA" or any
    other word merely starting with "US" — documents the deliberately
    unresolved short-acronym case."""
    preds = [pred(event_actors="USA officials", event_target="the summit")]
    with caplog.at_level("WARNING", logger="tm.extractor"):
        audit_named_entity_dyad_mismatch(preds, "Will the US attend the summit?")
    assert any(
        "event=entity_dyad_mismatch " in r.message for r in caplog.records
    )


def test_topic_vs_responder_shape_no_longer_logs(caplog):
    """retro#644 fix: "Ebola" in "the Ebola outbreak" is a topic modifier, not
    the actor the question is about, so a legitimately different,
    correctly-extracted actor/target pair (WHO / Africa CDC) must no longer
    spuriously fire."""
    preds = [pred(event_actors="WHO", event_target="Africa CDC")]
    with caplog.at_level("WARNING", logger="tm.extractor"):
        audit_named_entity_dyad_mismatch(
            preds, "Will the Ebola outbreak be declared over by Q4?"
        )
    assert not any(
        "event=entity_dyad_mismatch " in r.message for r in caplog.records
    )


def test_topic_head_noun_not_in_the_curated_list_still_logs(caplog):
    """Honest about non-exhaustiveness: _TOPIC_HEAD_NOUNS is a small closed
    list, not real NER — a topic-modifier shape it doesn't cover still fires,
    same as before retro#644. Uses "flare-up", deliberately absent from the
    list, to document the remaining gap rather than implying it's closed."""
    preds = [pred(event_actors="WHO", event_target="Africa CDC")]
    with caplog.at_level("WARNING", logger="tm.extractor"):
        audit_named_entity_dyad_mismatch(
            preds, "Will the Ebola flare-up be declared over by Q4?"
        )
    assert any(
        "event=entity_dyad_mismatch " in r.message for r in caplog.records
    )


# ── _extract_actor_shaped_entities (retro#644) ──────────────────────────────

def test_extract_actor_shaped_entities_drops_topic_modifier():
    entities = _extract_actor_shaped_entities(
        "Will the Ebola outbreak be declared over by Q4?"
    )
    assert "Ebola" not in entities


def test_extract_actor_shaped_entities_keeps_a_real_actor():
    assert "Yoaz Hendel" in _extract_actor_shaped_entities(QUESTION)


def test_extract_actor_shaped_entities_keeps_actor_before_stem_matched_alias():
    """Regression guard: the institutional-alias/adjectival-form fixes (#645)
    must keep working once the subject comes from the new extractor."""
    entities = _extract_actor_shaped_entities("Will Donald Trump sign the order?")
    assert "Donald Trump" in entities


def test_summary_log_line_reports_eligible_and_fired_counts(caplog):
    preds = [
        pred(event_actors=None),  # not eligible (missing dyad field)
        pred(event_actors="Almog Cohen", event_target="the Knesset"),  # eligible, fires
        pred(event_actors="Yoaz Hendel", event_target="the Knesset"),  # eligible, no-op
    ]
    with caplog.at_level("INFO", logger="tm.extractor"):
        audit_named_entity_dyad_mismatch(preds, QUESTION)
    summary = [r.message for r in caplog.records if "event=entity_dyad_mismatch_shadow" in r.message]
    assert len(summary) == 1
    assert "eligible=2" in summary[0]
    assert "fired=1" in summary[0]
    assert "n=3" in summary[0]


# ── _mentions_entity_stem unit tests ────────────────────────────────────────

def test_mentions_entity_stem_matches_surname_in_institutional_phrase():
    assert _mentions_entity_stem("Trump administration", "Donald Trump") is True


def test_mentions_entity_stem_matches_adjectival_form():
    assert _mentions_entity_stem("Israeli government", "Israel") is True


def test_mentions_entity_stem_rejects_short_acronym():
    assert _mentions_entity_stem("USA officials", "US") is False


def test_mentions_entity_stem_rejects_generic_anchor_word():
    assert _mentions_entity_stem("the Republican Party's leadership", "Democratic Party") is False


def test_mentions_entity_stem_falls_back_to_exact_match():
    """A superset of _mentions_entity, not a replacement: an exact match
    still passes even when the anchor-word logic wouldn't independently
    justify it (single-word entity, generic-sounding last word)."""
    assert _mentions_entity_stem("the Knesset speaker", "the Knesset") is True


# ── never mutates ───────────────────────────────────────────────────────────

def test_never_mutates_the_prediction():
    preds = [pred()]
    out = audit_named_entity_dyad_mismatch(preds, QUESTION)
    assert out is preds
    assert out[0].stance == 0.9
    assert out[0].claim_strength == 0.9
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
