"""retro#805 — the subject gate as a pure function, the subject-card derivation's fail-open
contract, and the card store. Fixture text: the Walla live-blog (W4) from the 2026-09-07
A/B — the article that voted +1.0 on "Gantz's party runs alone" without naming Gantz."""
from __future__ import annotations

from forecast_api import subject_card as sc
from forecast_api.subject_card import (
    SubjectActor,
    SubjectCard,
    derive_subject_card,
    evaluate_subject_gate,
)
from forecast_api.subject_card_store import get_subject_card, put_subject_card, subject_card_key
from tm.models import ArticleCard, ArticleCardActor

WALLA = (
    "למעמד ההגשה הגיעו איזנקוט ובנט בנפרד, מה שמרמז כי איחוד בין המפלגות לא יתרחש. "
    "חמש המפלגות הראשונות שצפויות להגיש את הרשימות: \"המילואימניקים\" ברשות יועז הנדל, "
    "\"ביחד\" ברשות נפתלי בנט לפיד, \"ישר!\" ברשות גדי איזנקוט. "
    "בד בבד, יונתן שימריז הודיע כי לא יתמודד בבחירות הקרובות."
)

GANTZ = SubjectCard(actors=[
    SubjectActor(name_en="Benny Gantz", type="person",
                 surface_forms=["Benny Gantz", "Gantz", "בני גנץ", "גנץ", "Бени Ганц", "Ганц"]),
    SubjectActor(name_en="Blue and White", type="party",
                 surface_forms=["Blue and White", "כחול לבן", "כחול-לבן"]),
])
HENDEL = SubjectCard(actors=[
    SubjectActor(name_en="Yoaz Hendel", type="person",
                 surface_forms=["Yoaz Hendel", "Hendel", "יועז הנדל", "הנדל", "Хендель"]),
])

# What Haiku actually returns for W4 once asked for verbatim spans: the article's real
# actors, plus (in the W3 failure shape) the question's subject smuggled in.
WALLA_CARD = ArticleCard(named_actors=[
    ArticleCardActor(span="איזנקוט", name_en="Gadi Eisenkot"),
    ArticleCardActor(span="בנט", name_en="Naftali Bennett"),
    ArticleCardActor(span="יועז הנדל", name_en="Yoaz Hendel"),
    ArticleCardActor(span="בני גנץ", name_en="Benny Gantz"),  # NOT in the text → dropped
], bears_on_question=True)


class TestEvaluate:
    def test_w4_fires_gantz_never_named(self):
        r = evaluate_subject_gate(WALLA_CARD, WALLA, GANTZ)
        assert r.evaluated and r.fired
        assert r.matched_via == "none"
        assert r.verified_spans == ["איזנקוט", "בנט", "יועז הנדל"]
        assert r.dropped_spans == ["בני גנץ"]
        assert r.subjects == ["Benny Gantz", "Blue and White"]
        assert r.bears_on_question is True  # logged, disagrees with the deterministic check

    def test_control_hendel_passes_via_surface_form(self):
        r = evaluate_subject_gate(WALLA_CARD, WALLA, HENDEL)
        assert r.evaluated and not r.fired
        assert r.matched_via == "surface_form"
        assert r.matched_actor == "Yoaz Hendel"

    def test_any_listed_actor_suffices_party_only_mention(self):
        text = WALLA + " בכחול לבן שותקים."
        r = evaluate_subject_gate(WALLA_CARD, text, GANTZ)
        assert not r.fired and r.matched_actor == "Blue and White"

    def test_hebrew_prefix_on_subject_surname(self):
        text = WALLA + " ולגנץ אין תשובה."
        r = evaluate_subject_gate(WALLA_CARD, text, GANTZ)
        assert not r.fired and r.matched_form == "גנץ"

    def test_gloss_route_is_reported_but_does_not_clear_by_default(self):
        # A verified span whose English gloss names the subject, no surface form in text.
        card = ArticleCard(named_actors=[ArticleCardActor(span="איזנקוט", name_en="Benny Gantz")])
        r = evaluate_subject_gate(card, WALLA, GANTZ)
        assert r.matched_via == "gloss" and r.fired
        r2 = evaluate_subject_gate(card, WALLA, GANTZ, trust_gloss=True)
        assert r2.matched_via == "gloss" and not r2.fired

    def test_english_article_matches_name_en(self):
        card = ArticleCard(named_actors=[ArticleCardActor(span="Gantz", name_en="Benny Gantz")])
        r = evaluate_subject_gate(card, "Gantz said his party would run alone.", GANTZ)
        assert not r.fired and r.matched_via == "surface_form"


class TestFailOpen:
    def test_no_subject_card(self):
        r = evaluate_subject_gate(WALLA_CARD, WALLA, None)
        assert not r.evaluated and not r.fired and r.skip_reason == "no_subject_card"

    def test_empty_subject_card(self):
        r = evaluate_subject_gate(WALLA_CARD, WALLA, SubjectCard(actors=[]))
        assert not r.fired and r.skip_reason == "subject_card_empty"

    def test_no_article_card(self):
        r = evaluate_subject_gate(None, WALLA, GANTZ)
        assert not r.fired and r.skip_reason == "no_article_card" and r.subjects

    def test_no_verified_spans(self):
        card = ArticleCard(named_actors=[ArticleCardActor(span="Benny Gantz", name_en="Benny Gantz")])
        r = evaluate_subject_gate(card, WALLA, GANTZ)
        assert not r.fired and r.skip_reason == "no_verified_spans"
        assert r.dropped_spans == ["Benny Gantz"]


class TestSubjectCardModel:
    def test_tidy_dedupes_forms_and_seeds_name_en(self):
        card = SubjectCard.model_validate({"actors": [
            {"name_en": " Benny Gantz ", "type": "person", "surface_forms": ["gantz", "Gantz", "", "בני גנץ"]},
            {"name_en": "", "type": "party", "surface_forms": ["x"]},
        ]})
        assert len(card.actors) == 1
        assert card.actors[0].name_en == "Benny Gantz"
        assert card.actors[0].surface_forms == ["Benny Gantz", "gantz", "בני גנץ"]

    async def test_derive_fails_open_on_error(self, monkeypatch):
        async def boom(*a, **kw):
            raise RuntimeError("bedrock down")
        monkeypatch.setattr(sc, "complete_structured", boom)
        assert await derive_subject_card("Q?", None, model="m", languages="English") is None

    async def test_derive_returns_card_and_empty_is_a_real_answer(self, monkeypatch):
        seen = {}

        async def fake(model, response_model, prompt, **kw):
            seen["prompt"] = prompt
            return SubjectCard(actors=[]), {}
        monkeypatch.setattr(sc, "complete_structured", fake)
        card = await derive_subject_card("Q?", "criteria", model="m", languages="English, Hebrew")
        assert card == SubjectCard(actors=[])
        assert "QUESTION: Q?" in seen["prompt"] and "RESOLUTION CRITERIA: criteria" in seen["prompt"]
        assert "English, Hebrew" in seen["prompt"]

    async def test_derive_empty_question_is_none(self):
        assert await derive_subject_card("", None, model="m", languages="x") is None


class TestStore:
    async def test_roundtrip_and_key_identity(self, tmp_path):
        k1 = subject_card_key("Q", "C", model="m")
        assert k1 == subject_card_key("Q", "C", model="m")
        assert k1 != subject_card_key("Q", "C2", model="m")
        assert k1 != subject_card_key("Q", "C", model="m2")
        assert await get_subject_card(tmp_path, k1) is None
        await put_subject_card(tmp_path, k1, GANTZ)
        assert await get_subject_card(tmp_path, k1) == GANTZ

    async def test_malformed_entry_reads_as_miss(self, tmp_path):
        from forecast_api import subject_card_store as store
        store._get_store(tmp_path).set("k", "{not json")
        assert await get_subject_card(tmp_path, "k") is None
