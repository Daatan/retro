"""retro#805 — the verified article card: the model nominates verbatim spans, code keeps
only the ones the article really contains. Fixture text is the Walla live-blog (W4) and the
Kikar HaShabbat item (W3) from the 2026-09-07 A/B (`~/.claude-runs/2026-09-07-ab545`), the
two prod articles whose wrong-entity extraction motivated the field."""
from __future__ import annotations

import logging

from tm.extractor import (
    PROMPT_SUFFIX,
    normalize_for_match,
    text_contains_alias,
    verify_article_card,
)
from tm.models import ArticleCard, ArticleCardActor, ExtractionOutput

# Walla, 2026-09-01: list submission at the Central Elections Committee. Eisenkot and Bennett
# arrive separately; Hendel's list is named; Shimriz bows out. Gantz is never mentioned.
WALLA = (
    "למעמד ההגשה הגיעו איזנקוט ובנט בנפרד, מה שמרמז כי איחוד בין המפלגות לא יתרחש. "
    "חמש המפלגות הראשונות שצפויות להגיש את הרשימות: \"המילואימניקים\" ברשות יועז הנדל, "
    "\"ביחד\" ברשות נפתלי בנט לפיד, \"ישר!\" ברשות גדי איזנקוט, \"עמך ישראל\" ברשות עופר וינטר "
    "ו\"ישראל תחילה\" ברשות שרן השכל. "
    "בד בבד, יונתן שימריז הודיע כי לא יתמודד בבחירות הקרובות. "
    "אחד המשוריינים של נתניהו בסבירות כמעט מוחלטת הוא עו\"ד דוד פטר."
)
# Kikar HaShabbat: Ben Gvir places Goldberger ninth on Otzma Yehudit's list. No Segalovitz.
KIKAR = (
    "רגע לפני סגירת הרשימות, יו״ר 'עוצמה יהודית' השר איתמר בן גביר מגביר את המאמצים. "
    "לראשונה בתולדות המפלגה, ישובץ נציג חרדי ברשימתה לכנסת: יוסי גולדברגר, שיוצב במקום התשיעי."
)


class TestNormalizeForMatch:
    def test_strips_niqqud_quotes_and_gershayim(self):
        assert normalize_for_match("עו\"ד יוֹאָב סגלוביץ׳") == "עוד יואב סגלוביצ"  # final ץ folds too

    def test_dashes_case_and_whitespace(self):
        assert normalize_for_match("  Blue-and-White   – Benny  GANTZ ") == "blue and white benny gantz"

    def test_gershayim_rewrite_is_invisible(self):
        # retro#801 rewrites in-word `"` to U+05F4 before the model sees the text; the model
        # may copy either form back. Both normalise identically.
        assert normalize_for_match('ח"כ בני גנץ') == normalize_for_match("ח״כ בני גנץ")

    def test_none_and_empty(self):
        assert normalize_for_match(None) == ""
        assert normalize_for_match("   ") == ""


class TestTextContainsAlias:
    def test_hebrew_proclitic_prefixes_allowed(self):
        t = normalize_for_match("הודיעו כי ולגנץ יש תוכנית, ובליכוד שותקים")
        assert text_contains_alias(t, "גנץ")       # ו+ל+גנץ
        assert text_contains_alias(t, "ליכוד")     # ו+ב+ליכוד
        # ב absorbs the article: "בליכוד" is "in the Likud", so the form WITH ה is not in
        # the text. A subject card lists both spellings; the matcher does not invent them.
        assert not text_contains_alias(t, "הליכוד")

    def test_word_start_boundary_and_suffix_tolerance(self):
        t = normalize_for_match("Мильвидского видели в Кнессете; Segalovitz was absent")
        assert text_contains_alias(t, "Мильвидский"[:-2])  # Russian stem, inflected in text
        assert text_contains_alias(t, "Segalovitz")
        assert not text_contains_alias(t, "egalovitz")  # not at a word start

    def test_hebrew_plene_defective_spellings_match(self):
        # Walla: שימריז; ynet and the derived subject card: שמריז / שמריץ. One name.
        t = normalize_for_match("יונתן שימריז הודיע כי לא יתמודד")
        assert text_contains_alias(t, "שמריז")
        assert text_contains_alias(t, "יונתן שמריז")
        assert text_contains_alias(t, "שימריז")
        t2 = normalize_for_match("יונתן שמריז הודיע")
        assert text_contains_alias(t2, "שימריז")
        # מילביצקי vs מילבידסקי (a consonant differs) is NOT a spelling variant — still a miss,
        # the card has to carry it.
        assert not text_contains_alias(normalize_for_match("חנוך מילבידסקי"), "מילביצקי")

    def test_hebrew_final_forms_fold(self):
        assert text_contains_alias(normalize_for_match("סגלוביץ'"), "סגלוביצ")
        assert text_contains_alias(normalize_for_match("של רעם"), "רעמ")

    def test_short_aliases_never_match(self):
        t = normalize_for_match("Li Qiang met Xi")
        assert not text_contains_alias(t, "Li")
        assert not text_contains_alias(t, "Xi")

    def test_walla_names_present_and_gantz_absent(self):
        t = normalize_for_match(WALLA)
        for present in ("הנדל", "יועז הנדל", "איזנקוט", "בנט", "שימריז", "וינטר", "נתניהו"):
            assert text_contains_alias(t, present), present
        for absent in ("גנץ", "בני גנץ", "Gantz", "כחול לבן"):
            assert not text_contains_alias(t, absent), absent


class TestVerifyArticleCard:
    def test_keeps_only_spans_the_article_contains(self, caplog):
        card = ArticleCard(named_actors=[
            ArticleCardActor(span="גדי איזנקוט", name_en="Gadi Eisenkot"),
            ArticleCardActor(span="יועז הנדל", name_en="Yoaz Hendel"),
            # W3's failure shape: the question's subject, relabelled as if the article named it
            ArticleCardActor(span="בני גנץ", name_en="Benny Gantz"),
            ArticleCardActor(span="Benny Gantz", name_en="Benny Gantz"),  # translated = not verbatim
        ], bears_on_question=False)
        with caplog.at_level(logging.INFO, logger="tm.extractor"):
            verified, dropped = verify_article_card(card, WALLA)
        assert [a.span for a in verified.named_actors] == ["גדי איזנקוט", "יועז הנדל"]
        assert dropped == ["בני גנץ", "Benny Gantz"]
        assert verified.bears_on_question is False
        assert sum("event=article_card_span_unverified" in r.message for r in caplog.records) == 2

    def test_copying_quirks_do_not_fail_a_genuine_span(self):
        # ASCII vs gershayim quote inside the abbreviation, and a dropped inner quote.
        card = ArticleCard(named_actors=[
            ArticleCardActor(span="עו״ד דוד פטר"),
            ArticleCardActor(span="'עוצמה יהודית'"),
            ArticleCardActor(span="בן גביר"),
        ])
        v, dropped = verify_article_card(card, WALLA + " " + KIKAR)
        assert dropped == []
        assert len(v.named_actors) == 3

    def test_none_card_is_none(self):
        assert verify_article_card(None, WALLA) == (None, [])

    def test_never_raises_on_empty_text(self):
        card = ArticleCard(named_actors=[ArticleCardActor(span="x")])
        v, dropped = verify_article_card(card, "")
        assert v.named_actors == [] and dropped == ["x"]


class TestArticleCardParsing:
    """The malformed-guard trade every shadow field makes (see `_drop_malformed_claim_actor`):
    a bad card is nulled and logged, never a failed article."""

    def test_bare_string_actors_become_spans(self):
        out = ExtractionOutput.model_validate({
            "predictions": [],
            "article_card": {"named_actors": ["הליכוד", {"span": "בנט", "name_en": "Bennett"}]},
        })
        assert [(a.span, a.name_en) for a in out.article_card.named_actors] == [
            ("הליכוד", None), ("בנט", "Bennett"),
        ]
        assert out.article_card.bears_on_question is None

    def test_over_long_list_is_truncated_not_rejected(self):
        out = ExtractionOutput.model_validate({
            "predictions": [],
            "article_card": {"named_actors": [f"actor {i}" for i in range(20)]},
        })
        assert len(out.article_card.named_actors) == 8

    def test_double_serialised_card_is_parsed(self):
        out = ExtractionOutput.model_validate({
            "predictions": [],
            "article_card": '{"named_actors": [{"span": "X"}], "bears_on_question": true}',
        })
        assert out.article_card.named_actors[0].span == "X"
        assert out.article_card.bears_on_question is True

    def test_malformed_card_is_nulled_and_logged(self, caplog):
        with caplog.at_level(logging.WARNING, logger="tm.models"):
            out = ExtractionOutput.model_validate({
                "predictions": [],
                "article_card": {"named_actors": "not a list"},
            })
        assert out.article_card is None
        assert any("event=article_card_malformed" in r.message for r in caplog.records)

    def test_absent_card_is_none(self):
        assert ExtractionOutput.model_validate({"predictions": []}).article_card is None


class TestPromptText:
    def test_suffix_asks_for_the_card_and_shows_it_on_every_example(self):
        assert "ARTICLE_CARD" in PROMPT_SUFFIX
        # The three worked examples each carry the field: an example reads as the
        # definitive enumeration of its block (retro#686), so none may omit it.
        assert PROMPT_SUFFIX.count('"article_card": {{"named_actors"') == 4  # 3 examples + inline
        assert "copied verbatim" in PROMPT_SUFFIX
        assert "even when \"predictions\" is empty" in PROMPT_SUFFIX
