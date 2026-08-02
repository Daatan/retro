"""Tests for schema validation of the Factum Atlas case studies (retro#348).

`data/events/*.json` is hand-curated, so a typo'd field name or an unparseable
outcome_date is a silent data defect: the backtest still runs, it just scores a
starved or mis-keyed case study. These tests pin each check, and the last one
asserts the real directory is clean — that is the regression barrier, so a bad
hand edit fails CI instead of quietly degrading the atlas.
"""

import datetime as dt
import importlib.util
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = REPO_ROOT / "pipeline" / "scripts" / "validate_events.py"

# scripts/ is not an importable package — load the module by path.
_spec = importlib.util.spec_from_file_location("validate_events", _SCRIPT)
validate_events = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(validate_events)

TODAY = dt.date(2026, 8, 2)

VALID_EVENT = {
    "id": "A01",
    "name": "Bennett-Lapid coalition sworn in as government",
    "outcome": True,
    "outcome_date": "2021-06-13",
    "search_keywords": ["Bennett-Lapid government", "sworn in"],
    "llm_referee_criteria": "Must discuss the formation of the 36th government.",
    "predictive_window_days": 30,
    "category": ["Israeli Politics"],
    "tags": ["coalition", "Bennett"],
    "polymarket": None,
}


# "Absent" and "explicitly null" are different states here — `outcome: null`
# means unresolved, while a missing `outcome` is a malformed file — so the
# helper needs a sentinel rather than overloading None.
DROP = object()


def check(**overrides):
    """Validate VALID_EVENT with overrides applied; DROP removes the key."""
    event = {**VALID_EVENT, **overrides}
    for key, value in overrides.items():
        if value is DROP:
            event.pop(key, None)
    return validate_events.validate_event(event, event.get("id", "A01"), today=TODAY)


class TestSchemaCompleteness:
    def test_valid_event_is_clean(self):
        assert check() == []

    def test_missing_required_field_is_flagged(self):
        assert "missing_field:name" in check(name=DROP)

    def test_missing_nullable_polymarket_key_is_flagged(self):
        assert "missing_field:polymarket" in check(polymarket=DROP)

    def test_explicit_null_polymarket_is_allowed(self):
        assert check(polymarket=None) == []

    def test_wrong_type_is_flagged(self):
        assert "wrong_type:search_keywords" in check(search_keywords="not a list")

    def test_int_in_bool_field_is_flagged(self):
        # bool is a subclass of int, so a naive isinstance check passes 1 as True.
        assert "wrong_type:outcome" in check(outcome=1)

    def test_bool_in_int_field_is_flagged(self):
        assert "wrong_type:predictive_window_days" in check(predictive_window_days=True)

    def test_unknown_field_is_flagged(self):
        # The point of this check: catches `outcome_dat` before the backtest does.
        assert "unknown_field:outcome_dat" in check(outcome_dat="2021-06-13")

    def test_optional_fields_are_allowed(self):
        assert check(oracle_question="Will X happen?", scope="global") == []


class TestIdentity:
    def test_malformed_id_is_flagged(self):
        assert "malformed_id" in check(id="a1")

    def test_three_digit_id_is_allowed(self):
        assert validate_events.validate_event(
            {**VALID_EVENT, "id": "A100"}, "A100", today=TODAY
        ) == []

    def test_id_must_match_filename(self):
        assert "id_filename_mismatch" in validate_events.validate_event(
            VALID_EVENT, "B07", today=TODAY
        )

    def test_empty_name_is_flagged(self):
        assert "empty_name" in check(name="   ")


class TestOutcomeDate:
    def test_unparseable_date_is_flagged(self):
        assert "unparseable_outcome_date" in check(outcome_date="13-06-2021")

    def test_impossible_date_is_flagged(self):
        assert "unparseable_outcome_date" in check(outcome_date="2021-02-30")

    def test_future_date_on_resolved_event_is_flagged(self):
        assert "future_outcome_date" in check(outcome_date="2027-01-01")

    def test_today_is_not_future(self):
        assert check(outcome_date=TODAY.isoformat()) == []

    def test_resolved_event_without_date_is_flagged(self):
        assert "resolved_without_outcome_date" in check(outcome_date=DROP)

    def test_false_outcome_still_requires_a_date(self):
        # `outcome: false` is resolved too — the event demonstrably did not happen.
        assert "resolved_without_outcome_date" in check(outcome=False, outcome_date=DROP)


class TestUnresolvedEvents:
    """data/README.md documents `outcome: null` as unresolved."""

    def test_unresolved_event_needs_no_date(self):
        assert check(outcome=None, outcome_date=DROP) == []

    def test_unresolved_event_may_carry_a_future_target_date(self):
        assert check(outcome=None, outcome_date="2027-01-01") == []

    def test_unresolved_event_still_needs_a_parseable_date_if_present(self):
        assert "unparseable_outcome_date" in check(outcome=None, outcome_date="soon")

    def test_optional_window_may_be_absent(self):
        # README documents predictive_window_days as optional with a default.
        assert check(predictive_window_days=DROP) == []


class TestListFields:
    def test_empty_keywords_flagged(self):
        assert "empty_search_keywords" in check(search_keywords=[])

    def test_blank_keyword_flagged(self):
        assert "blank_entry_in:search_keywords" in check(search_keywords=["ok", "  "])

    def test_non_string_keyword_flagged(self):
        assert "non_string_in:search_keywords" in check(search_keywords=["ok", 7])

    def test_empty_category_flagged(self):
        assert "empty_category" in check(category=[])

    def test_empty_duel_keywords_allowed(self):
        # duel_keywords is optional; present-but-empty is not a defect.
        assert check(duel_keywords=[]) == []

    def test_non_positive_window_flagged(self):
        assert "non_positive_window" in check(predictive_window_days=0)


class TestPolymarket:
    def test_valid_market_is_clean(self):
        assert check(
            polymarket={
                "question": "Will the coalition survive?",
                "url": "https://polymarket.com/event/x",
                "match_quality": "exact",
            }
        ) == []

    def test_invert_flag_is_allowed(self):
        assert check(
            polymarket={
                "question": "Q",
                "url": "https://polymarket.com/event/x",
                "match_quality": "close",
                "invert": True,
            }
        ) == []

    def test_missing_url_flagged(self):
        assert "polymarket_missing:url" in check(
            polymarket={"question": "Q", "match_quality": "exact"}
        )

    def test_unknown_match_quality_flagged(self):
        assert "polymarket_bad_match_quality" in check(
            polymarket={
                "question": "Q",
                "url": "https://polymarket.com/event/x",
                "match_quality": "vague",
            }
        )

    def test_non_bool_invert_flagged(self):
        assert "polymarket_bad_invert" in check(
            polymarket={
                "question": "Q",
                "url": "https://polymarket.com/event/x",
                "match_quality": "exact",
                "invert": "yes",
            }
        )


class TestDirectoryScan:
    def test_duplicate_ids_are_flagged(self, tmp_path):
        (tmp_path / "A01.json").write_text(json.dumps(VALID_EVENT))
        (tmp_path / "A02.json").write_text(json.dumps(VALID_EVENT))
        flagged, checked = validate_events.validate_events_dir(tmp_path, today=TODAY)
        assert checked == 2
        codes = [c for _, findings in flagged for c in findings]
        assert any(c.startswith("duplicate_id:") for c in codes)

    def test_invalid_json_is_reported_not_raised(self, tmp_path):
        (tmp_path / "A01.json").write_text("{not json")
        flagged, _ = validate_events.validate_events_dir(tmp_path, today=TODAY)
        assert any(
            c.startswith("invalid_json:") for _, findings in flagged for c in findings
        )

    def test_real_event_directory_is_clean(self):
        """The regression barrier: every committed case study must validate."""
        events_dir = REPO_ROOT / "data" / "events"
        assert events_dir.is_dir(), f"expected case studies at {events_dir}"
        flagged, checked = validate_events.validate_events_dir(events_dir)
        assert checked > 0, "no event files found"
        assert flagged == [], f"{len(flagged)} event(s) failed validation: {flagged}"
