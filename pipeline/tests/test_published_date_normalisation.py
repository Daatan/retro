"""retro#714 — a provider's publication date is normalised, never trusted as-is.

The failure this guards is silent and one-directional: an unparseable date string is
stored, `aggregation._parse_date` returns None on it, and `recency_weight` applies the
floor (0.02) rather than ~1.0. A correctly dated article in an unexpected format is
therefore read as maximally stale and loses 50x its weight without anything raising.
"""
from datetime import datetime

import pytest

from tm.web_search_ingest import normalise_published_date as norm

NOW = datetime(2026, 8, 29)


class TestFormatsWeAccept:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("2026-02-24", "2026-02-24"),
            ("2026-02-24T10:00:00Z", "2026-02-24"),
            ("2026-02-24 10:00:00", "2026-02-24"),
            ("2026-2-4", "2026-02-04"),
        ],
    )
    def test_iso_with_or_without_time(self, raw, expected):
        assert norm(raw, now=NOW) == expected

    @pytest.mark.parametrize(
        "raw",
        ["Feb 24, 2026", "February 24, 2026", "Feb. 24 2026", "FEB 24, 2026",
         "Feb 24th, 2026", "24 Feb 2026", "24 February, 2026", "24th Feb 2026"],
    )
    def test_english_month_names_either_order(self, raw):
        assert norm(raw, now=NOW) == "2026-02-24"

    def test_sept_is_september_not_a_prefix_collision(self):
        # "sept" must not truncate to "sep"+"t" or fall through to a 3-char lookup
        # that lands somewhere else. Both spellings are in the wild.
        assert norm("Sept 3, 2026", now=NOW) == "2026-09-03"
        assert norm("Sep 3, 2026", now=NOW) == "2026-09-03"

    def test_bidi_marks_are_stripped(self):
        """The 7 prod rows that started this issue: DD/MM/YYYY wrapped in U+200F.

        The marks are invisible, carry no date information, and defeat every parser.
        Removing them cannot change which date the string denotes.
        """
        assert norm("16‏/09‏/2026", now=NOW) == "2026-09-16"
        assert norm("﻿2026-02-24", now=NOW) == "2026-02-24"

    def test_relative_dates_are_absolutized(self):
        """Brave hands its `age` field over raw, and it is often relative.

        Delegated to web_search._absolutize_relative_date so retro#562 keeps one copy
        of the relative-date grammar rather than this module growing a second.
        """
        assert norm("2 days ago", now=NOW) == "2026-08-27"
        assert norm("1 week ago", now=NOW) == "2026-08-22"


class TestUnambiguousNumericOnly:
    def test_day_first_when_the_day_cannot_be_a_month(self):
        assert norm("16/09/2026", now=NOW) == "2026-09-16"
        assert norm("16-09-2026", now=NOW) == "2026-09-16"

    def test_month_first_when_the_second_field_cannot_be_a_month(self):
        assert norm("09/16/2026", now=NOW) == "2026-09-16"

    def test_ambiguous_numeric_is_refused_not_guessed(self):
        """05/09/2026 is 5 September to most of the world and 9 May in the US.

        Nothing in a SERP payload says which, and a guess would be *believed*:
        article_date is what _apply_relative_date_override walks the calendar against,
        so a wrong one propagates into event_date. Refusing costs the article its
        provider date; guessing would cost it its meaning.
        """
        assert norm("05/09/2026", now=NOW) is None
        assert norm("01/02/2026", now=NOW) is None


class TestRejected:
    @pytest.mark.parametrize(
        "raw", [None, "", "   ", "‏​", "not a date", "yesterday",
                "Feb 24, 20", "24/2026", "0000-00-00"],
    )
    def test_non_dates_return_none(self, raw):
        assert norm(raw, now=NOW) is None

    def test_impossible_calendar_dates_are_rejected(self):
        """A shape-valid string is not a date. datetime() is the arbiter, not the regex."""
        assert norm("Feb 30, 2026", now=NOW) is None
        assert norm("2026-02-30", now=NOW) is None
        assert norm("31/04/2026", now=NOW) is None

    def test_the_slice_that_used_to_manufacture_garbage(self):
        """`(published_date or "").strip()[:10]` turned "Feb 24, 2026" into "Feb 24, 20".

        That truncated value was then stored and could never parse. The slice is gone;
        this pins that its output is not silently accepted if it ever comes back.
        """
        assert norm("Feb 24, 20", now=NOW) is None


class TestDeterminism:
    def test_no_clock_dependence_for_absolute_inputs(self):
        """The replay harness needs the same input to give the same key forever.

        Only the relative branch may read `now`; every absolute form must not.
        """
        early = datetime(2020, 1, 1)
        late = datetime(2030, 12, 31)
        for raw in ("2026-02-24", "Feb 24, 2026", "16/09/2026", "24 Feb 2026"):
            assert norm(raw, now=early) == norm(raw, now=late)
