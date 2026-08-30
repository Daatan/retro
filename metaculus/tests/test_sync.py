from datetime import datetime, timedelta, timezone

import pytest

from sync import build_comment, needs_forecast, question_text, select_season_tournament

NOW = datetime(2026, 8, 22, 12, 0, tzinfo=timezone.utc)
STALE_AFTER = timedelta(hours=24)


class TestNeedsForecast:
    def test_never_forecasted(self):
        assert needs_forecast({"my_forecasts": {"latest": None}}, STALE_AFTER, NOW) is True

    def test_missing_my_forecasts_key(self):
        assert needs_forecast({}, STALE_AFTER, NOW) is True

    def test_recent_forecast_is_not_stale(self):
        recent = (NOW - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
        question = {"my_forecasts": {"latest": {"start_time": recent}}}
        assert needs_forecast(question, STALE_AFTER, NOW) is False

    def test_old_forecast_is_stale(self):
        old = (NOW - timedelta(hours=48)).isoformat().replace("+00:00", "Z")
        question = {"my_forecasts": {"latest": {"start_time": old}}}
        assert needs_forecast(question, STALE_AFTER, NOW) is True

    def test_latest_present_but_no_timestamp_field(self):
        question = {"my_forecasts": {"latest": {}}}
        assert needs_forecast(question, STALE_AFTER, NOW) is True


class TestQuestionText:
    def test_joins_description_criteria_and_fine_print(self):
        post = {"title": "Will X happen?"}
        question = {
            "description": "Background.",
            "resolution_criteria": "Resolves YES if X.",
            "fine_print": "Edge cases here.",
        }
        title, criteria = question_text(post, question)
        assert title == "Will X happen?"
        assert criteria == "Background.\n\nResolves YES if X.\n\nEdge cases here."

    def test_no_criteria_fields_yields_none(self):
        title, criteria = question_text({"title": "Q"}, {})
        assert title == "Q"
        assert criteria is None

    def test_falls_back_to_question_title_when_post_title_missing(self):
        title, _ = question_text({}, {"title": "Fallback title"})
        assert title == "Fallback title"


class TestBuildComment:
    def test_includes_probability_and_sources(self):
        result = {
            "reason": "Evidence favors YES.",
            "mean": 0.4,
            "articles_used": 7,
            "sources": ["a.com", "b.com", "c.com", "d.com", "e.com", "f.com"],
        }
        comment = build_comment(result, probability=0.7)
        assert "Evidence favors YES." in comment
        assert "p=0.70" in comment
        assert "mean stance 0.400" in comment
        assert "7 articles" in comment
        # Only the first 5 sources are included.
        assert "f.com" not in comment
        assert "e.com" in comment

    def test_missing_reason_falls_back(self):
        comment = build_comment({}, probability=0.5)
        assert "Forecast from Daatan Oracle." in comment


def test_required_env_rejects_unset_and_blank(monkeypatch):
    # An unset GitHub secret arrives as "" — must name the variable, not fail later in httpx.
    from sync import _required_env

    monkeypatch.delenv("METACULUS_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="METACULUS_API_KEY"):
        _required_env("METACULUS_API_KEY")
    monkeypatch.setenv("METACULUS_API_KEY", "   ")
    with pytest.raises(SystemExit, match="METACULUS_API_KEY"):
        _required_env("METACULUS_API_KEY")
    monkeypatch.setenv("METACULUS_API_KEY", " tok ")
    assert _required_env("METACULUS_API_KEY") == "tok"


class TestSelectSeasonTournament:
    NOW = datetime(2026, 9, 3, tzinfo=timezone.utc)

    def _t(self, slug, name, start, fc_end, close=None):
        return {"slug": slug, "name": name, "start_date": start, "forecasting_end_date": fc_end, "close_date": close}

    def test_picks_latest_started_open_season_on_overlap(self):
        # Real shapes from /api/projects/tournaments/ (2026-08-30): Summer keeps
        # opening questions until 09-06, Fall opens early September.
        tournaments = [
            self._t("spring-aib-2026", "Spring 2026 FutureEval Bot Tournament", "2026-01-05T00:00:00Z", "2026-05-06T00:00:00Z"),
            self._t("summer-futureeval-2026", "Summer 2026 FutureEval Bot Tournament", "2026-05-18T00:00:00Z", "2026-09-06T00:00:00Z"),
            self._t("fall-futureeval-2026", "Fall 2026 FutureEval Bot Tournament", "2026-09-01T00:00:00Z", "2027-01-06T00:00:00Z"),
            self._t("metaculus-cup-2026", "Metaculus Cup", "2026-05-01T00:00:00Z", "2026-12-01T00:00:00Z"),
        ]
        assert select_season_tournament(tournaments, self.NOW) == "fall-futureeval-2026"

    def test_falls_back_to_running_season_when_next_not_started(self):
        tournaments = [
            self._t("summer-futureeval-2026", "Summer 2026 FutureEval Bot Tournament", "2026-05-18T00:00:00Z", "2026-09-06T00:00:00Z"),
            self._t("fall-futureeval-2026", "Fall 2026 FutureEval Bot Tournament", "2026-09-10T00:00:00Z", "2027-01-06T00:00:00Z"),
        ]
        assert select_season_tournament(tournaments, self.NOW) == "summer-futureeval-2026"

    def test_none_when_every_season_window_is_closed(self):
        tournaments = [
            self._t("summer-futureeval-2026", "Summer 2026 FutureEval Bot Tournament", "2026-05-18T00:00:00Z", "2026-09-06T00:00:00Z", "2026-11-05T00:00:00Z"),
        ]
        assert select_season_tournament(tournaments, datetime(2026, 9, 7, tzinfo=timezone.utc)) is None

    def test_matches_marker_in_name_when_slug_is_opaque(self):
        tournaments = [self._t("aibq4", "AI Forecasting Benchmark Tournament - 2024 Q4", "2024-10-08T00:00:00Z", "2025-01-08T00:00:00Z")]
        assert select_season_tournament(tournaments, datetime(2024, 11, 1, tzinfo=timezone.utc)) == "aibq4"

    def test_ignores_rows_without_slug_or_start(self):
        tournaments = [
            {"slug": None, "name": "FutureEval mystery", "start_date": "2026-05-18T00:00:00Z", "forecasting_end_date": None},
            {"slug": "summer-futureeval-2026", "name": None, "start_date": None, "forecasting_end_date": None},
        ]
        assert select_season_tournament(tournaments, self.NOW) is None
