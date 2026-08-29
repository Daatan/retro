"""Pin the cadence-critical defaults (retro#728).

Tournament questions close 1.5h after opening and only the last forecast
before close is scored, so a stale window measured in hours-plural silently
throws away the spot-score mechanic. These are cheap guards against that
regressing back to the pre-retro#617 values.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sync import needs_forecast

SYNC_SRC = (Path(__file__).resolve().parent.parent / "sync.py").read_text()


def _default_stale_hours() -> float:
    m = re.search(r'STALE_AFTER_HOURS",\s*"([0-9.]+)"', SYNC_SRC)
    assert m, "STALE_AFTER_HOURS default not found in sync.py"
    return float(m.group(1))


def test_default_stale_window_fits_inside_a_question_window() -> None:
    # Questions close after 1.5h; a default at or above that means we never
    # re-forecast within the window at all.
    assert _default_stale_hours() < 1.5


def test_default_stale_window_allows_a_mid_window_update() -> None:
    # Room for at least two forecasts inside the 1.5h window.
    assert _default_stale_hours() <= 0.75


def test_question_forecast_50_minutes_ago_is_stale_at_the_default() -> None:
    # 50 min > the 45 min default, so it is due a re-forecast before close.
    now = datetime(2026, 8, 29, 12, 0, tzinfo=timezone.utc)
    question = {"my_forecasts": {"latest": {"start_time": "2026-08-29T11:10:00Z"}}}
    assert needs_forecast(question, timedelta(hours=_default_stale_hours()), now)


def test_question_forecast_5_minutes_ago_is_not_stale() -> None:
    now = datetime(2026, 8, 29, 12, 0, tzinfo=timezone.utc)
    question = {"my_forecasts": {"latest": {"start_time": "2026-08-29T11:55:00Z"}}}
    assert not needs_forecast(question, timedelta(hours=_default_stale_hours()), now)
