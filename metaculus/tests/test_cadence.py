"""Pin the cadence-critical defaults (retro#728, retro#755).

Two separate knobs, often confused:

* **How often we poll** — every 20 min, set by `infra/metaculus-sync.timer`.
  That is what catches a question at all inside its window, and it is not
  affected by anything here.
* **How often we submit** — `STALE_AFTER_HOURS`. The FutureEval rules say
  "bot makers should only submit one forecast per question in these bot-only
  tournaments", so this must sit ABOVE the question window, not below it.

An earlier version of this file pinned the opposite (`<= 0.75`, "room for at
least two forecasts inside the 1.5h window"). That maximised the spot peer
score — only the last forecast before close counts — but broke the rule.
retro#755 inverted the decision; these tests keep it inverted.

The window is **3.0h**, measured 2026-08-30 across all 60 questions of the
2026-08-24 MiniBench round (min = p50 = max = 3.00h) and unchanged for the five
rounds since 2026-06-15. Metaculus's own docs still say 1.5h; they are stale.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sync import needs_forecast

SYNC_SRC = (Path(__file__).resolve().parent.parent / "sync.py").read_text()

QUESTION_WINDOW_HOURS = 3.0


def _default_stale_hours() -> float:
    m = re.search(r'STALE_AFTER_HOURS",\s*"([0-9.]+)"', SYNC_SRC)
    assert m, "STALE_AFTER_HOURS default not found in sync.py"
    return float(m.group(1))


def test_default_stale_window_exceeds_the_question_window() -> None:
    # One forecast per question: by the time we could re-forecast, the question
    # has closed. A default below the window silently resumes double-submitting.
    assert _default_stale_hours() >= QUESTION_WINDOW_HOURS


def test_a_question_forecast_earlier_in_the_same_window_is_not_re_forecast() -> None:
    # 2h55m after our submission the question is about to close; we must not
    # submit again.
    now = datetime(2026, 8, 29, 12, 0, tzinfo=timezone.utc)
    question = {"my_forecasts": {"latest": {"start_time": "2026-08-29T09:05:00Z"}}}
    assert not needs_forecast(question, timedelta(hours=_default_stale_hours()), now)


def test_an_unforecast_question_is_still_picked_up() -> None:
    now = datetime(2026, 8, 29, 12, 0, tzinfo=timezone.utc)
    assert needs_forecast({"my_forecasts": {"latest": None}}, timedelta(hours=_default_stale_hours()), now)
