"""Tests for check_shadow_flag_telemetry.py's parsing/verdict logic (retro#806).

Against a fake log string, not the real Oracle box — see check_resolution_shadow_gate.py
for the sibling script this generalizes the "shadow flag believed live but silently
inert" half of (retro#601).
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from check_shadow_flag_telemetry import (  # noqa: E402
    FLAG_REGISTRY,
    check_flags,
    count_events,
)


def _settings(**overrides) -> SimpleNamespace:
    base = {flag.settings_attr: False for flag in FLAG_REGISTRY}
    base.update(overrides)
    return SimpleNamespace(**base)


def _log_line(ts: datetime, message: str) -> str:
    return f"{ts.strftime('%Y-%m-%d %H:%M:%S')},123 INFO forecast_api.forecaster — {message}"


class TestCountEvents:
    def test_counts_matching_lines_inside_the_window(self):
        now = datetime(2026, 9, 7, 12, 0, 0)
        since = now - timedelta(hours=48)
        lines = [
            _log_line(now - timedelta(hours=1), "event=premise_verifier dead=False errored=False question=abc"),
            _log_line(now - timedelta(hours=2), "event=premise_verifier_error err=boom"),
        ]
        n = count_events(lines, ("event=premise_verifier", "event=premise_verifier_error"), since)
        assert n == 2

    def test_lines_outside_the_window_are_not_counted(self):
        now = datetime(2026, 9, 7, 12, 0, 0)
        since = now - timedelta(hours=48)
        lines = [
            _log_line(now - timedelta(hours=100), "event=premise_verifier dead=True errored=False question=old"),
        ]
        assert count_events(lines, ("event=premise_verifier",), since) == 0

    def test_ignores_lines_with_no_parsable_timestamp(self):
        since = datetime(2026, 9, 1)
        lines = ["a stray line mentioning event=premise_verifier with no timestamp prefix"]
        assert count_events(lines, ("event=premise_verifier",), since) == 0

    def test_does_not_prefix_match_an_unrelated_event_name(self):
        now = datetime(2026, 9, 7, 12, 0, 0)
        since = now - timedelta(hours=1)
        lines = [_log_line(now, "event=premise_verifier_unrelated_thing foo=bar")]
        assert count_events(lines, ("event=premise_verifier",), since) == 0

    def test_parses_a_realistic_multiline_log_blob(self):
        # A fake log string in the exact shape main.py's logging.basicConfig
        # produces, the way it would arrive from the Oracle box.
        now = datetime(2026, 9, 7, 12, 0, 0)
        since = now - timedelta(hours=48)
        log_blob = "\n".join([
            _log_line(now - timedelta(hours=3), "event=precursor_match source=daatan status=ok question=abc candidate_id=1 candidate_score=0.9 relation_type=alias direction=fwd polarity=pos"),
            _log_line(now - timedelta(hours=2), "event=precursor_match_crash node_id=42"),
            _log_line(now - timedelta(hours=200), "event=precursor_match source=polymarket status=ok question=xyz candidate_id=2 candidate_score=0.5 relation_type=nested direction=fwd polarity=neg"),
        ])
        n = count_events(log_blob.splitlines(), ("event=precursor_match", "event=precursor_match_crash"), since)
        assert n == 2


class TestCheckFlags:
    def test_disabled_flag_is_off_regardless_of_logs(self):
        settings = _settings(premise_verifier_enabled=False)
        since = datetime(2026, 9, 1)
        results = check_flags(settings, [], since)
        row = next(r for r in results if r["flag"] == "premise_verifier_enabled")
        assert row["verdict"] == "OFF"
        assert row["event_count"] == 0

    def test_enabled_flag_with_no_matching_events_is_silent(self):
        settings = _settings(premise_verifier_enabled=True)
        since = datetime(2026, 9, 1)
        results = check_flags(settings, [], since)
        row = next(r for r in results if r["flag"] == "premise_verifier_enabled")
        assert row["verdict"] == "SILENT"
        assert row["event_count"] == 0

    def test_enabled_flag_with_matching_events_passes(self):
        now = datetime(2026, 9, 7, 12, 0, 0)
        since = now - timedelta(hours=48)
        settings = _settings(premise_verifier_enabled=True)
        lines = [_log_line(now - timedelta(hours=1), "event=premise_verifier dead=False errored=False question=abc")]
        results = check_flags(settings, lines, since)
        row = next(r for r in results if r["flag"] == "premise_verifier_enabled")
        assert row["verdict"] == "PASS"
        assert row["event_count"] == 1

    def test_multiple_flags_are_evaluated_independently(self):
        now = datetime(2026, 9, 7, 12, 0, 0)
        since = now - timedelta(hours=48)
        settings = _settings(premise_verifier_enabled=True, precursor_match_enabled=True)
        lines = [_log_line(now - timedelta(hours=1), "event=premise_verifier dead=False errored=False question=abc")]
        results = check_flags(settings, lines, since)
        by_flag = {r["flag"]: r for r in results}
        assert by_flag["premise_verifier_enabled"]["verdict"] == "PASS"
        # enabled but nothing in the log for it -> SILENT, not masked by the
        # other flag's telemetry
        assert by_flag["precursor_match_enabled"]["verdict"] == "SILENT"
        assert by_flag["settled_grounding_enabled"]["verdict"] == "OFF"
        assert by_flag["retry_relaxed_search_enabled"]["verdict"] == "OFF"

    def test_registry_covers_the_four_known_shadow_flags(self):
        names = {flag.settings_attr for flag in FLAG_REGISTRY}
        assert names == {
            "premise_verifier_enabled",
            "precursor_match_enabled",
            "settled_grounding_enabled",
            "retry_relaxed_search_enabled",
        }
