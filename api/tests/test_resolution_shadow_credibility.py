"""Credibility cutover — get_credibility_weight() sourced from the
resolution-informed shadow board instead of the legacy vault
(docs/ORACLE_VARIABLES.md §9, retro #337).

The flag ships OFF, so the first test here is the regression guard that the
default path is untouched; the rest exercise the flag-ON path.
"""

from __future__ import annotations

import asyncio
import json
import os
import time

import pytest

from forecast_api import leaderboard
from forecast_api.config import settings
from forecast_api.leaderboard import _weight_from_brier, get_credibility_weight


@pytest.fixture
def vault(monkeypatch):
    """Populate the legacy vault cache with a strongly-scored source."""
    monkeypatch.setattr(leaderboard, "_cache", {"ynet": {"skill_conservative": 10.0}})


@pytest.fixture
def shadow(monkeypatch):
    """Populate the resolution-shadow cache. Returns a setter for the global
    scoreable-resolution count so each test can sit either side of the gate."""
    board = {
        # brier 0.05 over 40 resolutions — a genuinely accurate outlet
        "sharp": {"id": "sharp", "brier_score": 0.05, "predictions": 40},
        # brier 0.55 over 40 — consistently confidently wrong
        "wrong": {"id": "wrong", "brier_score": 0.55, "predictions": 40},
        # brier 0.25 — uninformed / always hedging
        "hedger": {"id": "hedger", "brier_score": 0.25, "predictions": 40},
        # present in the vault too, so "replace, not blend" is observable
        "ynet": {"id": "ynet", "brier_score": 0.55, "predictions": 40},
    }
    monkeypatch.setattr(leaderboard, "_shadow_cache", board)
    monkeypatch.setattr(settings, "resolution_shadow_credibility_enabled", True)

    def set_total(n):
        monkeypatch.setattr(leaderboard, "_shadow_total_resolutions", n)

    set_total(settings.resolution_shadow_min_global_predictions)
    return set_total


class TestFlagOff:
    def test_default_is_off(self):
        assert settings.resolution_shadow_credibility_enabled is False

    def test_vault_path_is_untouched(self, vault):
        # 1.0 + 10.0/25.0 — the legacy transform, unchanged.
        assert get_credibility_weight("ynet") == pytest.approx(1.4)

    def test_unknown_source_is_neutral(self, vault):
        assert get_credibility_weight("never-seen") == 1.0


class TestGlobalGate:
    def test_under_the_gate_everything_is_neutral(self, shadow):
        shadow(settings.resolution_shadow_min_global_predictions - 1)
        assert get_credibility_weight("sharp") == 1.0
        assert get_credibility_weight("wrong") == 1.0

    def test_at_the_gate_scores_apply(self, shadow):
        shadow(settings.resolution_shadow_min_global_predictions)
        assert get_credibility_weight("sharp") > 1.0
        assert get_credibility_weight("wrong") < 1.0


class TestShadowWeighting:
    def test_accurate_source_is_boosted_and_wrong_source_penalised(self, shadow):
        assert get_credibility_weight("sharp") > 1.0 > get_credibility_weight("wrong")

    def test_uninformed_brier_is_exactly_neutral(self, shadow):
        """0.25 is the uninformed prior — shrinking it toward itself is a no-op,
        so a hedger lands on exactly 1.0 regardless of how much history it has."""
        assert get_credibility_weight("hedger") == pytest.approx(1.0)

    def test_source_with_no_resolution_history_is_neutral_not_vault(self, shadow, vault):
        assert get_credibility_weight("never-seen") == 1.0

    def test_vault_is_never_consulted_under_the_flag(self, shadow, vault):
        """'ynet' scores 1.4 from the vault but is a poor performer on real
        resolutions. Under the flag it must be penalised, not rescued by a
        2022 backtest — this is the 'replace, not blend' guarantee."""
        assert get_credibility_weight("ynet") < 1.0

    def test_malformed_board_row_falls_back_to_neutral(self, shadow, monkeypatch):
        monkeypatch.setattr(leaderboard, "_shadow_cache", {
            "no-brier": {"id": "no-brier", "predictions": 40},
            "no-count": {"id": "no-count", "brier_score": 0.05},
            "zero-count": {"id": "zero-count", "brier_score": 0.05, "predictions": 0},
        })
        assert get_credibility_weight("no-brier") == 1.0
        assert get_credibility_weight("no-count") == 1.0
        assert get_credibility_weight("zero-count") == 1.0


class TestWeightFromBrier:
    def test_shrinkage_protects_a_lucky_newcomer(self):
        """Two near-perfect calls must not buy the upper clamp. With prior_n=10
        the newcomer sits near neutral and earns its way out as n grows."""
        lucky = _weight_from_brier(0.01, 2)
        established = _weight_from_brier(0.01, 100)
        assert 1.0 < lucky < 1.15
        assert established > 1.3
        assert lucky < established

    def test_clamps_hold_at_both_ends(self, monkeypatch):
        monkeypatch.setattr(settings, "resolution_shadow_brier_slope", 100.0)
        assert _weight_from_brier(0.0, 10_000) == settings.resolution_shadow_weight_max
        assert _weight_from_brier(1.0, 10_000) == settings.resolution_shadow_weight_min

    def test_is_monotonic_in_accuracy(self):
        weights = [_weight_from_brier(b, 50) for b in (0.05, 0.15, 0.25, 0.35, 0.45)]
        assert weights == sorted(weights, reverse=True)


class TestLoadShadowFromDisk:
    def test_keys_the_board_list_by_source_id(self, tmp_path):
        board = tmp_path / "board.json"
        board.write_text(json.dumps([
            {"id": "bbc", "brier_score": 0.1, "predictions": 5},
            {"brier_score": 0.2, "predictions": 5},  # no id — dropped, not fatal
        ]))
        loaded = leaderboard._load_shadow_from_disk(board)
        assert set(loaded) == {"bbc"}

    def test_missing_board_is_empty_not_an_error(self, tmp_path):
        assert leaderboard._load_shadow_from_disk(tmp_path / "nope.json") == {}


class TestFallbackDefaultLogging:
    """retro#458 Phase 4: the frozen-credibility 1.0 fallback must be explicit
    in the logs, on both the legacy and shadow branches. Purely additive —
    every value assertion elsewhere in this file must stay unchanged; these
    tests only add caplog coverage of the new log event."""

    def test_legacy_branch_logs_on_unknown_source(self, vault, caplog):
        with caplog.at_level("INFO"):
            weight = get_credibility_weight("never-seen")
        assert weight == 1.0  # unchanged behavior
        assert "event=credibility_fallback_default" in caplog.text
        assert "source_id=never-seen" in caplog.text
        assert "branch=legacy" in caplog.text

    def test_legacy_branch_does_not_log_when_source_is_known(self, vault, caplog):
        with caplog.at_level("INFO"):
            weight = get_credibility_weight("ynet")
        assert weight == pytest.approx(1.4)  # unchanged behavior
        assert "event=credibility_fallback_default" not in caplog.text

    def test_shadow_branch_logs_under_the_global_gate(self, shadow, caplog):
        shadow(settings.resolution_shadow_min_global_predictions - 1)
        with caplog.at_level("INFO"):
            weight = get_credibility_weight("sharp")
        assert weight == 1.0  # unchanged behavior
        assert "event=credibility_fallback_default" in caplog.text
        assert "source_id=sharp" in caplog.text
        assert "branch=shadow" in caplog.text

    def test_shadow_branch_logs_when_source_absent_from_cache(self, shadow, caplog):
        with caplog.at_level("INFO"):
            weight = get_credibility_weight("never-seen")
        assert weight == 1.0  # unchanged behavior
        assert "event=credibility_fallback_default" in caplog.text
        assert "source_id=never-seen" in caplog.text
        assert "branch=shadow" in caplog.text

    def test_shadow_branch_logs_on_malformed_board_row(self, shadow, monkeypatch, caplog):
        monkeypatch.setattr(leaderboard, "_shadow_cache", {
            "no-brier": {"id": "no-brier", "predictions": 40},
        })
        with caplog.at_level("INFO"):
            weight = get_credibility_weight("no-brier")
        assert weight == 1.0  # unchanged behavior
        assert "event=credibility_fallback_default" in caplog.text
        assert "source_id=no-brier" in caplog.text
        assert "branch=shadow" in caplog.text

    def test_shadow_branch_does_not_log_when_score_applies(self, shadow, caplog):
        with caplog.at_level("INFO"):
            weight = get_credibility_weight("sharp")
        assert weight > 1.0  # unchanged behavior
        assert "event=credibility_fallback_default" not in caplog.text


class TestLeaderboardStalenessWarning:
    """retro#458 Phase 4: refresh_cache() should warn once per refresh when
    leaderboard.json's snapshot is older than the staleness threshold."""

    def _write_board_with_age(self, tmp_path, age_days):
        board = tmp_path / "leaderboard.json"
        board.write_text(json.dumps([{"id": "ynet", "skill_conservative": 1.0}]))
        old_time = time.time() - age_days * 86400
        os.utime(board, (old_time, old_time))
        return board

    def test_warns_when_snapshot_is_older_than_the_threshold(self, tmp_path, caplog, monkeypatch):
        monkeypatch.setattr(leaderboard, "_cache", {})
        monkeypatch.setattr(leaderboard, "_snapshot_mtime", None)
        board = self._write_board_with_age(tmp_path, leaderboard._STALE_THRESHOLD_DAYS + 5)
        with caplog.at_level("WARNING"):
            asyncio.run(leaderboard.refresh_cache(board))
        assert "event=credibility_leaderboard_stale" in caplog.text
        assert f"snapshot_date={leaderboard.leaderboard_snapshot_date()}" in caplog.text

    def test_does_not_warn_when_snapshot_is_fresh(self, tmp_path, caplog, monkeypatch):
        monkeypatch.setattr(leaderboard, "_cache", {})
        monkeypatch.setattr(leaderboard, "_snapshot_mtime", None)
        board = self._write_board_with_age(tmp_path, 1)
        with caplog.at_level("WARNING"):
            asyncio.run(leaderboard.refresh_cache(board))
        assert "event=credibility_leaderboard_stale" not in caplog.text

    def test_does_not_warn_when_the_file_is_missing(self, tmp_path, caplog, monkeypatch):
        monkeypatch.setattr(leaderboard, "_cache", {})
        monkeypatch.setattr(leaderboard, "_snapshot_mtime", None)
        with caplog.at_level("WARNING"):
            asyncio.run(leaderboard.refresh_cache(tmp_path / "nope.json"))
        assert "event=credibility_leaderboard_stale" not in caplog.text
