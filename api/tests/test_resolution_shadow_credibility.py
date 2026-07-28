"""Credibility cutover — get_credibility_weight() sourced from the
resolution-informed shadow board instead of the legacy vault
(docs/ORACLE_VARIABLES.md §9, retro #337).

The flag ships OFF, so the first test here is the regression guard that the
default path is untouched; the rest exercise the flag-ON path.
"""

from __future__ import annotations

import json

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
