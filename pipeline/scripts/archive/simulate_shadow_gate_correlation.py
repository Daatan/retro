#!/usr/bin/env python3
"""
Simulate corr(true source skill, resolution-shadow weight) at various sample
sizes, to justify `resolution_shadow_min_global_predictions`
(docs/ORACLE_VARIABLES.md §9) with a number instead of a guess.

retro#337 reported three data points from an uncommitted simulation — n=6 ->
corr +0.81, n=50 -> +0.97, n=100 -> +0.97 (flat) — and picked 50 as "where it
stabilises". retro#341 found the claim mix can't reach 50 in any reasonable
time (~3.7 scoreable resolutions/week) and asked whether a lower n is still
"bounded" enough to trust. There was no simulation for anything between 6 and
50, so this script rebuilds one against the real weight formula
(leaderboard._weight_from_brier) and reports the missing n values, instead of
interpolating by eye.

Methodology: each synthetic source gets a "true" per-resolution Brier skill
drawn uniform in [0.05, 0.45] (0.25 = uninformed prior; matches the real
board's observed weight spread). Each of its n resolutions draws a noisy
per-event Brier score from Beta(true_brier * kappa, (1 - true_brier) * kappa)
-- Beta because Brier scores are themselves bounded in [0, 1], and kappa
controls how noisy one resolution's Brier score is around the source's long-
run mean. kappa=3.0 was chosen because it reproduces retro#337's two
checkpoints most closely (n=6 -> 0.806 vs the reported 0.81, n=50 -> 0.969 vs
0.97); it does NOT reproduce the exact n=50->n=100 plateau (this sim keeps
climbing slowly, 0.969 -> 0.984), so treat this as a calibrated approximation
of the original methodology, not a reproduction of it.

Run: python3 pipeline/scripts/simulate_shadow_gate_correlation.py
No repo dependencies (stdlib only) — deliberately runnable without the
forecast_api package or a venv.
"""

from __future__ import annotations

import random
import statistics

PRIOR_N = 10.0  # must match ApiSettings.resolution_shadow_brier_prior_n
SLOPE = 2.0  # must match ApiSettings.resolution_shadow_brier_slope
WEIGHT_MIN, WEIGHT_MAX = 0.25, 2.0  # must match the weight clamps
KAPPA = 3.0  # calibrated against retro#337's n=6 and n=50 checkpoints, see module docstring
TRUE_BRIER_RANGE = (0.05, 0.45)


def weight_from_brier(brier_mean: float, predictions: int) -> float:
    """Mirrors forecast_api.leaderboard._weight_from_brier exactly."""
    shrunk = (brier_mean * predictions + PRIOR_N * 0.25) / (predictions + PRIOR_N)
    weight = 1.0 + (0.25 - shrunk) * SLOPE
    return max(WEIGHT_MIN, min(WEIGHT_MAX, weight))


def _pearson(xs: list[float], ys: list[float]) -> float:
    mx, my = statistics.mean(xs), statistics.mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = sum((x - mx) ** 2 for x in xs) ** 0.5
    sy = sum((y - my) ** 2 for y in ys) ** 0.5
    return cov / (sx * sy)


def _one_trial(n: int, n_sources: int, rng: random.Random) -> float:
    true_briers = [rng.uniform(*TRUE_BRIER_RANGE) for _ in range(n_sources)]
    weights = []
    for tb in true_briers:
        alpha, beta = tb * KAPPA, (1 - tb) * KAPPA
        draws = [rng.betavariate(alpha, beta) for _ in range(n)]
        weights.append(weight_from_brier(sum(draws) / n, n))
    true_skill = [-tb for tb in true_briers]  # lower Brier = more skilled
    return _pearson(weights, true_skill)


def corr_at_n(n: int, n_sources: int = 400, n_trials: int = 150, seed: int = 42) -> float:
    rng = random.Random(seed)
    return statistics.mean(_one_trial(n, n_sources, rng) for _ in range(n_trials))


if __name__ == "__main__":
    print(f"corr(true skill, shadow weight) by sample size (kappa={KAPPA}):\n")
    for n in (6, 10, 15, 20, 25, 30, 40, 50, 100):
        marker = "  <- retro#337 checkpoint" if n in (6, 50, 100) else ""
        print(f"  n={n:>3}: corr={corr_at_n(n):.4f}{marker}")
