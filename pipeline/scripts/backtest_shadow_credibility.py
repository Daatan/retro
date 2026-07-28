#!/usr/bin/env python3
"""
Backtest the resolution-shadow credibility weights against flat 1.0, so the
cutover flag (RESOLUTION_SHADOW_CREDIBILITY_ENABLED, docs/ORACLE_VARIABLES.md
§9) gets flipped on evidence rather than on vibes.

For each resolved forecast in resolution_feedback.jsonl, pool its sources'
stances into a probability twice — once with each source weighted by its
shadow-derived credibility, once with every source at 1.0 — and score both
against the real outcome with Brier. Leave-one-out: each resolution is scored
using a board built from the OTHER resolutions, so a resolution never helps
predict itself.

Run from api/ (it needs the forecast_api package for the scorer + weight fn).
ORACLE_API_KEY is only there because ApiSettings validates it at import — this
script makes no API calls:

  cd api && ORACLE_API_KEY=unused uv run python \\
      ../pipeline/scripts/backtest_shadow_credibility.py ../data/resolution_feedback.jsonl

Pull the live file off the Oracle box first — it is not in git:

  ~/.claude/skills/ssm-exec/ssm-run.sh oracle \\
      'cat /home/ubuntu/truthmachine/data/resolution_feedback.jsonl'

Reports mean Brier under both schemes, the win/loss split per resolution, and
the mean weight (which shifts evidence_mass, and with it decisiveness_floor and
thin-evidence CI widening — a real side effect worth eyeballing before the flip).
"""

from __future__ import annotations

import sys
from pathlib import Path

from forecast_api.config import settings
from forecast_api.leaderboard import _weight_from_brier
from forecast_api.resolution_scorer import _load_records, _usable_sources
from tm.scorer import brier_score, stance_to_prob


def board_from(records: list[dict]) -> dict[str, dict]:
    """Per-source mean Brier over the given resolutions, deduped per resolution
    exactly as rescore_from_disk does (one vote per source per outcome)."""
    stats: dict[str, dict] = {}
    for record in records:
        usable = _usable_sources(record)
        if not usable:
            continue
        outcome = record["outcome"]
        by_source: dict[str, list[float]] = {}
        for s in usable:
            by_source.setdefault(s["source"], []).append(s["stance"])
        for sid, stances in by_source.items():
            mean_stance = sum(stances) / len(stances)
            st = stats.setdefault(sid, {"brier_total": 0.0, "predictions": 0})
            st["brier_total"] += brier_score(stance_to_prob(mean_stance), outcome)
            st["predictions"] += 1
    return {
        sid: {"brier_score": st["brier_total"] / st["predictions"], "predictions": st["predictions"]}
        for sid, st in stats.items()
    }


def pooled_probability(record: dict, board: dict[str, dict] | None) -> float:
    """Credibility-weighted mean stance -> probability. board=None means flat 1.0.

    A deliberate simplification of aggregate_pool(): the ingest schema carries
    no relevance_score / published_date / certainty, so recency, relevance and
    class weighting cannot be reproduced here. It isolates the one term the
    cutover changes, which is what this backtest is for.
    """
    total_w = total_ws = 0.0
    by_source: dict[str, list[float]] = {}
    for s in _usable_sources(record):
        by_source.setdefault(s["source"], []).append(s["stance"])
    for sid, stances in by_source.items():
        mean_stance = sum(stances) / len(stances)
        if board is None:
            w = 1.0
        else:
            entry = board.get(sid)
            w = 1.0 if not entry else _weight_from_brier(entry["brier_score"], entry["predictions"])
        total_w += w
        total_ws += w * mean_stance
    return stance_to_prob(total_ws / total_w) if total_w else 0.5


def main(path: Path) -> int:
    records = [r for r in _load_records(path) if _usable_sources(r)]
    if not records:
        print(f"No scoreable resolutions in {path}")
        return 1

    gate = settings.resolution_shadow_min_global_predictions
    print(f"Scoreable resolutions: {len(records)}   (global gate: {gate})")
    if len(records) < gate:
        print(f"  NOTE: below the gate — with the flag on, every source would still get 1.0 today.")
    print(f"Shrinkage prior_n={settings.resolution_shadow_brier_prior_n}, "
          f"slope={settings.resolution_shadow_brier_slope}, "
          f"clamps=[{settings.resolution_shadow_weight_min}, {settings.resolution_shadow_weight_max}]")
    print()

    shadow_briers, flat_briers, wins, losses, ties = [], [], 0, 0, 0
    for i, record in enumerate(records):
        board = board_from(records[:i] + records[i + 1:])  # leave-one-out
        outcome = record["outcome"]
        b_shadow = brier_score(pooled_probability(record, board), outcome)
        b_flat = brier_score(pooled_probability(record, None), outcome)
        shadow_briers.append(b_shadow)
        flat_briers.append(b_flat)
        if b_shadow < b_flat - 1e-9:
            wins += 1
        elif b_shadow > b_flat + 1e-9:
            losses += 1
        else:
            ties += 1
        print(f"  {record.get('prediction_id', '?')[:24]:24s} outcome={str(outcome):5s} "
              f"flat={b_flat:.4f}  shadow={b_shadow:.4f}  {'better' if b_shadow < b_flat else 'worse' if b_shadow > b_flat else 'same'}")

    mean_shadow = sum(shadow_briers) / len(shadow_briers)
    mean_flat = sum(flat_briers) / len(flat_briers)
    full_board = board_from(records)
    weights = [_weight_from_brier(e["brier_score"], e["predictions"]) for e in full_board.values()]

    print()
    print(f"mean Brier  flat 1.0 : {mean_flat:.4f}")
    print(f"mean Brier  shadow   : {mean_shadow:.4f}   ({mean_flat - mean_shadow:+.4f}, "
          f"{'BETTER' if mean_shadow < mean_flat else 'WORSE' if mean_shadow > mean_flat else 'no change'})")
    print(f"per-resolution        : {wins} better, {losses} worse, {ties} unchanged")
    if weights:
        print(f"weights over {len(weights)} sources: min={min(weights):.3f} max={max(weights):.3f} "
              f"mean={sum(weights) / len(weights):.3f}")
        print("  (mean != 1.0 shifts evidence_mass, hence decisiveness_floor and CI widening)")
    print()
    print("Flip the flag only if shadow is meaningfully better AND the weight spread looks sane.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(Path(sys.argv[1])))
