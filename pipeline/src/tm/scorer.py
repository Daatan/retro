import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from openskill.models import PlackettLuce

from .utils import predates_outcome


def stance_to_prob(stance: float) -> float:
    """Convert stance [-1, 1] to probability [0, 1]."""
    return (stance + 1) / 2


def brier_score(prob: float, outcome: bool) -> float:
    """Brier score for a probability forecast."""
    return (prob - (1.0 if outcome else 0.0)) ** 2


def time_decay_weight(article_date: str, outcome_date: str, half_life_days: float = 30.0) -> float:
    """Weight predictions closer to the event more heavily.

    Uses exponential decay: weight = 2^(-days_before / half_life_days).
    A prediction made on the event date gets weight 1.0; one made
    `half_life_days` before gets 0.5. Returns 1.0 when dates are missing.
    """
    if not article_date or not outcome_date:
        return 1.0
    try:
        art_dt = datetime.fromisoformat(article_date[:10])
        evt_dt = datetime.fromisoformat(outcome_date[:10])
        days_before = max(0, (evt_dt - art_dt).days)
        return 2.0 ** (-days_before / half_life_days)
    except (ValueError, TypeError):
        return 1.0


def brier_decomposition(
    pairs: list[tuple[float, float]], n_bins: int = 5
) -> Optional[dict]:
    """Murphy (1973) Brier decomposition: BS = REL − RES + UNC.

    pairs: (predicted_prob, outcome) with outcome ∈ {0.0, 1.0}.
    Returns None when fewer than 5 data points.
    Note: when all outcomes are the same (all YES/NO), RES and UNC collapse to 0
    and the decomposition gives BS = REL trivially.
    """
    if len(pairs) < 5:
        return None

    bins: list[list[tuple[float, float]]] = [[] for _ in range(n_bins)]
    for p, o in pairs:
        idx = min(int(p * n_bins), n_bins - 1)
        bins[idx].append((p, o))

    n = len(pairs)
    o_bar = sum(o for _, o in pairs) / n

    rel = res = 0.0
    for b in bins:
        if not b:
            continue
        n_k = len(b)
        p_bar_k = sum(p for p, _ in b) / n_k
        o_bar_k = sum(o for _, o in b) / n_k
        rel += (n_k / n) * (p_bar_k - o_bar_k) ** 2
        res += (n_k / n) * (o_bar_k - o_bar) ** 2

    unc = o_bar * (1 - o_bar)

    return {
        "brier": round(rel - res + unc, 4),
        "rel":   round(rel, 4),
        "res":   round(res, 4),
        "unc":   round(unc, 4),
        "n":     n,
        "o_bar": round(o_bar, 4),
    }


def compute_calibration_bins(
    pairs: list[tuple[float, float]], n_bins: int = 10
) -> Optional[dict]:
    """Bin (implied_p, outcome) pairs and compute actual outcome rate per bin.

    Returns None when there are fewer than 10 data points.
    """
    bins: list[list[float]] = [[] for _ in range(n_bins)]
    for p, outcome in pairs:
        idx = min(int(p * n_bins), n_bins - 1)
        bins[idx].append(outcome)

    if sum(len(b) for b in bins) < 10:
        return None

    labels, predicted, actual, counts = [], [], [], []
    for i, b in enumerate(bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        labels.append(f"{int(lo*100)}–{int(hi*100)}%")
        predicted.append(round((lo + hi) / 2, 3))
        actual.append(round(sum(b) / len(b), 3) if b else 0.0)
        counts.append(len(b))

    return {"labels": labels, "predicted": predicted, "actual": actual, "counts": counts}


class Scorer:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.atlas_dir = data_dir / "atlas"
        self.events_dir = data_dir / "events"
        self.sources_dir = data_dir / "sources"
        self.scores_path = data_dir / "leaderboard.json"
        self.calibration_path = data_dir / "calibration.json"

    # ─────────────────────────────────────────────
    # Core scoring primitives
    # ─────────────────────────────────────────────

    def calculate_brier(self, stance: float, outcome: bool) -> float:
        """Brier score for a single prediction."""
        return brier_score(stance_to_prob(stance), outcome)

    def confidence_weight(self, certainty: float) -> float:
        """
        Higher certainty → higher weight when correct, higher penalty when wrong.
        Returns a multiplier in [0.5, 2.0] so low-certainty predictions still count.
        """
        certainty = max(0.0, min(1.0, certainty))
        return 0.5 + 1.5 * certainty

    def weighted_brier(self, stance: float, certainty: float, outcome: bool) -> float:
        """Confidence-weighted Brier score."""
        return self.calculate_brier(stance, outcome) * self.confidence_weight(certainty)

    def calculate_log_score(self, stance: float, outcome: bool) -> float:
        """Logarithmic scoring rule. Higher (less negative) is better. Range: (-∞, 0]."""
        prob = max(0.01, min(0.99, (stance + 1) / 2))
        return math.log(prob) if outcome else math.log(1 - prob)

    def calculate_accuracy(self, stance: float, outcome: bool) -> int:
        """1 if directionally correct, 0 otherwise. Stance=0 counts as wrong."""
        return 1 if (stance > 0) == outcome else 0

    # ─────────────────────────────────────────────
    # Stat buckets
    # ─────────────────────────────────────────────

    def _empty_bucket(self, name: str = "") -> dict:
        return {
            "name": name,
            "brier_total": 0.0,
            "weighted_brier_total": 0.0,
            "time_decay_brier_total": 0.0,
            "time_decay_weight_total": 0.0,
            "log_score_total": 0.0,
            "correct_count": 0,
            "prediction_count": 0,
            "events_covered": 0,
            "elo": 1200.0,
        }

    # ─────────────────────────────────────────────
    # Main run
    # ─────────────────────────────────────────────

    def run(self):
        # global stats:  sid → bucket
        global_stats: Dict[str, dict] = {}
        # category stats: sid → category → bucket
        category_stats: Dict[str, Dict[str, dict]] = defaultdict(dict)
        # OpenSkill ratings: sid → Rating (μ=25, σ=25/3 by default)
        skill_model = PlackettLuce()
        skill_ratings: Dict[str, object] = {}

        # Initialise buckets for all sources
        for source_file in self.sources_dir.glob("*.json"):
            source = json.loads(source_file.read_text())
            sid = source["id"]
            global_stats[sid] = self._empty_bucket(source["name"])
            skill_ratings[sid] = skill_model.rating()

        # Global (implied_p, outcome) pairs for calibration curve
        calib_pairs: list[tuple[float, float]] = []

        # Score every event
        for event_file in sorted(self.events_dir.glob("*.json")):
            event = json.loads(event_file.read_text())
            eid = event["id"]
            outcome = event["outcome"]
            outcome_date: str = event.get("outcome_date", "")
            categories: List[str] = event.get("category", [])

            atlas_event_dir = self.atlas_dir / eid
            if not atlas_event_dir.exists():
                continue

            event_predictions = []   # (sid, stance) for ELO update

            for source_dir in atlas_event_dir.iterdir():
                if not source_dir.is_dir():
                    continue
                sid = source_dir.name
                if sid not in global_stats:
                    continue

                entry_files = list(source_dir.glob("entry_*.json"))
                if not entry_files:
                    continue

                # Ensure per-category buckets exist for this source
                for cat in categories:
                    if cat not in category_stats[sid]:
                        category_stats[sid][cat] = self._empty_bucket(
                            global_stats[sid]["name"]
                        )

                covered = False
                for entry_file in entry_files:
                    entry = json.loads(entry_file.read_text())
                    preds = entry.get("predictions", [])
                    if not preds:
                        continue

                    # Anti-lookahead: never score a prediction published after the
                    # outcome — it would leak future knowledge into the leaderboard.
                    article_date = entry.get("article_date", "")
                    if not predates_outcome(article_date, outcome_date):
                        continue

                    stances     = [p.get("stance", 0.0) for p in preds]
                    certainties = [p.get("certainty", 0.5) for p in preds]
                    avg_stance    = sum(stances) / len(stances)
                    avg_certainty = sum(certainties) / len(certainties)

                    b   = self.calculate_brier(avg_stance, outcome)
                    wb  = self.weighted_brier(avg_stance, avg_certainty, outcome)
                    ls  = self.calculate_log_score(avg_stance, outcome)
                    acc = self.calculate_accuracy(avg_stance, outcome)

                    decay = time_decay_weight(article_date, outcome_date)

                    # Global bucket
                    global_stats[sid]["brier_total"]              += b
                    global_stats[sid]["weighted_brier_total"]     += wb
                    global_stats[sid]["time_decay_brier_total"]   += b * decay
                    global_stats[sid]["time_decay_weight_total"]  += decay
                    global_stats[sid]["log_score_total"]          += ls
                    global_stats[sid]["correct_count"]            += acc
                    global_stats[sid]["prediction_count"]         += 1

                    # Per-category buckets
                    for cat in categories:
                        category_stats[sid][cat]["brier_total"]          += b
                        category_stats[sid][cat]["weighted_brier_total"] += wb
                        category_stats[sid][cat]["log_score_total"]      += ls
                        category_stats[sid][cat]["correct_count"]        += acc
                        category_stats[sid][cat]["prediction_count"]     += 1

                    implied_p = (avg_stance + 1.0) / 2.0
                    calib_pairs.append((implied_p, 1.0 if outcome else 0.0))

                    event_predictions.append((sid, avg_stance))
                    covered = True

                if covered:
                    global_stats[sid]["events_covered"] += 1
                    for cat in categories:
                        category_stats[sid][cat]["events_covered"] += 1

            # ELO + OpenSkill updates (global only)
            if len(event_predictions) > 1:
                self._update_elo(global_stats, event_predictions, outcome)
                self._update_skill(skill_model, skill_ratings, event_predictions, outcome)

        # ── Build leaderboard ────────────────────
        leaderboard = []
        for sid, stats in global_stats.items():
            n = stats["prediction_count"]
            if n == 0:
                continue

            # Per-category scores for this source
            skill_r = skill_ratings.get(sid)
            per_category = {}
            for cat, cstats in category_stats.get(sid, {}).items():
                cn = cstats["prediction_count"]
                if cn == 0:
                    continue
                per_category[cat] = {
                    "brier_score":          round(cstats["brier_total"] / cn, 4),
                    "weighted_brier_score": round(cstats["weighted_brier_total"] / cn, 4),
                    "log_score":            round(cstats["log_score_total"] / cn, 4),
                    "accuracy":             round(cstats["correct_count"] / cn, 4),
                    "predictions":          cn,
                    "events":               cstats["events_covered"],
                }

            tw = stats["time_decay_weight_total"]
            time_decay_brier = (
                round(stats["time_decay_brier_total"] / tw, 4) if tw > 0 else None
            )

            leaderboard.append({
                "id":                     sid,
                "name":                   stats["name"],
                "brier_score":            round(stats["brier_total"] / n, 4),
                "weighted_brier_score":   round(stats["weighted_brier_total"] / n, 4),
                "time_decay_brier_score": time_decay_brier,
                "log_score":              round(stats["log_score_total"] / n, 4),
                "accuracy":               round(stats["correct_count"] / n, 4),
                "elo":                round(stats["elo"], 0),
                "skill_mu":           round(skill_r.mu, 2) if skill_r else 25.0,
                "skill_sigma":        round(skill_r.sigma, 2) if skill_r else 8.33,
                "skill_conservative": round(skill_r.mu - 3 * skill_r.sigma, 2) if skill_r else 0.0,
                "predictions":            n,
                "events":                 stats["events_covered"],
                "by_category":            per_category,
            })

        leaderboard.sort(key=lambda x: x["skill_conservative"], reverse=True)

        self.scores_path.write_text(json.dumps(leaderboard, indent=2))
        print(f"Scoring complete. {len(leaderboard)} sources scored → {self.scores_path}")

        # ── Calibration curve ──────────────────────────────────────────────────
        calibration = compute_calibration_bins(calib_pairs)
        self.calibration_path.write_text(
            json.dumps(
                {"n_predictions": len(calib_pairs), "calibration": calibration},
                indent=2,
            )
        )
        if calibration:
            print(f"Calibration written ({len(calib_pairs)} predictions) → {self.calibration_path}")
        else:
            print(f"Not enough predictions ({len(calib_pairs)}) for calibration curve")

        return leaderboard

    # ─────────────────────────────────────────────
    # ELO helper
    # ─────────────────────────────────────────────

    def _update_elo(self, stats: dict, predictions: list, outcome: bool, K: int = 32):
        """Zero-sum ELO adjustment: sources that predicted correctly gain from those that didn't."""
        for sid, stance in predictions:
            is_correct = (stance > 0) == outcome
            delta = K / len(predictions)
            stats[sid]["elo"] += delta if is_correct else -delta

    def _update_skill(
        self,
        model: PlackettLuce,
        ratings: Dict[str, object],
        predictions: list,
        outcome: bool,
    ) -> None:
        """Update OpenSkill ratings: correct predictors beat incorrect ones."""
        winners = [sid for sid, stance in predictions if (stance > 0) == outcome]
        losers  = [sid for sid, stance in predictions if (stance > 0) != outcome]
        if not winners or not losers:
            return  # all right or all wrong — no information

        winner_teams = [[ratings[sid]] for sid in winners]
        loser_teams  = [[ratings[sid]] for sid in losers]
        ranks = [0] * len(winner_teams) + [1] * len(loser_teams)
        new_ratings = model.rate(winner_teams + loser_teams, ranks=ranks)

        for i, sid in enumerate(winners):
            ratings[sid] = new_ratings[i][0]
        for i, sid in enumerate(losers):
            ratings[sid] = new_ratings[len(winners) + i][0]

    # Backwards-compat alias
    def update_elo(self, stats, predictions, outcome, K=32):
        self._update_elo(stats, predictions, outcome, K)


def main():
    import sys
    data_dir = Path("/app/data")
    if len(sys.argv) > 1:
        data_dir = Path(sys.argv[1])
    scorer = Scorer(data_dir)
    scorer.run()


if __name__ == "__main__":
    main()
