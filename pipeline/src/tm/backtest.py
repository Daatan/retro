"""
Bediavad Backtest Engine
========================
Retroactive source audit: given the Factum Atlas, does the extracted per-source
signal predict resolved event outcomes out-of-sample? Produces the source
credibility leaderboard (Brier per source), not a market comparison — the
Polymarket head-to-head is a separate project (the Duel, duel_report.py).

Prediction window: 3 to 30 days before event resolution.
Output: per-event Brier scores and per-source contribution breakdown.

Usage:
    uv run python -m tm.backtest --events A01 A02 B01 --output data/backtest/
    uv run python -m tm.backtest --all-resolved --output data/backtest/
"""

import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich import box

from .utils import split_scored_predictions

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

from .config import settings

console = Console()

# --- Constants ---

MIN_DAYS_BEFORE_EVENT = 3
MAX_DAYS_BEFORE_EVENT = 30

DEFAULT_SOURCE_BRIER = 0.25  # random-baseline fallback for unranked sources


# --- Data loading ---

def load_event(event_id: str) -> Optional[dict]:
    """Load event metadata from data/events/{event_id}.json"""
    path = settings.events_dir / f"{event_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def load_atlas_entries(event_id: str) -> list[dict]:
    """
    Load all extraction entries for an event from the Factum Atlas.
    Path: data/atlas/{event_id}/{source_id}/entry_*.json
    Returns flat list of entries, each with source_id injected.
    """
    atlas_dir = settings.data_dir / "atlas" / event_id
    entries = []
    if not atlas_dir.exists():
        return entries
    for source_dir in atlas_dir.iterdir():
        if not source_dir.is_dir():
            continue
        source_id = source_dir.name
        for entry_file in source_dir.glob("entry_*.json"):
            data = json.loads(entry_file.read_text())
            data["source_id"] = source_id
            entries.append(data)
    return entries


def load_source_briers() -> dict[str, float]:
    """
    Load the per-source historical Brier scores from data/leaderboard.json.
    Returns {source_id: brier_score}. This is the engine's "who was right"
    signal — it feeds both the source_brier feature and the weighted-average
    fallback weighting. Empty dict if the leaderboard hasn't been built yet.
    """
    path = settings.data_dir / "leaderboard.json"
    if not path.exists():
        return {}
    rows = json.loads(path.read_text())
    return {row["id"]: float(row["brier_score"]) for row in rows if "brier_score" in row}


# --- Prediction filtering ---

def filter_by_window(entries: list[dict], event_date: str) -> list[dict]:
    """
    Keep only entries published between MIN and MAX days before the event.
    Requires entries to have 'article_date' field (ISO format).
    """
    event_dt = datetime.fromisoformat(event_date)
    min_dt = event_dt - timedelta(days=MAX_DAYS_BEFORE_EVENT)
    max_dt = event_dt - timedelta(days=MIN_DAYS_BEFORE_EVENT)

    filtered = []
    for entry in entries:
        article_date = entry.get("article_date")
        if not article_date:
            continue
        try:
            article_dt = datetime.fromisoformat(article_date)
            if min_dt <= article_dt <= max_dt:
                filtered.append(entry)
        except ValueError:
            continue
    return filtered


# --- Feature extraction ---

def entry_to_features(entry: dict, source_brier: float) -> dict:
    """
    Convert a single atlas entry to a flat feature dict for LightGBM.
    `source_brier` is the publishing source's historical Brier from the
    leaderboard (lower = more accurate); falls back to DEFAULT_SOURCE_BRIER.
    """
    preds = entry.get("predictions", [])
    if not preds:
        return {}

    # Loud validation: stance/certainty are required. Skip (never neutral-default)
    # any prediction missing them so a schema regression is visible, not silent.
    preds, malformed = split_scored_predictions(preds)
    if malformed:
        console.print(
            f"[yellow]⚠️  skipped {len(malformed)} prediction(s) missing numeric "
            f"stance/certainty in an entry — not scored as neutral[/yellow]"
        )
    if not preds:
        return {}

    # Aggregate across predictions in this article (mean)
    stance = np.mean([p["stance"] for p in preds])
    certainty = np.mean([p["certainty"] for p in preds])
    specificity = np.mean([p.get("specificity", 0.5) for p in preds])
    hedge_ratio = np.mean([p.get("hedge_ratio", 0.5) for p in preds])
    conditionality = np.mean([p.get("conditionality", 0) for p in preds])
    magnitude = np.mean([p.get("magnitude", 0.5) for p in preds])
    source_authority = np.mean([p.get("source_authority", 0.5) for p in preds])
    sentiment = np.mean([p.get("sentiment", 0.5) for p in preds])

    # Time-to-event in days
    article_date = entry.get("article_date", "")
    event_date = entry.get("event_date", "")
    days_before = None
    if article_date and event_date:
        try:
            days_before = (
                datetime.fromisoformat(event_date) - datetime.fromisoformat(article_date)
            ).days
        except ValueError:
            pass

    return {
        "stance": stance,
        "certainty": certainty,
        "specificity": specificity,
        "hedge_ratio": hedge_ratio,
        "conditionality": conditionality,
        "magnitude": magnitude,
        "source_authority": source_authority,
        "sentiment": sentiment,
        "days_before": days_before if days_before is not None else MAX_DAYS_BEFORE_EVENT,
        "source_brier": source_brier,
        "prediction_count": len(preds),
    }


# --- Model ---

def weighted_average_prediction(entries: list[dict], source_briers: dict[str, float]) -> float:
    """
    Fallback when not enough data for LightGBM.
    Weighted average of stance scores, weighted by source Brier accuracy.
    Maps stance (-1 to 1) to probability (0 to 1).
    `source_briers` maps source_id -> historical Brier from the leaderboard.
    """
    weighted_sum = 0.0
    weight_total = 0.0
    for entry in entries:
        preds = entry.get("predictions", [])
        if not preds:
            continue
        preds, _ = split_scored_predictions(preds)  # skip malformed; never neutral-default
        if not preds:
            continue
        stance = np.mean([p["stance"] for p in preds])
        source_id = entry.get("source_id", "")
        score = source_briers.get(source_id, DEFAULT_SOURCE_BRIER)
        # Convert Brier score to weight: lower Brier = more accurate = higher weight
        weight = max(0.01, 1.0 - score)
        weighted_sum += stance * weight
        weight_total += weight

    if weight_total == 0:
        return 0.5

    # Map from [-1, 1] to [0, 1]
    raw = weighted_sum / weight_total
    return (raw + 1) / 2


def train_and_predict_lgbm(
    train_events: list[dict],
    target_features: list[dict],
) -> list[float]:
    """
    Train LightGBM on resolved training events, predict on target features.
    Returns list of probabilities.
    """
    if not HAS_LGB:
        raise ImportError("lightgbm not installed. Run: uv add lightgbm")

    X_train, y_train = [], []
    first_feat = None
    for ev in train_events:
        for feat in ev.get("features", []):
            if feat:
                X_train.append(list(feat.values()))
                y_train.append(float(ev["outcome"]))
                if first_feat is None:
                    first_feat = feat

    if len(X_train) < 10:
        raise ValueError(f"Not enough training data: {len(X_train)} samples (need 10+)")

    feature_names = list(first_feat.keys())

    model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        num_leaves=15,
        min_child_samples=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )
    model.fit(X_train, y_train, feature_name=feature_names)

    X_target = [list(f.values()) for f in target_features if f]
    if not X_target:
        return []
    probs = model.predict_proba(X_target)[:, 1]
    return probs.tolist()


# --- Scoring ---

def brier_score(prediction: float, outcome: bool) -> float:
    """Lower is better. Perfect = 0.0, random = 0.25."""
    return (prediction - float(outcome)) ** 2


# --- Calibration ---

def _pava(values: list[float]) -> list[float]:
    """
    Pool Adjacent Violators: monotonic non-decreasing least-squares fit of
    `values` (already ordered by the predictor). Pure numpy/stdlib, no sklearn.
    """
    blocks: list[list[float]] = []  # each block is [mean, count]
    for v in values:
        blocks.append([v, 1.0])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            v2, c2 = blocks.pop()
            v1, c1 = blocks.pop()
            blocks.append([(v1 * c1 + v2 * c2) / (c1 + c2), c1 + c2])
    out: list[float] = []
    for mean, count in blocks:
        out.extend([mean] * int(count))
    return out


def fit_isotonic(predictions: list[float], outcomes: list[bool]):
    """
    Fit an isotonic calibration map prediction -> P(outcome) via PAV.
    Returns a transform(p) callable. Used to correct over/under-confidence in
    the raw model probabilities.

    Caveat: fitting and evaluating on the same predictions is in-sample for the
    calibrator — report it as an upper bound until enough events exist to hold
    out a calibration fold (and until the outcome set has both classes).
    """
    order = np.argsort(predictions)
    xs = np.asarray(predictions, dtype=float)[order]
    ys = np.asarray([float(o) for o in outcomes], dtype=float)[order]
    fitted = np.asarray(_pava(list(ys)), dtype=float)

    def transform(p: float) -> float:
        return float(np.interp(p, xs, fitted))

    return transform


# --- Report ---

def save_backtest_result(output_dir: Path, event_id: str, result: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{event_id}_backtest.json"
    out_path.write_text(json.dumps(result, indent=2))


def print_report(results: list[dict], calibrated: bool = False) -> None:
    console.rule("[bold teal]Bediavad Backtest Report[/bold teal]")

    # Summary table
    table = Table(
        title="Per-Event Results",
        box=box.ROUNDED,
        show_lines=True,
    )
    table.add_column("Event", style="white", width=10)
    table.add_column("Outcome", style="bold", width=8)
    table.add_column("Our P(YES)", style="cyan", width=12)
    table.add_column("Our Brier", style="cyan", width=12)
    if calibrated:
        table.add_column("Cal. P(YES)", style="green", width=12)
        table.add_column("Cal. Brier", style="green", width=12)

    our_briers = []
    cal_briers = []
    # Always-YES base rate: the share of positives — the bar any model must clear.
    outcomes = [r["outcome"] for r in results]
    base_rate = np.mean([float(o) for o in outcomes]) if outcomes else 0.0

    for r in results:
        outcome = r["outcome"]
        our_p = r.get("our_prediction")
        cal_p = r.get("calibrated_prediction")

        our_b = brier_score(our_p, outcome) if our_p is not None else None
        if our_b is not None:
            our_briers.append(our_b)

        row = [
            r["event_id"],
            "✅ YES" if outcome else "❌ NO",
            f"{our_p:.3f}" if our_p is not None else "N/A",
            f"{our_b:.4f}" if our_b is not None else "N/A",
        ]
        if calibrated:
            cal_b = brier_score(cal_p, outcome) if cal_p is not None else None
            if cal_b is not None:
                cal_briers.append(cal_b)
            row.append(f"{cal_p:.3f}" if cal_p is not None else "N/A")
            row.append(f"{cal_b:.4f}" if cal_b is not None else "N/A")
        table.add_row(*row)

    console.print(table)

    # Aggregate
    if our_briers:
        # Brier of the trivial always-YES-at-base-rate baseline, for context.
        baseline_brier = np.mean([(base_rate - float(o)) ** 2 for o in outcomes])
        console.print(f"\n[bold]Aggregate Brier Score[/bold]  (lower is better)")
        console.print(f"  Ours:           [cyan]{np.mean(our_briers):.4f}[/cyan]  ({len(our_briers)} events)")
        if cal_briers:
            console.print(f"  Ours (calib.):  [green]{np.mean(cal_briers):.4f}[/green]  (isotonic, in-sample)")
        console.print(
            f"  Base rate (always-YES @ {base_rate:.2f}): [yellow]{baseline_brier:.4f}[/yellow] "
            f"— [dim]any honest model must beat this[/dim]"
        )

    # Source breakdown
    source_contributions: dict[str, list[float]] = {}
    for r in results:
        for src, contrib in r.get("source_contributions", {}).items():
            source_contributions.setdefault(src, []).append(contrib)

    if source_contributions:
        src_table = Table(title="Source Contribution (avg stance weight)", box=box.SIMPLE)
        src_table.add_column("Source", style="white")
        src_table.add_column("Avg Contribution", style="cyan")
        src_table.add_column("Events", style="gray50")
        for src, contribs in sorted(
            source_contributions.items(), key=lambda x: abs(np.mean(x[1])), reverse=True
        ):
            src_table.add_row(src, f"{np.mean(contribs):+.3f}", str(len(contribs)))
        console.print(src_table)


# --- Main ---

def run_backtest(event_ids: list[str], output_dir: Path, use_lgbm: bool = True) -> None:
    results = []
    all_event_data = []
    source_briers = load_source_briers()

    console.print(f"[bold]Running backtest on {len(event_ids)} events[/bold]")
    console.print(f"Window: {MIN_DAYS_BEFORE_EVENT}–{MAX_DAYS_BEFORE_EVENT} days before event\n")

    for event_id in event_ids:
        event = load_event(event_id)
        if not event:
            console.print(f"[yellow]⚠ Event {event_id} not found, skipping[/yellow]")
            continue

        outcome = event.get("outcome")
        if outcome is None:
            console.print(f"[yellow]⚠ Event {event_id} has no outcome, skipping[/yellow]")
            continue

        event_date = event.get("outcome_date")
        # Events carry `category` as a list; the first is the primary domain.
        categories = event.get("category") or []
        domain = categories[0] if categories else "general"

        # Load and filter atlas entries
        entries = load_atlas_entries(event_id)
        entries = filter_by_window(entries, event_date)

        if not entries:
            console.print(f"[yellow]⚠ No entries in window for {event_id}[/yellow]")
            continue

        # Inject event_date into entries for feature extraction
        for e in entries:
            e["event_date"] = event_date

        # Build features per entry
        source_contributions = {}
        features = []
        for entry in entries:
            source_id = entry.get("source_id", "unknown")
            source_brier = source_briers.get(source_id, DEFAULT_SOURCE_BRIER)
            feat = entry_to_features(entry, source_brier)
            if feat:
                features.append(feat)
                source_contributions[source_id] = feat.get("stance", 0) * (1 - source_brier)

        all_event_data.append({
            "event_id": event_id,
            "outcome": bool(outcome),
            "domain": domain,
            "features": features,
        })

        results.append({
            "event_id": event_id,
            "outcome": bool(outcome),
            "domain": domain,
            "entry_count": len(entries),
            "source_contributions": source_contributions,
            "our_prediction": None,  # filled below
        })

    # --- Predict ---
    use_lgbm_run = use_lgbm and HAS_LGB and len(all_event_data) >= 5
    if use_lgbm_run:
        console.print("[cyan]Training LightGBM model (leave-one-out)...[/cyan]")
        for i, result in enumerate(results):
            event_id = result["event_id"]
            target_features = next(
                (e["features"] for e in all_event_data if e["event_id"] == event_id), []
            )
            if not target_features:
                continue

            # Leave-one-out: train on all other events
            train_data = [e for e in all_event_data if e["event_id"] != event_id]
            try:
                probs = train_and_predict_lgbm(train_data, target_features)
                result["our_prediction"] = float(np.mean(probs)) if probs else None
            except ValueError as e:
                console.print(f"[yellow]LightGBM skipped for {event_id}: {e}[/yellow]")
                # Fallback to weighted average
                entries = load_atlas_entries(event_id)
                entries = filter_by_window(entries, load_event(event_id)["outcome_date"])
                result["our_prediction"] = weighted_average_prediction(entries, source_briers)
    else:
        console.print("[yellow]Using weighted average (not enough data for LightGBM or not installed)[/yellow]")
        for result in results:
            event_id = result["event_id"]
            event = load_event(event_id)
            entries = load_atlas_entries(event_id)
            entries = filter_by_window(entries, event["outcome_date"])
            result["our_prediction"] = weighted_average_prediction(entries, source_briers)

    # Isotonic calibration of the raw predictions (in-sample upper bound).
    predicted = [r for r in results if r.get("our_prediction") is not None]
    calibrated = False
    if len(predicted) >= 5:
        transform = fit_isotonic(
            [r["our_prediction"] for r in predicted],
            [r["outcome"] for r in predicted],
        )
        for r in predicted:
            r["calibrated_prediction"] = transform(r["our_prediction"])
        calibrated = True

    # Save individual results
    for result in results:
        save_backtest_result(output_dir, result["event_id"], result)

    # Print report
    print_report(results, calibrated=calibrated)

    # Save summary
    summary = {
        "run_at": datetime.utcnow().isoformat(),
        "events": len(results),
        "window_days": [MIN_DAYS_BEFORE_EVENT, MAX_DAYS_BEFORE_EVENT],
        "model": "lightgbm_loo" if use_lgbm_run else "weighted_average",
        "calibration": "isotonic_in_sample" if calibrated else "none",
        "results": results,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    console.print(f"\n[green]Results saved to {output_dir}[/green]")


def main():
    parser = argparse.ArgumentParser(description="Bediavad Backtest Engine")
    parser.add_argument("--events", nargs="+", help="Event IDs to backtest (e.g. A01 A02 B01)")
    parser.add_argument("--all-resolved", action="store_true", help="Use all resolved events from atlas")
    parser.add_argument("--output", default="data/backtest", help="Output directory")
    parser.add_argument("--no-lgbm", action="store_true", help="Force weighted average instead of LightGBM")
    args = parser.parse_args()

    output_dir = Path(args.output)

    if args.all_resolved:
        atlas_dir = settings.data_dir / "atlas"
        event_ids = [d.name for d in atlas_dir.iterdir() if d.is_dir()] if atlas_dir.exists() else []
    elif args.events:
        event_ids = args.events
    else:
        parser.error("Provide --events or --all-resolved")
        return

    run_backtest(event_ids, output_dir, use_lgbm=not args.no_lgbm)


if __name__ == "__main__":
    main()
