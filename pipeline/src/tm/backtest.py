"""
Bediavad Backtest Engine
========================
Compares TruthMachine oracle predictions against Polymarket prices
on resolved events from the Factum Atlas.

Prediction window: 3 to 30 days before event resolution.
Output: Brier scores, per-source breakdown, beat/loss report vs Polymarket.

Usage:
    uv run python -m tm.backtest --events A01 A02 B01 --output data/backtest/
    uv run python -m tm.backtest --all-resolved --output data/backtest/
"""

import json
import argparse
import httpx
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

POLYMARKET_API = "https://gamma-api.polymarket.com"


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


def load_source_scores(source_id: str) -> dict:
    """
    Load historical Brier scores for a source from data/sources/{source_id}.json
    Returns dict with domain -> brier_score mapping, or empty dict.
    """
    path = settings.sources_dir / f"{source_id}.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return data.get("brier_scores", {})


# --- Polymarket integration ---

def fetch_polymarket_price(event_title: str, event_date: str) -> Optional[float]:
    """
    Search Polymarket for a matching market and return the YES price
    closest to MAX_DAYS_BEFORE_EVENT days before the event.
    Returns None if no match found.
    """
    try:
        resp = httpx.get(
            f"{POLYMARKET_API}/markets",
            params={"q": event_title, "limit": 5},
            timeout=10,
        )
        resp.raise_for_status()
        markets = resp.json()
        if not markets:
            return None

        # Take best match (first result)
        market = markets[0]
        market_slug = market.get("slug") or market.get("id")
        if not market_slug:
            return None

        # Fetch time series prices
        target_date = datetime.fromisoformat(event_date) - timedelta(days=MAX_DAYS_BEFORE_EVENT)
        prices_resp = httpx.get(
            f"{POLYMARKET_API}/markets/{market_slug}/prices-history",
            params={"interval": "1d"},
            timeout=10,
        )
        prices_resp.raise_for_status()
        prices = prices_resp.json()

        if not prices:
            return None

        # Find price closest to target_date
        best_price = None
        best_delta = float("inf")
        for point in prices:
            ts = datetime.fromisoformat(point["t"].replace("Z", ""))
            delta = abs((ts - target_date).total_seconds())
            if delta < best_delta:
                best_delta = delta
                best_price = float(point["p"])

        return best_price

    except Exception as e:
        console.print(f"[yellow]Polymarket fetch failed for '{event_title}': {e}[/yellow]")
        return None


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

def entry_to_features(entry: dict, source_scores: dict, domain: str) -> dict:
    """
    Convert a single atlas entry to a flat feature dict for LightGBM.
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
            f"stance/certainty in a {domain} entry — not scored as neutral[/yellow]"
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

    # Source historical accuracy for this domain
    source_brier = source_scores.get(domain, source_scores.get("overall", 0.25))

    return {
        "stance": stance,
        "certainty": certainty,
        "specificity": specificity,
        "hedge_ratio": hedge_ratio,
        "conditionality": conditionality,
        "magnitude": magnitude,
        "source_authority": source_authority,
        "sentiment": sentiment,
        "days_before": days_before or MAX_DAYS_BEFORE_EVENT,
        "source_brier": source_brier,
        "prediction_count": len(preds),
    }


# --- Model ---

def weighted_average_prediction(entries: list[dict], source_scores: dict, domain: str) -> float:
    """
    Fallback when not enough data for LightGBM.
    Weighted average of stance scores, weighted by source Brier accuracy.
    Maps stance (-1 to 1) to probability (0 to 1).
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
        score = source_scores.get(source_id, {}).get(domain, 0.25)
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
    for ev in train_events:
        for feat in ev.get("features", []):
            if feat:
                X_train.append(list(feat.values()))
                y_train.append(float(ev["outcome"]))

    if len(X_train) < 10:
        raise ValueError(f"Not enough training data: {len(X_train)} samples (need 10+)")

    feature_names = list(train_events[0]["features"][0].keys())

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


# --- Report ---

def save_backtest_result(output_dir: Path, event_id: str, result: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{event_id}_backtest.json"
    out_path.write_text(json.dumps(result, indent=2))


def print_report(results: list[dict]) -> None:
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
    table.add_column("Poly P(YES)", style="magenta", width=12)
    table.add_column("Poly Brier", style="magenta", width=12)
    table.add_column("Winner", width=10)

    our_briers, poly_briers = [], []

    for r in results:
        outcome = r["outcome"]
        our_p = r.get("our_prediction")
        poly_p = r.get("polymarket_price")

        our_b = brier_score(our_p, outcome) if our_p is not None else None
        poly_b = brier_score(poly_p, outcome) if poly_p is not None else None

        if our_b is not None:
            our_briers.append(our_b)
        if poly_b is not None:
            poly_briers.append(poly_b)

        winner = "—"
        if our_b is not None and poly_b is not None:
            if our_b < poly_b - 0.01:
                winner = "[green]US[/green]"
            elif poly_b < our_b - 0.01:
                winner = "[red]POLY[/red]"
            else:
                winner = "[yellow]TIE[/yellow]"

        table.add_row(
            r["event_id"],
            "✅ YES" if outcome else "❌ NO",
            f"{our_p:.3f}" if our_p is not None else "N/A",
            f"{our_b:.4f}" if our_b is not None else "N/A",
            f"{poly_p:.3f}" if poly_p is not None else "N/A",
            f"{poly_b:.4f}" if poly_b is not None else "N/A",
            winner,
        )

    console.print(table)

    # Aggregate
    if our_briers and poly_briers:
        avg_our = np.mean(our_briers)
        avg_poly = np.mean(poly_briers)
        beats = sum(1 for o, p in zip(our_briers, poly_briers) if o < p - 0.01)
        loses = sum(1 for o, p in zip(our_briers, poly_briers) if p < o - 0.01)
        ties = len(our_briers) - beats - loses

        console.print(f"\n[bold]Aggregate Brier Score[/bold]")
        console.print(f"  Ours:       [cyan]{avg_our:.4f}[/cyan]")
        console.print(f"  Polymarket: [magenta]{avg_poly:.4f}[/magenta]")
        console.print(f"  Beat Poly:  [green]{beats}/{len(our_briers)}[/green] events")
        console.print(f"  Lost:       [red]{loses}/{len(our_briers)}[/red] events")
        console.print(f"  Tied:       [yellow]{ties}/{len(our_briers)}[/yellow] events")

        if avg_our < avg_poly:
            console.print(f"\n[bold green]✅ We outperform Polymarket overall[/bold green]")
        else:
            console.print(f"\n[bold red]❌ Polymarket outperforms us overall[/bold red]")

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
        domain = event.get("domain", "general")

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
            src_scores = load_source_scores(source_id)
            feat = entry_to_features(entry, src_scores, domain)
            if feat:
                features.append(feat)
                source_contributions[source_id] = feat.get("stance", 0) * (1 - feat.get("source_brier", 0.25))

        all_event_data.append({
            "event_id": event_id,
            "outcome": bool(outcome),
            "domain": domain,
            "features": features,
        })

        # Polymarket comparison
        poly_price = fetch_polymarket_price(event.get("name", event_id), event_date)

        results.append({
            "event_id": event_id,
            "outcome": bool(outcome),
            "domain": domain,
            "entry_count": len(entries),
            "polymarket_price": poly_price,
            "source_contributions": source_contributions,
            "our_prediction": None,  # filled below
        })

    # --- Predict ---
    if use_lgbm and HAS_LGB and len(all_event_data) >= 5:
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
                src_scores = {}
                result["our_prediction"] = weighted_average_prediction(entries, src_scores, result["domain"])
    else:
        console.print("[yellow]Using weighted average (not enough data for LightGBM or not installed)[/yellow]")
        for result in results:
            event_id = result["event_id"]
            event = load_event(event_id)
            entries = load_atlas_entries(event_id)
            entries = filter_by_window(entries, event["outcome_date"])
            src_scores = {e.get("source_id", ""): load_source_scores(e.get("source_id", "")) for e in entries}
            result["our_prediction"] = weighted_average_prediction(entries, src_scores, result["domain"])

    # Save individual results
    for result in results:
        save_backtest_result(output_dir, result["event_id"], result)

    # Print report
    print_report(results)

    # Save summary
    summary = {
        "run_at": datetime.utcnow().isoformat(),
        "events": len(results),
        "window_days": [MIN_DAYS_BEFORE_EVENT, MAX_DAYS_BEFORE_EVENT],
        "model": "lightgbm_loo" if (use_lgbm and HAS_LGB) else "weighted_average",
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
