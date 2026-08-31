"""
Duel Report — TruthMachine vs Polymarket on 70-event main dataset.

For each event that has both a Polymarket price cache and oracle_question,
computes temporally-valid Brier scores for both sides and renders duel.html.

Comparison protocol:
  T = 7 days before outcome_date
  PM probability  : last available price on or before T (inverted if ev["polymarket"]["invert"])
  TM probability  : Oracul API forecast using articles published <= T, converted (stance+1)/2

TM predictions are fetched live from oracle.daatan.com (or ORACLE_URL) and
cached in data/duel_oracle/{event_id}.json.  Set ORACLE_API_KEY, or rely on
the SSM fallback (/daatan/shared/secrets/ORACLE_API_KEY — the same shared
parameter daatan's app reads) before running.

Usage:
    ORACLE_API_KEY=sk-... DATA_DIR=data python -m tm.duel_report
    ORACLE_API_KEY=sk-... DATA_DIR=data python -m tm.duel_report --out duel.html --t-days 7
    DATA_DIR=data python -m tm.duel_report --html-only  # re-render from cache, no API calls
"""

import argparse
import html
import json
import math
import os
import random
import re
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import httpx
from rich.console import Console

from .models import is_negative_marker
from .scorer import brier_decomposition, brier_score, stance_to_prob
# Reuse the evergreen-domain helper rather than re-implement it (retro#439).
from .web_search_ingest import _is_evergreen_domain

console = Console()

T_DAYS_DEFAULT = 7


# ── Data loading ───────────────────────────────────────────────────────────────

def load_events_with_pm(data_dir: Path) -> list[dict]:
    """Return main events that have a polymarket URL and a non-empty price cache."""
    events = []
    pm_dir = data_dir / "polymarket"
    for f in sorted((data_dir / "events").glob("*.json")):
        ev = json.loads(f.read_text())
        pm_meta = ev.get("polymarket") or {}
        if not pm_meta.get("url"):
            continue
        cache = pm_dir / f"{ev['id']}.json"
        if not cache.exists():
            continue
        pm_data = json.loads(cache.read_text())
        if not pm_data.get("prices"):
            continue
        ev["_pm"] = pm_data
        events.append(ev)
    return events


def load_all_events(data_dir: Path) -> list[dict]:
    """Return all events, attaching _pm data where a price cache exists."""
    pm_dir = data_dir / "polymarket"
    events = []
    for f in sorted((data_dir / "events").glob("*.json")):
        ev = json.loads(f.read_text())
        cache = pm_dir / f"{ev['id']}.json"
        if cache.exists():
            pm_data = json.loads(cache.read_text())
            if pm_data.get("prices"):
                ev["_pm"] = pm_data
        events.append(ev)
    return events


def pm_probability(ev: dict, t_days: int) -> Optional[float]:
    """
    PM probability of our event happening, measured at T = outcome_date - t_days.
    Applies polarity inversion if ev["polymarket"]["invert"] is True.
    Falls back to earliest available price if no price exists before T.
    """
    prices = ev["_pm"].get("prices", [])
    if not prices:
        return None
    outcome_dt = datetime.strptime(ev["outcome_date"], "%Y-%m-%d").date()
    cutoff = outcome_dt - timedelta(days=t_days)
    candidates = [p for p in prices if datetime.strptime(p["date"], "%Y-%m-%d").date() <= cutoff]
    if not candidates:
        candidates = prices[:1]  # use earliest available if nothing before cutoff
    raw_prob = candidates[-1]["probability"]
    if (ev.get("polymarket") or {}).get("invert"):
        raw_prob = 1.0 - raw_prob
    return round(raw_prob, 4)


def _load_vault2_articles(data_dir: Path, eid: str, cutoff_str: str) -> list[dict]:
    """
    Return vault2 articles for event ``eid`` published on or before ``cutoff_str``.
    Each dict has url, title, text, published_date — ready for Oracul ArticleInput.
    """
    from urllib.parse import urlparse

    extractions_dir = data_dir / "vault2" / "extractions"
    articles_dir = data_dir / "vault2" / "articles"
    cutoff_dt = datetime.strptime(cutoff_str, "%Y-%m-%d").date()

    seen_hashes: set[str] = set()
    articles = []
    n_total = 0
    n_date_ok = 0
    for path in extractions_dir.glob(f"*_{eid}_v*.json"):
        article_hash = path.stem.split("_")[0]
        if article_hash in seen_hashes:
            continue
        # Negative markers (docs#57 item 2) record gate-rejected / no-prediction
        # articles — those must not be fed to the Oracul as forecast input.
        try:
            if is_negative_marker(json.loads(path.read_text())):
                continue
        except Exception:
            continue
        seen_hashes.add(article_hash)
        n_total += 1
        art_path = articles_dir / f"{article_hash}.json"
        if not art_path.exists():
            console.print(f"  [yellow]Warning: article file missing for hash {article_hash} (event {eid})[/yellow]")
            continue
        art = json.loads(art_path.read_text())
        pub = datetime.strptime(art["published_at"], "%Y-%m-%d").date()
        if pub > cutoff_dt:
            continue
        if art.get("estimated_date"):
            # estimated_date=True means we couldn't recover a real publish date;
            # don't trust it for temporal validity.
            console.print(f"    [dim yellow]{eid}: skipping estimated-date article {art['published_at']} — {art.get('url', '')[:70]}[/dim yellow]")
            continue
        url = art.get("url", "")
        domain = re.sub(r"^www\.", "", urlparse(url).netloc).lower()
        if _is_evergreen_domain(url):
            console.print(f"    [dim yellow]{eid}: skipping evergreen source {domain} — {art.get('headline','')[:60]}[/dim yellow]")
            continue
        n_date_ok += 1
        articles.append({
            "url": url,
            "title": art.get("headline", ""),
            "snippet": "",
            "source": domain,
            "published_date": art["published_at"],
            "text": art.get("text", ""),
        })
    console.print(f"  [dim]{eid}: {n_total} extractions, {n_date_ok} before {cutoff_str}, {len(articles)} returned[/dim]")
    return articles


def fetch_tm_probabilities_oracle(data_dir: Path, events: list[dict], t_days: int) -> dict[str, dict]:
    """
    For each event with oracle_question, run Oracul's full pipeline on vault2 articles.

    Flow per event:
      1. Gather vault2 articles published on or before outcome_date - t_days
      2. POST /forecast with question + articles (text= pre-fetched → skips trafilatura)
    Results cached in data/duel_oracle/{event_id}.json keyed on cutoff_date.

    Returns {event_id: {"probability": float, "articles_used": int, "mean_stance": float, ...}}
    """
    oracle_url = os.environ.get("ORACLE_URL", "https://oracle.daatan.com").rstrip("/")
    api_key = os.environ.get("ORACLE_API_KEY", "")
    if not api_key:
        try:
            import boto3
            ssm = boto3.client("ssm", region_name="eu-central-1")
            api_key = ssm.get_parameter(
                Name="/daatan/shared/secrets/ORACLE_API_KEY", WithDecryption=True
            )["Parameter"]["Value"].strip()
            console.print("[dim]Oracul API key loaded from SSM Parameter Store[/dim]")
        except Exception as e:
            console.print(f"[red]ORACLE_API_KEY not set and SSM fetch failed: {e}[/red]")

    if not api_key:
        console.print("[red bold]No Oracle API key — cannot fetch TM predictions. Set ORACLE_API_KEY.[/red bold]")
        return {}

    cache_dir = data_dir / "duel_oracle"
    cache_dir.mkdir(exist_ok=True)

    headers = {"x-api-key": api_key, "Content-Type": "application/json"}
    result: dict[str, dict] = {}
    forecast_calls = 0  # pace /forecast calls under 10/min

    for ev in events:
        question = ev.get("oracle_question")
        if not question:
            console.print(f"  [yellow]{ev['id']}: no oracle_question — skipped[/yellow]")
            continue

        eid = ev["id"]
        outcome_dt = datetime.strptime(ev["outcome_date"], "%Y-%m-%d").date()
        cutoff = outcome_dt - timedelta(days=t_days)
        cutoff_str = cutoff.strftime("%Y-%m-%d")

        # Serve from cache when cutoff and t_days match
        cache_path = cache_dir / f"{eid}_T{t_days}.json"
        if not cache_path.exists():
            # backward-compat: old single-file cache used {eid}.json
            old_cache = cache_dir / f"{eid}.json"
            if old_cache.exists():
                old_content = json.loads(old_cache.read_text())
                if old_content.get("t_days") == t_days:
                    cache_path = old_cache
        if cache_path.exists():
            cached = json.loads(cache_path.read_text())
            cache_valid = (
                cached.get("cutoff_date") == cutoff_str
                and cached.get("t_days") == t_days
                and (cached.get("probability") is not None or cached.get("placeholder"))
            )
            if cache_valid:
                result[eid] = cached
                label = "placeholder" if cached.get("placeholder") else f"p={cached['probability']} (n={cached.get('articles_used', '?')})"
                console.print(f"  [dim]{eid}: cache hit → {label}[/dim]")
                continue

        # Gather vault2 articles pre-dating the cutoff
        articles = _load_vault2_articles(data_dir, eid, cutoff_str)
        console.print(f"  [cyan]{eid}: {len(articles)} vault2 articles before {cutoff_str}[/cyan]")
        if not articles:
            console.print(f"  [yellow]{eid}: no vault2 articles before cutoff — skipped[/yellow]")
            continue

        # Rate-limit: /forecast is 10/min, space calls ≥7s apart
        if forecast_calls > 0:
            time.sleep(7)

        payload = {
            "question": question,
            "articles": [
                {
                    "url": a["url"],
                    "title": a["title"],
                    "snippet": a["snippet"],
                    "source": a["source"],
                    "published_date": a["published_date"],
                    "text": a["text"] or None,
                }
                for a in articles
            ],
        }
        forecast = None
        for attempt in range(3):
            if attempt > 0:
                backoff = 15 * (2 ** (attempt - 1))
                console.print(f"  [dim]Oracul retry {attempt}/2 — backoff {backoff}s[/dim]")
                time.sleep(backoff)
            try:
                fr = httpx.post(
                    f"{oracle_url}/forecast",
                    json=payload,
                    headers=headers,
                    timeout=120,
                )
                forecast_calls += 1
                if fr.status_code in (429, 500, 502, 503):
                    console.print(f"  [dim yellow]{eid}: /forecast {fr.status_code} — will retry[/dim yellow]")
                    continue
                if fr.status_code != 200:
                    console.print(f"  [red]{eid}: /forecast {fr.status_code} — {fr.text[:120]}[/red]")
                    break
                forecast = fr.json()
                break
            except Exception as e:
                console.print(f"  [dim yellow]{eid}: /forecast error: {e}[/dim yellow]")
        if forecast is None:
            continue

        is_placeholder = bool(forecast.get("placeholder"))
        if is_placeholder:
            console.print(f"  [yellow]{eid}: Oracul returned placeholder (gatekeeper rejected all articles)[/yellow]")

        mean = forecast.get("mean", 0.0) if not is_placeholder else None
        articles_used = forecast.get("articles_used", 0)
        probability = round(stance_to_prob(mean), 4) if mean is not None else None

        entry = {
            "event_id": eid,
            "probability": probability,
            "mean_stance": round(mean, 4) if mean is not None else None,
            "articles_used": articles_used,
            "question": question,
            "cutoff_date": cutoff_str,
            "t_days": t_days,
            "placeholder": is_placeholder,
        }
        write_path = cache_dir / f"{eid}_T{t_days}.json"
        write_path.write_text(json.dumps(entry, indent=2))
        result[eid] = entry
        console.print(f"  [green]{eid}: p={probability} (n={articles_used} Oracul articles)[/green]")

    return result


# ── Build comparison rows ──────────────────────────────────────────────────────

def build_rows(events: list[dict], tm_probs: dict, t_days: int) -> list[dict]:
    rows = []
    for ev in events:
        eid = ev["id"]
        pm_p = pm_probability(ev, t_days)
        tm_info = tm_probs.get(eid)
        tm_p = tm_info["probability"] if tm_info else None
        outcome = bool(ev.get("outcome", False))

        pm_brier = round(brier_score(pm_p, outcome), 4) if pm_p is not None else None
        tm_brier = round(brier_score(tm_p, outcome), 4) if tm_p is not None else None

        pm_meta = ev.get("polymarket") or {}
        tm_placeholder = bool(tm_info.get("placeholder")) if tm_info else False

        # Compute actual days-before-event that PM's snapshot price came from
        pm_price_days: Optional[int] = None
        outcome_dt_d = datetime.strptime(ev["outcome_date"], "%Y-%m-%d").date()
        prices = ev["_pm"].get("prices", [])
        cutoff_d = outcome_dt_d - timedelta(days=t_days)
        candidates = [p for p in prices if datetime.strptime(p["date"], "%Y-%m-%d").date() <= cutoff_d]
        if not candidates:
            candidates = prices[:1]
        if candidates:
            pm_actual_dt = datetime.strptime(candidates[-1]["date"], "%Y-%m-%d").date()
            pm_price_days = (outcome_dt_d - pm_actual_dt).days

        rows.append({
            "id": eid,
            "name": ev.get("name", ""),
            "outcome": outcome,
            "outcome_date": ev.get("outcome_date", ""),
            "category": ev.get("category", ""),
            "pm_question": ev["_pm"].get("question", ""),
            "pm_url": pm_meta.get("url", ""),
            "pm_invert": pm_meta.get("invert", False),
            "pm_p": pm_p,
            "pm_price_days": pm_price_days,
            "tm_p": tm_p,
            "tm_placeholder": tm_placeholder,
            "tm_articles_used": tm_info["articles_used"] if tm_info else 0,
            "tm_mean_stance": tm_info["mean_stance"] if tm_info else None,
            "pm_brier": pm_brier,
            "tm_brier": tm_brier,
            "pm_prices": ev["_pm"].get("prices", []),
            "winner": (
                "tm" if (tm_brier is not None and pm_brier is not None and tm_brier < pm_brier)
                else "pm" if (tm_brier is not None and pm_brier is not None and pm_brier < tm_brier)
                else "tie" if (tm_brier is not None and pm_brier is not None)
                else None
            ),
        })
    return rows


# ── Statistics ─────────────────────────────────────────────────────────────────

def avg_brier(rows: list[dict], key: str) -> Optional[float]:
    vals = [r[key] for r in rows if r[key] is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def bootstrap_ci(
    vals: list[float], n_boot: int = 10_000, seed: int = 42
) -> tuple[Optional[float], Optional[float]]:
    """Return (lo, hi) 95% bootstrap CI for the mean of vals."""
    if len(vals) < 2:
        return None, None
    rng = random.Random(seed)
    n = len(vals)
    boot_means = sorted(sum(rng.choices(vals, k=n)) / n for _ in range(n_boot))
    return round(boot_means[int(0.025 * n_boot)], 4), round(boot_means[int(0.975 * n_boot)], 4)


# ── Horizon sweep ──────────────────────────────────────────────────────────────

def run_sweep(
    data_dir: Path,
    events: list[dict],
    t_values: list[int],
    html_only: bool,
) -> dict[int, dict]:
    """Run the duel at multiple T-values; averages restricted to the common event set."""
    sweep: dict[int, dict] = {}
    for t in sorted(t_values):
        console.print(f"\n[bold cyan]── Sweep T={t}d ──[/bold cyan]")
        if html_only:
            cache_dir = data_dir / "duel_oracle"
            tm_probs: dict[str, dict] = {}
            for ev in events:
                eid = ev["id"]
                cp = cache_dir / f"{eid}_T{t}.json"
                if not cp.exists() and t == 7:
                    cp = cache_dir / f"{eid}.json"  # backward compat
                if cp.exists():
                    cached = json.loads(cp.read_text())
                    if cached.get("t_days") == t and (
                        cached.get("probability") is not None or cached.get("placeholder")
                    ):
                        tm_probs[eid] = cached
            console.print(f"  Loaded {len(tm_probs)} events from cache (--html-only)")
        else:
            tm_probs = fetch_tm_probabilities_oracle(data_dir, events, t)

        rows = build_rows(events, tm_probs, t)
        compared = [r for r in rows if r["pm_p"] is not None and r["tm_p"] is not None]
        sweep[t] = {
            "rows": rows,
            "compared": compared,
            "eids": {r["id"] for r in compared},
        }

    # Restrict summary averages to events present at every T
    common_eids: set[str] = set()
    for s in sweep.values():
        if not common_eids:
            common_eids = s["eids"].copy()
        else:
            common_eids &= s["eids"]
    console.print(f"\n[dim]Common events across all T values: {sorted(common_eids)}[/dim]")

    for t, s in sweep.items():
        common = [r for r in s["compared"] if r["id"] in common_eids]
        s["common"] = common
        s["tm_avg"] = avg_brier(common, "tm_brier")
        s["pm_avg"] = avg_brier(common, "pm_brier")
        s["tm_wins"] = sum(1 for r in common if r["winner"] == "tm")
        s["pm_wins"] = sum(1 for r in common if r["winner"] == "pm")
        s["n_common"] = len(common)
        console.print(
            f"  T={t}: n_common={len(common)}, TM={s['tm_avg']}, PM={s['pm_avg']}, "
            f"TM wins {s['tm_wins']}/{len(common)}"
        )

    return sweep


# ── HTML rendering ─────────────────────────────────────────────────────────────

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #0f1117; color: #e2e8f0; min-height: 100vh; }
.container { max-width: 1100px; margin: 0 auto; padding: 2rem 1rem; }

/* Header scorecard */
.scorecard { display: flex; gap: 2rem; justify-content: center; margin: 2rem 0; flex-wrap: wrap; }
.scorecard-side { flex: 1; min-width: 200px; max-width: 280px; background: #1e2130;
                  border-radius: 12px; padding: 1.5rem; text-align: center; }
.scorecard-side h2 { font-size: 1rem; color: #94a3b8; margin-bottom: 0.5rem; }
.scorecard-side .brier { font-size: 2.5rem; font-weight: 700; }
.scorecard-side .label { font-size: 0.8rem; color: #64748b; margin-top: 0.25rem; }
.winner-badge { background: #facc15; color: #0f1117; border-radius: 8px;
                padding: 0.3rem 1rem; font-weight: 700; font-size: 0.85rem;
                display: inline-block; margin-top: 0.75rem; }
.tm-color { color: #60a5fa; }
.pm-color { color: #fb923c; }

.vs-divider { display: flex; align-items: center; font-size: 1.5rem;
              color: #475569; font-weight: 700; padding-top: 1rem; }

/* Section */
h1 { font-size: 1.8rem; font-weight: 700; margin-bottom: 0.5rem; }
.subtitle { color: #64748b; font-size: 0.9rem; margin-bottom: 2rem; }
h2 { font-size: 1.2rem; font-weight: 600; margin: 2.5rem 0 1rem; color: #cbd5e1; border-bottom: 1px solid #2d3748; padding-bottom: 0.5rem; }
.note { background: #1e2130; border-left: 3px solid #475569; padding: 0.75rem 1rem;
        font-size: 0.82rem; color: #94a3b8; margin-bottom: 1.5rem; border-radius: 0 6px 6px 0; }

/* Per-event cards */
.events-grid { display: flex; flex-direction: column; gap: 1rem; }
.event-card { background: #1e2130; border-radius: 10px; padding: 1rem 1.25rem;
              border-left: 4px solid #2d3748; }
.event-card.tm-wins { border-left-color: #3b82f6; }
.event-card.pm-wins { border-left-color: #f97316; }
.event-card.tie     { border-left-color: #64748b; }
.event-header { display: flex; justify-content: space-between; align-items: flex-start; gap: 1rem; }
.event-name { font-weight: 600; font-size: 0.95rem; }
.event-meta { font-size: 0.75rem; color: #64748b; margin-top: 0.2rem; }
.event-pm-q { font-size: 0.75rem; color: #475569; margin-top: 0.15rem; font-style: italic; }
.event-pm-q .inv { color: #94a3b8; }
.badge { font-size: 0.7rem; font-weight: 700; padding: 0.2rem 0.6rem; border-radius: 999px;
         white-space: nowrap; }
.badge-tm { background: #1e3a5f; color: #60a5fa; }
.badge-pm { background: #431407; color: #fb923c; }
.badge-tie { background: #1e293b; color: #94a3b8; }

.event-body { display: flex; gap: 2rem; margin-top: 0.85rem; flex-wrap: wrap; }
.prob-block { flex: 1; min-width: 120px; }
.prob-label { font-size: 0.7rem; color: #64748b; margin-bottom: 0.2rem; }
.prob-bar-wrap { background: #0f1117; border-radius: 4px; height: 8px; margin: 4px 0; overflow: hidden; }
.prob-bar { height: 100%; border-radius: 4px; }
.prob-bar.tm { background: #3b82f6; }
.prob-bar.pm { background: #f97316; }
.prob-val { font-size: 0.85rem; font-weight: 600; }
.brier-val { font-size: 0.75rem; color: #64748b; }
.sparkline-wrap { flex: 2; min-width: 180px; }

/* Bar chart */
.bar-chart { display: flex; flex-direction: column; gap: 0.5rem; }
.bar-row { display: flex; align-items: center; gap: 0.75rem; font-size: 0.78rem; }
.bar-label { width: 120px; text-align: right; color: #94a3b8; white-space: nowrap;
             overflow: hidden; text-overflow: ellipsis; flex-shrink: 0; }
.bar-pair { flex: 1; display: flex; flex-direction: column; gap: 2px; }
.bar-seg { height: 12px; border-radius: 2px; min-width: 2px; display: inline-block; }
.bar-seg.tm { background: #3b82f6; }
.bar-seg.pm { background: #f97316; }
.bar-num { width: 42px; font-size: 0.72rem; color: #64748b; }

/* Scatter */
#scatter-wrap { background: #1e2130; border-radius: 10px; padding: 1rem; }

/* Table */
table { width: 100%; border-collapse: collapse; font-size: 0.82rem; margin-top: 1rem; }
th { text-align: left; color: #64748b; font-weight: 600; padding: 0.5rem 0.75rem;
     border-bottom: 1px solid #2d3748; }
td { padding: 0.5rem 0.75rem; border-bottom: 1px solid #1a2030; vertical-align: middle; }
tr:hover td { background: #242840; }
.outcome-yes { color: #4ade80; font-weight: 600; }
.outcome-no  { color: #f87171; font-weight: 600; }
.better { color: #4ade80; font-weight: 600; }
.worse  { color: #f87171; }
.no-data { color: #475569; font-style: italic; }

footer { margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #2d3748;
         font-size: 0.75rem; color: #475569; text-align: center; }

/* Coverage table */
.cov-both { color: #4ade80; }
.cov-pm   { color: #fb923c; }
.cov-none { color: #475569; }
.cov-dot  { font-size: 1rem; }
"""

_SPARKLINE_JS = """
function drawSparkline(canvas, prices, outcomeDate, tmPoints, invertedPm) {
  if (!canvas || !prices || prices.length === 0) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  const parseDate = s => new Date(s + 'T00:00:00Z');
  const priceMs = prices.map(p => parseDate(p.date).getTime());
  const minMs = priceMs[0];
  const maxMs = outcomeDate ? parseDate(outcomeDate).getTime() : priceMs[priceMs.length - 1];
  const rangeMs = (maxMs - minMs) || 1;

  function xForMs(ms) { return 10 + (ms - minMs) / rangeMs * (W - 20); }
  function yFor(v) { return H - 8 - v * (H - 16); }

  // Grid lines
  ctx.strokeStyle = '#2d3748';
  ctx.lineWidth = 0.5;
  [0.25, 0.5, 0.75].forEach(g => {
    ctx.beginPath();
    ctx.moveTo(0, yFor(g));
    ctx.lineTo(W, yFor(g));
    ctx.stroke();
  });

  // PM price line
  const vals = prices.map(p => invertedPm ? 1 - p.probability : p.probability);
  ctx.beginPath();
  ctx.strokeStyle = '#f97316';
  ctx.lineWidth = 1.5;
  prices.forEach((p, i) => {
    const x = xForMs(priceMs[i]);
    if (i === 0) ctx.moveTo(x, yFor(vals[i]));
    else ctx.lineTo(x, yFor(vals[i]));
  });
  ctx.stroke();

  // TM points: [{t: days_before_outcome, p: prob}]
  if (tmPoints && tmPoints.length > 0) {
    const tms = tmPoints
      .map(pt => ({ x: xForMs(maxMs - pt.t * 86400000), y: yFor(pt.p) }))
      .filter(pt => pt.x >= 8 && pt.x <= W - 8);

    if (tms.length > 0) {
      ctx.strokeStyle = '#3b82f6';
      ctx.lineWidth = 1.5;

      if (tms.length === 1) {
        // Single measurement: dashed horizontal line
        ctx.setLineDash([4, 3]);
        ctx.beginPath();
        ctx.moveTo(10, tms[0].y);
        ctx.lineTo(W - 10, tms[0].y);
        ctx.stroke();
        ctx.setLineDash([]);
      } else {
        // Multi-horizon: connected line through dated points
        ctx.beginPath();
        tms.forEach((pt, i) => {
          if (i === 0) ctx.moveTo(pt.x, pt.y);
          else ctx.lineTo(pt.x, pt.y);
        });
        ctx.stroke();
        ctx.fillStyle = '#3b82f6';
        tms.forEach(pt => {
          ctx.beginPath();
          ctx.arc(pt.x, pt.y, 2.5, 0, 2 * Math.PI);
          ctx.fill();
        });
      }
    }
  }

  // Outcome date marker (right edge)
  const xEnd = xForMs(maxMs);
  ctx.beginPath();
  ctx.strokeStyle = '#475569';
  ctx.lineWidth = 1;
  ctx.setLineDash([2, 2]);
  ctx.moveTo(xEnd, 0);
  ctx.lineTo(xEnd, H);
  ctx.stroke();
  ctx.setLineDash([]);
}

function _getSparkTooltip() {
  let tip = document.getElementById('_spark-tip');
  if (!tip) {
    tip = document.createElement('div');
    tip.id = '_spark-tip';
    tip.style.cssText = 'position:fixed;background:#1e2130;border:1px solid #475569;border-radius:6px;padding:4px 10px;font-size:11px;color:#e2e8f0;pointer-events:none;display:none;z-index:1000;white-space:nowrap;line-height:1.6;';
    document.body.appendChild(tip);
  }
  return tip;
}

function setupSparklineHover(canvas, prices, outcomeDate, tmPoints, invertedPm) {
  if (!canvas || !prices || prices.length === 0) return;
  const W = canvas.width;
  const parseDate = s => new Date(s + 'T00:00:00Z');
  const priceMs = prices.map(p => parseDate(p.date).getTime());
  const minMs = priceMs[0];
  const maxMs = outcomeDate ? parseDate(outcomeDate).getTime() : priceMs[priceMs.length - 1];
  const rangeMs = (maxMs - minMs) || 1;
  const xForMs = ms => 10 + (ms - minMs) / rangeMs * (W - 20);

  const pts = prices.map((p, i) => ({
    date: p.date,
    val: invertedPm ? 1 - p.probability : p.probability,
    x: xForMs(priceMs[i]),
    type: 'pm',
  }));
  if (tmPoints && tmPoints.length > 0) {
    tmPoints.forEach(pt => {
      const ms = maxMs - pt.t * 86400000;
      pts.push({ date: new Date(ms).toISOString().slice(0, 10), val: pt.p, x: xForMs(ms), type: 'tm' });
    });
  }

  const tip = _getSparkTooltip();

  canvas.addEventListener('mousemove', e => {
    const rect = canvas.getBoundingClientRect();
    const mx = (e.clientX - rect.left) * (canvas.width / rect.width);
    let nearest = null, minDist = Infinity;
    pts.forEach(pt => {
      const d = Math.abs(pt.x - mx);
      if (d < minDist) { minDist = d; nearest = pt; }
    });
    if (nearest && minDist < 20) {
      const color = nearest.type === 'tm' ? '#60a5fa' : '#fb923c';
      const label = nearest.type === 'tm' ? 'TM' : 'PM';
      tip.innerHTML = '<span style="color:' + color + '">' + label + '</span> · ' + nearest.date + ' · <strong>' + Math.round(nearest.val * 100) + '%</strong>';
      tip.style.display = 'block';
      tip.style.left = (e.clientX + 14) + 'px';
      tip.style.top = (e.clientY - 38) + 'px';
    } else {
      tip.style.display = 'none';
    }
  });
  canvas.addEventListener('mouseleave', () => { tip.style.display = 'none'; });
}
"""


def _event_tm_points(r: dict, t_days: int, horizon_data: Optional[dict]) -> list:
    """Build [{t, p}] TM horizon points for a single event row.

    Uses sweep data when available; falls back to the single-T measurement.
    Points are sorted by t descending so the JS draws left→right in time.
    """
    if horizon_data:
        pts = []
        for t, s in horizon_data.items():
            by_id = {row["id"]: row for row in s["rows"]}
            if r["id"] in by_id and by_id[r["id"]]["tm_p"] is not None:
                pts.append({"t": t, "p": by_id[r["id"]]["tm_p"]})
        return sorted(pts, key=lambda x: x["t"], reverse=True)
    if r["tm_p"] is not None:
        return [{"t": t_days, "p": r["tm_p"]}]
    return []


def _bar_width(brier_val: Optional[float], max_brier: float = 0.5) -> int:
    if brier_val is None:
        return 0
    return max(2, int((brier_val / max_brier) * 300))


def _render_coverage_table(coverage_rows: list[dict] | None) -> str:
    if not coverage_rows:
        return ""
    n_both = sum(1 for r in coverage_rows if r["has_pm"] and r["has_tm"])
    n_pm_only = sum(1 for r in coverage_rows if r["has_pm"] and not r["has_tm"])
    n_none = sum(1 for r in coverage_rows if not r["has_pm"])

    trs = []
    for r in coverage_rows:
        if r["has_pm"] and r["has_tm"]:
            css, pm_dot, tm_dot = "cov-both", "●", "●"
        elif r["has_pm"]:
            css, pm_dot, tm_dot = "cov-pm", "●", "○"
        else:
            css, pm_dot, tm_dot = "cov-none", "○", "○"
        trs.append(
            f'<tr class="{css}">'
            f'<td>{r["id"]}</td>'
            f'<td>{html.escape(r["name"])}</td>'
            f'<td>{r["category"]}</td>'
            f'<td>{r["outcome_date"]}</td>'
            f'<td class="cov-dot">{pm_dot}</td>'
            f'<td class="cov-dot">{tm_dot}</td>'
            f'</tr>'
        )
    rows_html = "\n".join(trs)
    return f"""
<!-- Coverage table -->
<h2 style="margin-top:3rem">Dataset coverage
  <span style="font-size:0.75rem;color:#475569">({len(coverage_rows)} events · <span class="cov-both">{n_both} both</span> · <span class="cov-pm">{n_pm_only} PM only</span> · <span class="cov-none">{n_none} neither</span>)</span>
</h2>
<p style="font-size:0.82rem;color:#64748b;margin-bottom:1rem">
  ● = data present · ○ = not yet fetched · only "both" rows enter the Brier comparison above
</p>
<div style="overflow-x:auto">
<table>
  <thead>
    <tr>
      <th>ID</th><th>Event</th><th>Category</th><th>Outcome date</th>
      <th>PM</th><th>TM</th>
    </tr>
  </thead>
  <tbody>
{rows_html}
  </tbody>
</table>
</div>"""


def _render_horizon_sweep(sweep: dict[int, dict]) -> str:
    t_values = sorted(sweep.keys())

    # Summary table — common-event averages
    table_rows_html = []
    for t in t_values:
        s = sweep[t]
        tm_avg, pm_avg = s["tm_avg"], s["pm_avg"]
        tm_is_better = tm_avg is not None and pm_avg is not None and tm_avg < pm_avg
        tm_style = "color:#4ade80;font-weight:600" if tm_is_better else "color:#f87171"
        pm_style = "color:#f87171" if tm_is_better else "color:#4ade80;font-weight:600"
        tm_str = f'<span style="{tm_style}">{tm_avg}</span>' if tm_avg is not None else "—"
        pm_str = f'<span style="{pm_style}">{pm_avg}</span>' if pm_avg is not None else "—"
        table_rows_html.append(
            f"<tr>"
            f'<td style="color:#94a3b8">T−{t}d</td>'
            f'<td>{s["n_common"]}</td>'
            f'<td style="color:#60a5fa">{tm_str}</td>'
            f'<td style="color:#fb923c">{pm_str}</td>'
            f'<td style="color:#60a5fa">{s["tm_wins"]}</td>'
            f'<td style="color:#fb923c">{s["pm_wins"]}</td>'
            f"</tr>"
        )

    # Per-event winner matrix
    all_eids: dict[str, str] = {}
    for s in sweep.values():
        for r in s["compared"]:
            all_eids[r["id"]] = r["name"]

    matrix_rows_html = []
    for eid in sorted(all_eids):
        cells = [
            f'<td style="font-size:0.78rem;color:#94a3b8">{eid}</td>',
            f'<td style="font-size:0.78rem">{html.escape(all_eids[eid][:35])}</td>',
        ]
        for t in t_values:
            by_id = {r["id"]: r for r in sweep[t]["compared"]}
            if eid not in by_id:
                cells.append('<td class="no-data">—</td>')
            elif by_id[eid]["winner"] == "tm":
                cells.append('<td style="color:#60a5fa;font-weight:600">TM</td>')
            elif by_id[eid]["winner"] == "pm":
                cells.append('<td style="color:#fb923c;font-weight:600">PM</td>')
            else:
                cells.append('<td style="color:#64748b">tie</td>')
        matrix_rows_html.append(f'<tr>{"".join(cells)}</tr>')

    t_headers = "".join(f"<th>T−{t}d</th>" for t in t_values)

    chart_data = json.dumps({
        "t_values": t_values,
        "tm": [sweep[t]["tm_avg"] for t in t_values],
        "pm": [sweep[t]["pm_avg"] for t in t_values],
        "n": [sweep[t]["n_common"] for t in t_values],
    })

    return f"""
<h2>Prediction horizon sweep
  <span style="font-size:0.75rem;color:#475569"> — how Brier changes as the information cutoff moves further from the event</span>
</h2>
<p style="font-size:0.82rem;color:#64748b;margin-bottom:1rem">
  Both sides measured at the <em>same</em> T: PM uses its last CLOB price ≤ T, TM uses vault2 articles published ≤ T.
  Smaller T = more recent info (easier). Larger T = further ahead (harder). Averages restricted to events with TM data at every T value (n shown per point).
</p>
<div style="background:#1e2130;border-radius:10px;padding:1.25rem;margin-bottom:1.5rem">
  <canvas id="horizon-canvas" width="680" height="280" style="max-width:100%;display:block;margin:0 auto"></canvas>
</div>
<script>
(function(){{
  var d={chart_data};
  var canvas=document.getElementById('horizon-canvas');
  if(!canvas||!d.t_values.length)return;
  var ctx=canvas.getContext('2d');
  var W=canvas.width,H=canvas.height;
  var pl=60,pr=20,pt=30,pb=50;
  var cw=W-pl-pr,ch=H-pt-pb;
  var n=d.t_values.length;
  var maxBrier=0.5;
  function xFor(i){{return pl+(i/(n-1))*cw;}}
  function yFor(v){{return pt+ch-(v/maxBrier)*ch;}}

  ctx.fillStyle='#1e2130';ctx.fillRect(0,0,W,H);

  // Grid
  ctx.strokeStyle='#2d3748';ctx.lineWidth=0.5;
  [0,0.1,0.2,0.3,0.4,0.5].forEach(function(g){{
    var y=yFor(g);
    ctx.beginPath();ctx.moveTo(pl,y);ctx.lineTo(pl+cw,y);ctx.stroke();
    ctx.fillStyle='#475569';ctx.font='11px sans-serif';ctx.textAlign='right';
    ctx.fillText(g.toFixed(1),pl-6,y+4);
  }});

  // Axes
  ctx.strokeStyle='#475569';ctx.lineWidth=1;
  ctx.beginPath();ctx.moveTo(pl,pt);ctx.lineTo(pl,pt+ch);ctx.lineTo(pl+cw,pt+ch);ctx.stroke();

  // X labels + n annotation
  d.t_values.forEach(function(t,i){{
    var x=xFor(i);
    ctx.fillStyle='#94a3b8';ctx.font='12px sans-serif';ctx.textAlign='center';
    ctx.fillText('T−'+t+'d',x,pt+ch+18);
    if(d.n[i]!==null){{
      ctx.fillStyle='#475569';ctx.font='10px sans-serif';
      ctx.fillText('n='+d.n[i],x,pt+ch+32);
    }}
  }});

  // Y label
  ctx.save();ctx.translate(13,pt+ch/2);ctx.rotate(-Math.PI/2);
  ctx.fillStyle='#64748b';ctx.font='11px sans-serif';ctx.textAlign='center';
  ctx.fillText('Avg Brier (lower = better)',0,0);ctx.restore();

  function drawLine(vals,color){{
    ctx.beginPath();ctx.strokeStyle=color;ctx.lineWidth=2.5;
    var started=false;
    vals.forEach(function(v,i){{
      if(v===null){{started=false;return;}}
      var y=yFor(v);
      if(!started){{ctx.moveTo(xFor(i),y);started=true;}}
      else ctx.lineTo(xFor(i),y);
    }});
    ctx.stroke();
    vals.forEach(function(v,i){{
      if(v===null)return;
      var x=xFor(i),y=yFor(v);
      ctx.beginPath();ctx.arc(x,y,5,0,2*Math.PI);
      ctx.fillStyle=color;ctx.fill();
      ctx.fillStyle=color;ctx.font='bold 11px sans-serif';ctx.textAlign='center';
      ctx.fillText(v.toFixed(3),x,y-10);
    }});
  }}
  drawLine(d.pm,'#f97316');
  drawLine(d.tm,'#3b82f6');

  // Legend
  [['TruthMachine','#3b82f6'],['Polymarket','#f97316']].forEach(function(item,i){{
    var lx=pl+10+i*140,ly=pt+14;
    ctx.fillStyle=item[1];ctx.fillRect(lx,ly-7,18,3);
    ctx.fillStyle='#e2e8f0';ctx.font='12px sans-serif';ctx.textAlign='left';
    ctx.fillText(item[0],lx+22,ly);
  }});
}})();
</script>

<div style="overflow-x:auto;margin-bottom:2rem">
<table>
  <thead>
    <tr>
      <th>Horizon</th><th>n (common)</th>
      <th style="color:#60a5fa">TM avg Brier</th>
      <th style="color:#fb923c">PM avg Brier</th>
      <th style="color:#60a5fa">TM wins</th>
      <th style="color:#fb923c">PM wins</th>
    </tr>
  </thead>
  <tbody>{"".join(table_rows_html)}</tbody>
</table>
</div>

<h3 style="font-size:1rem;font-weight:600;margin:1.5rem 0 0.75rem;color:#cbd5e1">Per-event winner at each horizon</h3>
<div style="overflow-x:auto">
<table>
  <thead><tr><th>ID</th><th>Event</th>{t_headers}</tr></thead>
  <tbody>{"".join(matrix_rows_html)}</tbody>
</table>
</div>"""


def render_html(
    rows: list[dict],
    t_days: int,
    out_path: Path,
    coverage_rows: list[dict] | None = None,
    horizon_data: dict | None = None,
) -> None:
    compared = [r for r in rows if r["pm_p"] is not None and r["tm_p"] is not None]
    tm_avg = avg_brier(compared, "tm_brier")
    pm_avg = avg_brier(compared, "pm_brier")
    tm_ci = bootstrap_ci([r["tm_brier"] for r in compared if r["tm_brier"] is not None])
    pm_ci = bootstrap_ci([r["pm_brier"] for r in compared if r["pm_brier"] is not None])

    if tm_avg is not None and pm_avg is not None:
        if tm_avg < pm_avg:
            overall_winner = "TruthMachine"
            winner_css = "tm-color"
        elif pm_avg < tm_avg:
            overall_winner = "Polymarket"
            winner_css = "pm-color"
        else:
            overall_winner = "Tie"
            winner_css = ""
    else:
        overall_winner = "—"
        winner_css = ""

    # Brier decomposition (Murphy 1973)
    pairs_tm = [(r["tm_p"], 1.0 if r["outcome"] else 0.0) for r in compared if r["tm_p"] is not None]
    pairs_pm = [(r["pm_p"], 1.0 if r["outcome"] else 0.0) for r in compared if r["pm_p"] is not None]
    decomp_tm = brier_decomposition(pairs_tm)
    decomp_pm = brier_decomposition(pairs_pm)

    # Scatter data
    scatter_data = json.dumps([
        {"x": r["pm_p"], "y": r["tm_p"], "label": r["id"]}
        for r in compared
    ])

    # Build event cards HTML
    cards_html = []
    for r in rows:
        card_class = {"tm": "tm-wins", "pm": "pm-wins", "tie": "tie"}.get(r["winner"] or "", "")
        winner_badge = ""
        if r["winner"] == "tm":
            winner_badge = '<span class="badge badge-tm">TM wins</span>'
        elif r["winner"] == "pm":
            winner_badge = '<span class="badge badge-pm">PM wins</span>'
        elif r["winner"] == "tie":
            winner_badge = '<span class="badge badge-tie">Tie</span>'

        inv_note = ' <span class="inv">(inverted)</span>' if r["pm_invert"] else ""
        pm_q_text = html.escape(r["pm_question"])
        if r.get("pm_url"):
            pm_q_text = f'<a href="{html.escape(r["pm_url"])}" target="_blank" rel="noopener" style="color:#f97316;text-decoration:none" onmouseover="this.style.textDecoration=\'underline\'" onmouseout="this.style.textDecoration=\'none\'">{pm_q_text}</a>'
        pm_q_html = f'<div class="event-pm-q">PM: {pm_q_text}{inv_note}</div>'

        # Probability bars
        def _prob_block(label, val, css_class, brier_val, n_art=None, placeholder=False, pm_days=None):
            if val is None:
                msg = "insufficient data" if placeholder else "no data"
                return f'<div class="prob-block"><div class="prob-label">{label}</div><div class="no-data">{msg}</div></div>'
            pct = int(val * 100)
            w = int(val * 100)
            if n_art is not None:
                sub = f'<span style="font-size:0.7rem;color:#475569">n={n_art} articles</span>'
            elif pm_days is not None:
                sub = f'<span style="font-size:0.7rem;color:#475569">−{pm_days}d before event</span>'
            else:
                sub = ""
            return f"""
            <div class="prob-block">
              <div class="prob-label">{label}</div>
              <div class="prob-bar-wrap"><div class="prob-bar {css_class}" style="width:{w}%"></div></div>
              <span class="prob-val">{pct}%</span>
              <span class="brier-val"> Brier: {brier_val if brier_val is not None else '—'}</span>
              {sub}
            </div>"""

        tm_block = _prob_block("TruthMachine", r["tm_p"], "tm", r["tm_brier"], r["tm_articles_used"], r["tm_placeholder"])
        pm_block = _prob_block(f"Polymarket (T-{t_days}d)", r["pm_p"], "pm", r["pm_brier"], pm_days=r.get("pm_price_days"))

        # Sparkline
        prices_json = json.dumps(r["pm_prices"])
        tm_pts = _event_tm_points(r, t_days, horizon_data)
        tm_pts_js = json.dumps(tm_pts) if tm_pts else "null"
        outcome_date_js = f'"{r["outcome_date"]}"'
        invert_js = "true" if r["pm_invert"] else "false"
        canvas_id = f"spark-{r['id']}"
        multi_label = " (multi-horizon)" if len(tm_pts) > 1 else ""
        sparkline = f"""
        <div class="sparkline-wrap">
          <div class="prob-label">PM price history  <span style="color:#f97316">━</span> PM  <span style="color:#3b82f6">●━</span> TM{multi_label}</div>
          <canvas id="{canvas_id}" width="260" height="60" style="width:100%;max-width:260px;height:60px"></canvas>
          <script>
            (function(){{
              var el=document.getElementById('{canvas_id}');
              drawSparkline(el,{prices_json},{outcome_date_js},{tm_pts_js},{invert_js});
              setupSparklineHover(el,{prices_json},{outcome_date_js},{tm_pts_js},{invert_js});
            }})();
          </script>
        </div>"""

        cards_html.append(f"""
        <div class="event-card {card_class}">
          <div class="event-header">
            <div>
              <div class="event-name">{r['id']} — {r['name']}</div>
              <div class="event-meta">{r['outcome_date']} · outcome: <span class="outcome-yes">YES</span> · {r['category']}</div>
              {pm_q_html}
            </div>
            {winner_badge}
          </div>
          <div class="event-body">
            {tm_block}
            {pm_block}
            {sparkline}
          </div>
        </div>""")

    cards_joined = "\n".join(cards_html)

    # Bar chart
    max_brier = max(
        (r["pm_brier"] or 0 for r in compared),
        default=0.5
    )
    max_brier = max(max_brier, 0.1)
    bar_rows = []
    for r in sorted(compared, key=lambda x: abs((x["tm_brier"] or 0) - (x["pm_brier"] or 0)), reverse=True):
        tw = _bar_width(r["tm_brier"], max_brier)
        pw = _bar_width(r["pm_brier"], max_brier)
        bar_rows.append(f"""
        <div class="bar-row">
          <div class="bar-label">{r['id']}</div>
          <div class="bar-pair">
            <div><span class="bar-seg tm" style="width:{tw}px"></span></div>
            <div><span class="bar-seg pm" style="width:{pw}px"></span></div>
          </div>
          <div class="bar-num">{r['tm_brier']}</div>
          <div class="bar-num">{r['pm_brier']}</div>
        </div>""")
    bar_chart_html = "\n".join(bar_rows)

    # Table
    table_rows = []
    for r in rows:
        tm_p_str = f"{int(r['tm_p']*100)}%" if r["tm_p"] is not None else '<span class="no-data">—</span>'
        pm_p_str = f"{int((r['pm_p'] or 0)*100)}%" if r["pm_p"] is not None else '<span class="no-data">—</span>'
        pm_days_str = f'<span style="color:#64748b">−{r["pm_price_days"]}d</span>' if r["pm_price_days"] is not None else '<span class="no-data">—</span>'
        n_str = f'<span style="color:#94a3b8">{r["tm_articles_used"]}</span>' if r["tm_p"] is not None else '<span class="no-data">—</span>'
        inv_mark = " ↻" if r["pm_invert"] else ""
        tm_b = r["tm_brier"]
        pm_b = r["pm_brier"]
        if tm_b is not None and pm_b is not None:
            tm_b_str = f'<span class="{"better" if tm_b < pm_b else "worse"}">{tm_b}</span>'
            pm_b_str = f'<span class="{"better" if pm_b < tm_b else "worse"}">{pm_b}</span>'
        else:
            tm_b_str = f"{tm_b}" if tm_b is not None else '<span class="no-data">—</span>'
            pm_b_str = f"{pm_b}" if pm_b is not None else '<span class="no-data">—</span>'
        winner_str = {"tm": "TM ✓", "pm": "PM ✓", "tie": "Tie"}.get(r["winner"] or "", "—")
        table_rows.append(f"""
        <tr>
          <td>{r['id']}</td>
          <td>{r['name'][:45]}</td>
          <td class="outcome-yes">YES</td>
          <td>{"<a href='" + html.escape(r['pm_url']) + "' target='_blank' rel='noopener' style='color:#f97316'>" + html.escape(r['pm_question'][:40]) + "</a>" if r.get("pm_url") else html.escape(r['pm_question'][:40])}{inv_mark}</td>
          <td>{pm_p_str}</td>
          <td>{pm_days_str}</td>
          <td>{tm_p_str}</td>
          <td>{n_str}</td>
          <td>{pm_b_str}</td>
          <td>{tm_b_str}</td>
          <td>{winner_str}</td>
        </tr>""")
    table_rows_html = "\n".join(table_rows)

    # Score summary line
    n_compared = len(compared)
    tm_wins = sum(1 for r in compared if r["winner"] == "tm")
    pm_wins = sum(1 for r in compared if r["winner"] == "pm")
    ties = sum(1 for r in compared if r["winner"] == "tie")

    # Brier decomposition block
    if decomp_tm and decomp_pm:
        all_yes_note = (
            '<div style="font-size:0.72rem;color:#475569;margin-top:0.75rem">'
            '⚠ All outcomes are YES — RES and UNC trivially = 0; decomposition is indicative only (n is small).'
            '</div>'
        ) if decomp_tm["o_bar"] >= 0.99 else ""
        decomp_block = f"""
<div style="background:#1e2130;border-radius:10px;padding:1rem 1.5rem;max-width:560px;margin:0 auto 1.5rem">
  <h3 style="font-size:0.78rem;color:#64748b;margin-bottom:0.75rem;text-transform:uppercase;letter-spacing:0.05em">Brier decomposition · BS = REL − RES + UNC</h3>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;font-size:0.8rem">
    <div>
      <div style="color:#60a5fa;font-weight:600;margin-bottom:0.4rem">TruthMachine</div>
      <div style="display:flex;justify-content:space-between;margin-bottom:2px"><span>REL <span style="color:#475569;font-size:0.7rem">(↓ better)</span></span><span style="color:#60a5fa">{decomp_tm['rel']}</span></div>
      <div style="display:flex;justify-content:space-between;margin-bottom:2px"><span>RES <span style="color:#475569;font-size:0.7rem">(↑ better)</span></span><span style="color:#60a5fa">{decomp_tm['res']}</span></div>
      <div style="display:flex;justify-content:space-between"><span>UNC</span><span style="color:#94a3b8">{decomp_tm['unc']}</span></div>
    </div>
    <div>
      <div style="color:#fb923c;font-weight:600;margin-bottom:0.4rem">Polymarket</div>
      <div style="display:flex;justify-content:space-between;margin-bottom:2px"><span>REL</span><span style="color:#fb923c">{decomp_pm['rel']}</span></div>
      <div style="display:flex;justify-content:space-between;margin-bottom:2px"><span>RES</span><span style="color:#fb923c">{decomp_pm['res']}</span></div>
      <div style="display:flex;justify-content:space-between"><span>UNC</span><span style="color:#94a3b8">{decomp_pm['unc']}</span></div>
    </div>
  </div>
  {all_yes_note}
</div>"""
    else:
        decomp_block = ""

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>TruthMachine vs Polymarket — Duel</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>{_CSS}</style>
</head>
<body>
<div class="container">

<h1>TruthMachine <span style="color:#475569">vs</span> Polymarket</h1>
<p class="subtitle">Phase 1 · {n_compared} events · T = {t_days} days before resolution · all outcomes YES</p>

<div class="note">
  <strong>Methodology:</strong>
  PM probability = last CLOB price on or before <em>outcome_date − {t_days} days</em>
  (inverted where PM question is framed as "will X survive?" vs our event "X was killed").<br>
  TM probability = Oracul API forecast (gatekeeper+extractor+credibility weighting) on vault2 articles published ≤ T, converted (stance+1)/2.<br>
  All {n_compared} events resolved YES (our dataset captures things that happened).
  Brier score: lower = better. Range [0, 1].
</div>

<!-- Scorecard -->
<div class="scorecard">
  <div class="scorecard-side">
    <h2>TruthMachine</h2>
    <div class="brier tm-color">{tm_avg if tm_avg is not None else "—"}</div>
    <div class="label">avg Brier ({n_compared} events)</div>
    {'<div class="label" style="margin-top:0.25rem;font-size:0.72rem">95% CI: [' + str(tm_ci[0]) + '–' + str(tm_ci[1]) + ']</div>' if tm_ci[0] is not None else ''}
    <div class="label" style="margin-top:0.5rem">wins: {tm_wins} / {n_compared}</div>
  </div>
  <div class="vs-divider">VS</div>
  <div class="scorecard-side">
    <h2>Polymarket</h2>
    <div class="brier pm-color">{pm_avg if pm_avg is not None else "—"}</div>
    <div class="label">avg Brier ({n_compared} events)</div>
    {'<div class="label" style="margin-top:0.25rem;font-size:0.72rem">95% CI: [' + str(pm_ci[0]) + '–' + str(pm_ci[1]) + ']</div>' if pm_ci[0] is not None else ''}
    <div class="label" style="margin-top:0.5rem">wins: {pm_wins} / {n_compared}</div>
  </div>
</div>
<div style="text-align:center;margin-bottom:1.5rem">
  <span class="winner-badge {winner_css}">
    {"🏆 " if overall_winner not in ("Tie","—") else ""}{overall_winner} wins
  </span>
  &nbsp;
  <span style="color:#64748b;font-size:0.82rem">
    (ties: {ties} · ⚠ n={n_compared} is small — results are indicative, not conclusive)
  </span>
</div>

{decomp_block}

<!-- Sparkline helper -->
<script>{_SPARKLINE_JS}</script>

<!-- Per-event cards -->
<h2>Per-event breakdown</h2>
<p style="color:#94a3b8;font-size:0.85rem;margin-top:-0.5rem;margin-bottom:1rem">
  Showing {len(rows)} of 70 atlas events — the {len(rows)} that have Polymarket CLOB price history.
  Pre-2023 markets (Ukraine invasion, Kherson, JCPOA, Mahsa Amini, ChatGPT launch) are not in
  Polymarket's CLOB system and cannot be scored. See the Dataset coverage table below for full
  atlas coverage.
</p>
<div class="events-grid">
{cards_joined}
</div>

<!-- Bar chart -->
<h2>Brier score comparison <span style="font-size:0.75rem;color:#475569">(shorter = better · <span style="color:#3b82f6">■</span> TM  <span style="color:#f97316">■</span> PM · sorted by |Δ|)</span></h2>
<div style="background:#1e2130;border-radius:10px;padding:1rem">
  <div style="font-size:0.72rem;color:#64748b;margin-bottom:0.75rem">
    TM (top bar) vs PM (bottom bar) per event
  </div>
  <div class="bar-chart">
{bar_chart_html}
  </div>
</div>

<!-- Scatter -->
<h2>Agreement scatter <span style="font-size:0.75rem;color:#475569">(diagonal = perfect agreement)</span></h2>
<div id="scatter-wrap">
  <canvas id="scatter-canvas" width="500" height="400" style="max-width:100%;display:block;margin:0 auto"></canvas>
</div>

<!-- Data table -->
<h2>Full data table</h2>
<div style="overflow-x:auto">
<table>
  <thead>
    <tr>
      <th>ID</th><th>TM Event</th><th>Out</th><th>PM Question</th>
      <th>PM@T-{t_days}d</th><th title="days before event that PM price was recorded">T_PM</th>
      <th>TM prob</th><th title="Oracul articles used">n</th>
      <th>PM Brier</th><th>TM Brier</th><th>Winner</th>
    </tr>
  </thead>
  <tbody>
{table_rows_html}
  </tbody>
</table>
</div>

{_render_horizon_sweep(horizon_data) if horizon_data else ""}

{_render_coverage_table(coverage_rows)}

<footer>
  Generated {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC ·
  TruthMachine data: Oracle API (oracle.daatan.com) ·
  Polymarket data: CLOB API (gamma-api.polymarket.com) ·
  T = {t_days} days before outcome_date
</footer>

<script>
// Scatter plot
(function() {{
  var data = {scatter_data};
  var canvas = document.getElementById('scatter-canvas');
  if (!canvas || !data.length) return;
  var ctx = canvas.getContext('2d');
  var W = canvas.width, H = canvas.height;
  var pad = 50;
  function sx(v) {{ return pad + v * (W - 2*pad); }}
  function sy(v) {{ return H - pad - v * (H - 2*pad); }}

  // Axes
  ctx.strokeStyle = '#2d3748';
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pad, pad); ctx.lineTo(pad, H-pad); ctx.lineTo(W-pad, H-pad); ctx.stroke();

  // Diagonal
  ctx.strokeStyle = '#334155';
  ctx.setLineDash([4,4]);
  ctx.beginPath(); ctx.moveTo(pad,H-pad); ctx.lineTo(W-pad,pad); ctx.stroke();
  ctx.setLineDash([]);

  // Grid
  ctx.strokeStyle = '#1e2130';
  [0.25,0.5,0.75].forEach(g => {{
    ctx.beginPath(); ctx.moveTo(sx(g),pad); ctx.lineTo(sx(g),H-pad);
    ctx.moveTo(pad,sy(g)); ctx.lineTo(W-pad,sy(g)); ctx.stroke();
  }});

  // Axis labels
  ctx.fillStyle = '#64748b';
  ctx.font = '12px sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('PM probability at T-{t_days}', W/2, H-10);
  ctx.save(); ctx.translate(14, H/2); ctx.rotate(-Math.PI/2);
  ctx.fillText('TM probability', 0, 0); ctx.restore();
  [0,0.25,0.5,0.75,1].forEach(v => {{
    ctx.fillText(Math.round(v*100)+'%', sx(v), H-pad+18);
    ctx.textAlign='right';
    ctx.fillText(Math.round(v*100)+'%', pad-6, sy(v)+4);
    ctx.textAlign='center';
  }});

  // Points
  data.forEach(d => {{
    ctx.beginPath();
    ctx.arc(sx(d.x), sy(d.y), 7, 0, 2*Math.PI);
    ctx.fillStyle = '#22c55e';
    ctx.fill();
    ctx.strokeStyle = '#15803d';
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.fillStyle = '#e2e8f0';
    ctx.font = 'bold 9px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(d.label, sx(d.x), sy(d.y)-10);
  }});
}})();
</script>

</div>
</body>
</html>"""

    out_path.write_text(html_doc)
    console.print(f"[bold green]Wrote {out_path}[/bold green] ({out_path.stat().st_size // 1024} KB)")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "data"))
    ap.add_argument("--out", default="duel.html")
    ap.add_argument("--t-days", type=int, default=T_DAYS_DEFAULT)
    ap.add_argument(
        "--html-only",
        action="store_true",
        help="Re-render duel.html from existing duel_oracle cache — no API calls",
    )
    ap.add_argument(
        "--clear-cache",
        action="store_true",
        help="Delete all duel_oracle/*.json cache files before fetching",
    )
    ap.add_argument(
        "--t-sweep",
        action="store_true",
        help="Run at multiple T values and add a prediction-horizon sweep section to the output",
    )
    ap.add_argument(
        "--sweep-values",
        default="3,7,14,30",
        help="Comma-separated T values for the sweep (default: 3,7,14,30)",
    )
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_path = Path(args.out)

    if args.clear_cache:
        cache_dir = data_dir / "duel_oracle"
        removed = list(cache_dir.glob("*.json"))
        for f in removed:
            f.unlink()
        console.print(f"[yellow]Cleared {len(removed)} cached Oracul results[/yellow]")

    events = load_events_with_pm(data_dir)
    console.print(f"Events with PM price data: {len(events)}")

    if args.html_only:
        # Load from cache only — no API calls
        cache_dir = data_dir / "duel_oracle"
        tm_probs: dict[str, dict] = {}
        for ev in events:
            eid = ev["id"]
            # Try new per-T naming, fall back to old single-file cache
            cache_path = cache_dir / f"{eid}_T{args.t_days}.json"
            if not cache_path.exists():
                cache_path = cache_dir / f"{eid}.json"
            if cache_path.exists():
                cached = json.loads(cache_path.read_text())
                if cached.get("probability") is not None or cached.get("placeholder"):
                    tm_probs[eid] = cached
        console.print(f"Events loaded from cache (--html-only): {len(tm_probs)}")
    else:
        console.print(f"Fetching TM predictions from Oracul API (T-{args.t_days}d cutoff)...")
        tm_probs = fetch_tm_probabilities_oracle(data_dir, events, args.t_days)
        console.print(f"Events with Oracul predictions: {len(tm_probs)}")

    rows = build_rows(events, tm_probs, args.t_days)

    compared = [r for r in rows if r["pm_p"] is not None and r["tm_p"] is not None]
    console.print(f"Events in Brier comparison: {len(compared)}")

    for r in rows:
        tm_str = f"TM={r['tm_p']}" if r["tm_p"] is not None else "TM=—"
        pm_str = f"PM={r['pm_p']}" if r["pm_p"] is not None else "PM=—"
        w = r["winner"] or "—"
        console.print(f"  {r['id']}: {tm_str} {pm_str} → {w}")

    horizon_data = None
    if args.t_sweep:
        sweep_t_values = [int(v.strip()) for v in args.sweep_values.split(",")]
        console.print(f"\nRunning horizon sweep at T={sweep_t_values}...")
        horizon_data = run_sweep(data_dir, events, sweep_t_values, args.html_only)

    all_events = load_all_events(data_dir)
    coverage_rows = [
        {
            "id": ev["id"],
            "name": ev.get("name", ""),
            "category": ev.get("category", ""),
            "outcome_date": ev.get("outcome_date", ""),
            "has_pm": "_pm" in ev,
            "has_tm": ev["id"] in tm_probs and not tm_probs[ev["id"]].get("placeholder"),
        }
        for ev in all_events
    ]

    render_html(rows, args.t_days, out_path, coverage_rows, horizon_data=horizon_data)


if __name__ == "__main__":
    main()
