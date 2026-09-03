#!/usr/bin/env python3
"""Calibration backtest driver for retro#619 — validate or refute retro#615.

Three modes:

    uv run python scripts/metaculus_backtest.py fetch --token $METACULUS_TOKEN \\
        --out questions.json [--limit 50]
        Pull resolved binary questions from the Metaculus API. **Resolution
        and community-prediction (`aggregations`) require the Bot
        Benchmarking Access Tier** (apply via the Metaculus Data Needs Form,
        linked from post 38928 / docs/... — see retro#619). Confirmed
        empirically 2026-08-24 that the standard bot-account tier returns
        both as null/empty even on a fully `resolved` question. Without that
        tier this mode still fetches question text + close times (enough to
        drive `run-oracle` below) — it just can't supply ground truth, and
        every row is written with `resolution: null`.

    uv run python scripts/metaculus_backtest.py run-oracle questions.json \\
        --out results.json [--deployed-commit SHA] [--limit N]
        For each question, temporally slice the news corpus (search via the
        Oracul's provider chain with date_to = the day before the question's
        close) and run the real Oracul pipeline **in-process** against
        exactly that slice, via `/forecast`'s existing `articles` field —
        no Oracul API change needed, this is retro#619's "Option 2".
        Verifies no returned article's `published_date` postdates the
        cutoff (retro#619's explicit ask); a question with a leaked article
        is recorded with `error: "leak"` rather than silently scored on
        hindsight.

    uv run python scripts/metaculus_backtest.py self-resolve questions.json \\
        --out self_resolutions.json [--deployed-commit SHA] [--limit N] \\
        [--window-days N] [--confidence 0.9]
        retro#737 — a fallback ground truth for questions whose Metaculus
        `resolution` is withheld (no Bot Benchmarking Access Tier). Runs the
        real Oracul pipeline **in-process**, exactly like `run-oracle`, but
        searches a window **after** `actual_close_time` instead of before it
        — this is deliberately the mirror image of `run-oracle`'s leak check,
        not a reuse of it: `run-oracle` forecasts, this resolves. A
        confident post-close probability (>= --confidence or <= 1 -
        --confidence) is recorded as a self-resolved outcome; anything in
        between is left unresolved rather than guessed. This is **strictly
        weaker evidence than a real Metaculus resolution** — see the honesty
        constraint in retro#737 — and every row this produces is labelled
        `ground_truth_source: "self_resolved"`, never "validated".

    uv run python scripts/metaculus_backtest.py score results.json questions.json \\
        [--self-resolved self_resolutions.json]
        Join Oracul's sliced probability against resolution + community
        prediction and print calibration buckets, log score, and Brier
        score. A question without a resolution/CP value is reported as
        unscored, never silently dropped or treated as a pass (retro#395's
        rule for measurement scripts — see scan_outlier_estimates.py).
        With --self-resolved, a question that has no Metaculus resolution
        falls back to that file's self-resolved outcome when present,
        scored and reported **separately** under a "self-resolved (weaker
        evidence — retro#737)" heading — never merged into the primary
        Metaculus-backed numbers.

Why in-process for run-oracle, not the live HTTP API: no 10/min `/forecast`
rate limit for a run over dozens of questions, matching
`scan_outlier_estimates.py`'s convention. That is only legitimate while local
code equals deployed code — pass `--deployed-commit` from
`GET https://oracle.daatan.com/version` (`git_sha` field) to enforce it; the
run aborts on a mismatch rather than warning.

**No thresholds, no verdict.** This prints per-question scores and bucketed
summaries (by whether the question has a hard numeric/date threshold, and by
event base rate); a human decides whether retro#615's bias hypothesis holds
and what the fallback ladder (retro#621) should look like.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

METACULUS_API = "https://www.metaculus.com/api"


# ---------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------


def _get(url: str, token: str) -> dict:
    # Cloudflare 403s urllib's default User-Agent even with a valid token
    # (curl and browser automation both pass) — send a real one.
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Token {token}", "User-Agent": "Mozilla/5.0 (compatible; daatan-oracle-backtest/1.0)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as e:
        print(f"Metaculus API error {e.code} on {url}: {e.read().decode()[:500]}", file=sys.stderr)
        raise


def fetch_resolved_questions(token: str, limit: int) -> list[dict]:
    rows: list[dict] = []
    url = f"{METACULUS_API}/posts/?statuses=resolved&limit=50&order_by=-actual_close_time"
    while url and len(rows) < limit:
        data = _get(url, token)
        for post in data.get("results", []):
            q = post.get("question") or {}
            if q.get("type") != "binary":
                continue  # backtest scope is binary questions only, per retro#619
            agg = (q.get("aggregations") or {}).get("unweighted") or {}
            rows.append(
                {
                    "post_id": post["id"],
                    "question_id": q.get("id"),
                    "title": post["title"],
                    "resolution_criteria": q.get("resolution_criteria", ""),
                    "fine_print": q.get("fine_print", ""),
                    "actual_close_time": post.get("actual_close_time"),
                    "cp_reveal_time": q.get("cp_reveal_time"),
                    "spot_scoring_time": q.get("spot_scoring_time"),
                    "resolution": q.get("resolution"),
                    "community_prediction_latest": agg.get("latest"),
                }
            )
            if len(rows) >= limit:
                break
        url = data.get("next")
    return rows


def cmd_fetch(args: argparse.Namespace) -> int:
    token = args.token or os.environ.get("METACULUS_TOKEN")
    if not token:
        print("Need a token: --token or $METACULUS_TOKEN", file=sys.stderr)
        return 1
    rows = fetch_resolved_questions(token, args.limit)
    unresolved = [r for r in rows if r["resolution"] is None]
    no_cp = [r for r in rows if r["community_prediction_latest"] is None]
    Path(args.out).write_text(json.dumps(rows, indent=2))
    print(f"Wrote {len(rows)} binary resolved questions to {args.out}")
    if unresolved:
        print(
            f"WARNING: {len(unresolved)}/{len(rows)} have resolution=null — this token lacks the "
            "Bot Benchmarking Access Tier (retro#619's Metaculus Data Needs Form). "
            "`run-oracle` still works; `score` will report these as unscored."
        )
    if no_cp:
        print(f"WARNING: {len(no_cp)}/{len(rows)} have no community_prediction_latest for the same reason.")
    return 0


# ---------------------------------------------------------------------------
# run-oracle
# ---------------------------------------------------------------------------


def _check_deployed_commit(expected_sha: str) -> None:
    req = urllib.request.Request("https://oracle.daatan.com/version")
    with urllib.request.urlopen(req, timeout=15) as resp:
        live = json.load(resp)
    local_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent.parent, capture_output=True, text=True
    ).stdout.strip()
    if live.get("git_sha") != expected_sha:
        print(
            f"--deployed-commit={expected_sha} does not match live oracle.daatan.com "
            f"(git_sha={live.get('git_sha')}). Refusing to run.",
            file=sys.stderr,
        )
        sys.exit(1)
    if local_sha != expected_sha:
        print(
            f"WARNING: local HEAD ({local_sha[:8]}) differs from --deployed-commit "
            f"({expected_sha[:8]}) — this in-process run will NOT match what production would return.",
            file=sys.stderr,
        )


def _postdates(published_date: str, cutoff_date) -> bool:
    """True if published_date parses to strictly after cutoff_date.

    Providers return published_date in mixed formats — ISO ("2026-08-10") and
    human-readable ("May 21, 2026") both occur from the same search call — so
    this must actually parse the string, not slice/compare it lexically (a
    naive `published_date[:10] > cutoff` false-flagged every non-ISO date as
    a leak, e.g. "May 21, 2026" > "2026-08-11" is true as a *string* compare).
    Unparseable dates are NOT treated as leaks — a stale/empty published_date
    is filtered elsewhere in the pipeline (`_filter_by_date`); treating parse
    failure as leakage here would drop good articles the pipeline already
    trusts enough to have returned.
    """
    if not published_date:
        return False
    try:
        from dateutil import parser as _dateutil_parser

        return _dateutil_parser.parse(published_date).date() > cutoff_date
    except (ValueError, OverflowError):
        return False


async def _slice_and_forecast(run_search, run_forecast, SearchRequest, ForecastRequest, ArticleInput, q: dict, article_limit: int) -> dict:
    close = q.get("actual_close_time")
    if not close:
        return {"post_id": q["post_id"], "error": "no_close_time"}
    close_dt = datetime.fromisoformat(close.replace("Z", "+00:00"))
    cutoff_dt = (close_dt - timedelta(days=1)).date()
    cutoff = cutoff_dt.isoformat()

    search_resp = await run_search(SearchRequest(query=q["title"], date_to=cutoff, limit=article_limit))

    leaked = [r for r in search_resp.results if _postdates(r.published_date, cutoff_dt)]
    if leaked:
        return {
            "post_id": q["post_id"],
            "cutoff": cutoff,
            "error": "leak",
            "leaked_urls": [r.url for r in leaked],
        }

    articles = [
        ArticleInput(url=r.url, title=r.title, snippet=r.snippet, source=r.source, published_date=r.published_date)
        for r in search_resp.results
    ]
    fc = await run_forecast(ForecastRequest(question=q["title"], articles=articles))

    return {
        "post_id": q["post_id"],
        "cutoff": cutoff,
        "oracle_probability": None if fc.insufficient_data else (fc.mean + 1) / 2,
        "std": fc.std,
        "insufficient_data": fc.insufficient_data,
        "reason": fc.reason,
        "articles_used": fc.articles_used,
        "articles_found_pre_slice": search_resp.count,
        "provider": search_resp.provider,
    }


def cmd_run_oracle(args: argparse.Namespace) -> int:
    os.environ.setdefault("ORACLE_API_KEY", "dummy")
    os.environ.setdefault("SETTLEMENT_VERIFIER_ENABLED", "false")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

    if args.deployed_commit:
        _check_deployed_commit(args.deployed_commit)
    else:
        print(
            "WARNING: no --deployed-commit given — this run is not verified against "
            "what production actually serves. Pass the sha from GET https://oracle.daatan.com/version.",
            file=sys.stderr,
        )

    from forecast_api.forecaster import run_forecast  # noqa: E402
    from forecast_api.models import ArticleInput, ForecastRequest, SearchRequest  # noqa: E402
    from forecast_api.searcher import run_search  # noqa: E402

    questions = json.loads(Path(args.questions).read_text())
    if args.limit:
        questions = questions[: args.limit]

    async def run_all():
        results = []
        for i, q in enumerate(questions):
            print(f"[{i+1}/{len(questions)}] {q['title'][:80]}", file=sys.stderr)
            r = await _slice_and_forecast(run_search, run_forecast, SearchRequest, ForecastRequest, ArticleInput, q, args.article_limit)
            results.append(r)
        return results

    results = asyncio.run(run_all())
    Path(args.out).write_text(json.dumps(results, indent=2))

    leaks = [r for r in results if r.get("error") == "leak"]
    scored = [r for r in results if "oracle_probability" in r and r["oracle_probability"] is not None]
    print(f"Wrote {len(results)} rows to {args.out}: {len(scored)} produced a probability, {len(leaks)} leaked (dropped).")
    return 0


# ---------------------------------------------------------------------------
# self-resolve (retro#737)
# ---------------------------------------------------------------------------


def classify_self_resolution(probability: float, confidence: float) -> str | None:
    """Map a post-close Oracul probability to a self-resolved outcome.

    Returns "yes"/"no" only when the post-close evidence is decisive (>=
    confidence either direction); otherwise None — an ambiguous post-close
    read must never be silently forced into a verdict.
    """
    eps = 1e-9  # float slop guard: 1 - 0.9 != 0.1 exactly in binary float
    if probability >= confidence - eps:
        return "yes"
    if probability <= (1 - confidence) + eps:
        return "no"
    return None


async def _self_resolve_one(run_search, run_forecast, SearchRequest, ForecastRequest, ArticleInput, q: dict, article_limit: int, window_days: int, confidence: float) -> dict:
    close = q.get("actual_close_time")
    if not close:
        return {"post_id": q["post_id"], "error": "no_close_time"}
    close_dt = datetime.fromisoformat(close.replace("Z", "+00:00")).date()
    window_start = close_dt
    window_end = close_dt + timedelta(days=window_days)

    search_resp = await run_search(
        SearchRequest(query=q["title"], date_from=window_start.isoformat(), date_to=window_end.isoformat(), limit=article_limit)
    )
    articles = [
        ArticleInput(url=r.url, title=r.title, snippet=r.snippet, source=r.source, published_date=r.published_date)
        for r in search_resp.results
    ]
    fc = await run_forecast(ForecastRequest(question=q["title"], articles=articles))

    if fc.insufficient_data:
        return {
            "post_id": q["post_id"],
            "window": [window_start.isoformat(), window_end.isoformat()],
            "self_resolution": None,
            "reason": "insufficient_post_close_evidence",
            "ground_truth_source": "self_resolved",
        }

    probability = (fc.mean + 1) / 2
    resolution = classify_self_resolution(probability, confidence)
    return {
        "post_id": q["post_id"],
        "window": [window_start.isoformat(), window_end.isoformat()],
        "self_resolution": resolution,
        "self_resolution_probability": probability,
        "reason": None if resolution else "ambiguous_post_close_evidence",
        "articles_used": fc.articles_used,
        "articles_found": search_resp.count,
        "provider": search_resp.provider,
        "ground_truth_source": "self_resolved",
    }


def cmd_self_resolve(args: argparse.Namespace) -> int:
    os.environ.setdefault("ORACLE_API_KEY", "dummy")
    os.environ.setdefault("SETTLEMENT_VERIFIER_ENABLED", "false")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

    if args.deployed_commit:
        _check_deployed_commit(args.deployed_commit)
    else:
        print(
            "WARNING: no --deployed-commit given — this run is not verified against "
            "what production actually serves. Pass the sha from GET https://oracle.daatan.com/version.",
            file=sys.stderr,
        )

    from forecast_api.forecaster import run_forecast  # noqa: E402
    from forecast_api.models import ArticleInput, ForecastRequest, SearchRequest  # noqa: E402
    from forecast_api.searcher import run_search  # noqa: E402

    questions = json.loads(Path(args.questions).read_text())
    if args.limit:
        questions = questions[: args.limit]

    async def run_all():
        results = []
        for i, q in enumerate(questions):
            print(f"[{i+1}/{len(questions)}] {q['title'][:80]}", file=sys.stderr)
            r = await _self_resolve_one(
                run_search, run_forecast, SearchRequest, ForecastRequest, ArticleInput, q, args.article_limit, args.window_days, args.confidence
            )
            results.append(r)
        return results

    results = asyncio.run(run_all())
    Path(args.out).write_text(json.dumps(results, indent=2))

    resolved = [r for r in results if r.get("self_resolution") is not None]
    print(
        f"Wrote {len(results)} rows to {args.out}: {len(resolved)}/{len(results)} self-resolved "
        f"at confidence >= {args.confidence}. This is weaker evidence than a Metaculus resolution "
        "(retro#737) — never report these as 'validated'."
    )
    return 0


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------


def _brier(p: float, outcome: int) -> float:
    return (p - outcome) ** 2


def _log_score(p: float, outcome: int, eps: float = 1e-6) -> float:
    p = min(max(p, eps), 1 - eps)
    return math.log(p) if outcome == 1 else math.log(1 - p)


def _has_hard_threshold(text: str) -> bool:
    """Crude heuristic — a human should re-check the bucketing, this is a first pass."""
    import re

    return bool(re.search(r"\d[\d,.]*\s*%|\bby\b.*\d{4}\b|\$\d|\bat least\b\s*\d|\bmore than\b\s*\d", text, re.I))


def cmd_score(args: argparse.Namespace) -> int:
    results = {r["post_id"]: r for r in json.loads(Path(args.results).read_text())}
    questions = {q["post_id"]: q for q in json.loads(Path(args.questions).read_text())}
    self_resolutions = {}
    if args.self_resolved:
        self_resolutions = {r["post_id"]: r for r in json.loads(Path(args.self_resolved).read_text())}

    rows = []
    self_resolved_rows = []
    unscored = []
    for post_id, r in results.items():
        q = questions.get(post_id)
        if not q:
            continue
        if r.get("error"):
            unscored.append((post_id, q["title"], f"run-oracle error: {r['error']}"))
            continue
        if r.get("oracle_probability") is None:
            unscored.append((post_id, q["title"], f"insufficient_data ({r.get('reason')})"))
            continue
        p = r["oracle_probability"]
        cp = q.get("community_prediction_latest")
        hard_threshold = _has_hard_threshold(q.get("resolution_criteria", "") or q["title"])

        resolution = q.get("resolution")
        if resolution in ("yes", "no", True, False, 1, 0):
            outcome = 1 if resolution in ("yes", True, 1) else 0
            rows.append(
                {
                    "post_id": post_id,
                    "title": q["title"],
                    "oracle_p": p,
                    "community_p": cp,
                    "outcome": outcome,
                    "oracle_brier": _brier(p, outcome),
                    "oracle_log_score": _log_score(p, outcome),
                    "community_brier": _brier(cp, outcome) if cp is not None else None,
                    "community_log_score": _log_score(cp, outcome) if cp is not None else None,
                    "hard_threshold": hard_threshold,
                }
            )
            continue

        # No Metaculus resolution on this token — fall back to our own
        # self-resolved outcome (retro#737) if one was supplied and decisive.
        sr = self_resolutions.get(post_id)
        sr_outcome = sr.get("self_resolution") if sr else None
        if sr_outcome in ("yes", "no"):
            outcome = 1 if sr_outcome == "yes" else 0
            self_resolved_rows.append(
                {
                    "post_id": post_id,
                    "title": q["title"],
                    "oracle_p": p,
                    "outcome": outcome,
                    "oracle_brier": _brier(p, outcome),
                    "oracle_log_score": _log_score(p, outcome),
                    "hard_threshold": hard_threshold,
                }
            )
            continue

        unscored.append((post_id, q["title"], "no resolution on this token — needs Bot Benchmarking Access Tier"))

    print(f"Scored {len(rows)}/{len(results)} questions on Metaculus resolutions; {len(unscored)} unscored.")
    if unscored:
        print("\nUnscored (not silently dropped — see retro#395):")
        for post_id, title, why in unscored:
            print(f"  #{post_id} {title[:70]}: {why}")

    if not rows and not self_resolved_rows:
        print("\nNothing scorable yet. This is expected until the Bot Benchmarking Access Tier is granted — see retro#619.")
        return 1  # a run that measured nothing must never read as a pass

    if not rows:
        print(
            "\nNo Metaculus-resolution-backed rows — only self-resolved fallback data below "
            "(needs the Bot Benchmarking Access Tier for real Metaculus resolutions, see retro#619)."
        )
        _print_self_resolved_summary(self_resolved_rows)
        return 0

    n = len(rows)
    mean_brier = sum(r["oracle_brier"] for r in rows) / n
    mean_log = sum(r["oracle_log_score"] for r in rows) / n
    cp_rows = [r for r in rows if r["community_p"] is not None]
    print(f"\nOracle: mean Brier {mean_brier:.4f}, mean log score {mean_log:.4f} (n={n})")
    if cp_rows:
        cn = len(cp_rows)
        cp_brier = sum(r["community_brier"] for r in cp_rows) / cn
        cp_log = sum(r["community_log_score"] for r in cp_rows) / cn
        # Peer score vs. a single peer (the community aggregate), not Metaculus's
        # true multi-forecaster spot peer score — a documented simplification
        # until we have per-forecaster data, which the API does not expose.
        peer_log = mean_log - cp_log
        print(f"Community: mean Brier {cp_brier:.4f}, mean log score {cp_log:.4f} (n={cn})")
        print(f"Oracul vs community (log score delta, our simplified 'peer score'): {peer_log:+.4f}")

    for label, subset in (("hard-threshold", [r for r in rows if r["hard_threshold"]]), ("loose-worded", [r for r in rows if not r["hard_threshold"]])):
        if subset:
            b = sum(r["oracle_brier"] for r in subset) / len(subset)
            print(f"  {label}: n={len(subset)}, mean Brier {b:.4f}")

    # Calibration buckets
    buckets: dict[int, list[int]] = {b: [] for b in range(0, 100, 10)}
    for r in rows:
        b = min(int(r["oracle_p"] * 10) * 10, 90)
        buckets[b].append(r["outcome"])
    print("\nCalibration (predicted bucket -> observed frequency):")
    for b in sorted(buckets):
        outcomes = buckets[b]
        if outcomes:
            print(f"  [{b}-{b+10}%): n={len(outcomes)}, observed {sum(outcomes)/len(outcomes)*100:.0f}%")

    if self_resolved_rows:
        _print_self_resolved_summary(self_resolved_rows)

    return 0


def _print_self_resolved_summary(self_resolved_rows: list[dict]) -> None:
    """retro#737 — always printed under its own heading, never merged into the
    Metaculus-backed numbers above: this is strictly weaker evidence."""
    n = len(self_resolved_rows)
    mean_brier = sum(r["oracle_brier"] for r in self_resolved_rows) / n
    mean_log = sum(r["oracle_log_score"] for r in self_resolved_rows) / n
    print(f"\n--- Self-resolved (weaker evidence — retro#737, not a Metaculus resolution) ---")
    print(f"Oracle vs. our own settlement: mean Brier {mean_brier:.4f}, mean log score {mean_log:.4f} (n={n})")


# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pf = sub.add_parser("fetch")
    pf.add_argument("--token")
    pf.add_argument("--out", default="metaculus_questions.json")
    pf.add_argument("--limit", type=int, default=50)
    pf.set_defaults(func=cmd_fetch)

    pr = sub.add_parser("run-oracle")
    pr.add_argument("questions")
    pr.add_argument("--out", default="oracle_results.json")
    pr.add_argument("--deployed-commit")
    pr.add_argument("--limit", type=int, default=0)
    pr.add_argument("--article-limit", type=int, default=15)
    pr.set_defaults(func=cmd_run_oracle)

    psr = sub.add_parser("self-resolve")
    psr.add_argument("questions")
    psr.add_argument("--out", default="self_resolutions.json")
    psr.add_argument("--deployed-commit")
    psr.add_argument("--limit", type=int, default=0)
    psr.add_argument("--article-limit", type=int, default=15)
    psr.add_argument("--window-days", type=int, default=14, help="days after actual_close_time to search for outcome evidence")
    psr.add_argument("--confidence", type=float, default=0.9, help="min |p-0.5| distance to call the outcome resolved")
    psr.set_defaults(func=cmd_self_resolve)

    ps = sub.add_parser("score")
    ps.add_argument("results")
    ps.add_argument("questions")
    ps.add_argument("--self-resolved", default=None, help="retro#737 fallback outcomes for questions with no Metaculus resolution")
    ps.set_defaults(func=cmd_score)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
