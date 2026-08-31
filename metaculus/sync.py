#!/usr/bin/env python3
"""Submit Daatan Oracul forecasts into a Metaculus tournament.

Usage:
    cd /home/mark/projects/retro
    METACULUS_API_KEY=... ORACLE_API_KEY=... python metaculus/sync.py

Env vars:
    METACULUS_API_KEY      bot token for the target Metaculus bot account (required)
    ORACLE_API_KEY          named Oracle API key for this relay (required) — see oracle_client.py
    ORACLE_BASE_URL          default "https://oracle.daatan.com"
    METACULUS_TOURNAMENT      default "bot-testing-area" — deliberately NOT a
                              scored tournament. Point this at a real AIB/MiniBench
                              slug only once a clean run history exists here.
                              "auto" picks the current FutureEval/AIB season by
                              itself (retro#726): the latest-started tournament
                              whose slug or name says futureeval/aib and whose
                              forecasting window is still open. Seasons start
                              every September, January and May and there is no
                              per-season registration, so nobody has to watch
                              metaculus.com for the new slug. If no season is
                              open the run logs that and exits without doing
                              anything.
    MAX_QUESTIONS_PER_RUN     default 5 — caps Oracul calls per run (each one
                              costs real LLM spend and can take minutes).
    STALE_AFTER_HOURS          default 4 — re-forecast a question only if our
                              last submission is older than this, or missing.
                              Set ABOVE the question window on purpose, so we
                              submit exactly once: the FutureEval rules say
                              "bot makers should only submit one forecast per
                              question in these bot-only tournaments"
                              (retro#755). An earlier 0.75 default deliberately
                              produced ~2 forecasts to exploit the spot score —
                              correct for the score, against the rules.
                              The window is 3.0h, measured 2026-08-30 across all
                              60 questions of the current MiniBench round (the
                              1.5h in Metaculus's own docs is stale), so 4h
                              clears it with margin.
    DRY_RUN                   "true" to log what would be submitted without
                              calling Metaculus's write endpoints.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone

from metaculus_client import MetaculusClient
from oracle_client import OracleClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("metaculus-sync")


AUTO_TOURNAMENT = "auto"
_SEASON_MARKERS = ("futureeval", "aib")


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def select_season_tournament(tournaments: list[dict], now: datetime) -> str | None:
    """Slug of the FutureEval/AIB season to forecast in right now, or None.

    A season counts when its slug or name carries a marker (``futureeval``,
    ``aib``), it has started, and its ``forecasting_end_date`` (the day questions
    stop opening — ``close_date`` is months later, after resolution) is still
    ahead. Summer's window ends ~a week after Fall opens, so on the overlap the
    **latest-started** season wins; MiniBench is a separate project type and is
    never matched here.
    """
    best: tuple[datetime, str] | None = None
    for t in tournaments:
        slug = str(t.get("slug") or "")
        haystack = f"{slug} {t.get('name') or ''}".lower()
        if not slug or not any(m in haystack for m in _SEASON_MARKERS):
            continue
        start = _parse_ts(t.get("start_date"))
        end = _parse_ts(t.get("forecasting_end_date")) or _parse_ts(t.get("close_date"))
        if start is None or start > now or (end is not None and end <= now):
            continue
        if best is None or start > best[0]:
            best = (start, slug)
    return best[1] if best else None


def _parse_forecast_time(value: object) -> datetime | None:
    """Read `my_forecasts.latest.start_time`, which is NOT an ISO string.

    Metaculus returns this one field as a float epoch in seconds
    (`1788083931.233327`), unlike every other timestamp in the API. Until
    2026-08-30 the bot had never forecast anything, so `latest` was always
    `None` and this branch never ran — the first real submission turned every
    subsequent poll into `AttributeError: 'float' object has no attribute
    'replace'`, which aborts the whole run, not just that question (retro#727).

    Both shapes are accepted: the payload is undocumented and older code read a
    `timestamp` key that may still arrive as a string.
    """
    if isinstance(value, bool):  # bool is an int subclass; not a timestamp
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(value, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def needs_forecast(question: dict, stale_after: timedelta, now: datetime) -> bool:
    latest = (question.get("my_forecasts") or {}).get("latest")
    if not latest:
        return True
    start_time = latest.get("start_time") or latest.get("timestamp")
    if not start_time:
        return True
    forecast_time = _parse_forecast_time(start_time)
    if forecast_time is None:
        # A forecast exists but cannot be dated. Skip rather than re-submit:
        # bot-only tournaments ask for one forecast per question (retro#755),
        # so a duplicate is a worse failure than a missed refresh.
        log.warning(
            "Undatable forecast timestamp %r on question %s — skipping",
            start_time, question.get("id"),
        )
        return False
    return now - forecast_time > stale_after


def question_text(post: dict, question: dict) -> tuple[str, str | None]:
    title = post.get("title") or question.get("title") or ""
    criteria_parts = [
        part
        for part in (question.get("description"), question.get("resolution_criteria"), question.get("fine_print"))
        if part
    ]
    criteria = "\n\n".join(criteria_parts) if criteria_parts else None
    return title, criteria


def build_comment(result: dict, probability: float) -> str:
    rationale = result.get("reason") or "Forecast from Daatan Oracul."
    comment = (
        f"{rationale}\n\nOracle p={probability:.2f} "
        f"(mean stance {result.get('mean', 0):.3f}, {result.get('articles_used', 0)} articles)."
    )
    sources = result.get("sources") or []
    if sources:
        comment += "\n\nSources: " + ", ".join(str(s) for s in sources[:5])
    return comment


def run(
    metaculus_token: str,
    oracle_api_key: str,
    tournament: str,
    oracle_base_url: str,
    max_questions: int,
    stale_after: timedelta,
    dry_run: bool = False,
) -> int:
    processed = 0
    now = datetime.now(timezone.utc)
    with MetaculusClient(metaculus_token) as mc, OracleClient(oracle_api_key, base_url=oracle_base_url) as oc:
        if tournament == AUTO_TOURNAMENT:
            resolved = select_season_tournament(mc.list_tournaments(), now)
            if resolved is None:
                log.info("METACULUS_TOURNAMENT=auto: no FutureEval/AIB season is open right now, nothing to do")
                return 0
            log.info("METACULUS_TOURNAMENT=auto resolved to %s", resolved)
            tournament = resolved
        candidates = mc.open_binary_questions(tournament, limit=max_questions * 3)
        for post in candidates:
            if processed >= max_questions:
                log.info("Reached MAX_QUESTIONS_PER_RUN=%d, stopping", max_questions)
                break
            post_id = post["id"]
            detail = mc.get_question(post_id)
            question = detail["question"]
            if not needs_forecast(question, stale_after, now):
                continue

            title, criteria = question_text(detail, question)
            log.info("Forecasting post %d: %s", post_id, title[:80])
            try:
                result = oc.forecast(title, resolution_criteria=criteria)
            except Exception:
                log.exception("Oracul forecast failed for post %d, skipping", post_id)
                continue

            if result.get("insufficient_data") or result.get("mean") is None:
                log.info("Oracul returned insufficient_data for post %d, skipping", post_id)
                continue

            probability = (result["mean"] + 1) / 2
            if dry_run:
                log.info("[dry-run] would submit p=%.3f for post %d", probability, post_id)
                processed += 1
                continue

            question_id = question["id"]
            mc.submit_binary_forecast(question_id, probability)
            mc.post_comment(post_id, build_comment(result, probability), private=True)
            processed += 1

    log.info("Done — forecasted %d question(s)", processed)
    return processed


def _required_env(name: str) -> str:
    """Fail with the variable's *name* when it is unset or blank.

    An unset GitHub secret is passed to the job as an empty string, not a
    missing variable — so ``os.environ[name]`` succeeds and the failure surfaces
    much later as httpx's ``Illegal header value b'Token '`` (retro#727 dry run,
    2026-08-29). Say which secret is missing instead.
    """
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"{name} is unset or empty — check the workflow's repo secrets")
    return value


def main() -> None:
    metaculus_token = _required_env("METACULUS_API_KEY")
    oracle_api_key = _required_env("ORACLE_API_KEY")
    tournament = os.environ.get("METACULUS_TOURNAMENT", "bot-testing-area")
    oracle_base_url = os.environ.get("ORACLE_BASE_URL", "https://oracle.daatan.com")
    max_questions = int(os.environ.get("MAX_QUESTIONS_PER_RUN", "5"))
    stale_after = timedelta(hours=float(os.environ.get("STALE_AFTER_HOURS", "4")))
    dry_run = os.environ.get("DRY_RUN", "").lower() in ("1", "true", "yes")

    run(metaculus_token, oracle_api_key, tournament, oracle_base_url, max_questions, stale_after, dry_run=dry_run)


if __name__ == "__main__":
    main()
