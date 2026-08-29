#!/usr/bin/env python3
"""Submit Daatan Oracle forecasts into a Metaculus tournament.

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
    MAX_QUESTIONS_PER_RUN     default 5 — caps Oracle calls per run (each one
                              costs real LLM spend and can take minutes).
    STALE_AFTER_HOURS          default 24 — re-forecast a question only if our
                              last submission is older than this, or missing.
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


def needs_forecast(question: dict, stale_after: timedelta, now: datetime) -> bool:
    latest = (question.get("my_forecasts") or {}).get("latest")
    if not latest:
        return True
    start_time = latest.get("start_time") or latest.get("timestamp")
    if not start_time:
        return True
    forecast_time = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
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
    rationale = result.get("reason") or "Forecast from Daatan Oracle."
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
                log.exception("Oracle forecast failed for post %d, skipping", post_id)
                continue

            if result.get("insufficient_data") or result.get("mean") is None:
                log.info("Oracle returned insufficient_data for post %d, skipping", post_id)
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


def main() -> None:
    metaculus_token = os.environ["METACULUS_API_KEY"]
    oracle_api_key = os.environ["ORACLE_API_KEY"]
    tournament = os.environ.get("METACULUS_TOURNAMENT", "bot-testing-area")
    oracle_base_url = os.environ.get("ORACLE_BASE_URL", "https://oracle.daatan.com")
    max_questions = int(os.environ.get("MAX_QUESTIONS_PER_RUN", "5"))
    stale_after = timedelta(hours=float(os.environ.get("STALE_AFTER_HOURS", "24")))
    dry_run = os.environ.get("DRY_RUN", "").lower() in ("1", "true", "yes")

    run(metaculus_token, oracle_api_key, tournament, oracle_base_url, max_questions, stale_after, dry_run=dry_run)


if __name__ == "__main__":
    main()
