"""
Resolution-informed shadow scoring — credibility feedback loop, step 3
(docs/ORACLE_VARIABLES.md §9).

Replays every accumulated data/resolution_feedback.jsonl entry through a
fresh OpenSkill model on every ingest, mirroring tm.scorer.Scorer.run()'s
existing replay-from-scratch pattern rather than mutating a persisted Rating
incrementally: Scorer.run() has never worked the incremental way in this
codebase, replay is trivial work at realistic resolution volumes, and a
future scoring-logic fix applies to all history on the next call instead of
only to new data.

Shadow only: writes to resolution_leaderboard.json, a file separate from the
vault-curated leaderboard.json get_credibility_weight() reads. Nothing here
touches get_credibility_weight() or leaderboard.json.

rescore_authors_from_disk() is the author-scoring lane (author-scoring
redesign, Phase 1 step 3): the same replay over each record's author_signals,
keyed per (byline author, outlet) instead of per source, written to its own
resolution_author_leaderboard.json.
"""
import json
import logging
from pathlib import Path

from openskill.models import PlackettLuce

from tm.scorer import brier_score, stance_to_prob

logger = logging.getLogger(__name__)


def _load_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records: list[dict] = []
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            logger.warning("Skipping malformed resolution-feedback line %d in %s", lineno, path)
    return records


def rescore_from_disk(ingest_path: Path, output_path: Path) -> dict:
    """
    Replay every ingested resolution through a fresh PlackettLuce model and
    write the resulting per-source shadow scores to output_path. Returns a
    summary dict (for logging) of how many resolutions/sources were scored
    or skipped and why. Never raises — a malformed line is logged and
    skipped by _load_records, not fatal to the whole rescore.
    """
    records = _load_records(ingest_path)

    skill_model = PlackettLuce()
    ratings: dict[str, object] = {}
    stats: dict[str, dict] = {}
    resolutions_scored = 0
    resolutions_skipped_incomplete = 0
    resolutions_single_source = 0
    articles_skipped_opinion = 0

    for record in records:
        outcome = record.get("outcome")
        sources = record.get("sources", [])
        if outcome is None or not sources:
            resolutions_skipped_incomplete += 1
            continue

        # Same exclusion policy as the credibility feedback loop's Q7
        # decision — enforced here too, not just trusted from the caller:
        # daatan already filters opinion-class before pushing, but a future
        # caller of /leaderboard/ingest shouldn't be able to silently bypass
        # the rule by omission.
        usable = [
            s for s in sources
            if s.get("evidence_class") != "opinion" and s.get("stance") is not None and s.get("source")
        ]
        articles_skipped_opinion += sum(1 for s in sources if s.get("evidence_class") == "opinion")
        if not usable:
            resolutions_skipped_incomplete += 1
            continue

        resolutions_scored += 1
        if len(usable) < 2:
            resolutions_single_source += 1  # still scored below (brier), just no ranking counterparty

        event_predictions: list[tuple[str, float]] = []
        for s in usable:
            sid = s["source"]
            stance = s["stance"]
            if sid not in ratings:
                ratings[sid] = skill_model.rating()
                stats[sid] = {"predictions": 0, "brier_total": 0.0}
            stats[sid]["predictions"] += 1
            stats[sid]["brier_total"] += brier_score(stance_to_prob(stance), outcome)
            event_predictions.append((sid, stance))

        # A lone source (or a unanimous group) has no ranking counterparty —
        # same "all right or all wrong, no information" case Scorer.run()
        # hits — so OpenSkill doesn't move, but brier/predictions above still
        # captured its plain accuracy.
        winners = [sid for sid, stance in event_predictions if (stance > 0) == outcome]
        losers = [sid for sid, stance in event_predictions if (stance > 0) != outcome]
        if winners and losers:
            winner_teams = [[ratings[sid]] for sid in winners]
            loser_teams = [[ratings[sid]] for sid in losers]
            ranks = [0] * len(winner_teams) + [1] * len(loser_teams)
            new_ratings = skill_model.rate(winner_teams + loser_teams, ranks=ranks)
            for i, sid in enumerate(winners):
                ratings[sid] = new_ratings[i][0]
            for i, sid in enumerate(losers):
                ratings[sid] = new_ratings[len(winners) + i][0]

    shadow_leaderboard = []
    for sid, s in stats.items():
        n = s["predictions"]
        r = ratings[sid]
        shadow_leaderboard.append({
            "id": sid,
            "brier_score": round(s["brier_total"] / n, 4),
            "skill_mu": round(r.mu, 2),
            "skill_sigma": round(r.sigma, 2),
            "skill_conservative": round(r.mu - 3 * r.sigma, 2),
            "predictions": n,
        })
    shadow_leaderboard.sort(key=lambda x: x["skill_conservative"], reverse=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(shadow_leaderboard, indent=2))

    summary = {
        "resolutions_total": len(records),
        "resolutions_scored": resolutions_scored,
        "resolutions_skipped_incomplete": resolutions_skipped_incomplete,
        "resolutions_single_source": resolutions_single_source,
        "articles_skipped_opinion": articles_skipped_opinion,
        "sources_scored": len(shadow_leaderboard),
    }
    logger.info("event=credibility_feedback_scored %s", summary)
    return summary


NO_BYLINE = "(no byline)"


def _normalize_identity(value) -> str:
    return " ".join(value.split()) if value else ""


def rescore_authors_from_disk(ingest_path: Path, output_path: Path) -> dict:
    """
    Author-scoring lane: replay every ingested resolution's author_signals
    through a fresh PlackettLuce model and write per-(byline author, outlet)
    shadow scores to output_path. Same replay-from-scratch pattern as
    rescore_from_disk — which also means a future byline-identity merge (the
    known HE/EN duplicates and byline-parse artifacts) applies to all history
    on the next call, no re-ingest needed; until then keys are the raw
    whitespace-normalized byline strings.

    Two deliberate departures from the stance lane: opinion-class rows are
    INCLUDED (author_lean is the author's own lean — opinion is the signal,
    not noise), and within one resolution an author's rows are averaged first
    so a prolific author is scored once per outcome, not once per article.
    """
    records = _load_records(ingest_path)

    skill_model = PlackettLuce()
    ratings: dict[tuple[str, str], object] = {}
    stats: dict[tuple[str, str], dict] = {}
    resolutions_scored = 0
    resolutions_without_signals = 0

    for record in records:
        outcome = record.get("outcome")
        signals = record.get("author_signals") or []
        usable = [s for s in signals if s.get("author_lean") is not None]
        if outcome is None or not usable:
            resolutions_without_signals += 1
            continue

        leans_by_key: dict[tuple[str, str], list[float]] = {}
        for s in usable:
            key = (_normalize_identity(s.get("author")) or NO_BYLINE, _normalize_identity(s.get("outlet_name")))
            leans_by_key.setdefault(key, []).append(s["author_lean"])

        resolutions_scored += 1
        event_predictions: list[tuple[tuple[str, str], float]] = []
        for key, leans in leans_by_key.items():
            mean_lean = sum(leans) / len(leans)
            if key not in ratings:
                ratings[key] = skill_model.rating()
                stats[key] = {"predictions": 0, "articles": 0, "brier_total": 0.0}
            stats[key]["predictions"] += 1
            stats[key]["articles"] += len(leans)
            stats[key]["brier_total"] += brier_score(stance_to_prob(mean_lean), outcome)
            event_predictions.append((key, mean_lean))

        winners = [k for k, lean in event_predictions if (lean > 0) == outcome]
        losers = [k for k, lean in event_predictions if (lean > 0) != outcome]
        if winners and losers:
            winner_teams = [[ratings[k]] for k in winners]
            loser_teams = [[ratings[k]] for k in losers]
            ranks = [0] * len(winner_teams) + [1] * len(loser_teams)
            new_ratings = skill_model.rate(winner_teams + loser_teams, ranks=ranks)
            for i, k in enumerate(winners):
                ratings[k] = new_ratings[i][0]
            for i, k in enumerate(losers):
                ratings[k] = new_ratings[len(winners) + i][0]

    author_board = []
    for (author, outlet), s in stats.items():
        r = ratings[(author, outlet)]
        author_board.append({
            "id": f"{author} — {outlet}" if outlet else author,
            "author": author,
            "outlet_name": outlet or None,
            "brier_score": round(s["brier_total"] / s["predictions"], 4),
            "skill_mu": round(r.mu, 2),
            "skill_sigma": round(r.sigma, 2),
            "skill_conservative": round(r.mu - 3 * r.sigma, 2),
            "predictions": s["predictions"],
            "articles": s["articles"],
        })
    author_board.sort(key=lambda x: x["skill_conservative"], reverse=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # ensure_ascii=False: byline keys are routinely Hebrew/Arabic — keep the
    # shadow file human-inspectable on the box.
    output_path.write_text(json.dumps(author_board, indent=2, ensure_ascii=False))

    summary = {
        "resolutions_total": len(records),
        "resolutions_scored": resolutions_scored,
        "resolutions_without_signals": resolutions_without_signals,
        "authors_scored": len(author_board),
    }
    logger.info("event=author_shadow_scored %s", summary)
    return summary


def load_shadow_leaderboard(output_path: Path) -> list[dict]:
    if not output_path.exists():
        return []
    return json.loads(output_path.read_text())
