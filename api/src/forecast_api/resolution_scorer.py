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
import re
from pathlib import Path

import httpx
from openskill.models import PlackettLuce

from tm.scorer import brier_score, stance_to_prob
from tm.web_search import NEWS_INDEXER_API_KEY, NEWS_INDEXER_URL

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


def _usable_sources(record: dict) -> list[dict]:
    """The rows rescore_from_disk() actually scores for one record.

    Opinion-class rows are excluded — the same exclusion policy as the
    credibility feedback loop's Q7 decision, enforced here rather than merely
    trusted from the caller: daatan already filters opinion-class before
    pushing, but a future /leaderboard/ingest caller shouldn't be able to
    silently bypass the rule by omission. Rows without a stance or a source id
    are unscoreable and dropped too. Returns [] for an incomplete record.

    Shared with count_resolutions() so the credibility gate counts exactly the
    resolutions that were scored, and the two can't drift apart.
    """
    if record.get("outcome") is None or not record.get("sources"):
        return []
    return [
        s for s in record["sources"]
        if s.get("evidence_class") != "opinion" and s.get("stance") is not None and s.get("source")
    ]


def count_resolutions(ingest_path: Path) -> int:
    """How many ingested resolutions are actually scoreable — the global gate
    for resolution-shadow credibility (settings.resolution_shadow_min_global_predictions).
    Counts resolutions, not source rows or articles."""
    return sum(1 for record in _load_records(ingest_path) if _usable_sources(record))


def archetype_base_rate(
    ingest_path: Path,
    archetype: str,
    *,
    prior_p: float,
    prior_n: float,
) -> tuple[float, int]:
    """Resolved base rate for one claim archetype, shrunk toward ``prior_p``.

    ``(successes + prior_n * prior_p) / (n + prior_n)`` — the same shrinkage
    construction as ``resolution_shadow_brier_prior_n`` (config.py), and for
    the same reason: it degrades smoothly rather than at a minimum-n cliff. The
    retro#356 hazard needs a base rate to drift toward, and demanding a
    calibrated one before shipping anything would block the shadow
    indefinitely — with zero resolved claims of this archetype the rate simply
    IS ``prior_p``, and it earns its way toward the empirical value as
    resolutions land.

    Returns ``(rate, n)`` where ``n`` counts records actually matching the
    archetype, so a caller can report how much of the rate is real evidence
    rather than prior. Records with no ``claim_archetype`` (every push that
    predates daatan sending it) match nothing and are excluded — absence is not
    evidence for any particular archetype.

    Unlike :func:`count_resolutions` this deliberately does NOT filter on
    ``_usable_sources``: that gate asks "can this record score per-source
    credibility", which needs usable source rows, while a base rate asks only
    "how did the claim resolve" — a resolution with no scoreable sources still
    resolved, and dropping it would bias the rate toward claims that happened
    to attract usable coverage.
    """
    want = (archetype or "").strip().lower()
    n = 0
    successes = 0
    for record in _load_records(ingest_path):
        if (record.get("claim_archetype") or "").strip().lower() != want:
            continue
        outcome = record.get("outcome")
        if outcome is None:
            continue
        n += 1
        if outcome is True:
            successes += 1
    denom = n + prior_n
    if denom <= 0:
        return prior_p, n
    return (successes + prior_n * prior_p) / denom, n


def rescore_from_disk(ingest_path: Path, output_path: Path) -> dict:
    """
    Replay every ingested resolution through a fresh PlackettLuce model and
    write the resulting per-source shadow scores to output_path. Returns a
    summary dict (for logging) of how many resolutions/sources were scored
    or skipped and why. Never raises — a malformed line is logged and
    skipped by _load_records, not fatal to the whole rescore.

    Within one resolution a source's rows are averaged before scoring, so a
    source is scored once per outcome rather than once per article — the same
    shape rescore_authors_from_disk() already uses. This is not cosmetic: the
    row-level version put a source with mixed-sign rows into BOTH `winners`
    and `losers`, so it competed against itself in rate() and the loser
    write-back then clobbered its winner update (28 such source-resolutions
    across the first 6 real ingests). It also made `predictions` count
    articles, overstating how much independent evidence a prolific source
    had — the count downstream shrinkage divides by.
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

        articles_skipped_opinion += sum(1 for s in sources if s.get("evidence_class") == "opinion")
        usable = _usable_sources(record)
        if not usable:
            resolutions_skipped_incomplete += 1
            continue

        stances_by_source: dict[str, list[float]] = {}
        for s in usable:
            stances_by_source.setdefault(s["source"], []).append(s["stance"])

        resolutions_scored += 1
        if len(stances_by_source) < 2:
            resolutions_single_source += 1  # still scored below (brier), just no ranking counterparty

        event_predictions: list[tuple[str, float]] = []
        for sid, stances in stances_by_source.items():
            mean_stance = sum(stances) / len(stances)
            if sid not in ratings:
                ratings[sid] = skill_model.rating()
                stats[sid] = {"predictions": 0, "articles": 0, "brier_total": 0.0}
            stats[sid]["predictions"] += 1
            stats[sid]["articles"] += len(stances)
            stats[sid]["brier_total"] += brier_score(stance_to_prob(mean_stance), outcome)
            event_predictions.append((sid, mean_stance))

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
            "articles": s["articles"],
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

# Hebrew cantillation/niqqud marks (U+0591-U+05C7) and geresh/gershayim
# (U+05F3-U+05F4) — stripped so e.g. "שילה פריד" and "שילֹה פריד" (a stray
# gershayim mark) normalize to the same key. The exact case ni#161 hand-
# curated for Ynet/Ynetnews before an identity map existed to catch it.
_HEBREW_DIACRITICS_RE = re.compile("[֑-ׇ׳״]")


def _normalize_identity(value) -> str:
    if not value:
        return ""
    return " ".join(_HEBREW_DIACRITICS_RE.sub("", value).split())


def _load_identity_map() -> dict[str, str]:
    """Fetch news-indexer's curated Person/PersonAlias identity map (GET
    /authors/admin/people) and flatten it into {normalized_alias: canonical_name}.

    Fails open to {} on any error — missing config, timeout, non-200,
    malformed response — so author grouping falls back to raw normalized
    byline strings, same fail-open convention as every other news-indexer-
    backed dependency in this codebase (retro/CLAUDE.md's _secret() note).
    Currently a small (~dozen), slowly-growing curated set, so no caching:
    one GET per rescore call is negligible next to the replay itself.
    """
    if not (NEWS_INDEXER_URL and NEWS_INDEXER_API_KEY):
        return {}
    try:
        r = httpx.get(
            f"{NEWS_INDEXER_URL}/authors/admin/people",
            headers={"x-api-key": NEWS_INDEXER_API_KEY},
            timeout=httpx.Timeout(connect=2.0, read=5.0, write=2.0, pool=2.0),
        )
        r.raise_for_status()
        people = r.json().get("people", [])
        identity_map: dict[str, str] = {}
        for person in people:
            canonical = person.get("canonical_name")
            if not canonical:
                continue
            for alias_row in person.get("aliases", []):
                normalized = _normalize_identity(alias_row.get("alias"))
                if normalized:
                    identity_map[normalized] = canonical
        return identity_map
    except Exception as exc:
        logger.debug("news_indexer identity map fetch failed (non-fatal): %s", exc)
        return {}


def rescore_authors_from_disk(ingest_path: Path, output_path: Path) -> dict:
    """
    Author-scoring lane: replay every ingested resolution's author_signals
    through a fresh PlackettLuce model and write per-(byline author, outlet)
    shadow scores to output_path. Same replay-from-scratch pattern as
    rescore_from_disk — which also means the byline-identity merge below
    applies to all history on the next call, no re-ingest needed.

    Byline identity: raw bylines are grouped through news-indexer's curated
    Person/PersonAlias map (_load_identity_map()) before keying — the known
    HE/EN duplicates and byline-parse artifacts collapse into one row when
    curated there. An unmatched raw byline still keys on its own
    whitespace-normalized string, same as before the map existed.

    Two deliberate departures from the stance lane: opinion-class rows are
    INCLUDED (author_lean is the author's own lean — opinion is the signal,
    not noise), and within one resolution an author's rows are averaged first
    so a prolific author is scored once per outcome, not once per article.
    """
    records = _load_records(ingest_path)
    identity_map = _load_identity_map()

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
            raw_author = _normalize_identity(s.get("author"))
            author = identity_map.get(raw_author, raw_author) if raw_author else NO_BYLINE
            key = (author, _normalize_identity(s.get("outlet_name")))
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
