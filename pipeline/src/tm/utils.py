"""Shared utilities used across tm.* modules."""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path


def existing_articles(cell_dir: Path) -> list[Path]:
    """List already-ingested ``article_*.json`` files in a cell directory.

    Every batch ingestor (gdelt, gnews, site_search, web_search) opens with the
    same cache-check idiom — ``if existing and not force: return len(existing)``
    — and several reuse the count to continue article numbering. This centralises
    the glob (and the missing-directory guard); callers keep their own
    force/numbering logic, which legitimately differs between ingestors.
    """
    return list(cell_dir.glob("article_*.json")) if cell_dir.exists() else []


def save_article(cell_dir: Path, idx: int, article: dict) -> Path:
    """Write one article to ``cell_dir/article_{idx:02d}.json`` and return its path.

    Creates ``cell_dir`` if needed. The pretty-printed, ``ensure_ascii=False``
    JSON dump was repeated at every ingestor save site (gnews alone had five);
    the per-ingestor index logic stays with the caller, which is where it
    legitimately differs.
    """
    cell_dir.mkdir(parents=True, exist_ok=True)
    out = cell_dir / f"article_{idx:02d}.json"
    out.write_text(json.dumps(article, indent=2, ensure_ascii=False))
    return out


# Deliberately NOT ``*.json``: ``infra/ec2_run.sh`` and
# ``Orchestrator.local_file_search`` both glob ``*.json`` in a cell directory to
# find articles, and pathlib's glob (unlike the shell's) matches dotfiles — a
# ``.empty.json`` would be counted, and then parsed, as an article.
EMPTY_MARKER_NAME = ".empty"
EMPTY_MARKER_TTL_DAYS = 30
# A window that closed only days ago can still gain results as indexes catch up,
# so its emptiness is not final yet and is not recorded.
EMPTY_MARKER_WINDOW_GRACE_DAYS = 7


def cell_marked_empty(cell_dir: Path, window_end: datetime, ttl_days: int | None = None) -> bool:
    """True if a previous run searched this cell, found nothing, and that is still fresh.

    The ``existing_articles`` idiom only remembers cells that *saved* something,
    so an empty cell re-walked its whole provider ladder — paid SERP legs
    included — on every batch cycle (retro#836). The marker expires after
    ``ttl_days`` (env ``EMPTY_CELL_TTL_DAYS``, default 30) so a new provider or a
    keyword change eventually gets a second look; ``--force`` callers skip this
    check altogether. A marker written for a different ``window_end`` (an edited
    ``outcome_date``, another ``--t-days``) answers a different question and is
    ignored, as is an unreadable one.
    """
    if ttl_days is None:
        ttl_days = int(os.environ.get("EMPTY_CELL_TTL_DAYS", EMPTY_MARKER_TTL_DAYS))
    marker = cell_dir / EMPTY_MARKER_NAME
    try:
        data = json.loads(marker.read_text())
        checked_at = datetime.fromisoformat(data["checked_at"])
        if data["window_end"] != window_end.strftime("%Y-%m-%d"):
            return False
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return datetime.now() - checked_at < timedelta(days=ttl_days)


def mark_cell_empty(cell_dir: Path, window_end: datetime, **meta) -> bool:
    """Record that a completed search of this cell saved nothing. Returns True if written.

    Call it only on the normal completion path — never from an exception handler,
    or an outage turns into a month of "nothing to find". Skipped while the search
    window is still open (or closed less than ``EMPTY_MARKER_WINDOW_GRACE_DAYS``
    ago): only a historical window's empty answer is final.
    """
    if window_end + timedelta(days=EMPTY_MARKER_WINDOW_GRACE_DAYS) > datetime.now():
        return False
    cell_dir.mkdir(parents=True, exist_ok=True)
    payload = {"checked_at": datetime.now().isoformat(timespec="seconds"),
               "window_end": window_end.strftime("%Y-%m-%d"), **meta}
    (cell_dir / EMPTY_MARKER_NAME).write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return True


def _is_number(v) -> bool:
    """True for a real numeric value. Excludes bool (a subclass of int)."""
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def split_scored_predictions(preds: list[dict]) -> tuple[list[dict], list[dict]]:
    """Partition predictions into (usable, malformed) for scoring.

    Scoring requires a numeric ``stance`` and ``certainty`` on every prediction
    — the extractor's Pydantic model (PredictionExtraction) guarantees them, so
    a missing/non-numeric value here means upstream corruption or a schema
    regression. Callers must NOT silently substitute a neutral default
    (stance=0, certainty=0.5): that would score a broken prediction as a
    legitimate neutral one and quietly poison the leaderboard. Instead they log
    the malformed ones loudly and skip them.

    Optional fields (specificity, hedge_ratio, …) are intentionally not checked
    — they are declared Optional and a default for them is correct, not a bug.
    """
    usable: list[dict] = []
    malformed: list[dict] = []
    for p in preds:
        if _is_number(p.get("stance")) and _is_number(p.get("certainty")):
            usable.append(p)
        else:
            malformed.append(p)
    return usable, malformed


def predates_outcome(article_date: str, outcome_date: str) -> bool:
    """Anti-lookahead guard: True if the article is known to predate the outcome.

    Scoring must only count predictions published on/before the event's
    outcome date — otherwise a post-event "prediction" leaks future knowledge
    into the source's Brier/credibility score (which the live Oracul reads).

    Returns False *only* when the article date parses and is strictly after the
    outcome date. Missing/unparseable dates return True (conservative — don't
    silently drop entries we can't evaluate; the ingest-time filters are
    responsible for undated articles). Compares on the date (first 10 chars).
    """
    if not article_date or not outcome_date:
        return True
    try:
        art_dt = datetime.fromisoformat(article_date[:10])
        evt_dt = datetime.fromisoformat(outcome_date[:10])
    except (ValueError, TypeError):
        return True
    return art_dt <= evt_dt


def _is_ascii(s: str) -> bool:
    try:
        s.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


# Source IDs the orchestrator recognises as named-source cells.
# Each entry must have a matching data/sources/{id}.json file.
KNOWN_SOURCE_IDS: list[str] = [
    "ynet", "haaretz", "haaretz_he", "toi", "globes", "reuters", "jpost",
    "israel_hayom", "walla", "n12", "maariv", "ch13", "calcalist",
    "bloomberg", "bbc",
    "aljazeera", "nyt", "ft", "guardian", "axios",
    # kan11:      TV-only, no indexable web article corpus (Phase 2)
    # wapost:     hard paywall, scraping returns subscription walls
    # web_search: meta-source / synthetic, not a scoreable news outlet
    # gdelt:      synthetic aggregator, not a scoreable news outlet
]
