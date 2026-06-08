"""Shared utilities used across tm.* modules."""

import json
from datetime import datetime
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
    into the source's Brier/credibility score (which the live Oracle reads).

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
