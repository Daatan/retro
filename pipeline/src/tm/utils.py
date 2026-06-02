"""Shared utilities used across tm.* modules."""

from datetime import datetime


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
    "israel_hayom", "walla", "n12", "maariv", "ch13", "kan11",
    "web_search", "gdelt",
    "bloomberg", "bbc",
    "aljazeera", "nyt", "ft", "guardian", "wapost", "axios",
]
