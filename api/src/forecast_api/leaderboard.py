"""
Leaderboard cache — loads leaderboard.json at startup and refreshes every N seconds.

The leaderboard is written by the batch pipeline (render_atlas.py / scorer.py)
and read here to weight source credibility during forecast aggregation.
"""
import asyncio
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# source_id → leaderboard entry dict
_cache: dict[str, dict] = {}
_cache_lock = asyncio.Lock()


def _load_from_disk(path: Path) -> dict[str, dict]:
    if not path.exists():
        logger.warning("leaderboard.json not found at %s — all sources get neutral weight", path)
        return {}
    data = json.loads(path.read_text())
    # Support both list format and dict format
    if isinstance(data, list):
        return {entry["id"]: entry for entry in data if "id" in entry}
    if isinstance(data, dict) and "sources" in data:
        return {entry["id"]: entry for entry in data["sources"] if "id" in entry}
    return {}


async def refresh_cache(path: Path) -> None:
    loaded = await asyncio.to_thread(_load_from_disk, path)
    async with _cache_lock:
        _cache.clear()
        _cache.update(loaded)
    logger.info("Leaderboard refreshed: %d sources loaded", len(_cache))


async def background_refresh_loop(path: Path, interval_seconds: int) -> None:
    while True:
        await asyncio.sleep(interval_seconds)
        try:
            await refresh_cache(path)
        except Exception as exc:
            logger.error("Leaderboard refresh failed: %s", exc)


def get_credibility_weight(source_id: str) -> float:
    """
    Credibility weight derived from TrueSkill conservative estimate (μ − 3σ).
    Falls back to ELO/Brier if TrueSkill fields are absent.
    Sources absent from leaderboard get neutral weight 1.0.

    TrueSkill conservative score: higher = more trusted.
    We normalise so that the default uninformed prior (μ=25, σ=8.33 → conservative≈0)
    maps to weight 1.0.

    Formula: max(0.1, 1.0 + conservative / 25.0)
    - conservative =  0  → weight = 1.0  (new/unknown source)
    - conservative = 10  → weight = 1.4  (trusted)
    - conservative = -5  → weight = 0.8  (distrusted)
    """
    entry = _cache.get(source_id)
    if entry is None:
        return 1.0

    if "trueskill_conservative" in entry:
        conservative = float(entry["trueskill_conservative"])
        return max(0.1, 1.0 + conservative / 25.0)

    # Fallback: ELO / Brier
    elo = float(entry.get("elo", 1200.0))
    brier = float(entry.get("brier_score", 0.25))
    return elo / (1200.0 + brier * 200.0)


def leaderboard_size() -> int:
    return len(_cache)


def get_leaderboard_data() -> list[dict]:
    """Return a snapshot of all leaderboard entries, sorted by TrueSkill conservative score."""
    entries = list(_cache.values())
    entries.sort(key=lambda e: float(e.get("trueskill_conservative", 0.0)), reverse=True)
    return entries
