"""
Leaderboard cache — loads leaderboard.json at startup and refreshes every N seconds.

The leaderboard is written by the batch pipeline (render_atlas.py / scorer.py)
and read here to weight source credibility during forecast aggregation.

Two boards live here, and which one feeds get_credibility_weight() is decided by
settings.resolution_shadow_credibility_enabled (docs/ORACLE_VARIABLES.md §9):

- leaderboard.json — the LEGACY vault, scored by tm.scorer.Scorer.run() against
  the hand-curated data/events/*.json backtest. Nothing in production has
  regenerated it since 2026-03-28 (no cron, timer or workflow calls Scorer.run()),
  and it only ever covered 5 Israeli outlets, so in practice it returns neutral
  1.0 for almost every live source. Retained solely as the flag-OFF path.
- resolution_leaderboard.json — the resolution-informed shadow board, replayed
  from real daatan resolutions by resolution_scorer.rescore_from_disk() on every
  ingest. The flag-ON path.
"""
import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from .config import settings
from .resolution_scorer import count_resolutions, load_shadow_leaderboard

logger = logging.getLogger(__name__)

# source_id → leaderboard entry dict
_cache: dict[str, dict] = {}
# mtime of leaderboard.json as of the last successful refresh_cache(), or None
# if the file didn't exist. Surfaced via leaderboard_snapshot_date() so callers
# can tell how stale the vault is (retro#340) instead of trusting a "live"
# label that stopped being true once nothing regenerated the file.
_snapshot_mtime: float | None = None
# source_id → resolution-shadow entry dict, plus the global scoreable-resolution
# count that gates it. Kept separate from _cache rather than merged: the two
# boards have different provenance and different lifecycles, and the flag has to
# be able to fall back to the vault untouched.
_shadow_cache: dict[str, dict] = {}
_shadow_total_resolutions: int = 0
# One lock guards both — they are only ever refreshed together, on one event loop.
_cache_lock = asyncio.Lock()

# Age (days) past which leaderboard.json is flagged as stale on refresh
# (retro#458 Phase 4). The vault hasn't regenerated since 2026-03-28 — this
# just makes that fact observable instead of silent.
_STALE_THRESHOLD_DAYS = 90


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


def _mtime_or_none(path: Path) -> float | None:
    return path.stat().st_mtime if path.exists() else None


async def refresh_cache(path: Path) -> None:
    global _snapshot_mtime
    loaded = await asyncio.to_thread(_load_from_disk, path)
    mtime = await asyncio.to_thread(_mtime_or_none, path)
    async with _cache_lock:
        _cache.clear()
        _cache.update(loaded)
        _snapshot_mtime = mtime
    logger.info("Leaderboard refreshed: %d sources loaded", len(_cache))
    if mtime is not None:
        age_days = (datetime.now(tz=timezone.utc) - datetime.fromtimestamp(mtime, tz=timezone.utc)).days
        if age_days > _STALE_THRESHOLD_DAYS:
            logger.warning(
                "event=credibility_leaderboard_stale snapshot_date=%s age_days=%d",
                leaderboard_snapshot_date(), age_days,
            )


def leaderboard_snapshot_date() -> str | None:
    """ISO date leaderboard.json was last written on disk, or None if it's
    missing. This is the vault's provenance, not a "last checked" timestamp —
    nothing has regenerated the file since 2026-03-28, so this date has been
    static for months and callers should treat it as a frozen baseline."""
    if _snapshot_mtime is None:
        return None
    return datetime.fromtimestamp(_snapshot_mtime, tz=timezone.utc).date().isoformat()


def _load_shadow_from_disk(board_path: Path) -> dict[str, dict]:
    """Key the resolution-shadow board (a bare list) by source id, mirroring
    _load_from_disk's shape. Fails open to {} — a missing board just means no
    resolution has been ingested yet, which is not an error."""
    return {entry["id"]: entry for entry in load_shadow_leaderboard(board_path) if entry.get("id")}


async def refresh_shadow_cache(board_path: Path, feedback_path: Path) -> None:
    global _shadow_total_resolutions
    loaded = await asyncio.to_thread(_load_shadow_from_disk, board_path)
    total = await asyncio.to_thread(count_resolutions, feedback_path)
    async with _cache_lock:
        _shadow_cache.clear()
        _shadow_cache.update(loaded)
        _shadow_total_resolutions = total
    logger.info(
        "Resolution-shadow board refreshed: %d sources, %d scoreable resolutions (gate: %d)",
        len(_shadow_cache), _shadow_total_resolutions,
        settings.resolution_shadow_min_global_predictions,
    )


async def background_refresh_loop(
    path: Path,
    interval_seconds: int,
    shadow_board_path: Path | None = None,
    shadow_feedback_path: Path | None = None,
) -> None:
    while True:
        await asyncio.sleep(interval_seconds)
        try:
            await refresh_cache(path)
        except Exception as exc:
            logger.error("Leaderboard refresh failed: %s", exc)
        if shadow_board_path is not None and shadow_feedback_path is not None:
            try:
                await refresh_shadow_cache(shadow_board_path, shadow_feedback_path)
            except Exception as exc:
                logger.error("Resolution-shadow refresh failed: %s", exc)


def _conservative_score(entry: dict):
    """skill_conservative, falling back to the legacy trueskill_conservative.
    Returns None only when neither is present — 0.0 is a valid score, not 'missing'."""
    v = entry.get("skill_conservative")
    return v if v is not None else entry.get("trueskill_conservative")


def _conservative_or_zero(entry: dict) -> float:
    v = _conservative_score(entry)
    return float(v) if v is not None else 0.0


def _weight_from_brier(brier_mean: float, predictions: int) -> float:
    """Credibility from a source's mean Brier score over resolved forecasts,
    shrunk toward the uninformed prior (0.25) by
    settings.resolution_shadow_brier_prior_n pseudo-resolutions.

    Brier 0.25 — an uninformed source, or one that consistently hedges — maps
    to exactly 1.0, so an unknown source is neutral by construction and this
    term only ever moves a source relative to that. Shrinkage is what makes a
    minimum per-source count unnecessary: a new source with two lucky calls
    lands near 1.0 instead of at the upper clamp, and earns its way out
    smoothly as its own record accumulates.

    Deliberately NOT the vault's μ−3σ transform: σ barely moves in these large
    multi-team OpenSkill matches, so conservative stays pinned near 0 and every
    source would map to ≈1.0 (measured spread 1.03× on the real board).
    """
    prior_n = settings.resolution_shadow_brier_prior_n
    shrunk = (brier_mean * predictions + prior_n * 0.25) / (predictions + prior_n)
    weight = 1.0 + (0.25 - shrunk) * settings.resolution_shadow_brier_slope
    return max(settings.resolution_shadow_weight_min,
               min(settings.resolution_shadow_weight_max, weight))


def _resolution_shadow_weight(source_id: str) -> float:
    """Flag-ON path. Neutral 1.0 until the whole dataset clears the global gate,
    and neutral for any source with no resolution history — never falling back
    to the frozen vault, which would reintroduce exactly the stale-scores
    ambiguity the cutover exists to remove."""
    if _shadow_total_resolutions < settings.resolution_shadow_min_global_predictions:
        logger.info("event=credibility_fallback_default source_id=%s branch=%s", source_id, "shadow")
        return 1.0
    entry = _shadow_cache.get(source_id)
    if entry is None:
        logger.info("event=credibility_fallback_default source_id=%s branch=%s", source_id, "shadow")
        return 1.0
    brier = entry.get("brier_score")
    predictions = entry.get("predictions")
    if brier is None or not predictions:
        logger.info("event=credibility_fallback_default source_id=%s branch=%s", source_id, "shadow")
        return 1.0
    return _weight_from_brier(float(brier), int(predictions))


def get_credibility_weight(source_id: str) -> float:
    """
    Per-source credibility multiplier for the aggregation weight. 1.0 is
    neutral in both paths, and an unknown source always gets 1.0.

    When settings.resolution_shadow_credibility_enabled is set, this comes from
    the resolution-informed shadow board (see _resolution_shadow_weight). The
    legacy path below is the flag-OFF default:

    Credibility weight derived from OpenSkill conservative estimate (μ − 3σ).
    Falls back to ELO/Brier if skill fields are absent.
    Sources absent from leaderboard get neutral weight 1.0.

    Conservative score: higher = more trusted.
    Normalised so that the default uninformed prior (μ=25, σ=8.33 → conservative≈0)
    maps to weight 1.0.

    Formula: max(0.1, 1.0 + conservative / 25.0)
    - conservative =  0  → weight = 1.0  (new/unknown source)
    - conservative = 10  → weight = 1.4  (trusted)
    - conservative = -5  → weight = 0.8  (distrusted)
    """
    if settings.resolution_shadow_credibility_enabled:
        return _resolution_shadow_weight(source_id)

    entry = _cache.get(source_id)
    if entry is None:
        logger.info("event=credibility_fallback_default source_id=%s branch=%s", source_id, "legacy")
        return 1.0

    # Support new field name (skill_conservative) and old (trueskill_conservative).
    # A real 0.0 score must map to weight 1.0, not fall through to the ELO/Brier path.
    conservative_raw = _conservative_score(entry)
    if conservative_raw is not None:
        return max(0.1, 1.0 + float(conservative_raw) / 25.0)

    # Fallback: ELO / Brier
    elo = float(entry.get("elo", 1200.0))
    brier = float(entry.get("brier_score", 0.25))
    return elo / (1200.0 + brier * 200.0)


def leaderboard_size() -> int:
    return len(_cache)


def get_leaderboard_data() -> list[dict]:
    """Return a snapshot of all leaderboard entries, sorted by skill conservative score."""
    entries = list(_cache.values())
    entries.sort(key=_conservative_or_zero, reverse=True)
    return entries
