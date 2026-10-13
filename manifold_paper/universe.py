"""Which Manifold markets the paper bot tracks (retro#918).

Manifold is mostly personal, joke and self-resolving markets; a calibration
set needs the opposite. The filter keeps liquid, public-interest binary
markets that close soon enough to give feedback inside the runway. Every
threshold is an env var (see ``paper.py``); the keyword blocklist is the only
hard-coded taste.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta

from manifold_client import Market

# Personal / self-referential / unfalsifiable-by-us questions: the Oracle reads
# news, and "will I like X" has no news to read.
_PERSONAL = re.compile(
    r"\b(i|i\'ll|i\'m|my|me|mine|we|our|myself)\b|\bthis (market|poll)\b|\bmana\b|\bmanifold\b"
    r"|\[short fuse\]|\bsubsidy\b|\bwill (this|the) (market|question)\b"
    r"|coin ?flip|^\s*daily\b|shitpost",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class UniverseParams:
    min_bettors: int = 15
    min_volume: float = 500.0
    min_hours_to_close: float = 120.0
    max_days_to_close: float = 30.0


def is_personal(question: str) -> bool:
    return bool(_PERSONAL.search(question))


def eligible(m: Market, p: UniverseParams, now: datetime) -> bool:
    """New-market admission test (already-tracked markets skip it)."""
    if m.closed or not m.has_price or m.resolution_raw is not None:
        return False
    if m.unique_bettors < p.min_bettors or m.volume < p.min_volume:
        return False
    if is_personal(m.question):
        return False
    if m.end_date is None:
        return False
    close = datetime.fromisoformat(m.end_date)
    return now + timedelta(hours=p.min_hours_to_close) <= close <= now + timedelta(days=p.max_days_to_close)
