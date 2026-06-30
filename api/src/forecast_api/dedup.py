"""Collapse syndicated near-duplicate articles before they reach the pool.

A single wire story (Reuters, AP) is routinely re-hosted across aggregators
(MSN, Yahoo, Google News) under near-identical headlines. Left alone, each copy
is fetched, extracted and *counted as an independent source*, so one story can
triple its weight in the aggregation and pad the evidence mass. We collapse such
clusters to a single highest-credibility representative up front — saving
extraction cost and keeping the pool honest about how many *distinct* sources
actually bear on the claim.

Deliberately conservative: clustering is by title-token Jaccard at a high
threshold (default 0.8), so genuinely different stories that merely share a topic
word ("Anthropic", "Mythos") are *not* merged — only true re-prints are.
"""

from __future__ import annotations

import re
from typing import Callable, Sequence, TypeVar

T = TypeVar("T")

_WORD = re.compile(r"[a-z0-9]+")

# Jaccard on a handful of tokens is noisy, and a real syndicated headline is long.
# Below this many meaningful tokens we don't cluster by title (URL-dedup still runs),
# so a terse or degenerate title can't accidentally merge distinct stories.
_MIN_CLUSTER_TOKENS = 4

# Function words that shouldn't drive a syndication match. Topic/entity words are
# kept — they're exactly what distinguishes two stories on the same subject.
_STOP = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "of", "to", "in", "on", "for",
        "with", "at", "by", "from", "as", "is", "are", "was", "were", "be",
        "will", "would", "could", "should", "has", "have", "had", "its", "it",
        "this", "that", "these", "those", "what", "why", "how", "when", "who",
        "s", "here",
    }
)


def _title_tokens(title: str) -> frozenset[str]:
    return frozenset(t for t in _WORD.findall((title or "").lower()) if t not in _STOP)


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if not inter:
        return 0.0
    return inter / len(a | b)


def dedupe_syndicated(
    results: Sequence[T],
    *,
    title_of: Callable[[T], str],
    url_of: Callable[[T], str],
    priority_of: Callable[[T], float],
    threshold: float = 0.8,
) -> list[T]:
    """Return ``results`` with syndicated near-duplicates collapsed.

    Two items are duplicates when their URLs match or their title-token Jaccard is
    ``>= threshold``. Within a cluster the item with the greatest ``priority_of``
    (e.g. source credibility) is kept; ties resolve to the first seen, preserving
    input order. ``threshold <= 0`` disables clustering (URL-dedup still applies).
    """
    reps: list[T] = []
    rep_tokens: list[frozenset[str]] = []
    url_to_cluster: dict[str, int] = {}

    for item in results:
        url = (url_of(item) or "").strip()
        idx: int | None = url_to_cluster.get(url) if url else None

        if idx is None:
            tokens = _title_tokens(title_of(item))
            if threshold > 0.0 and len(tokens) >= _MIN_CLUSTER_TOKENS:
                best = threshold
                for i, rt in enumerate(rep_tokens):
                    if len(rt) < _MIN_CLUSTER_TOKENS:
                        continue
                    sim = _jaccard(tokens, rt)
                    if sim >= best:
                        best, idx = sim, i
            if idx is None:
                # New cluster.
                idx = len(reps)
                reps.append(item)
                rep_tokens.append(tokens)
                if url:
                    url_to_cluster[url] = idx
                continue

        # Merge into existing cluster idx: keep the higher-priority representative.
        if priority_of(item) > priority_of(reps[idx]):
            reps[idx] = item
        if url:
            url_to_cluster.setdefault(url, idx)

    return reps
