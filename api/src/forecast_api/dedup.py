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

import math
import re
from collections import Counter, defaultdict
from typing import Callable, Sequence, TypeVar

T = TypeVar("T")

# Script-agnostic: `\w` minus digits/underscore matches a "letter" in any
# script Python's `str.lower()` and `re` know about (Latin, Hebrew, Cyrillic,
# Arabic, ...), not just ASCII a-z0-9. The old `[a-z0-9]+` silently tokenized
# non-Latin titles to nothing, so a Hebrew or Russian re-print never cleared
# `_MIN_CLUSTER_TOKENS` and could never cluster (retro#458 Phase 3).
# `re.UNICODE` is already the default for a `str` pattern in Python 3 — kept
# explicit here so the intent reads without needing to know that default.
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)

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

    **Exact prefix filtering (retro#1056).** The original loop compared every item
    against every representative, which is quadratic: 618k Jaccard calls on a
    1,150-row ``/pool/aggregate`` pool. It now only compares candidates that can
    reach the threshold, and the result is bit-identical:

    * ``J(a, b) >= t`` implies ``|a & b| >= ceil(t * |a|)``, so at most
      ``|a| - ceil(t * |a|)`` of ``a``'s tokens can be missing from ``b``. Any
      ``|a| - ceil(t * |a|) + 1`` tokens of ``a`` must therefore include one that
      ``b`` has. Each representative is indexed under its rarest such *prefix*
      (one global rarest-first order), and two sets that pass have to share a
      prefix token.
    * ``J(a, b) >= t`` also implies ``t * |a| <= |b| <= |a| / t`` (length filter).
    * Each surviving candidate still gets the exact ``_jaccard``. Candidates are
      walked in ascending representative index under the same ``sim >= best``
      rule, so ties pick the same representative as the quadratic loop.

    ``tests/test_dedup.py`` pins equivalence against the quadratic reference.
    """
    tokens_of: list[frozenset[str]] = []
    rank: dict[str, int] = {}
    if threshold > 0.0:
        tokens_of = [_title_tokens(title_of(item)) for item in results]
        freq = Counter(t for ts in tokens_of for t in ts)
        rank = {t: k for k, t in enumerate(sorted(freq, key=lambda t: (freq[t], t)))}

    def prefix(tokens: frozenset[str]) -> list[str]:
        # Rounded DOWN by an epsilon: a shorter overlap bound means a longer
        # prefix, which can only add candidates, never drop a true match.
        min_overlap = math.ceil(threshold * len(tokens) - 1e-9)
        return sorted(tokens, key=rank.__getitem__)[: len(tokens) - min_overlap + 1]

    reps: list[T] = []
    rep_tokens: list[frozenset[str]] = []
    url_to_cluster: dict[str, int] = {}
    postings: defaultdict[str, list[int]] = defaultdict(list)

    for k, item in enumerate(results):
        url = (url_of(item) or "").strip()
        idx: int | None = url_to_cluster.get(url) if url else None

        if idx is None:
            tokens = tokens_of[k] if threshold > 0.0 else frozenset()
            clusterable = threshold > 0.0 and len(tokens) >= _MIN_CLUSTER_TOKENS
            pre: list[str] = []
            if clusterable:
                pre = prefix(tokens)
                lo = threshold * len(tokens) - 1e-9
                hi = len(tokens) / threshold + 1e-9
                best = threshold
                for i in sorted({i for t in pre for i in postings.get(t, ())}):
                    rt = rep_tokens[i]
                    if not lo <= len(rt) <= hi:
                        continue
                    sim = _jaccard(tokens, rt)
                    if sim >= best:
                        best, idx = sim, i
            if idx is None:
                # New cluster.
                idx = len(reps)
                reps.append(item)
                rep_tokens.append(tokens)
                # Short-title representatives are never matched (the quadratic
                # loop skipped them), so they are never indexed.
                for t in pre:
                    postings[t].append(idx)
                if url:
                    url_to_cluster[url] = idx
                continue

        # Merge into existing cluster idx: keep the higher-priority representative.
        if priority_of(item) > priority_of(reps[idx]):
            reps[idx] = item
        if url:
            url_to_cluster.setdefault(url, idx)

    return reps
