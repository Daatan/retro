"""Correlated-evidence clustering (retro#355).

``aggregate_pool`` treats every pool row as independent evidence. Many rows are
not: a single development ("sources say the Kremlin is considering a new wave")
is reported by twenty outlets, and pooling reads twenty echoes as twenty facts.
Per-article classification quality cannot fix this — however good the extractor
gets, N reports of one development are still one development. This module finds
the echoes; ``aggregation.cluster_downweight_factors`` decides what they are
worth.

**Lexical, not semantic, and deliberately so.** The issue sketch proposed reusing
news-indexer's pgvector near-dup machinery, but that lives in another service:
retro has no embedding dependency, and adding one buys a per-row API call on
every recompute plus a model whose drift would silently re-cluster history. The
gate harness (#350) replays past pools to measure this change; a
non-deterministic clusterer cannot be replayed. Shingle Jaccard is free, exact,
and reproducible forever. If it proves too blunt once there is enough resolved
evidence to measure against, the seam to swap it is this module's public
surface, not the estimator.

Single-linkage is the right transitivity here: A echoing B and B echoing C means
all three cover one development, even when A and C word it differently enough to
miss the pairwise bar.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Sequence

#: Words carry the signal; punctuation and case do not. Digits are kept — dates,
#: counts and place numbers are often exactly what makes two reports the same story.
_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def shingles(text: str, size: int) -> frozenset[tuple[str, ...]]:
    """Overlapping word n-grams of ``text``.

    Word order matters for story identity — "Russia strikes Ukraine" and
    "Ukraine strikes Russia" share every unigram and are not the same story — so
    shingles, not a bag of words.

    Texts shorter than ``size`` words degrade to a single whole-text shingle
    rather than producing nothing: a two-word claim is still a claim, and
    returning an empty set would silently make it unclusterable.
    """
    toks = _tokens(text)
    if not toks:
        return frozenset()
    if len(toks) <= size:
        return frozenset({tuple(toks)})
    return frozenset(tuple(toks[i:i + size]) for i in range(len(toks) - size + 1))


def jaccard(a: frozenset, b: frozenset) -> float:
    """|A ∩ B| / |A ∪ B|, with the empty pair scoring 0.

    Two rows with no extractable text are *unknown*, not identical — scoring
    them 1.0 would cluster every text-less legacy row into one giant pseudo-story
    and downweight the lot.
    """
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if not inter:
        return 0.0
    return inter / len(a | b)


#: Number of equal-width bands the pairwise-similarity histogram is bucketed into.
#: Band ``b`` covers ``[b/10, (b+1)/10)``, with an exact 1.0 clamped into the last.
SIMILARITY_BANDS = 10


@dataclass(frozen=True)
class ClusterStats:
    """What one clustering pass observed, including everything BELOW the threshold.

    The cluster ids alone cannot distinguish the three ways a pool produces no
    echo, and those three mean opposite things:

    1. fewer than two rows — nothing to compare;
    2. rows present but text-less — structurally unclusterable, so the pass says
       nothing about the world, only about ``claims_detail`` coverage;
    3. genuinely comparable text that no pair matched — a real observation that
       live pools carry little lexical echo.

    Only (3) is evidence. Reporting just the clusters collapses all three to an
    identical silence, which is what made ``event=evidence_clusters`` produce a
    single line in a month of traffic and left the number uninterpretable.

    ``max_jaccard`` and ``histogram`` are the near-miss record. ``config.py`` says
    to tune ``cluster_jaccard_threshold`` "against the logged cluster structure" —
    but the log only ever fired ABOVE the threshold, so there was no data below it
    to lower the bar onto. That instruction was uncloseable until these fields
    existed.
    """

    #: Rows handed to the clusterer.
    rows: int
    #: Rows with a non-empty shingle set — the denominator that actually matters.
    textful: int
    #: Pairwise comparisons performed, i.e. ``C(textful, 2)``.
    pairs: int
    #: Highest similarity seen, threshold notwithstanding. 0.0 when no pairs.
    max_jaccard: float
    #: Counts per similarity band; sums to ``pairs``. See ``SIMILARITY_BANDS``.
    histogram: tuple[int, ...]


def cluster_texts(
    texts: Sequence[Optional[str]],
    *,
    threshold: float,
    shingle_size: int = 3,
) -> tuple[int, ...]:
    """Group indices whose texts report the same development.

    Returns one cluster id per input, ids assigned in first-appearance order so
    the output is stable under nothing but the input itself (no dict iteration
    order, no clock, no RNG) — the property the replay harness depends on.

    A row with no usable text is always its own singleton. That is the
    conservative reading: an unclusterable row keeps its full weight, so missing
    text can never *cost* a source its vote. Legacy pool rows written before
    ``claims_detail`` existed all land here.

    Thin wrapper over :func:`cluster_texts_with_stats` for callers that only want
    the grouping. The stats cost nothing extra to produce — see that function.
    """
    ids, _ = cluster_texts_with_stats(
        texts, threshold=threshold, shingle_size=shingle_size,
    )
    return ids


def cluster_texts_with_stats(
    texts: Sequence[Optional[str]],
    *,
    threshold: float,
    shingle_size: int = 3,
) -> tuple[tuple[int, ...], ClusterStats]:
    """:func:`cluster_texts`, plus what the pass saw below the threshold.

    **The statistics are free.** The pairwise loop already computes every
    ``jaccard(...)`` and today discards each score the instant it fails the
    ``>= threshold`` test. Recording the max and a band count adds no pass, no
    allocation beyond one fixed-size counter list, and no comparison — it keeps a
    number that was being thrown away.

    Deliberately returns the stats rather than logging them. This module is the
    documented swap seam and the replay harness depends on it being pure and
    deterministic; the caller owns the log line, where the question hash lives.
    """
    n = len(texts)
    sets = [shingles(t, shingle_size) if t else frozenset() for t in texts]

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path halving
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            # Lower index wins, so root identity never depends on visit order.
            parent[max(rx, ry)] = min(rx, ry)

    max_jaccard = 0.0
    pairs = 0
    histogram = [0] * SIMILARITY_BANDS

    for i in range(n):
        if not sets[i]:
            continue
        for j in range(i + 1, n):
            if not sets[j]:
                continue
            score = jaccard(sets[i], sets[j])
            pairs += 1
            if score > max_jaccard:
                max_jaccard = score
            # An exact 1.0 would index one past the end; it belongs in the top band.
            histogram[min(int(score * SIMILARITY_BANDS), SIMILARITY_BANDS - 1)] += 1
            if score >= threshold:
                union(i, j)

    ids: dict[int, int] = {}
    out: list[int] = []
    for i in range(n):
        root = find(i)
        if root not in ids:
            ids[root] = len(ids)
        out.append(ids[root])

    stats = ClusterStats(
        rows=n,
        textful=sum(1 for s in sets if s),
        pairs=pairs,
        max_jaccard=max_jaccard,
        histogram=tuple(histogram),
    )
    return tuple(out), stats


def cluster_text_for_claims(
    claims_detail: Optional[Sequence], fallback: Optional[str] = None
) -> Optional[str]:
    """The text one pool row is clustered on.

    **Both weight sites must call this.** ``/forecast`` and ``/pool/aggregate``
    deriving cluster text differently would let a recompute re-cluster rows the
    live path already clustered — the same failure mode retro#404 guarded
    against for the relevance band table, and the reason the recompute path was
    given ``claims_detail`` at all (daatan#1264).

    Uses each claim's summary plus its verbatim quote: the summary is what the
    extractor thought the article said, the quote is what it actually said, and
    echo is visible in both.

    **``claims_detail`` is the only text either caller can supply today.** Neither
    ``SourceSignal`` nor ``PoolSourceInput`` carries a title, so both call sites pass
    ``fallback=None`` and a row without ``claims_detail`` is unclusterable — it stays a
    singleton at full weight. That is the safe direction (missing text can never *cost* a
    source its vote), but it does bound what ``event=evidence_clusters`` can currently
    observe: rows extracted before ``claims_detail`` existed contribute nothing to the
    measurement, and coverage grows only as the pool re-extracts.

    ``fallback`` is kept as the seam for that fix — adding a title to those models is the
    one change that would make legacy rows clusterable — but it is inert until something
    actually supplies one.
    """
    if not claims_detail:
        return fallback
    parts: list[str] = []
    for c in claims_detail:
        claim = getattr(c, "claim", None) or (c.get("claim") if isinstance(c, dict) else None)
        quote = getattr(c, "quote", None) or (c.get("quote") if isinstance(c, dict) else None)
        if claim:
            parts.append(str(claim))
        if quote:
            parts.append(str(quote))
    if not parts:
        return fallback
    return " ".join(parts)
