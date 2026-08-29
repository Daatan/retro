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


# ── event-key clustering (retro#682) ─────────────────────────────────────────
#
# The Jaccard key above asks "did the extractor use the same words?". Measured over
# 19,926,967 pairwise comparisons in prod, **99.72% of pairs score below 0.1** and
# `max_jaccard` is exactly 0.0 in two thirds of pools — because pool rows are LLM
# paraphrases of twenty different outlets' prose, and two rows can describe one
# development while sharing almost no trigram. Lowering the threshold does not help:
# the [0.3,0.4) band holds 0.012% of pairs, the same order as what already fires.
#
# The event key asks a different question — "same (who, what, when)?" — and is
# paraphrase-invariant by construction. It reads the retro#313 facets, which were
# elicited but barely consumed before this.
#
# Reporting only: `cluster_downweight_exponent` stays 0.0 and the discount still runs
# off the Jaccard ids. This changes the key the seam will use *when* it turns on, not
# whether it turns on — that remains #355's December backtest (#403).

#: Leading noise that never distinguishes two entities.
_LEADING_ARTICLE_RE = re.compile(r"^(?:the|a|an)\s+")
#: Everything that is not a word character or an inter-word space.
_ENTITY_PUNCT_RE = re.compile(r"[^a-z0-9 ]+")

#: Deliberately tiny and hand-maintained. Every entry is a nation-state or institution
#: synonym where the merge is not a judgement call. It is NOT an attempt at entity
#: resolution: "Washington" for the US administration, "Number 10" for the UK PM's
#: office and the like are metonyms whose correctness depends on context, and guessing
#: wrong silently merges two developments into one. news-indexer's entity ids are the
#: Phase 2 escalation if string-level agreement proves insufficient — the plan says so,
#: and the measured 18.8% collapse rate says it is not needed yet.
_ENTITY_ALIASES = {
    "us": "united states", "usa": "united states", "u s": "united states",
    "u s a": "united states", "america": "united states",
    "uk": "united kingdom", "u k": "united kingdom",
    "britain": "united kingdom", "great britain": "united kingdom",
    "uae": "united arab emirates",
    "eu": "european union",
    "un": "united nations",
    "drc": "democratic republic of the congo",
    "north korea": "north korea", "dprk": "north korea",
    "south korea": "south korea", "rok": "south korea",
}


def _normalise_entity(value: Optional[str]) -> Optional[str]:
    """One actor or target, reduced to a comparable form.

    Multi-actor strings are split on commas, normalised individually and **sorted**, so
    "United States, Israel" and "Israel, the United States" produce one key. Sorting is
    what makes that deterministic; without it the key would depend on the order the
    extractor happened to list them in, which is not a fact about the world.
    """
    if not value:
        return None
    parts: list[str] = []
    for raw in str(value).split(","):
        text = _ENTITY_PUNCT_RE.sub(" ", raw.lower())
        text = " ".join(text.split())
        text = _LEADING_ARTICLE_RE.sub("", text).strip()
        if not text:
            continue
        parts.append(_ENTITY_ALIASES.get(text, text))
    if not parts:
        return None
    return ", ".join(sorted(set(parts)))


def _day_of(value: Optional[str]) -> Optional[str]:
    """The day component of a date string, by **slice rather than parse**.

    `published_date` is free text in the pool and does hold non-ISO junk (retro#714
    normalises it going forward, but stored rows keep whatever they were written with).
    A slice yields a stable, weird-but-consistent key for such a value; a parse raises,
    and an exception inside the clusterer would take down a `/forecast` request over a
    reporting-only measurement. Slicing also keeps this stdlib-deterministic, which the
    replay harness (#350/#403) depends on for the same reason `cluster_texts` does.
    """
    if not value:
        return None
    text = str(value).strip()[:10]
    return text or None


def event_key(
    actors: Optional[str], target: Optional[str], day: Optional[str],
) -> Optional[str]:
    """``(who, what, when)`` as one comparable string, or None if underdetermined.

    **All three parts are required.** The dyad alone is a *relationship*, not an event:
    on live pools the largest actor-target-only cluster is 171 rows on
    `united states -> iran`, which is months of coverage collapsed into one "story".
    With the discount enabled that would crush the pool to `n_eff ~ 1`. Adding the day
    splits that same group into 34 sub-clusters, largest 24.
    """
    a, t, d = _normalise_entity(actors), _normalise_entity(target), _day_of(day)
    if not (a and t and d):
        return None
    return f"{a}\x1f{t}\x1f{d}"


def _pick_claim_facets(claims_detail: Optional[Sequence]):
    """The facets of the claim that best represents the row.

    Highest ``claim_strength`` wins, ties broken by array position — deterministic, and
    the same rule the prod measurement behind this change used, so the numbers in
    `docs/ORACLE_VARIABLES.md` describe what actually ships.
    """
    best = None
    best_rank = None
    for i, c in enumerate(claims_detail or []):
        def _get(name):
            v = getattr(c, name, None)
            if v is None and isinstance(c, dict):
                v = c.get(name)
            return v

        if _get("event_actors") is None or _get("event_target") is None:
            continue
        try:
            strength = float(_get("claim_strength") or 0.0)
        except (TypeError, ValueError):
            strength = 0.0
        rank = (-strength, i)
        if best_rank is None or rank < best_rank:
            best_rank, best = rank, (
                _get("event_actors"), _get("event_target"), _get("event_date"),
            )
    return best


def event_key_for_row(
    claims_detail: Optional[Sequence],
    *,
    event_actors: Optional[str] = None,
    event_target: Optional[str] = None,
    settlement_event_date: Optional[str] = None,
    published_date: Optional[str] = None,
) -> Optional[str]:
    """The event key of one pool row, or None when it cannot be keyed.

    **Both weight sites must call this**, for the reason spelled out on
    :func:`cluster_text_for_claims`: `/forecast` and `/pool/aggregate` deriving a key
    differently would let a recompute re-cluster rows the live path already clustered
    (retro#404).

    Per-claim facets first, then the row-level columns as a fallback — measured on prod,
    the fallback lifts keyed rows from 5,403 to **6,886** (+27%) with the largest cluster
    unchanged at 24, so it is coverage for free.

    The date falls back to `published_date`, and that is not a detail: `event_date` is
    present on only 6% of voting rows ("omit entirely when the article states no date"),
    so requiring it would key 793 rows instead of ~6,900. Publication day is the right
    proxy — twenty outlets covering one development do so within a day or two of each
    other, which is precisely the shape being counted.
    """
    facets = _pick_claim_facets(claims_detail)
    if facets is not None:
        actors, target, edate = facets
    else:
        actors, target, edate = event_actors, event_target, settlement_event_date
    return event_key(
        actors or event_actors,
        target or event_target,
        edate or settlement_event_date or published_date,
    )


@dataclass(frozen=True)
class EventKeyStats:
    """What one event-key pass observed.

    Mirrors :class:`ClusterStats`' reason for existing: the cluster count alone cannot
    tell "no echo in a keyable pool" (a real observation) from "nothing was keyable"
    (a statement about facet coverage, not about the world).
    """

    #: Rows handed to the pass.
    rows: int
    #: Rows that produced a key at all — the denominator that matters.
    keyed: int
    #: Distinct keys among those rows.
    clusters: int
    #: Rows sitting in a cluster of two or more.
    echoed_rows: int
    #: Size of the largest cluster; 1 when nothing echoes.
    largest: int


def cluster_by_event_key(
    keys: Sequence[Optional[str]],
) -> tuple[tuple[int, ...], EventKeyStats]:
    """Group rows by exact key equality; unkeyed rows are singletons.

    No threshold and no pairwise loop — equality is the whole comparison, which is why
    this is O(n) where the Jaccard pass is O(n^2). Ids are assigned in first-appearance
    order so the output depends on nothing but the input, the property
    :func:`cluster_texts` documents and the replay harness requires.

    An unkeyed row keeps its own cluster and therefore its full weight — the same
    conservative direction the text clusterer takes for a row with no text. Missing
    facets must never cost a source its vote.
    """
    ids: dict[str, int] = {}
    counts: dict[int, int] = {}
    out: list[int] = []
    keyed = 0
    next_id = 0
    for k in keys:
        if k is None:
            # A row we cannot key is its own singleton, allocated in place so cluster
            # ids stay in first-appearance order across keyed and unkeyed rows alike.
            cid = next_id
            next_id += 1
        else:
            keyed += 1
            if k not in ids:
                ids[k] = next_id
                next_id += 1
            cid = ids[k]
        out.append(cid)
        counts[cid] = counts.get(cid, 0) + 1

    echoed = sum(n for n in counts.values() if n > 1)
    return tuple(out), EventKeyStats(
        rows=len(keys),
        keyed=keyed,
        clusters=len(counts),
        echoed_rows=echoed,
        largest=max(counts.values()) if counts else 0,
    )


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
