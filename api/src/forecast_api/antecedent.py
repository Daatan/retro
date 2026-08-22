"""Antecedent-conditioned pool filtering (retro#573, "pool-split").

``/forecast`` and ``/pool/aggregate`` price one flat pool per consequent:
"given bloc >= 61, will Netanyahu be PM?" and "will Netanyahu be PM?" search
the same evidence and return the same number, because nothing in the pool
distinguishes a claim scoped to one antecedent from ordinary unconditional
evidence or from a claim scoped to a DIFFERENT antecedent (retro#573 — the
manual decomposition run that measured antecedent sensitivity ~= 0 on a
36-edge re-pricing). This module is Option 1 from that issue ("pool-split,
method 2, as primary"): filter an ALREADY-EXTRACTED pool by antecedent before
the existing weight loop runs. No new prompt, no new search, no extraction
change — every field this reads (``is_conditional`` / ``antecedent_text_en`` /
``antecedent_polarity``) has been live on ``ClaimDetail`` since PR#570.

Lexical, not semantic, on the same rationale as ``clustering.py`` (retro#355):
reuses its ``shingles``/``jaccard`` rather than an embedding call, so a
recompute over the same pool with the same antecedent text is deterministic
and replayable forever, not dependent on a model's drift.
"""

from __future__ import annotations

from typing import Optional, Sequence, TypeVar

from .clustering import jaccard, shingles

T = TypeVar("T")

# Antecedents are one short clause ("the ceasefire holds"), not full-article
# text — clustering.py's default trigram window (tuned for paragraph-length
# story text) would rarely find two independently-worded restatements of the
# same clause overlapping at all. A bigram is the smallest window that still
# requires word ORDER, not just shared vocabulary, to count as a match.
_SHINGLE_SIZE = 2

# How much shingle overlap counts as "the same antecedent". Deliberately
# lower than clustering.py's cluster_jaccard_threshold (0.40, tuned for
# near-duplicate ARTICLES restating one story): two independently-worded
# restatements of a one-clause condition share less surface text than two
# wire-service copies of the same paragraph do. Unvalidated against resolved
# antecedent pairs — retro#573 is explicit that this is provisional, the same
# caveat clustering.py's own threshold carries, and the number to refit once
# there is enough conditional-claim volume to measure against.
DEFAULT_ANTECEDENT_JACCARD_THRESHOLD = 0.25


def _claim_field(claim, name: str):
    """Duck-typed field read: a ``ClaimDetail`` or the raw dict test fixtures
    and callers use interchangeably (see ``clustering.cluster_text_for_claims``,
    the existing precedent for this exact duck-typing)."""
    val = getattr(claim, name, None)
    if val is None and isinstance(claim, dict):
        val = claim.get(name)
    return val


def _antecedent_matches(
    claim, query_shingles: frozenset, query_polarity: bool, threshold: float
) -> bool:
    antecedent_en = _claim_field(claim, "antecedent_text_en")
    if not antecedent_en:
        # Conditional but the extractor didn't populate the English form —
        # nothing to match against; not this antecedent.
        return False
    polarity = _claim_field(claim, "antecedent_polarity")
    if polarity is None:
        polarity = True  # ClaimDetail's own convention: None == affirmative.
    if bool(polarity) != query_polarity:
        return False
    return jaccard(query_shingles, shingles(antecedent_en, _SHINGLE_SIZE)) >= threshold


def source_matches_antecedent(
    claims_detail: Optional[Sequence],
    query_shingles: frozenset,
    query_polarity: bool,
    threshold: float = DEFAULT_ANTECEDENT_JACCARD_THRESHOLD,
) -> bool:
    """True when a source carrying ``claims_detail`` stays in an
    antecedent-filtered pool.

    Kept when it has at least one claim that either isn't conditional at all
    (ordinary evidence — in scope no matter which antecedent is being asked
    about) or is conditional on a matching antecedent. Dropped only when
    EVERY claim on it is conditional on some OTHER antecedent: evidence
    scoped to a hypothetical that isn't the one being asked about doesn't
    inform this question.

    A source with no ``claims_detail`` at all (pre-dates PR#570, or a caller
    that never populated it) is kept — the shadow-field pattern's rule,
    already applied by ``clustering.cluster_text_for_claims``, is that
    missing per-claim data can never *cost* a source its vote.
    """
    if not claims_detail:
        return True
    saw_conditional = False
    for claim in claims_detail:
        if not _claim_field(claim, "is_conditional"):
            return True
        saw_conditional = True
        if _antecedent_matches(claim, query_shingles, query_polarity, threshold):
            return True
    # Every claim was conditional and none matched — unless there were no
    # claims to begin with (empty list), which is the "no data" case, kept
    # for the same reason a None claims_detail is kept.
    return not saw_conditional


def antecedent_keep_mask(
    claims_detail_rows: Sequence[Optional[Sequence]],
    antecedent_query: Optional[str],
    antecedent_query_polarity: bool = True,
    threshold: float = DEFAULT_ANTECEDENT_JACCARD_THRESHOLD,
) -> list[bool]:
    """Per-row antecedent-match boolean mask, one entry per ``claims_detail_rows``
    element, in order.

    The primitive both pool-filtering shapes in this codebase reduce to.
    ``/pool/aggregate`` holds one list of sources and can filter it directly
    (``filter_pool_by_antecedent`` below). The live ``/forecast`` path
    (retro#583) instead builds several PARALLEL arrays in lockstep inside its
    per-article loop (stance, weight, relevance, ... one list per field, not
    one list of source objects) — there is no single sequence to filter, so
    the caller needs the keep/drop decision as a mask it can apply to every
    array itself. This function is that mask; ``filter_pool_by_antecedent``
    is the convenience wrapper for callers that DO have one sequence.

    ``antecedent_query`` is ``None`` on every existing caller (the field is
    new and optional): returns an all-``True`` mask (nothing dropped) — the
    same inert-by-default contract as ``filter_pool_by_antecedent``.
    """
    if not antecedent_query:
        return [True] * len(claims_detail_rows)
    query_shingles = shingles(antecedent_query, _SHINGLE_SIZE)
    return [
        source_matches_antecedent(claims_detail, query_shingles, antecedent_query_polarity, threshold)
        for claims_detail in claims_detail_rows
    ]


def filter_pool_by_antecedent(
    sources: Sequence[T],
    antecedent_query: Optional[str],
    antecedent_query_polarity: bool = True,
    threshold: float = DEFAULT_ANTECEDENT_JACCARD_THRESHOLD,
) -> list[T]:
    """Return ``sources`` filtered to those relevant to ``antecedent_query``.

    ``antecedent_query`` is ``None`` on every existing caller (the field is
    new and optional): returns ``sources`` unchanged — this filter is fully
    inert until a caller opts in, same "additive and fail-open" contract as
    ``claim_direction``/``claim_deadline`` on ``ForecastRequest``.
    """
    mask = antecedent_keep_mask(
        [getattr(s, "claims_detail", None) for s in sources],
        antecedent_query, antecedent_query_polarity, threshold,
    )
    return [s for s, keep in zip(sources, mask) if keep]
