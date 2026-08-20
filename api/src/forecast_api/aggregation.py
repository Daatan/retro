"""Pure aggregation helpers for the live forecaster.

These functions are dependency-free and side-effect-free so they are trivially
unit-testable. The forecaster converts each source's stance to a probability and
pools the sources in **log-odds (logit) space**, weighting by
``credibility × evidence_class_weight × recency × relevance²``. (Certainty is
*not* in that product for a classified claim — :func:`evidence_class_weight`
reads it only on the ``unclassified`` fallback. It shapes the within-article
stance instead, via :func:`claim_weighted_stance`.)

Why logit pooling instead of an arithmetic mean of stance?
  An arithmetic mean is easily dragged toward the middle by a few off-topic or
  stale sources (the original "73% on a decided series" bug). Pooling in log-odds
  space (a logarithmic opinion pool) is robust to such outliers: a single weak
  dissenter cannot pull a confident consensus back toward 0.5, and the pooled
  estimate stays bounded by its members. Paired with certainty weighting within
  an article and recency weighting across sources, a decided event reads as
  decisive instead of wishy-washy.

Stance is on [-1, 1] (−1 = event won't happen, +1 = will happen); probability is
``(stance + 1) / 2``.
"""
from __future__ import annotations

import logging
import math
from datetime import date, datetime, timedelta
from typing import NamedTuple, Optional, Sequence

logger = logging.getLogger(__name__)

# Two-sided 95% normal quantile. Hoisted so the pooled standard error and the
# dispersion floor provably use the same one.
_Z95 = 1.96

# Saturation bounds on the settlement pin's own band — the module's THIRD set of
# probability bounds, after logit_clamp (per-source + pooled CI + thin widening)
# and the [0,1] validity bound F16 removed. Named rather than unified; see
# _settlement_pin for why.
_SETTLEMENT_CI_MIN_P = 0.005
_SETTLEMENT_CI_MAX_P = 0.995


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def stance_to_prob(stance: float) -> float:
    """Map stance [-1, 1] → probability [0, 1]."""
    return (stance + 1.0) / 2.0


def prob_to_stance(p: float) -> float:
    """Inverse of :func:`stance_to_prob`: probability [0, 1] → stance [-1, 1]."""
    return 2.0 * p - 1.0


def logit(p: float) -> float:
    """Log-odds of a probability. Caller must pass p in the open interval (0, 1)."""
    return math.log(p / (1.0 - p))


def sigmoid(x: float) -> float:
    """Inverse of :func:`logit`, computed in a numerically stable way."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _parse_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value[:10]).date()
    except (ValueError, TypeError):
        return None


def event_date_state(value: Optional[str]) -> str:
    """``absent`` | ``unparseable`` | ``parsed`` — HOW a settlement vote's
    ``settlement_event_date`` reached ``settlement_vote_validity`` (retro#554).

    The undated demotion reasons (``missing_event_date``,
    ``undated_foreclosure``, ``stale_undated_foreclosure``) all fire on
    ``_parse_date(...) is None``, which conflates two very different inputs:
    an article that genuinely carried no date (``absent``) and one whose date
    string exists but fails ISO parsing (``unparseable`` — a fixable
    extraction defect, not a property of the article). Audit log lines carry
    this alongside the raw value so the two populations can be separated
    read-only. ``parsed`` is the ordinary dated case, present so the field
    can ride on every demotion line, not just the undated reasons.
    """
    if not value or not str(value).strip():
        return "absent"
    return "parsed" if _parse_date(value) is not None else "unparseable"


def recency_weight(
    article_date: Optional[str],
    ref_date: Optional[str],
    half_life_days: float,
    floor: float = 0.02,
) -> float:
    """Exponential recency decay: ``0.5 ** (age_days / half_life_days)``.

    ``age_days`` is measured from ``article_date`` up to ``ref_date`` (the newest
    article / "now"). The result is floored at ``floor`` so very old articles
    still count a little.

    An article with **no usable date decays straight to** ``floor`` (F13, design
    rule R3: missing data never increases influence). It used to return a
    neutral 1.0 — the single best multiplier the term can produce — so an
    article that never said when it was written outweighed an honest, dated
    three-week-old report by 50×, and the less we knew about a source the more
    it was worth. The floor is the other end of the same scale: an undated
    article is treated as maximally stale rather than maximally fresh. It still
    votes; it just cannot buy influence with an absence.

    ``ref_date`` missing or ``half_life_days <= 0`` is a different thing — the
    caller has switched recency off, or has no reference point at all — and
    still returns a neutral 1.0 for every article alike.
    """
    art = _parse_date(article_date)
    ref = _parse_date(ref_date)
    if ref is None or half_life_days <= 0:
        return 1.0
    if art is None:
        return floor
    age_days = max(0, (ref - art).days)
    w = 0.5 ** (age_days / half_life_days)
    return max(floor, w)


def weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    """Weighted arithmetic mean; falls back to a plain mean when weights sum to 0."""
    total = sum(weights)
    if total == 0:
        return sum(values) / len(values)
    return sum(v * w for v, w in zip(values, weights)) / total


def claim_weighted_stance(
    stances: Sequence[float],
    certainties: Sequence[float],
    specificities: Optional[Sequence[Optional[float]]] = None,
) -> float:
    """Within-article aggregation: certainty(×specificity)-weighted mean stance.

    A decisive claim ("they clinched it", certainty ≈ 1) dominates tangential,
    hedged claims (certainty ≈ 0.2) extracted from the same article, instead of
    being washed out by a flat mean. ``specificity`` is optional (the live
    extractor does not emit it) and defaults to a neutral 1.0.
    """
    if not stances:
        raise ValueError("claim_weighted_stance requires at least one claim")
    if specificities is None:
        specificities = [None] * len(stances)
    weights = [
        c * (sp if sp is not None else 1.0)
        for c, sp in zip(certainties, specificities)
    ]
    return weighted_mean(list(stances), weights)


def resolve_stance_certainty(
    stance: float,
    certainty: float,
    quantitative_estimate: Optional[float],
    *,
    evidence_class: Optional[str] = None,
    min_certainty: float = 0.9,
) -> tuple[float, float]:
    """Resolve a source's (stance, certainty), overriding both from an explicit
    cited ``quantitative_estimate`` rather than trusting the extractor to have
    aligned them itself.

    extractor.py's prompt instructs the LLM to set stance/certainty to match a
    cited quantitative_estimate, but nothing enforces that — a misaligned
    stance would get amplified, not fixed, by evidence_class_weight's
    cited_probability premium below (a correctly-extracted number weighted onto
    the wrong direction is worse than not extracting it at all).

    The rewrite applies ONLY to ``evidence_class == "cited_probability"``: that
    class is defined as "the same figure that would populate
    quantitative_estimate as a genuine probability of the event occurring". Any
    other class carrying a number — a cited_share vote share, a seat count, a
    reported measurement — is a different quantity whose value is NOT a
    probability of the event, so ``prob_to_stance`` on it is a category error
    (retro#362: 47 of the 117 qe-carrying prod pool rows were cited_share seat
    shares rewritten to stance = 2×share−1 at certainty 0.9). Unclassified
    (``None``) claims are also left alone — missing data must not increase
    influence. Returns ``(stance, certainty)`` unchanged in all those cases.
    """
    if quantitative_estimate is None or evidence_class != "cited_probability":
        return stance, certainty
    return prob_to_stance(quantitative_estimate), max(certainty, min_certainty)


def settlement_grade(
    stance: float,
    certainty: float,
    *,
    min_stance: float,
    min_certainty: float,
) -> bool:
    """Whether a ``settled`` claim is decisive enough to count toward the
    settlement pin.

    The extractor's prompt mandates that an accomplished-fact claim carry the
    full ±1.0 stance at certainty ≥ 0.9 — but prompts are advisory. A hedged,
    half-confident "settlement" (the 2026-07-08 F-35 false pin rode on claims
    at stance −0.8 / certainty 0.52) is treated as ordinary evidence instead:
    it still votes in the pool, it just cannot pin the estimate to a boundary.
    """
    return abs(stance) >= min_stance and certainty >= min_certainty


def settlement_direction_allowed(
    direction: float,
    claim_direction: Optional[str],
    claim_deadline: Optional[str],
    today: Optional[str] = None,
) -> bool:
    """Whether a settlement pin in ``direction`` is temporally coherent.

    Before a claim's deadline, the only reportable accomplished fact is that
    the event OCCURRED — for an *arrival* claim that settles YES (+1), for a
    *survival* claim ("X will NOT happen by D") it settles NO (−1). The
    opposite direction ("it didn't happen") is not a fact until the window
    closes: the F-35 forecast was falsely pinned to 3% five months before its
    deadline by background history, which no genuine early settlement could
    assert. Once the deadline has passed, either direction is coherent.

    Fail-open: without direction metadata or a parseable deadline (callers
    that don't classify claims), behavior is unchanged.
    """
    if claim_direction not in ("arrival", "survival"):
        return True
    deadline = _parse_date(claim_deadline)
    if deadline is None:
        return True
    ref = _parse_date(today) or datetime.now().date()
    if deadline <= ref:
        return True
    return direction > 0 if claim_direction == "arrival" else direction < 0


def settlement_vote_validity(
    stance: float,
    settlement_event_date: Optional[str],
    published_date: Optional[str],
    claim_direction: Optional[str],
    claim_deadline: Optional[str],
    claim_created_at: Optional[str],
    claim_archetype: Optional[str],
    today: Optional[str] = None,
    post_deadline_grace_days: int = 14,
) -> Optional[str]:
    """Why this single settlement vote must NOT count — or ``None`` if it stands.

    Extraction-time guards (``enforce_settlement_event_date``, retro #276/#291)
    only protect fresh extractions; a pool recompute replays stored ``settled``
    bits written before those guards existed, or poisoned since (the 2026-07-16
    audit: stale rows re-pinned wrong estimates on every recompute, and
    re-pushes re-flipped cleaned flags within hours). This makes the anchor
    requirement an invariant re-checked on every aggregation.

    A vote's obligations depend on which way it points relative to the claim's
    temporal direction (raw sign is not enough — a survival claim settles
    POSITIVE precisely when nothing happened, which is inherently undatable):

    - **Occurrence-direction** (arrival:+, survival:−, unclassified:+ — the
      vote asserts the underlying event HAPPENED): must carry a parseable
      ``settlement_event_date``; the date must not fall after ``claim_deadline``
      (an occurrence after the window contradicts an arrival claim rather than
      settling it), must not precede ``claim_created_at`` — any archetype
      since 2026-08-16; previously ``scheduled`` only (a dated fact from
      before the claim existed: the 2021/2022-article class, and the
      2024-election rows that pinned the survival-class Putin claim) — and
      must not post-date the article that
      "reported" it (a schedule, not a fact — re-asserting the extraction
      guard on stored rows).
    - **Non-occurrence-direction** (arrival:−, survival:+ — the vote asserts
      the event didn't/can't happen): valid once the deadline has passed (the
      absence itself is then the evidence, no date needed), or when it carries
      a dated FORECLOSING event within the window (France eliminated by Spain
      settles "France wins" NO five days before the final). An undated
      non-occurrence vote before (or without) a known deadline has no anchor —
      that is the F-35/Netanyahu background-history class — and is demoted.
      A non-occurrence vote ANCHORED on an occurrence dated far past the
      closed window is a non-sequitur, not an expiry observation: the
      2026-07-19 pool audit's "US bombs Iran in 2025" rows were settled NO at
      0.93+ by articles reporting the US actively bombing Iran in July 2026 —
      an out-of-window occurrence of a REPEATABLE event says nothing about the
      window (and ground truth there was YES). Only a small grace past the
      deadline (``post_deadline_grace_days``) is honored, for the flipped
      late-arrival class where the occurrence itself proves the miss: the
      Knesset dissolving July 17 against a July 15 deadline settles NO, the
      same claim "settled" by a strike seven months later does not. The same
      non-sequitur reaches an UNDATED vote too, via ``published`` instead of
      ``event`` (retro#295, the #293 residue): 12 of that pool's rows had no
      ``settlement_event_date`` at all but were extracted from articles
      *published* ~7 months after the deadline reporting the SAME event class
      recurring in a later year — timeframe misattribution the extractor
      prompt already forbids (retro#295's fix), demoted here as a backstop
      for whatever slips through. An undated vote from an article published
      within grace of the deadline is unaffected — that is the ordinary,
      honest "window closed quietly" case.

    Both directions share the creation-date lower bound
    (``event_before_claim_window``): any DATED vote whose event precedes
    ``claim_created_at`` is demoted, on every archetype (widened from
    ``scheduled``-only after the 2026-08-16 audit — see the inline comment).

    Every individual check is fail-open on absent metadata (no deadline → no
    deadline comparison), but the date requirement itself is fail-closed: an
    occurrence vote with no date never counts. Demoted votes keep their stance
    (the row still moves the pooled mean as ordinary evidence).
    """
    event = _parse_date(settlement_event_date)
    deadline = _parse_date(claim_deadline)
    published = _parse_date(published_date)
    ref = _parse_date(today) or datetime.now().date()

    occurrence_sign_positive = claim_direction != "survival"
    is_occurrence_vote = (stance >= 0) == occurrence_sign_positive

    if is_occurrence_vote:
        if event is None:
            return "missing_event_date"
        if deadline is not None and event > deadline:
            return "event_after_deadline"
    else:
        window_closed = deadline is not None and deadline <= ref
        if event is None:
            if window_closed:
                if (
                    published is not None
                    and deadline is not None
                    and published > deadline + timedelta(days=post_deadline_grace_days)
                ):
                    # An undated "nothing happened" vote from an article published
                    # long after the window closed is more likely reporting a LATER,
                    # different-timeframe occurrence of a recurring event — misread
                    # as silence on the closed window — than genuine retrospective
                    # silence (the 2026-07-19 "US bombs Iran 2025" class: mid-2026
                    # articles about active 2026 strikes, extracted as an undated NO
                    # for the closed 2025 window). The extractor prompt already
                    # forbids cross-timeframe extraction (retro#295); this is the
                    # aggregation-time backstop for rows that slip through it.
                    return "stale_undated_foreclosure"
                return None  # window closed — absence of the event is the anchor
            return "undated_foreclosure"
        if deadline is not None and event > deadline:
            if not window_closed:
                # A foreclosure dated after a still-open window forecloses
                # nothing in it.
                return "event_after_deadline"
            if event > deadline + timedelta(days=post_deadline_grace_days):
                # Dated anchor far past the closed window — the repeatable-event
                # non-sequitur ("it is happening NOW"), not evidence of
                # in-window absence. Undated expiry votes (event None above)
                # remain the honest way to settle NO on a closed window.
                return "post_window_occurrence"
            # Within grace: the flipped late arrival — the occurrence itself
            # proves the event missed the window.
    # Both directions, EVERY archetype: a dated fact from before this claim
    # existed can neither settle nor foreclose it. Originally scoped to
    # ``scheduled`` claims (2021 "Bennett formed the government" rows voting
    # on 2026 claims), on the theory that a threshold may legitimately be
    # crossed before its claim is created (models.py records that rationale
    # as historical). The 2026-08-16 pool audit showed the un-scoped
    # archetypes leak the identical failure: the survival-class
    # "Putin not president" claim pinned at 57% by two articles about the
    # 2024 election (events 2024-03-17/21), a 2022 maritime-deal story
    # re-pinning the 2026 Lebanon-agreement claim at 97%, and a 2022
    # swearing-in article voting on the Netanyahu-2026 claim. Ruling
    # (2026-08-16): a claim created mid-window reads future-facing — a
    # pre-creation crossing may inform the pooled mean (demoted rows still
    # vote as ordinary evidence) but must not settle. Strict ``<`` at date
    # granularity keeps creation-day events valid (the Brent $100 pin,
    # event on its claim's creation day, survives). Fail-open when
    # ``claim_created_at`` is absent or unparseable. The undated-foreclosure
    # branches above return earlier and are deliberately out of scope — they
    # carry no event date to bound.
    created = _parse_date(claim_created_at)
    if created is not None and event < created:
        return "event_before_claim_window"
    if published is not None and event > published:
        return "event_after_article"
    return None


def evidence_window_outside(
    n: int,
    settlement_event_dates: Optional[Sequence[Optional[str]]],
    published_dates: Optional[Sequence[Optional[str]]],
    claim_created_at: Optional[str],
    claim_deadline: Optional[str],
    claim_archetype: Optional[str],
    *,
    lookback_days: int,
) -> tuple:
    """Which rows fall outside the claim's evidence window
    ``[claim_created_at − lookback_days, claim_deadline]`` — as
    ``((index, reason), …)``, empty when every row is inside.

    Shadow instrumentation for retro#545 slice (iii), per the 2026-08-19
    Gate-0 decision (Daatan/docs decisions.md): nothing here excludes or
    demotes anything — the caller logs the result so the movement of
    bounding the window can be measured before it is enforced (the F4/F20
    pattern). The lookback keeps legitimate precursor/trend coverage that
    makes a young forecast estimable; rows before it are the adjacent-event
    class — an earlier, similar incident counted as evidence for the
    forecasted one.

    Scoped to non-``scheduled`` archetypes: the decision bounds the window
    exactly where today's is ``(−∞, deadline]``, and ``scheduled`` claims
    already consult ``claim_created_at`` in :func:`settlement_vote_validity`
    under their own historical rationale (models.py records it).

    A row's date is its ``settlement_event_date`` when parseable, else its
    ``published_date``; a row with neither is skipped (fail-open on absent
    metadata, matching every other check in this module — an unboundable row
    is not evidence *against* the window). Reasons: ``before_window`` (date
    precedes ``created − lookback``) and ``after_deadline`` (date past the
    deadline — expected ≈0 in the shadow numbers if the upper edge is as
    bound as believed; a non-zero count is itself a finding). Negative
    ``lookback_days`` disables the check entirely (the config kill switch).
    """
    if lookback_days < 0 or claim_archetype == "scheduled":
        return ()
    created = _parse_date(claim_created_at)
    if created is None:
        return ()
    deadline = _parse_date(claim_deadline)
    window_start = created - timedelta(days=lookback_days)
    outside: list[tuple[int, str]] = []
    for i in range(n):
        date = _parse_date(settlement_event_dates[i]) if settlement_event_dates else None
        if date is None:
            date = _parse_date(published_dates[i]) if published_dates else None
        if date is None:
            continue
        if date < window_start:
            outside.append((i, "before_window"))
        elif deadline is not None and date > deadline:
            outside.append((i, "after_deadline"))
    return tuple(outside)


def evidence_class_weight(
    evidence_class: Optional[str],
    certainty: float,
    *,
    weights: dict[str, float],
    default: float = 0.6,
    unclassified_cap: float = 0.25,
) -> float:
    """Per-claim weight component for the cross-article ``weight`` term (S2
    cutover, retro docs/ORACLE_VARIABLES.md §5).

    Classified evidence (``evidence_class`` set by the extractor) is weighted by
    its evidence type via ``weights`` — one lookup table replacing
    certainty-as-weight, the 0.9 certainty floor :func:`resolve_stance_certainty`
    applied on a cited quantitative estimate, and the old standalone ×4
    quantitative-anchor multiplier. A single named-model/poll/market baseline
    (``cited_probability``) is materially stronger evidence than qualitative
    "favorite"/"strong candidate" framing and must not be diluted by volume when
    several such qualitative articles are pooled alongside it — the France 2026
    World Cup regression (a 75% pooled estimate against a cited Opta baseline of
    18.83%) is exactly this failure; see ``weights["cited_probability"]``.

    Unclassified evidence (``evidence_class`` is ``None`` — the extractor omitted
    it) falls back to the claim's own ``certainty``, so partial classification
    coverage doesn't regress weighting quality for claims the classifier skipped
    — but **capped at** ``unclassified_cap`` (F10, design rule R3: missing data
    never increases influence). Certainty and the class table are two
    incommensurable scales sharing one slot: uncapped, a confident unlabelled
    claim resolved to 0.95 and out-weighed an identically confident claim the
    classifier *did* label ``reporting`` (0.6) — silence about the evidence type
    bought more influence than any answer would have. With the cap at the
    weakest class's weight, an unlabelled claim can tie the weakest classified
    one and never beat it, while a hedged unlabelled claim still resolves below
    that on its own certainty (the conservative direction is preserved, not
    flattened).
    """
    if evidence_class is None:
        return min(certainty, unclassified_cap)
    return weights.get(evidence_class, default)


def effective_sample_size(weights: Sequence[float]) -> float:
    """Kish's effective sample size, ``(Σ w)² / Σ w²``.

    Equal weights give exactly ``n``; a pool one row dominates gives ``1``.
    Mirrors :func:`pool_sources`' zero-total fallback (flat weights ⇒ ``n``) so
    the two can never disagree about how many sources a pool really has.

    A row count is not a sample size when the rows carry unequal weight (F16,
    retro#365). Measured over the 118 live pools on 2026-08-02 the median
    ``n_eff / n`` is 0.50 and 79.7% of pools sit below 0.8 — driven by recency
    decay across a weeks-long pool, not by near-zero weights.
    """
    n = len(weights)
    if n == 0:
        return 0.0
    total_w = sum(weights)
    if total_w <= 0:
        return float(n)
    sum_w2 = sum(w * w for w in weights)
    if sum_w2 <= 0:
        return float(n)
    return clamp((total_w * total_w) / sum_w2, 1.0, float(n))


def capped_weight_count(weights: Sequence[float], cap: float) -> float:
    """Capped-weight count, ``Σ min(wᵢ, cap) / cap`` (retro#382).

    How many rows' worth of evidence the pool carries if no single row may
    claim more than ``cap`` of it. Kish's ``n_eff`` is exactly the row count
    for equal weights, so a pool of N identical low-mass rows — the funnel's
    76× fan-out shape — reads as N full samples no matter how little mass each
    carries (matrix case C15). This statistic charges those rows for their
    mass: fifty rows at w=0.02 count as one.

    Not a sample size on its own — it is unbounded below 1 and overcounts a
    pool one heavy row dominates (the D1 objection recorded in retro#382, ~22%
    too narrow). Callers must take ``min(effective_sample_size(w), k)``: each
    statistic overcounts in the failure mode the other catches, and the min is
    conservative in both.

    ``cap`` ≤ 0 is a caller error; guarded at the call site (the cap tracks
    ``decisiveness_floor``, whose kill switch is 0).
    """
    if cap <= 0:
        raise ValueError("capped_weight_count requires cap > 0")
    return sum(min(w, cap) for w in weights) / cap


def pool_sources(
    stances: Sequence[float],
    weights: Sequence[float],
    *,
    clamp_eps: float = 0.01,
) -> tuple[float, float, float, float]:
    """Logit-pool source stances into ``(mean, std, ci_low, ci_high)``.

    Each stance is converted to a probability, clamped to ``[eps, 1-eps]`` (so the
    log-odds are finite), pooled in log-odds space (weighted), then converted back
    to the stance scale. Spread / 95% CI are computed in probability space (the
    weighted std of the per-source probabilities) and converted back to stance.

    All four returned values are on the stance scale [-1, 1], matching
    :class:`forecast_api.models.ForecastResponse`.

    The standard error divides by Kish's **effective** sample size,
    :func:`effective_sample_size` — exactly ``n`` for equal weights, exactly
    ``1`` for a single source or a pool one row dominates (F16, retro#365).

    Note this function reports *observed* dispersion, so a unanimous pool still
    returns a zero-width interval. The published minimum width is policy and
    lives in :func:`widen_ci_for_unresolved_dispersion`, which
    :func:`aggregate_pool` always applies.
    """
    n = len(stances)
    if n == 0:
        raise ValueError("pool_sources requires at least one source")

    probs = [clamp(stance_to_prob(s), clamp_eps, 1.0 - clamp_eps) for s in stances]
    w = list(weights)
    total_w = sum(w)
    if total_w <= 0:
        w = [1.0] * n
        total_w = float(n)

    pooled_logit = sum(wi * logit(p) for p, wi in zip(probs, w)) / total_w
    pooled_p = sigmoid(pooled_logit)

    var_p = sum(wi * (p - pooled_p) ** 2 for p, wi in zip(probs, w)) / total_w
    std_p = math.sqrt(var_p)
    # The old divisor was the raw row count, with an `if n > 1 else std_p`
    # special case. n_eff subsumes both exactly: it is 1.0 for a single source,
    # so sqrt(n_eff) is a no-op there.
    sem_p = std_p / math.sqrt(effective_sample_size(w))
    ci_low_p = clamp(pooled_p - _Z95 * sem_p, clamp_eps, 1.0 - clamp_eps)
    ci_high_p = clamp(pooled_p + _Z95 * sem_p, clamp_eps, 1.0 - clamp_eps)

    # stance = 2·p − 1 is linear, so a probability-scale std maps to 2× on the
    # stance scale.
    mean = prob_to_stance(pooled_p)
    std = 2.0 * std_p
    ci_low = prob_to_stance(ci_low_p)
    ci_high = prob_to_stance(ci_high_p)
    return mean, std, ci_low, ci_high


def widen_ci_for_thin_evidence(
    mean: float,
    ci_low: float,
    ci_high: float,
    std: float,
    *,
    deficit: float,
    max_inflation: float,
    clamp_eps: float,
) -> tuple[float, float, float]:
    """Widen a pooled CI to reflect *thin* evidence.

    :func:`pool_sources` derives its interval from how much the sources *disagree*
    and how many there are — not from how *much* evidence backs them. So a handful
    of hedged articles that happen to agree get a deceptively tight CI. Given a
    ``deficit`` ∈ [0, 1] (how far the certainty-weighted evidence mass falls below
    the decisiveness floor, 1 = no evidence) we add up to ``max_inflation`` of
    probability-space half-width symmetrically around the mean, so a thin on-topic
    pool self-reports as a low-confidence estimate with a wide band.

    Returns ``(ci_low, ci_high, std)`` on the stance scale [-1, 1]; a no-op when
    ``deficit`` or ``max_inflation`` is ≤ 0.
    """
    if deficit <= 0.0 or max_inflation <= 0.0:
        return ci_low, ci_high, std
    extra_p = clamp(deficit, 0.0, 1.0) * max_inflation
    p = stance_to_prob(mean)
    # F16(c): one endpoint convention for the module. ``[0.0, 1.0]`` was a type
    # constraint standing where a policy bound belongs — it made the widening
    # term, whose whole purpose is to express LESS confidence, the only path by
    # which the Oracle could publish "0%–100%": endpoints its own clamp declares
    # unreachable. pool_sources cannot relax to match; its clamp is logit()'s
    # domain requirement.
    lo_p = clamp(min(stance_to_prob(ci_low), p) - extra_p, clamp_eps, 1.0 - clamp_eps)
    hi_p = clamp(max(stance_to_prob(ci_high), p) + extra_p, clamp_eps, 1.0 - clamp_eps)
    # std is secondary (the CI is what callers display); bump it monotonically so
    # it can't claim more precision than the widened band.
    new_std = max(std, 2.0 * extra_p)
    return prob_to_stance(lo_p), prob_to_stance(hi_p), new_std


def widen_ci_for_unresolved_dispersion(
    mean: float,
    ci_low: float,
    ci_high: float,
    std: float,
    *,
    min_dispersion: float,
    n_eff: float,
    clamp_eps: float,
) -> tuple[float, float, float]:
    """Enforce a minimum published interval width (F16, retro#365).

    :func:`pool_sources` measures between-source *disagreement*. When there is
    none — a unanimous pool, or a pool of one — the measurement is exactly zero
    and the 95% band collapses onto the point estimate: maximum confidence from
    one article, with no gate in the way, because a single source that clears
    ``decisiveness_floor`` on its own never reaches
    :func:`widen_ci_for_thin_evidence`. Zero observed disagreement is a property
    of the sample, not of the world.

    So the band is floored as if the sources had scattered with a standard
    deviation of ``min_dispersion``, run through the same ``z / √n_eff``
    shrinkage :func:`pool_sources` applies. It therefore *decays with
    corroboration*: unanimity among twenty effective sources still buys
    precision, unanimity among one buys none. Because both the observed and the
    floored half-width divide by the same ``√n_eff``, the floor binds exactly
    when ``std_p < min_dispersion`` — a condition on dispersion alone,
    independent of pool size.

    Keyed on multiplicity, not mass, and that is deliberate: in prod the two
    populations are *anti*-correlated. Single-article pools already carry the
    widest median interval of any bucket (56pp) because thin-evidence widening
    dominates them; the zero-width cases are precisely the subset strong enough
    to escape it. A mass-keyed floor would fire on the pools that are already
    wide and miss the ones that are not. (The *divisor* is the exception:
    :func:`aggregate_pool` passes ``min(n_eff, capped_weight_count)`` so that
    equal-weight row volume cannot shrink the floor on multiplicity alone —
    retro#382, matrix case C15. The firing condition stays dispersion-only.)

    A floor, not an addition: ``min``/``max`` against whatever the band already
    is, so a pool with real dispersion — or one already widened for thin
    evidence — comes back untouched. Adding instead would charge a thin *and*
    unanimous pool twice for one deficiency.

    Returns ``(ci_low, ci_high, std)`` on the stance scale; a no-op when
    ``min_dispersion`` ≤ 0 (the kill switch).
    """
    if min_dispersion <= 0.0:
        return ci_low, ci_high, std
    half_p = _Z95 * min_dispersion / math.sqrt(max(n_eff, 1.0))
    p = stance_to_prob(mean)
    lo_p = clamp(min(stance_to_prob(ci_low), p - half_p), clamp_eps, 1.0 - clamp_eps)
    hi_p = clamp(max(stance_to_prob(ci_high), p + half_p), clamp_eps, 1.0 - clamp_eps)
    # std is a dispersion and does not shrink with count; only the standard
    # error does. So it is floored at min_dispersion itself, not at half_p.
    return prob_to_stance(lo_p), prob_to_stance(hi_p), max(std, 2.0 * min_dispersion)


class PoolAggregateResult(NamedTuple):
    """Result of :func:`aggregate_pool` pooling a set of already-weighted
    per-source signals into a final estimate.

    ``insufficient_reason`` is ``None`` on a usable result. When set
    (``"all_articles_off_topic"``, ``"no_usable_weight"`` or
    ``"no_decisive_signal"``), ``mean``/``std``/``ci_low``/``ci_high``/
    ``settled`` are not computed (pooling was skipped entirely, matching the
    live pipeline's early-return) and carry placeholder zeros — callers must
    check ``insufficient_reason`` first, the same way the live pipeline checks
    it before touching those fields.

    A pool that would abstain but carries a valid settlement pin is **not** one
    of those results: per §6.2's publish-time precedence the pin outranks
    abstention, so ``insufficient_reason`` is ``None``, ``settled`` is True and
    the fields carry the pin (retro#396). ``thin_evidence`` and
    ``evidence_mass`` still describe the pool the pin overrode, so a caller
    logging them sees the abstention that was outranked.

    ``n_eff`` and ``age_adjusted_mass`` are reporting-only additions
    (retro#458 Phase 2): nothing in this module or its callers reads them back
    into the pooled estimate, they exist so a caller can report on the pool's
    shape without re-deriving it. Both are computed on every branch, including
    abstained/off-topic results, mirroring ``evidence_mass``/``valve_mass``.
    """
    mean: float
    std: float
    ci_low: float
    ci_high: float
    settled: bool
    n: int
    # Kish's effective sample size of the voting `weights` (retro#458 Phase 2):
    # exactly `n` for equal weights, shrinking toward 1 as one row dominates.
    # See effective_sample_size() — this is that exact call's result, computed
    # once and reused for the CI floor's divisor below so the two can never
    # disagree about how many sources a pool really has.
    n_eff: float
    evidence_mass: float
    thin_evidence: bool
    # The mass the VALVES actually read (retro#397): recency-weighted and
    # UN-floored, where ``evidence_mass`` is the floored voting mass. Equal to
    # ``evidence_mass`` when the caller supplies no ``valve_weights``. Both are
    # kept because they answer different questions — "how much did each row get
    # to say" vs "do we still know anything" — and collapsing them is exactly
    # the defect §6.1 records.
    valve_mass: float
    # Reporting-only twin of `evidence_mass` with time decay switched off
    # (retro#458 Phase 2): `credibility * evidence_weight * relevance_weight**2`,
    # no `recency_weight` factor. Answers "how much would this pool weigh if
    # nothing had aged" — a caller-visible signal of how much of the pool's
    # current mass is a function of elapsed time rather than the evidence
    # itself. Equal to `evidence_mass` when the caller supplies no
    # `age_adjusted_weights`, same fallback convention as `valve_mass`.
    age_adjusted_mass: float
    insufficient_reason: Optional[str]
    # Settlement diagnostics for callers that want to log the outcome without
    # re-deriving settled_directions themselves. settlement_suppressed is True
    # when a would-be pin was blocked — by the temporal direction guard on the
    # legacy path (suppression_reason="settlement_direction"), by conflicting
    # valid votes on the revalidation path ("settlement_conflict"), or by the
    # winning direction's votes not clearing settlement_quality_floor despite
    # clearing settlement_min_sources ("settlement_quality_floor");
    # settled/settled_sources still reflect the *applied* outcome, i.e. no pin.
    settled_sources: int
    settlement_suppressed: bool
    suppression_reason: Optional[str] = None
    # Revalidation path only: (source index, reason) per settlement vote that
    # was NOT counted (settlement_vote_validity) — the row still voted as
    # ordinary evidence. Callers log these; aggregation stays log-free.
    settlement_demotions: tuple = ()
    # Source indices of the votes that actually carried the pin: settled, not
    # demoted, pointing the winning way. Empty when nothing pinned. Exposed so a
    # caller can name the evidence a pin rests on without re-deriving the
    # validity rules (retro#388's match gate reads exactly these rows) —
    # aggregation itself neither logs nor judges them.
    settlement_vote_indices: tuple = ()
    # Shadow instrumentation only (retro#545 slice iii): (source index, reason)
    # per row dated outside the claim's evidence window — see
    # evidence_window_outside(). Nothing in this module reads it back into the
    # pooled estimate; callers log it, same contract as settlement_demotions.
    # Empty when the check is disabled or nothing falls outside.
    evidence_window_outside_rows: tuple = ()


# Layer C's weight, per relevance band (retro#394).
#
# The gatekeeper does not emit a graded score. Across 84,254 gate-passing judgments the
# value takes 13 discrete 0.1-grained values with **exactly zero mass in the open interval
# (0.60, 0.70)** — and none in 0.71-0.79 or 0.81-0.84 either:
#
#   1.00->112  0.90->291  0.85->5  0.80->2,374  0.70->5,859  0.60->6,760
#   0.50->28,557  0.40->29,234  0.30->9,736  0.20->267  0.10->414  0.05->104  0.00->541
#
# Those are the edge labels of the prompt's own bands. So ``relevance ** 2`` was arithmetic
# applied to a categorical label: in the live daatan pool 51.9% of voting rows sit at exactly
# 0.70 and 25.2% at 0.80, making the "how much does this article count" dial a three-position
# switch whose positions nobody chose — 0.49 and 0.64 are simply what squaring produces, a
# ratio of 1.31x between "just passed" and "clearly relevant".
#
# **These values are deliberately initialised to exactly ``band ** 2``, so this table is a
# no-op today.** That is the point: the numbers become one named place to tune, instead of
# being implicit in an exponent. Retuning them needs outcome data, and there is not enough
# yet — as of 2026-08-04 only 6 resolved BINARY forecasts have a usable evidence pool (see
# retro#393), which cannot carry a Brier comparison. Change them when that sample grows, and
# say why in the commit.
RELEVANCE_BAND_WEIGHTS: dict[float, float] = {
    0.00: 0.0000,
    0.05: 0.0025,
    0.10: 0.0100,
    0.20: 0.0400,
    0.30: 0.0900,
    0.40: 0.1600,
    0.50: 0.2500,
    0.60: 0.3600,
    0.70: 0.4900,
    0.80: 0.6400,
    0.85: 0.7225,
    0.90: 0.8100,
    1.00: 1.0000,
}


def relevance_weight(relevance: float) -> float:
    """Layer C's contribution for one article's relevance score.

    Falls back to ``relevance ** 2`` for any value not in {@link RELEVANCE_BAND_WEIGHTS} —
    the table covers every band the gatekeeper has ever emitted, but a model change (or a
    caller-supplied verdict from another judge) could produce an off-band value, and
    silently rounding it to a neighbouring band would be a quiet re-weighting. The fallback
    is also what keeps this function identical to the old expression everywhere.
    """
    return RELEVANCE_BAND_WEIGHTS.get(round(relevance, 2), relevance ** 2)


def cluster_downweight_factors(
    cluster_ids: Optional[Sequence[int]], exponent: float,
) -> Optional[list[float]]:
    """Per-row weight multipliers that discount correlated evidence (retro#355).

    A cluster of ``k`` rows echoing one development contributes ``k ** (1 -
    exponent)`` times a single row's weight instead of ``k``. At
    ``exponent = 0`` this is all-ones — the identity, and the shipped default —
    so the seam costs nothing until someone chooses to spend it. At ``0.5`` a
    cluster of 4 carries the weight of 2; at ``1.0`` a cluster carries exactly
    one row's worth however loudly it is repeated.

    Deliberately a smooth exponent rather than a hard cap: the honest amount of
    independence in k reports of one story is unknown and certainly not 0, so a
    cap at one row would overcorrect — twenty outlets independently confirming a
    development *is* worth more than one, just not twenty times more.

    Returns ``None`` when the discount cannot apply (no ids, or a no-op
    exponent), which callers treat as "leave the weights alone" rather than
    multiplying by ones.
    """
    if not cluster_ids or exponent <= 0.0:
        return None
    sizes: dict[int, int] = {}
    for cid in cluster_ids:
        sizes[cid] = sizes.get(cid, 0) + 1
    return [sizes[cid] ** -exponent for cid in cluster_ids]


def cap_source_mass(
    weights: Sequence[float],
    source_ids: Optional[Sequence[str]],
    max_share: float,
) -> list[float]:
    """Per-row weights after capping any one source's share of total pool mass (retro#458).

    Distinct from `cluster_downweight_factors` above, which discounts near-
    duplicate TEXT ("two outlets wrote up one wire report"): this caps
    identity. A source that supplies many rows of genuinely distinct
    coverage — five different analysts quoted across five articles, none of
    them echoing each other's text — still contributes one outlet's editorial
    judgment about which analysts to platform, and a text-similarity check
    has nothing to say about that. Measured on prod 2026-08-08: one live pool
    (the S&P-crash forecast) carried 87.4% of its evidence mass from a single
    aggregator (finance.yahoo.com), with no near-duplicate text in sight —
    exactly the shape clustering cannot catch.

    Unlike `cluster_downweight_factors`, which returns per-row multipliers
    because a cluster's discount needs nothing else to change, this returns
    absolute weights: capping a dominant source only means something if the
    mass it loses is handed back to every other row, so `sum(weights)` is
    unchanged and every downstream consumer (evidence_mass, the decisiveness
    floor, pool_sources, effective_sample_size) keeps reading one honest
    number instead of a pool that quietly lost mass.

    For any `source_id` group whose share of total weight exceeds
    `max_share`, that group's weights are scaled down so its total equals
    exactly `max_share * total_weight`; the freed mass is redistributed
    proportionally over every row NOT in a capped group (whether or not that
    row's own source is still under the bar), preserving the grand total. A
    row with no `source_id` (legacy pool rows predating retro#364, or a
    caller that never threaded the join key through) cannot be attributed to
    any outlet, so it is treated as its own singleton group rather than
    lumped in with every other anonymous row — lumping would cap unrelated,
    unidentified sources together on an identity none of them actually
    share.

    If redistributing one group's excess pushes another group over the same
    bar (several large sources at once), that group is capped in a later
    pass too — this repeats until every remaining group clears the bar. If a
    pass finds every remaining group over the bar (most simply: the whole
    pool is one source), there is no surviving group left to hand the freed
    mass to, so capping would have to shrink the grand total to enforce the
    share — which the contract above forbids. That pass's offenders are left
    uncapped instead: a single-source pool cannot be capped down to a
    minority share of itself without inventing evidence that isn't there.

    No-op (returns `weights` as a fresh, unmodified ``list``) when
    `source_ids is None` (no identity to group on) or `max_share >= 1.0` (no
    cap configured — the shipped default). Mirrors
    `cluster_downweight_factors`'s inert-by-default contract: nothing moves
    until both an identity column AND a sub-1.0 share are supplied.
    """
    if source_ids is None or max_share >= 1.0:
        return list(weights)
    n = len(weights)
    if n == 0 or len(source_ids) != n:
        return list(weights)
    total = sum(weights)
    if total <= 0:
        return list(weights)

    groups: dict[object, list[int]] = {}
    for i, sid in enumerate(source_ids):
        key = sid if sid is not None else object()
        groups.setdefault(key, []).append(i)

    out = list(weights)
    cap_amount = max_share * total
    capped: set[object] = set()
    # Each pass caps at least one more group or exits, so this always
    # terminates within len(groups) passes.
    for _ in range(len(groups)):
        free = [k for k in groups if k not in capped]
        sums = {k: sum(out[i] for i in groups[k]) for k in free}
        offenders = [k for k in free if sums[k] > cap_amount]
        if not offenders:
            break
        survivors = [k for k in free if k not in offenders]
        if not survivors:
            # No group left anywhere to receive the freed mass without
            # shrinking the grand total — leave these rows as they are.
            break
        for k in offenders:
            gsum = sums[k]
            share = gsum / total
            factor = cap_amount / gsum
            for i in groups[k]:
                out[i] *= factor
            capped.add(k)
            logger.info(
                "event=source_mass_capped source_id=%s pre_cap_share=%.4f "
                "max_share=%.2f rows=%d total_weight=%.4f",
                k if isinstance(k, str) else None,
                share, max_share, len(groups[k]), total,
            )
        freed = sum(sums[k] for k in offenders) - cap_amount * len(offenders)
        survivor_indices = [i for k in survivors for i in groups[k]]
        survivor_total = sum(out[i] for i in survivor_indices)
        if survivor_total > 0:
            scale = (survivor_total + freed) / survivor_total
            for i in survivor_indices:
                out[i] *= scale
    return out


def _settlement_pin(
    settlement_stance: float, direction: float,
) -> tuple[float, float, float, float]:
    """The pinned ``(mean, ci_low, ci_high, std)`` for a settled direction.

    The pin uses _SETTLEMENT_CI_MIN_P/_MAX_P, NOT logit_clamp, and F16
    (retro#365) deliberately left it that way while unifying the other two
    conventions. Being allowed past the pooling clamp is the override's whole
    point (aggregation is bounded by its most confident member, so a decided
    event can never read as decided from averaging alone). Pulling the pin down
    to logit_clamp would move ci_high on every settled forecast in prod — 85.4%
    of them sit exactly on this bound — for no epistemic gain, since the pinned
    mean is already inside the pool clamp. The dispersion floor is applied
    BEFORE this on the pooled path, so a pinned interval is not subject to it
    either; that inconsistency is tracked, not hidden.

    Note that nothing here reads the pooled estimate: the pin is a function of
    ``settlement_stance`` and a sign alone. That is what makes it applicable to
    a pool that never got pooled (retro#396).
    """
    pinned_mean = settlement_stance * direction
    pinned_p = stance_to_prob(pinned_mean)
    if direction > 0:
        lo_p, hi_p = pinned_p - 0.06, min(_SETTLEMENT_CI_MAX_P, pinned_p + 0.025)
    else:
        lo_p, hi_p = max(_SETTLEMENT_CI_MIN_P, pinned_p - 0.025), pinned_p + 0.06
    return pinned_mean, prob_to_stance(lo_p), prob_to_stance(hi_p), 0.06


class SettlementDecision(NamedTuple):
    """Whether the settlement votes carry a pin, and the diagnostics of why not.

    ``direction`` is ``+1.0``/``-1.0`` when a pin is owed and ``None`` when it
    is not — either because no votes qualified or because a guard suppressed
    them (``suppressed`` says which). Deliberately decides nothing about the
    interval: :func:`_settlement_pin` owns that. Splitting the *decision* from
    its *application* is what lets the decision run before the abstention gate
    (retro#396) without pooling a set the gate has already judged unusable.
    """
    direction: Optional[float]
    settled_sources: int
    suppressed: bool
    suppression_reason: Optional[str]
    demotions: tuple
    vote_indices: tuple


def settlement_decision(
    stances: Sequence[float],
    weights: Sequence[float],
    settled_flags: Sequence[bool],
    *,
    settlement_min_sources: int,
    claim_direction: Optional[str] = None,
    claim_deadline: Optional[str] = None,
    settlement_event_dates: Optional[Sequence[Optional[str]]] = None,
    published_dates: Optional[Sequence[Optional[str]]] = None,
    claim_created_at: Optional[str] = None,
    claim_archetype: Optional[str] = None,
    settlement_revalidate: bool = False,
    settlement_post_deadline_grace_days: int = 14,
    settlement_quality_floor: float = 0.0,
    cluster_ids: Optional[Sequence[int]] = None,
) -> SettlementDecision:
    """Count the settlement votes and decide whether they pin the estimate.

    Pure and pooling-free: it reads the same per-source ``weights``
    :func:`aggregate_pool` pools, but only to score the winning direction
    against ``settlement_quality_floor`` — it never averages anything.

    The count is over **independent clusters**, not rows, when a cluster
    assignment is available (retro#372, the second half of F12): two syndicated
    copies of one report are one observation, and ``settlement_min_sources``
    exists to demand corroboration, which an echo is not. Rows without cluster
    text are singletons (:func:`clustering.cluster_text_for_claims`), so a
    missing claim layer can never cost a vote — it just counts as itself, the
    pre-#372 behavior. ``settlement_quality_floor`` stays a sum over rows:
    mass is mass, however correlated; independence is the count's axis.
    Legacy (non-revalidation) path deliberately untouched, kill-switch shape.
    """
    if settlement_revalidate and settlement_min_sources > 0:
        # Revalidation path (SETTLEMENT_REVALIDATE, default on): every settled
        # flag must re-prove its anchor on every aggregation — see
        # settlement_vote_validity. Replaces settlement_direction_allowed
        # (the per-vote rules strictly subsume the pin-level guard: an
        # early non-occurrence pin is exactly an undated_foreclosure unless
        # it carries a dated in-window foreclosing event, which is the case
        # the old guard wrongly suppressed). Majority vote is replaced by
        # unanimity: valid votes in BOTH directions mean one extraction is
        # provably wrong, and facts are not decided by outvoting — suppress
        # the pin, keep the pooled mean, and let the conflict log line drive
        # a human (or the admin `excluded` flag) to resolve it.
        demotions: list[tuple[int, str]] = []
        pos = 0
        neg = 0
        pos_weight = 0.0
        neg_weight = 0.0
        pos_indices: list[int] = []
        neg_indices: list[int] = []
        for i, (s, is_settled) in enumerate(zip(stances, settled_flags)):
            if not is_settled:
                continue
            reason = settlement_vote_validity(
                s,
                settlement_event_dates[i] if settlement_event_dates else None,
                published_dates[i] if published_dates else None,
                claim_direction, claim_deadline, claim_created_at, claim_archetype,
                post_deadline_grace_days=settlement_post_deadline_grace_days,
            )
            if reason is not None:
                demotions.append((i, reason))
            elif s >= 0:
                pos += 1
                pos_weight += weights[i]
                pos_indices.append(i)
            else:
                neg += 1
                neg_weight += weights[i]
                neg_indices.append(i)
        demoted = tuple(demotions)
        if pos > 0 and neg > 0:
            return SettlementDecision(None, 0, True, "settlement_conflict", demoted, ())
        # Independent-cluster count (retro#372). Falls back to the row count
        # when no assignment exists or it doesn't cover this pool (length
        # mismatch would mis-map votes to clusters — worse than not counting).
        if cluster_ids is not None and len(cluster_ids) == len(stances):
            pos_count = len({cluster_ids[i] for i in pos_indices})
            neg_count = len({cluster_ids[i] for i in neg_indices})
        else:
            pos_count, neg_count = pos, neg
        if max(pos_count, neg_count) >= settlement_min_sources:
            winning_weight = pos_weight if pos > neg else neg_weight
            if settlement_quality_floor > 0 and winning_weight < settlement_quality_floor:
                # Count clears the bar, but the settling votes are uniformly
                # weak evidence (low credibility, thin relevance, decayed
                # recency) — out-counting quality is not the same as agreeing
                # decisively (retro#279). The pooled mean stands, unpinned.
                #
                # This floor is also the guard that keeps retro#396's
                # pin-beats-abstention ordering honest: the votes that survive
                # an unusable pool are exactly the ones carrying real weight,
                # so "the pool is worthless" and "the pin is worthless" stay
                # two separate questions, each answered by its own threshold.
                return SettlementDecision(
                    None, 0, True, "settlement_quality_floor", demoted, (),
                )
            return SettlementDecision(
                1.0 if pos > 0 else -1.0,
                # Independent observations, not rows — the honest number for
                # settled_sources now that the trigger counts the same way.
                max(pos_count, neg_count),
                False,
                None,
                demoted,
                tuple(pos_indices if pos > 0 else neg_indices),
            )
        return SettlementDecision(None, 0, False, None, demoted, ())

    if settlement_min_sources > 0:
        # Legacy path (kill switch off): trust the flags, majority vote,
        # pin-level direction guard — byte-for-byte the pre-revalidation
        # behavior.
        settled_directions = [
            1.0 if s >= 0 else -1.0
            for s, is_settled in zip(stances, settled_flags)
            if is_settled
        ]
        if settled_directions:
            pos = sum(1 for d in settled_directions if d > 0)
            neg = len(settled_directions) - pos
            direction = 1.0 if pos > neg else -1.0 if neg > pos else 0.0
            if (
                direction != 0.0
                and max(pos, neg) >= settlement_min_sources
                and not settlement_direction_allowed(direction, claim_direction, claim_deadline)
            ):
                # Temporally incoherent early settlement (e.g. "won't happen"
                # pinned months before an arrival claim's deadline — the F-35
                # false pin): suppress the pin, keep the pooled estimate.
                return SettlementDecision(None, 0, True, "settlement_direction", (), ())
            if direction != 0.0 and max(pos, neg) >= settlement_min_sources:
                return SettlementDecision(
                    direction,
                    max(pos, neg),
                    False,
                    None,
                    (),
                    tuple(
                        i for i, (s, is_settled) in enumerate(zip(stances, settled_flags))
                        if is_settled and ((s >= 0) == (direction > 0))
                    ),
                )

    return SettlementDecision(None, 0, False, None, (), ())


def aggregate_pool(
    stances: Sequence[float],
    weights: Sequence[float],
    relevances: Sequence[float],
    settled_flags: Sequence[bool],
    *,
    relevance_weight_floor: float,
    decisiveness_floor: float,
    thin_evidence_ci_inflation: float,
    defer_on_thin_evidence: bool,
    settlement_min_sources: int,
    settlement_stance: float,
    logit_clamp: float,
    pool_dispersion_floor: float,
    claim_direction: Optional[str] = None,
    claim_deadline: Optional[str] = None,
    settlement_event_dates: Optional[Sequence[Optional[str]]] = None,
    published_dates: Optional[Sequence[Optional[str]]] = None,
    claim_created_at: Optional[str] = None,
    claim_archetype: Optional[str] = None,
    settlement_revalidate: bool = False,
    settlement_post_deadline_grace_days: int = 14,
    settlement_quality_floor: float = 0.0,
    cluster_ids: Optional[Sequence[int]] = None,
    cluster_downweight_exponent: float = 0.0,
    valve_weights: Optional[Sequence[float]] = None,
    age_adjusted_weights: Optional[Sequence[float]] = None,
    source_ids: Optional[Sequence[str]] = None,
    max_source_share: float = 1.0,
    evidence_window_lookback_days: int = -1,
) -> Optional[PoolAggregateResult]:
    """Pool a set of already-extracted, already-weighted per-source signals
    into a final estimate: the relevance off-topic safety net, logit
    pooling, thin-evidence CI widening, and the settlement override.

    This is every step ``forecast_api.forecaster.run_forecast`` performs
    *after* its per-article extraction loop, extracted here so a recompute
    over an accumulated evidence pool (retro docs/ORACLE_VARIABLES.md,
    recompute-over-pool — no search, no LLM, just already-extracted signals)
    can never silently drift from what a fresh run would produce. Each
    ``stances[i]``/``weights[i]``/``relevances[i]``/``settled_flags[i]``
    corresponds to one source's already-computed
    ``avg_stance``/``credibility * evidence_weight * recency * relevance**2``/
    ``relevance``/``bool(settled_preds)`` — exactly the fields a caller
    already has from ``SourceSignal`` (live) or a persisted evidence-pool row
    (recompute).

    Returns ``None`` only when there is nothing to pool at all (empty
    input) — an unambiguous "no sources" case every caller handles the same
    way. When sources exist but the set is entirely off-topic
    (``relevance_weight_floor``), carries no weight at all (F14), or is too
    thin to trust with ``defer_on_thin_evidence`` set, pooling is skipped and
    the result's ``insufficient_reason`` says why
    (``"all_articles_off_topic"`` / ``"no_usable_weight"`` /
    ``"no_decisive_signal"``) — each caller builds its own
    insufficient-data response around that reason (the live pipeline adds
    request-specific fetch/gate/extract diagnostics a pool recompute has no
    equivalent of). The one exception is a pool that also carries a valid
    settlement pin: the pin outranks abstention (retro#396), so that result is
    returned settled and usable rather than insufficient.

    ``age_adjusted_weights`` (retro#458 Phase 2) is a reporting-only third
    weights list, mirroring how ``valve_weights`` is threaded through: when
    supplied it feeds ``PoolAggregateResult.age_adjusted_mass`` and nothing
    else — it is never read by the pooling math itself, exactly like
    ``valve_weights``.
    """
    if not stances:
        return None

    # Shadow-only (retro#545 slice iii): which rows sit outside the evidence
    # window. Computed on every branch, like n_eff — a pool that abstains or
    # pins still has a window worth measuring — and read back by nothing here.
    # Disabled at this function's default (-1); Settings turns it on.
    window_outside = evidence_window_outside(
        len(stances), settlement_event_dates, published_dates,
        claim_created_at, claim_deadline, claim_archetype,
        lookback_days=evidence_window_lookback_days,
    )

    # Correlated-evidence discount (retro#355), applied FIRST so that every
    # downstream consumer — evidence_mass, the decisiveness floor, pool_sources,
    # effective_sample_size, the settlement vote weighting — sees one consistent
    # set of weights. Discounting later would let the thin-evidence test pass on
    # inflated mass and only then shrink it, which is how a pool of twenty
    # echoes would keep reading as decisive.
    #
    # Inert at the shipped default (exponent 0.0 ⇒ factors is None). No caller
    # can turn it on by accident: it takes both a cluster assignment AND a
    # non-zero exponent.
    factors = cluster_downweight_factors(cluster_ids, cluster_downweight_exponent)
    if factors is not None and len(factors) == len(weights):
        weights = [w * f for w, f in zip(weights, factors)]

    # Per-source mass cap (retro#458, Phase 1), applied AFTER the cluster
    # discount so a correlated cluster's rows are judged on their
    # already-discounted weight, not their raw pre-cluster weight — otherwise
    # a cluster that clustering already shrank could get penalized twice for
    # the same echo (once by the discount, again by a cap computed as if the
    # discount hadn't happened). Inert at the shipped default
    # (max_source_share = 1.0 ⇒ cap_source_mass is a no-op); see config.py.
    weights = cap_source_mass(weights, source_ids, max_source_share)

    # NOT discounted, deliberately — same reasoning as retro#404 kept it on the
    # raw square: this answers "is the whole set off-topic", a question about
    # topicality that echo does not bear on, and relevance_weight_floor was
    # tuned against the undiscounted sum.
    relevance_mass = sum(r * r for r in relevances)
    all_off_topic = relevance_mass < relevance_weight_floor

    evidence_mass = sum(weights)
    # Kish's effective sample size of the voting weights (retro#458 Phase 2),
    # computed once here and reused below as the CI floor's divisor (`floor_n`)
    # so the reporting field and the pooling math can never disagree about how
    # many sources this pool really has.
    n_eff = effective_sample_size(weights)
    # The VALVE mass (retro#397, system-model §6.1). `weights` carry recency
    # floored at `recency_floor` (0.02), which exists so an old row's *voting*
    # influence never goes to exactly zero. Reusing that floored mass to decide
    # whether we still know anything makes the fade-out impossible: 50
    # fully-decayed rows still sum to 1.0, so a large enough pool clears
    # decisiveness_floor forever and an abandoned question keeps publishing a
    # confident-ish number sourced entirely from stale coverage. The valves
    # therefore read an UN-floored, recency-weighted mass, supplied by the
    # caller alongside the voting weights.
    #
    # Absent (None) it falls back to the voting mass, which is exactly today's
    # behaviour — so a caller that has not been taught to compute it is
    # unchanged rather than silently switched to a stricter rule.
    valve_mass = sum(valve_weights) if valve_weights is not None else evidence_mass
    # Reporting-only twin of evidence_mass with recency decay switched off
    # (retro#458 Phase 2) — see PoolAggregateResult.age_adjusted_mass. Same
    # fallback convention as valve_mass: absent, it equals evidence_mass.
    age_adjusted_mass = (
        sum(age_adjusted_weights) if age_adjusted_weights is not None else evidence_mass
    )
    # Glide-eligible AND still running: a deadline-shaped question whose
    # deadline has not passed. After it passes there is no glide left to
    # protect, so the carve-out below stops applying and a stale pool may
    # abstain like any other.
    _deadline = _parse_date(claim_deadline)
    glide_active = _deadline is not None and _deadline >= datetime.now().date()
    # F14 (design rule R3): every source weighs exactly nothing — every one of
    # them was blocked by credibility, or zeroed by relevance, or both. Pooling
    # anyway means falling through pool_sources' zero-total guard, which
    # replaces the weights with a flat 1.0 each: the answer would then come
    # from precisely the rows the weighting judged worthless, and it would come
    # out unweighted. Abstain instead — "we have nothing usable" is the honest
    # reading, and the caller already knows how to render it.
    #
    # Deliberately still the VOTING mass: this asks "did the weighting judge
    # every row worthless", a question about credibility and relevance that age
    # does not bear on. An ancient pool's un-floored mass is tiny but never
    # exactly zero, so routing F14 through the valve mass would not add an
    # abstention — it would only make F14's reason misdescribe why.
    no_usable_weight = not all_off_topic and evidence_mass <= 0.0
    thin_evidence = not all_off_topic and valve_mass < decisiveness_floor
    no_decisive_signal = thin_evidence and defer_on_thin_evidence and not no_usable_weight
    if no_decisive_signal and glide_active:
        # §6.1's one carve-out, load-bearing against §6.2: on a glide-eligible
        # question decayed mass widens the CI but never aborts an ACTIVE glide
        # into abstention. The glide is the deadline clock pricing the silence,
        # not the pool whispering its last headline, and it converges on the
        # very boundary the impossibility pin will declare from metadata alone.
        # Abstention outranks a glide only in its §6.2 sense — relevance mass
        # ≈ 0, i.e. no valid anchor ever existed or the pool was killed — which
        # is `all_off_topic` above and is deliberately NOT suppressed here.
        no_decisive_signal = False

    # The settlement decision is taken BEFORE the abstention gate, because
    # §6.2's publish-time precedence is
    #   settlement pin > impossibility pin > abstention > glide > pooled estimate
    # and evaluating it after the gate's early return inverted the top of that
    # list: a pool tripping any abstention reason could never publish a pin,
    # however well-founded (retro#396). A settled fact does not need a
    # topically-dense pool to be true — the clearest shape is a question whose
    # evidence is mostly off-topic noise but which carries two valid settling
    # votes reporting the outcome. The vote's own guards (revalidation, the
    # unanimity rule and settlement_quality_floor) decide whether those votes
    # are worth anything; the pool's usability is a different question and no
    # longer answers this one.
    #
    # It is also cheap to compute here: the decision reads no pooled quantity,
    # so hoisting it costs one pass over the settled flags on every pool.
    decision = settlement_decision(
        stances, weights, settled_flags,
        settlement_min_sources=settlement_min_sources,
        claim_direction=claim_direction,
        claim_deadline=claim_deadline,
        settlement_event_dates=settlement_event_dates,
        published_dates=published_dates,
        claim_created_at=claim_created_at,
        claim_archetype=claim_archetype,
        settlement_revalidate=settlement_revalidate,
        settlement_post_deadline_grace_days=settlement_post_deadline_grace_days,
        settlement_quality_floor=settlement_quality_floor,
        # The same assignment the downweight uses (inert until its exponent is
        # non-zero) — but the settlement COUNT spends it unconditionally
        # (retro#372): two rows echoing one report are one settling source.
        cluster_ids=cluster_ids,
    )

    if all_off_topic or no_usable_weight or no_decisive_signal:
        if decision.direction is not None:
            # The pin outranks the abstention. Nothing is pooled: the interval
            # is a function of settlement_stance and a sign, so there is no
            # pooled estimate for it to replace, and pooling a zero-weight set
            # here would fall through pool_sources' flat-weight fallback — the
            # very thing F14 abstains to avoid. insufficient_reason is None
            # because this result IS publishable; thin_evidence/evidence_mass
            # still report the pool honestly for anyone logging it.
            pinned_mean, pinned_lo, pinned_hi, pinned_std = _settlement_pin(
                settlement_stance, decision.direction,
            )
            return PoolAggregateResult(
                mean=pinned_mean, std=pinned_std, ci_low=pinned_lo, ci_high=pinned_hi,
                settled=True, n=len(stances), n_eff=n_eff, evidence_mass=evidence_mass,
                valve_mass=valve_mass, age_adjusted_mass=age_adjusted_mass,
                thin_evidence=thin_evidence, insufficient_reason=None,
                settled_sources=decision.settled_sources,
                settlement_suppressed=False,
                suppression_reason=None,
                settlement_demotions=decision.demotions,
                settlement_vote_indices=decision.vote_indices,
                evidence_window_outside_rows=window_outside,
            )
        reason = (
            "all_articles_off_topic" if all_off_topic
            else "no_usable_weight" if no_usable_weight
            else "no_decisive_signal"
        )
        # The settlement diagnostics ride along even though nothing pinned: a
        # pin suppressed on an abstaining pool is exactly the case a reader of
        # this result needs to be able to tell apart from "no votes at all".
        return PoolAggregateResult(
            mean=0.0, std=0.0, ci_low=0.0, ci_high=0.0, settled=False,
            n=len(stances), n_eff=n_eff, evidence_mass=evidence_mass, valve_mass=valve_mass,
            age_adjusted_mass=age_adjusted_mass,
            thin_evidence=thin_evidence,
            insufficient_reason=reason, settled_sources=0,
            settlement_suppressed=decision.suppressed,
            suppression_reason=decision.suppression_reason,
            settlement_demotions=decision.demotions,
            settlement_vote_indices=(),
            evidence_window_outside_rows=window_outside,
        )

    n = len(stances)
    mean, std, ci_low, ci_high = pool_sources(stances, weights, clamp_eps=logit_clamp)

    # Reads the valve mass for the same reason `thin_evidence` does: §6.1's
    # intended consequence is that an aging pool's CI widens toward "we barely
    # know", and a deficit computed off the floored mass asymptotes to a
    # constant instead of continuing to open.
    if decisiveness_floor > 0 and valve_mass < decisiveness_floor:
        deficit = (decisiveness_floor - valve_mass) / decisiveness_floor
        ci_low, ci_high, std = widen_ci_for_thin_evidence(
            mean, ci_low, ci_high, std,
            deficit=deficit,
            max_inflation=thin_evidence_ci_inflation,
            clamp_eps=logit_clamp,
        )

    # F16: last, so it composes as a floor on whatever the pooled + thin path
    # produced, and BEFORE the settlement pin, which replaces the interval
    # outright with its own policy band.
    # The floor's divisor is min(n_eff, k) (retro#382): Kish alone is exactly
    # the row count for equal weights, so equal-weight volume — fifty rows at
    # w=0.02 — buys a fifty-row-tight band on one row's worth of evidence,
    # while k alone overcounts a pool one heavy row dominates. The min is
    # conservative in both failure modes. Falls back to n_eff alone when
    # decisiveness_floor is switched off (k needs a positive cap).
    floor_n = n_eff
    if decisiveness_floor > 0:
        floor_n = min(floor_n, capped_weight_count(weights, decisiveness_floor))
    ci_low, ci_high, std = widen_ci_for_unresolved_dispersion(
        mean, ci_low, ci_high, std,
        min_dispersion=pool_dispersion_floor,
        n_eff=floor_n,
        clamp_eps=logit_clamp,
    )

    # Settlement override: pooling is bounded by its most confident member, so
    # a decided event can never read as decided from averaging alone (the
    # Knicks "82% the day after the title" case). settled_flags[i] is already
    # the settlement-grade-filtered value (bool(settled_preds) in the live
    # loop) — its direction is simply the sign of that same source's stance,
    # which the live loop also already computed preferentially from
    # settled_preds when present.
    # settled_flags[i] is already the settlement-grade-filtered value
    # (bool(settled_preds) in the live loop) — its direction is simply the sign
    # of that same source's stance, which the live loop also already computed
    # preferentially from settled_preds when present. The decision itself was
    # taken above, before the abstention gate (retro#396); all that is left
    # here is applying it on top of the pooled interval it replaces.
    settled = decision.direction is not None
    if settled:
        mean, ci_low, ci_high, std = _settlement_pin(settlement_stance, decision.direction)

    return PoolAggregateResult(
        mean=mean, std=std, ci_low=ci_low, ci_high=ci_high, settled=settled,
        n=n, n_eff=n_eff, evidence_mass=evidence_mass, valve_mass=valve_mass,
        age_adjusted_mass=age_adjusted_mass,
        thin_evidence=thin_evidence,
        insufficient_reason=None, settled_sources=decision.settled_sources,
        settlement_suppressed=decision.suppressed,
        suppression_reason=decision.suppression_reason,
        settlement_demotions=decision.demotions,
        settlement_vote_indices=decision.vote_indices,
        evidence_window_outside_rows=window_outside,
    )
