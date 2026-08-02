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

import math
from datetime import date, datetime, timedelta
from typing import NamedTuple, Optional, Sequence


# Two-sided 95% normal quantile. Hoisted so the pooled standard error and the
# dispersion floor provably use the same one.
_Z95 = 1.96

# Saturation bounds on the settlement pin's own band — the module's THIRD set of
# probability bounds, after logit_clamp (per-source + pooled CI + thin widening)
# and the [0,1] validity bound F16 removed. Named rather than unified; see
# _apply_pin for why.
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
      settling it), must not precede ``claim_created_at`` on a ``scheduled``
      claim (a dated fact about an EARLIER instance of the recurring event —
      the 2021/2022-article class), and must not post-date the article that
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
    if claim_archetype == "scheduled":
        # Both directions: a dated fact from before this claim existed belongs
        # to an EARLIER instance of the scheduled event — it can neither settle
        # nor foreclose this one (2021 "Bennett formed the government" rows
        # were negative votes on 2026 claims).
        created = _parse_date(claim_created_at)
        if created is not None and event < created:
            return "event_before_claim_window"
    if published is not None and event > published:
        return "event_after_article"
    return None


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
    wide and miss the ones that are not.

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
    """
    mean: float
    std: float
    ci_low: float
    ci_high: float
    settled: bool
    n: int
    evidence_mass: float
    thin_evidence: bool
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
    equivalent of).
    """
    if not stances:
        return None

    relevance_mass = sum(r * r for r in relevances)
    all_off_topic = relevance_mass < relevance_weight_floor

    evidence_mass = sum(weights)
    # F14 (design rule R3): every source weighs exactly nothing — every one of
    # them was blocked by credibility, or zeroed by relevance, or both. Pooling
    # anyway means falling through pool_sources' zero-total guard, which
    # replaces the weights with a flat 1.0 each: the answer would then come
    # from precisely the rows the weighting judged worthless, and it would come
    # out unweighted. Abstain instead — "we have nothing usable" is the honest
    # reading, and the caller already knows how to render it.
    no_usable_weight = not all_off_topic and evidence_mass <= 0.0
    thin_evidence = not all_off_topic and evidence_mass < decisiveness_floor
    no_decisive_signal = thin_evidence and defer_on_thin_evidence and not no_usable_weight

    if all_off_topic or no_usable_weight or no_decisive_signal:
        reason = (
            "all_articles_off_topic" if all_off_topic
            else "no_usable_weight" if no_usable_weight
            else "no_decisive_signal"
        )
        return PoolAggregateResult(
            mean=0.0, std=0.0, ci_low=0.0, ci_high=0.0, settled=False,
            n=len(stances), evidence_mass=evidence_mass, thin_evidence=thin_evidence,
            insufficient_reason=reason, settled_sources=0, settlement_suppressed=False,
        )

    n = len(stances)
    mean, std, ci_low, ci_high = pool_sources(stances, weights, clamp_eps=logit_clamp)

    if decisiveness_floor > 0 and evidence_mass < decisiveness_floor:
        deficit = (decisiveness_floor - evidence_mass) / decisiveness_floor
        ci_low, ci_high, std = widen_ci_for_thin_evidence(
            mean, ci_low, ci_high, std,
            deficit=deficit,
            max_inflation=thin_evidence_ci_inflation,
            clamp_eps=logit_clamp,
        )

    # F16: last, so it composes as a floor on whatever the pooled + thin path
    # produced, and BEFORE the settlement pin, which replaces the interval
    # outright with its own policy band.
    ci_low, ci_high, std = widen_ci_for_unresolved_dispersion(
        mean, ci_low, ci_high, std,
        min_dispersion=pool_dispersion_floor,
        n_eff=effective_sample_size(weights),
        clamp_eps=logit_clamp,
    )

    # Settlement override: pooling is bounded by its most confident member, so
    # a decided event can never read as decided from averaging alone (the
    # Knicks "82% the day after the title" case). settled_flags[i] is already
    # the settlement-grade-filtered value (bool(settled_preds) in the live
    # loop) — its direction is simply the sign of that same source's stance,
    # which the live loop also already computed preferentially from
    # settled_preds when present.
    settled = False
    settled_sources = 0
    settlement_suppressed = False
    suppression_reason: Optional[str] = None
    settlement_demotions: tuple = ()
    settlement_vote_indices: tuple = ()

    def _apply_pin(direction: float) -> tuple[float, float, float, float]:
        # The pin uses _SETTLEMENT_CI_MIN_P/_MAX_P, NOT logit_clamp, and F16
        # (retro#365) deliberately left it that way while unifying the other two
        # conventions. Being allowed past the pooling clamp is the override's
        # whole point (aggregation is bounded by its most confident member, so a
        # decided event can never read as decided from averaging alone). Pulling
        # the pin down to logit_clamp would move ci_high on every settled
        # forecast in prod — 85.4% of them sit exactly on this bound — for no
        # epistemic gain, since the pinned mean is already inside the pool clamp.
        # The dispersion floor is applied BEFORE this, so a pinned interval is
        # not subject to it either; that inconsistency is tracked, not hidden.
        pinned_mean = settlement_stance * direction
        pinned_p = stance_to_prob(pinned_mean)
        if direction > 0:
            lo_p, hi_p = pinned_p - 0.06, min(_SETTLEMENT_CI_MAX_P, pinned_p + 0.025)
        else:
            lo_p, hi_p = max(_SETTLEMENT_CI_MIN_P, pinned_p - 0.025), pinned_p + 0.06
        return pinned_mean, prob_to_stance(lo_p), prob_to_stance(hi_p), 0.06

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
        settlement_demotions = tuple(demotions)
        if pos > 0 and neg > 0:
            settlement_suppressed = True
            suppression_reason = "settlement_conflict"
        elif max(pos, neg) >= settlement_min_sources:
            winning_weight = pos_weight if pos > neg else neg_weight
            if settlement_quality_floor > 0 and winning_weight < settlement_quality_floor:
                # Count clears the bar, but the settling votes are uniformly
                # weak evidence (low credibility, thin relevance, decayed
                # recency) — out-counting quality is not the same as agreeing
                # decisively (retro#279). The pooled mean stands, unpinned.
                settlement_suppressed = True
                suppression_reason = "settlement_quality_floor"
            else:
                settled = True
                settled_sources = max(pos, neg)
                settlement_vote_indices = tuple(pos_indices if pos > 0 else neg_indices)
                mean, ci_low, ci_high, std = _apply_pin(1.0 if pos > 0 else -1.0)
    elif settlement_min_sources > 0:
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
                settlement_suppressed = True
                suppression_reason = "settlement_direction"
                direction = 0.0
            if direction != 0.0 and max(pos, neg) >= settlement_min_sources:
                settled = True
                settled_sources = max(pos, neg)
                settlement_vote_indices = tuple(
                    i for i, (s, is_settled) in enumerate(zip(stances, settled_flags))
                    if is_settled and ((s >= 0) == (direction > 0))
                )
                mean, ci_low, ci_high, std = _apply_pin(direction)

    return PoolAggregateResult(
        mean=mean, std=std, ci_low=ci_low, ci_high=ci_high, settled=settled,
        n=n, evidence_mass=evidence_mass, thin_evidence=thin_evidence,
        insufficient_reason=None, settled_sources=settled_sources,
        settlement_suppressed=settlement_suppressed,
        suppression_reason=suppression_reason,
        settlement_demotions=settlement_demotions,
        settlement_vote_indices=settlement_vote_indices,
    )
