"""Pure aggregation helpers for the live forecaster.

These functions are dependency-free and side-effect-free so they are trivially
unit-testable. The forecaster converts each source's stance to a probability and
pools the sources in **log-odds (logit) space**, weighting by
``credibility × certainty × recency``.

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
from datetime import date, datetime
from typing import Optional, Sequence


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
    article / "now"). Returns ``1.0`` (neutral) when either date is missing or
    unparseable — an article is never penalised merely for lacking a date. The
    result is floored at ``floor`` so very old articles still count a little.
    """
    art = _parse_date(article_date)
    ref = _parse_date(ref_date)
    if art is None or ref is None or half_life_days <= 0:
        return 1.0
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


def quantitative_anchor_multiplier(
    quantitative_estimates: Sequence[Optional[float]],
    *,
    multiplier: float,
) -> float:
    """Weight premium for a source that cites an explicit modeled/poll/market
    probability for the event itself, rather than only qualitative momentum.

    A single named-model/poll/market baseline is materially stronger evidence than
    qualitative "favorite"/"strong candidate" framing, and must not be diluted by
    volume when several such qualitative articles are pooled alongside it — the
    France 2026 World Cup regression (a 75% pooled estimate against a cited Opta
    baseline of 18.83%) is exactly this failure. Returns ``multiplier`` when at
    least one extracted claim from this source carries an explicit
    ``quantitative_estimate``, else ``1.0`` (neutral, no change to today's weighting).
    """
    if any(q is not None for q in quantitative_estimates):
        return multiplier
    return 1.0


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
    sem_p = std_p / math.sqrt(n) if n > 1 else std_p
    ci_low_p = clamp(pooled_p - 1.96 * sem_p, clamp_eps, 1.0 - clamp_eps)
    ci_high_p = clamp(pooled_p + 1.96 * sem_p, clamp_eps, 1.0 - clamp_eps)

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
    lo_p = clamp(min(stance_to_prob(ci_low), p) - extra_p, 0.0, 1.0)
    hi_p = clamp(max(stance_to_prob(ci_high), p) + extra_p, 0.0, 1.0)
    # std is secondary (the CI is what callers display); bump it monotonically so
    # it can't claim more precision than the widened band.
    new_std = max(std, 2.0 * extra_p)
    return prob_to_stance(lo_p), prob_to_stance(hi_p), new_std
