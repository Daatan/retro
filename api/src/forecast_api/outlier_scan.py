"""Stage A of retro#526: measure stored Oracle estimates for outlier-ness
against **the evidence they were actually computed from**.

The question this answers is not "is this forecast right" — nothing here reads
an outcome — but "does this published number follow from its own pool, under
today's rules?". Five signals, all raw continuous values.

**There are deliberately no thresholds in this module.** Stage A produces
distributions; a human picks the cuts in the issue afterwards and Stage B acts
on them. A constant like ``if gap > 0.3`` appearing here would forfeit the whole
discipline — the point is that nobody has yet seen what these numbers look like,
so nobody is in a position to choose. ``"156 forecasts scored, zero outliers"``
is a complete and acceptable result.

Three things make this tractable, and each is load-bearing:

**The frozen roster.** ``context_snapshots.oracle_snapshot.sources[]`` is
exactly the rows the published number averaged (``snapshotSources =
pool.usableArticles`` in daatan's ``pooled-estimate.ts``), and
``EnrichedOracleSource`` carries every field ``PoolSourceInput`` needs. So a
recompute scores the pool as it stood, not today's grown pool — pool growth
cannot contaminate the measurement at all. This is what makes "a recompute is
not a replay" a solvable problem rather than a caveat.

**In-process aggregation.** :func:`~forecast_api.forecaster.run_pool_aggregate`
is importable and runs fully offline — no search, no LLM, ~0.03 s per call
against ~hours over HTTP under the 60/min rate limit. Every number below comes
from the real estimator functions; nothing here reimplements pooling math.

**Leave-one-out attribution.** ``contribution_i = p(all) − p(all − i)`` run
through the actual aggregator, not a client-side weight share. A weight share
would happily report "this row dominates" about a row the estimator had already
collapsed as a syndicated duplicate, downweighted as a cluster member, capped by
``max_source_share`` or demoted as an invalid settlement vote. LOO cannot,
because it goes through all of those.

Two clocks are used on purpose, and the difference between them is itself a
result:

* **today's rules** (wall clock) — the primary artifact. Recency is recomputed
  against now, exactly as a live recompute would, so the signals describe what
  the estimator says about this pool *today*.
* **as-published** (:func:`frozen_clock` at the snapshot's ``createdAt``) — the
  reproduction check. Without it, every stored number would appear to disagree
  with its recompute purely because its articles have aged since, and a real
  field-mapping defect would be invisible inside that decay.
"""
from __future__ import annotations

import math
import statistics
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Awaitable, Callable, Iterator, Optional, Sequence

from . import aggregation as _aggregation_mod
from . import forecaster as _forecaster_mod
from .aggregation import (
    effective_sample_size,
    recency_weight,
    relevance_weight,
    settlement_vote_validity,
    stance_to_prob,
)
from .config import settings
from .models import PoolAggregateRequest, PoolAggregateResponse, PoolSourceInput

#: The clean-corpus boundary from the system model — a *column to stratify on*
#: (S5), not a footnote re-derived by hand at reporting time.
CLEAN_CORPUS_START = "2026-08-04"

#: daatan's publish-time clamp (`stanceToPercent`, `oracle-snapshot.ts:17`).
#: Reproduced here rather than approximated, because the reproduction check (V6)
#: compares against a *stored* value that went through exactly this map — an
#: unclamped comparison would report a disagreement at every interval endpoint.
PUBLISH_PERCENT_MIN = 1
PUBLISH_PERCENT_MAX = 99

Aggregator = Callable[[PoolAggregateRequest], Awaitable[PoolAggregateResponse]]


def stance_to_percent(v: float) -> int:
    """daatan's `stanceToPercent`: stance [-1,1] → published integer percent."""
    percent = round(((v + 1.0) / 2.0) * 100)
    return min(PUBLISH_PERCENT_MAX, max(PUBLISH_PERCENT_MIN, percent))


def config_fingerprint() -> dict:
    """The settings that decide what this run measured.

    Written into the artifact so a scan posted to an issue can be told apart
    from one run under different tuning — the whole Stage A/B design assumes the
    two halves were measured under the same estimator.
    """
    return {
        "settlement_stance": settings.settlement_stance,
        "settlement_quality_floor": settings.settlement_quality_floor,
        "settlement_min_sources": settings.settlement_min_sources,
        "settlement_revalidate": settings.settlement_revalidate,
        "pool_dispersion_floor": settings.pool_dispersion_floor,
        "recency_half_life_days": settings.recency_half_life_days,
        "recency_floor": settings.recency_floor,
        "relevance_weight_floor": settings.relevance_weight_floor,
        "decisiveness_floor": settings.decisiveness_floor,
        "cluster_downweight_exponent": settings.cluster_downweight_exponent,
        "max_source_share": settings.max_source_share,
        "evidence_class_weight_unclassified_cap": settings.evidence_class_weight_unclassified_cap,
    }


# ── Clock freezing ──────────────────────────────────────────────────────────


@contextmanager
def frozen_clock(moment: datetime) -> Iterator[None]:
    """Pin ``datetime.now()`` for the duration of a recompute.

    **Both** namespaces must be patched. ``forecaster`` and ``aggregation`` each
    do ``from datetime import datetime``, so each holds its own module-level
    binding: ``forecaster`` stamps ``ref_date`` for every row's recency decay,
    and ``aggregation`` reads the clock independently inside settlement-vote
    revalidation and the deadline-glide check. Patching one and not the other
    produces a recompute whose recency is as-published but whose settlement
    validity is judged today — a silent hybrid that is neither clock, and the
    unit test pins that both patches are load-bearing.
    """
    frozen = type(
        "FrozenDatetime",
        (datetime,),
        {"now": classmethod(lambda cls, tz=None: moment)},
    )
    originals = [(mod, mod.datetime) for mod in (_aggregation_mod, _forecaster_mod)]
    for mod, _ in originals:
        mod.datetime = frozen
    try:
        yield
    finally:
        for mod, original in originals:
            mod.datetime = original


# ── Input parsing ───────────────────────────────────────────────────────────

#: What daatan's own `usable` filter (`recomputeFromPool`, evidence-pool.ts:563)
#: requires of a pool row before it may vote. Mirrored exactly: a roster row
#: missing any of these was never in the published average either, so dropping
#: it here reproduces the published pool rather than shrinking it.
REQUIRED_SOURCE_SCALARS = ("stance", "certainty", "credibilityWeight", "relevanceScore")


def build_pool_sources(sources: Sequence[dict]) -> tuple[list[PoolSourceInput], int, int]:
    """Map a frozen ``oracle_snapshot.sources[]`` roster onto the wire model.

    Returns ``(inputs, n_incomplete, n_invalid)``. The field mapping is
    deliberately identical to daatan's ``recomputeFromPool`` body — that
    equivalence is asserted against a captured prod payload (V7), because a
    field-mapping slip does not fail, it just reads as an outlier.

    ``source_id`` is **never** set: ``recomputeFromPool`` does not send it, and
    the snapshot's ``sourceId`` is the pool-row cuid, not the leaderboard outlet
    id ``cap_source_mass`` groups on. Sending it would group rows by a key that
    is unique per row — inert today at ``max_source_share = 1.0``, wrong the
    moment anyone lowers it.
    """
    inputs: list[PoolSourceInput] = []
    incomplete = 0
    invalid = 0
    for s in sources:
        if any(s.get(k) is None for k in REQUIRED_SOURCE_SCALARS):
            incomplete += 1
            continue
        try:
            inputs.append(
                PoolSourceInput(
                    stance=s["stance"],
                    certainty=s["certainty"],
                    credibility_weight=s["credibilityWeight"],
                    relevance_score=s["relevanceScore"],
                    evidence_weight=s.get("evidenceWeight"),
                    published_date=s.get("publishedAt"),
                    settled=bool(s.get("settled") or False),
                    settlement_event_date=s.get("settlementEventDate"),
                    claims_detail=s.get("claimsDetail"),
                    url=s.get("url"),
                    outlet=s.get("outletName"),
                    evidence_class=s.get("evidenceClass"),
                )
            )
        except Exception:  # noqa: BLE001 — a row the model rejects is a result
            invalid += 1
    return inputs, incomplete, invalid


@dataclass(frozen=True)
class SnapshotRecord:
    """One scorable published estimate: the claim, the number it published, and
    the roster it published that number from."""

    prediction_id: str
    claim: str
    status: Optional[str]
    outcome_type: Optional[str]
    resolved_at: Optional[str]
    claim_created_at: Optional[str]
    claim_direction: Optional[str]
    claim_deadline: Optional[str]
    claim_archetype: Optional[str]
    confidence: Optional[float]
    ai_ci_low: Optional[float]
    ai_ci_high: Optional[float]
    snapshot_id: str
    snapshot_created_at: str
    origin: Optional[str]
    kind: Optional[str]
    stored_mean: Optional[float]
    stored_std: Optional[float]
    stored_ci_low: Optional[float]
    stored_ci_high: Optional[float]
    stored_articles_used: Optional[int]
    stored_settled: Optional[bool]
    sources: tuple = ()

    @property
    def settled_status(self) -> bool:
        return bool(self.status and self.status.startswith("RESOLVED"))


def _lower_direction(value: Optional[str]) -> Optional[str]:
    """daatan stores ``ARRIVAL``/``SURVIVAL``; the Oracle accepts only the
    lowercase pair and treats anything else as absent."""
    v = (value or "").lower() or None
    return v if v in ("arrival", "survival") else None


def _lower_archetype(value: Optional[str]) -> Optional[str]:
    v = (value or "").lower() or None
    return v if v in ("scheduled", "diffuse", "threshold", "none") else None


def parse_record(raw: dict) -> tuple[Optional[SnapshotRecord], Optional[str]]:
    """Parse one dumped row. Returns ``(record, skip_reason)`` — never raises.

    A skip is a *reported* outcome, not a silent drop: the counts are printed
    and land in the artifact, because "156 of 192 were scorable" is part of the
    Stage A answer and an unexplained shortfall is the failure mode.
    """
    snap = raw.get("oracle_snapshot")
    if not isinstance(snap, dict):
        return None, "no_oracle_snapshot"
    if snap.get("insufficient") or snap.get("empty"):
        return None, "abstained"
    sources = snap.get("sources")
    if not isinstance(sources, list) or not sources:
        return None, "no_sources"
    return (
        SnapshotRecord(
            prediction_id=str(raw.get("pid")),
            claim=str(raw.get("claim") or ""),
            status=raw.get("status"),
            outcome_type=raw.get("outcome_type"),
            resolved_at=raw.get("resolved_at"),
            claim_created_at=raw.get("claim_created_at"),
            claim_direction=_lower_direction(raw.get("claim_direction")),
            claim_deadline=(raw.get("claim_deadline") or None),
            claim_archetype=_lower_archetype(raw.get("claim_archetype")),
            confidence=raw.get("confidence"),
            ai_ci_low=raw.get("ai_ci_low"),
            ai_ci_high=raw.get("ai_ci_high"),
            snapshot_id=str(raw.get("snapshot_id")),
            snapshot_created_at=str(raw.get("snapshot_created_at") or ""),
            origin=raw.get("origin"),
            kind=raw.get("kind"),
            stored_mean=snap.get("mean"),
            stored_std=snap.get("std"),
            stored_ci_low=snap.get("ciLow"),
            stored_ci_high=snap.get("ciHigh"),
            stored_articles_used=snap.get("articlesUsed"),
            stored_settled=snap.get("settled"),
            sources=tuple(s for s in sources if isinstance(s, dict)),
        ),
        None,
    )


# ── Per-row weights ─────────────────────────────────────────────────────────


def row_weights(sources: Sequence[PoolSourceInput], ref_date: str) -> list[float]:
    """Re-derive each row's voting weight exactly as ``run_pool_aggregate``'s
    per-source loop does (``forecaster.py``, the ``weight = ...`` line).

    This is the **pre-pooling** weight: syndication dedup, cluster downweighting,
    ``cap_source_mass`` and settlement demotion all happen inside
    ``aggregate_pool`` and are not reflected here. That is exactly why the LOO
    attribution exists alongside it, and why V8 checks the two agree on which row
    dominates — when they stop agreeing, this re-derivation has drifted from the
    estimator and the mass signals (S4) are no longer describing the real pool.
    """
    out = []
    for s in sources:
        evidence_weight = (
            s.evidence_weight
            if s.evidence_weight is not None
            else min(s.certainty, settings.evidence_class_weight_unclassified_cap)
        )
        rweight = recency_weight(
            s.published_date, ref_date, settings.recency_half_life_days, floor=settings.recency_floor
        )
        out.append(
            s.credibility_weight * evidence_weight * rweight * relevance_weight(s.relevance_score)
        )
    return out


def _weighted_median(values: Sequence[float], weights: Sequence[float]) -> Optional[float]:
    """Weighted median — the value at which cumulative weight crosses half.

    Paired with the unweighted median on purpose (S3): when the two disagree,
    the pool's centre is a property of its weighting rather than of its
    evidence, which is a different kind of finding from a genuinely split pool.
    """
    pairs = sorted(zip(values, weights), key=lambda vw: vw[0])
    total = sum(w for _, w in pairs)
    if not pairs:
        return None
    if total <= 0:
        return statistics.median([v for v, _ in pairs])
    seen = 0.0
    for v, w in pairs:
        seen += w
        if seen >= total / 2.0:
            return v
    return pairs[-1][0]


# ── Scoring ─────────────────────────────────────────────────────────────────


@dataclass
class ScanRow:
    """One scored snapshot. Every field is a raw measurement — no verdicts."""

    prediction_id: str
    snapshot_id: str
    claim: str
    status: Optional[str] = None
    resolved: bool = False
    snapshot_created_at: str = ""
    origin: Optional[str] = None
    kind: Optional[str] = None

    # roster shape
    n_roster: int = 0
    n_scored: int = 0
    n_incomplete: int = 0
    n_invalid: int = 0

    # the published number, and what it recomputes to
    stored_mean_pct: Optional[float] = None
    recomputed_mean_pct: Optional[int] = None
    repro_mean_pct: Optional[int] = None
    repro_delta_pct: Optional[float] = None
    repro_agrees: Optional[bool] = None
    recompute_reason: Optional[str] = None
    confidence_pct: Optional[float] = None
    confidence_divergence_pct: Optional[float] = None

    # S1 — pinned-extreme gap
    settled_now: Optional[bool] = None
    s1_pin_gap: Optional[float] = None
    s1_pooled_p_no_pin: Optional[float] = None
    s1_no_pin_reason: Optional[str] = None
    # S1b — what the pin rests on
    s1b_pin_votes: int = 0
    s1b_valid_pin_votes: int = 0
    s1b_winning_weight: Optional[float] = None
    s1b_winning_weight_share: Optional[float] = None
    s1b_votes_demoted: int = 0
    s1b_suppressed: Optional[bool] = None
    s1b_suppression_reason: Optional[str] = None

    # S2 — band width (percentage points)
    s2_stored_band_pct: Optional[float] = None
    s2_snapshot_band_pct: Optional[float] = None
    s2_recomputed_band_pct: Optional[float] = None

    # S3 — centre gap (probability units, 0-1)
    s3_centre_gap: Optional[float] = None
    s3_centre_gap_weighted: Optional[float] = None
    s3_median_p: Optional[float] = None
    s3_weighted_median_p: Optional[float] = None

    # S4 — mass and multiplicity
    s4_n_eff: Optional[float] = None
    s4_n_eff_ratio: Optional[float] = None
    s4_max_weight_share: Optional[float] = None
    s4_evidence_mass: Optional[float] = None
    s4_age_adjusted_mass: Optional[float] = None
    s4_articles_used: Optional[int] = None
    s4_near_zero_weight_rows: int = 0
    s4_single_article: Optional[bool] = None

    # S5 — corpus-hygiene covariates (stratifiers, not outlier signals)
    s5_snapshot_age_days: Optional[float] = None
    s5_clean_corpus: Optional[bool] = None
    s5_claims_detail_coverage: Optional[float] = None
    s5_carried_forward_share: Optional[float] = None
    s5_settled_row_share: Optional[float] = None

    # leave-one-out attribution
    loo_max_abs_delta: Optional[float] = None
    loo_max_delta_url: Optional[str] = None
    loo_agrees_with_max_weight: Optional[bool] = None


def _mean_pct_of(res: PoolAggregateResponse) -> Optional[int]:
    if res.insufficient_data:
        return None
    return stance_to_percent(res.mean)


def _request_for(rec: SnapshotRecord, sources: Sequence[PoolSourceInput]) -> PoolAggregateRequest:
    """Build the recompute request the way daatan builds it.

    ``question`` is sent whenever the claim clears retro's own ``min_length=5``:
    without it the settlement match gate cannot run and skips with
    ``reason=no_question``, so a pin the live path would have gated would sail
    through the recompute and read as a legitimate published pin.
    """
    claim = (rec.claim or "").strip()
    return PoolAggregateRequest(
        sources=list(sources),
        question=claim if len(claim) >= 5 else None,
        claim_direction=rec.claim_direction,
        claim_deadline=rec.claim_deadline,
        claim_created_at=rec.claim_created_at,
        claim_archetype=rec.claim_archetype,
    )


def _parse_moment(value: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00").replace(" ", "T")).replace(tzinfo=None)
    except (ValueError, AttributeError):
        return None


async def score_record(
    rec: SnapshotRecord,
    aggregate: Aggregator,
    *,
    now: Optional[datetime] = None,
    loo: bool = True,
) -> tuple[Optional[ScanRow], Optional[str]]:
    """Score one snapshot. Returns ``(row, skip_reason)`` — never raises.

    ``aggregate`` is injected rather than imported so the unit tests can score
    against a stub without importing the whole estimator, and so the driver can
    swap in an HTTP client for the parity check (V5) without a second code path.
    """
    pool, incomplete, invalid = build_pool_sources(rec.sources)
    if not pool:
        return None, "no_usable_sources"

    now = now or datetime.now()
    ref_date = now.strftime("%Y-%m-%d")
    req = _request_for(rec, pool)

    res = await aggregate(req)
    weights = row_weights(pool, ref_date)
    total_w = sum(weights)

    row = ScanRow(
        prediction_id=rec.prediction_id,
        snapshot_id=rec.snapshot_id,
        claim=rec.claim,
        status=rec.status,
        resolved=rec.settled_status,
        snapshot_created_at=rec.snapshot_created_at,
        origin=rec.origin,
        kind=rec.kind,
        n_roster=len(rec.sources),
        n_scored=len(pool),
        n_incomplete=incomplete,
        n_invalid=invalid,
        stored_mean_pct=rec.stored_mean,
        confidence_pct=rec.confidence,
        recomputed_mean_pct=_mean_pct_of(res),
        recompute_reason=res.reason,
        settled_now=res.settled,
    )

    # ── as-published reproduction (V6) ─────────────────────────────────────
    # Recompute under the clock the snapshot was written at, so a disagreement
    # means a mapping or rules change rather than a fortnight of recency decay.
    moment = _parse_moment(rec.snapshot_created_at)
    if moment is not None:
        with frozen_clock(moment):
            frozen_res = await aggregate(req)
        row.repro_mean_pct = _mean_pct_of(frozen_res)
        if row.repro_mean_pct is not None and rec.stored_mean is not None:
            row.repro_delta_pct = row.repro_mean_pct - rec.stored_mean
            row.repro_agrees = abs(row.repro_delta_pct) < 0.5

    # `predictions.confidence` is NOT a reproduction target — the requote cron
    # overwrites it daily for origin='clock' — so its distance from the snapshot
    # is reported as a measure of clock glide, never as a disagreement.
    if rec.confidence is not None and rec.stored_mean is not None:
        row.confidence_divergence_pct = rec.confidence - rec.stored_mean

    if res.insufficient_data:
        # The pool no longer produces an estimate at all. That IS the finding
        # for this row; the remaining signals describe a pooled mean that was
        # never computed, so they stay null rather than carrying placeholders.
        row.s4_evidence_mass = res.evidence_mass
        row.s4_n_eff = res.n_eff
        row.s4_age_adjusted_mass = res.age_adjusted_mass
        _fill_s5(row, rec, pool, moment, now)
        return row, None

    published_p = (rec.stored_mean / 100.0) if rec.stored_mean is not None else stance_to_prob(res.mean)

    # ── S1 / S1b — the pin ─────────────────────────────────────────────────
    if res.settled:
        unpinned = [s.model_copy(update={"settled": False}) for s in pool]
        no_pin = await aggregate(_request_for(rec, unpinned))
        if no_pin.insufficient_data:
            # A pool that abstains without its pin is a strictly stronger
            # version of the same finding, and reporting a gap of 0 here would
            # bury it among the well-supported pins.
            row.s1_no_pin_reason = no_pin.reason
        else:
            row.s1_pooled_p_no_pin = stance_to_prob(no_pin.mean)
            row.s1_pin_gap = abs(stance_to_prob(res.mean) - row.s1_pooled_p_no_pin)

        winning_positive = res.mean >= 0
        winning_weight = 0.0
        for i, s in enumerate(pool):
            if not s.settled:
                continue
            row.s1b_pin_votes += 1
            reason = settlement_vote_validity(
                s.stance,
                s.settlement_event_date,
                s.published_date,
                rec.claim_direction,
                rec.claim_deadline,
                rec.claim_created_at,
                rec.claim_archetype,
                today=ref_date,
                post_deadline_grace_days=settings.settlement_post_deadline_grace_days,
            )
            if reason is not None:
                continue
            row.s1b_valid_pin_votes += 1
            if (s.stance >= 0) == winning_positive:
                winning_weight += weights[i]
        row.s1b_winning_weight = winning_weight
        row.s1b_winning_weight_share = (winning_weight / total_w) if total_w > 0 else None
    row.s1b_votes_demoted = res.settlement_votes_demoted
    row.s1b_suppressed = res.settlement_suppressed
    row.s1b_suppression_reason = res.settlement_suppression_reason

    # ── S2 — band width, in published percentage points ────────────────────
    if rec.ai_ci_low is not None and rec.ai_ci_high is not None:
        row.s2_stored_band_pct = rec.ai_ci_high - rec.ai_ci_low
    if rec.stored_ci_low is not None and rec.stored_ci_high is not None:
        row.s2_snapshot_band_pct = rec.stored_ci_high - rec.stored_ci_low
    # Stance-space CI onto the same scale: `stanceToPercent`'s slope is 50.
    row.s2_recomputed_band_pct = (res.ci_high - res.ci_low) * 50.0

    # ── S3 — where the pool's centre sits vs what was published ────────────
    row_ps = [stance_to_prob(s.stance) for s in pool]
    row.s3_median_p = statistics.median(row_ps)
    row.s3_weighted_median_p = _weighted_median(row_ps, weights)
    row.s3_centre_gap = abs(published_p - row.s3_median_p)
    if row.s3_weighted_median_p is not None:
        row.s3_centre_gap_weighted = abs(published_p - row.s3_weighted_median_p)

    # ── S4 — mass and multiplicity ─────────────────────────────────────────
    row.s4_n_eff = res.n_eff
    row.s4_articles_used = res.articles_used
    row.s4_n_eff_ratio = (res.n_eff / res.articles_used) if res.articles_used else None
    row.s4_evidence_mass = res.evidence_mass
    row.s4_age_adjusted_mass = res.age_adjusted_mass
    row.s4_max_weight_share = (max(weights) / total_w) if total_w > 0 else None
    row.s4_near_zero_weight_rows = sum(1 for w in weights if w < 0.01)
    row.s4_single_article = res.articles_used == 1

    _fill_s5(row, rec, pool, moment, now)

    # ── leave-one-out attribution ──────────────────────────────────────────
    if loo and len(pool) > 1:
        base_p = stance_to_prob(res.mean)
        deltas: list[float] = []
        for i in range(len(pool)):
            trimmed = pool[:i] + pool[i + 1 :]
            one_out = await aggregate(_request_for(rec, trimmed))
            # An abstention without row i means that row was holding the pool
            # up — a maximal contribution, not a missing measurement.
            deltas.append(
                float("nan") if one_out.insufficient_data else base_p - stance_to_prob(one_out.mean)
            )
        finite = [(abs(d), i) for i, d in enumerate(deltas) if not math.isnan(d)]
        abstaining = [i for i, d in enumerate(deltas) if math.isnan(d)]
        top_i = abstaining[0] if abstaining else (max(finite)[1] if finite else None)
        if top_i is not None:
            row.loo_max_abs_delta = None if abstaining else max(finite)[0]
            row.loo_max_delta_url = pool[top_i].url
            row.loo_agrees_with_max_weight = top_i == max(range(len(weights)), key=lambda i: weights[i])

    return row, None


def _fill_s5(
    row: ScanRow,
    rec: SnapshotRecord,
    pool: Sequence[PoolSourceInput],
    moment: Optional[datetime],
    now: datetime,
) -> None:
    """Corpus-hygiene covariates. These are stratifiers, not outlier signals —
    a row is not suspicious for being old, but a signal that only fires on old
    rows is a corpus artefact rather than an estimator finding."""
    if moment is not None:
        row.s5_snapshot_age_days = (now - moment).total_seconds() / 86400.0
    row.s5_clean_corpus = rec.snapshot_created_at[:10] >= CLEAN_CORPUS_START if rec.snapshot_created_at else None
    if pool:
        row.s5_claims_detail_coverage = sum(1 for s in pool if s.claims_detail) / len(pool)
        row.s5_settled_row_share = sum(1 for s in pool if s.settled) / len(pool)
    if rec.sources:
        row.s5_carried_forward_share = sum(
            1 for s in rec.sources if s.get("carriedForward")
        ) / len(rec.sources)


# ── Reporting ───────────────────────────────────────────────────────────────

#: The signals whose distributions get posted to the issue. Ordered as the plan
#: lists them so the table reads the same way the design does.
SIGNAL_FIELDS: tuple[tuple[str, str], ...] = (
    ("s1_pin_gap", "S1  pin gap |p_pinned - p_no_pin|"),
    ("s1b_winning_weight", "S1b pin winning-direction weight"),
    ("s1b_winning_weight_share", "S1b pin weight share of pool"),
    ("s2_stored_band_pct", "S2  stored CI band (pp)"),
    ("s2_snapshot_band_pct", "S2  snapshot CI band (pp)"),
    ("s2_recomputed_band_pct", "S2  recomputed CI band (pp)"),
    ("s3_centre_gap", "S3  centre gap (unweighted)"),
    ("s3_centre_gap_weighted", "S3  centre gap (weighted)"),
    ("s4_n_eff", "S4  n_eff"),
    ("s4_n_eff_ratio", "S4  n_eff / articles_used"),
    ("s4_max_weight_share", "S4  max(w) / sum(w)"),
    ("s4_evidence_mass", "S4  evidence_mass"),
    ("s4_near_zero_weight_rows", "S4  rows with w < 0.01"),
    ("s5_snapshot_age_days", "S5  snapshot age (days)"),
    ("s5_claims_detail_coverage", "S5  claims_detail coverage"),
    ("s5_carried_forward_share", "S5  carried_forward share"),
    ("loo_max_abs_delta", "LOO max |contribution|"),
    ("confidence_divergence_pct", "clock glide: confidence - snapshot (pp)"),
)


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile over an already-sorted-able sequence."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[int(pos)]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


@dataclass(frozen=True)
class Distribution:
    field: str
    label: str
    n: int
    mean: float
    p10: float
    p50: float
    p90: float
    max: float


def distributions(rows: Sequence[ScanRow]) -> list[Distribution]:
    """Per-signal ``n | mean | p10 | p50 | p90 | max``.

    ``n`` counts rows where the signal is *defined*, not rows scanned — S1 only
    exists on pinned pools, and averaging it over unpinned ones would report a
    pin gap that is mostly zeros contributed by pools with no pin.
    """
    out = []
    for fieldname, label in SIGNAL_FIELDS:
        vals = [
            float(getattr(r, fieldname))
            for r in rows
            if getattr(r, fieldname, None) is not None and not isinstance(getattr(r, fieldname), bool)
        ]
        if not vals:
            out.append(Distribution(fieldname, label, 0, float("nan"), float("nan"),
                                    float("nan"), float("nan"), float("nan")))
            continue
        out.append(
            Distribution(
                field=fieldname,
                label=label,
                n=len(vals),
                mean=statistics.fmean(vals),
                p10=percentile(vals, 0.10),
                p50=percentile(vals, 0.50),
                p90=percentile(vals, 0.90),
                max=max(vals),
            )
        )
    return out


def top_rows(rows: Sequence[ScanRow], fieldname: str, k: int = 10) -> list[ScanRow]:
    """The k rows with the largest value of ``fieldname`` — the actual forecasts
    a human then looks at. A distribution alone cannot be acted on."""
    scored = [r for r in rows if getattr(r, fieldname, None) is not None
              and not isinstance(getattr(r, fieldname), bool)]
    return sorted(scored, key=lambda r: float(getattr(r, fieldname)), reverse=True)[:k]


@dataclass
class ScanReport:
    rows: list[ScanRow] = field(default_factory=list)
    skipped: dict = field(default_factory=dict)
    total_input: int = 0

    @property
    def scored(self) -> int:
        return len(self.rows)

    @property
    def repro_agreement(self) -> Optional[float]:
        """Share of scored rows whose as-published recompute reproduced the
        stored number. Reported, never asserted: the post-2026-08-04 corpus
        should be near 100% and older rows should not be, and the shape of that
        curve is itself a Stage A result."""
        checked = [r for r in self.rows if r.repro_agrees is not None]
        if not checked:
            return None
        return sum(1 for r in checked if r.repro_agrees) / len(checked)

    @property
    def loo_weight_agreement(self) -> Optional[float]:
        """V8: share of pools where the heaviest row is also the most
        influential by leave-one-out. Well below 1 means ``row_weights`` has
        drifted from the estimator's own per-source loop and the S4 mass signals
        are describing a pool nobody computed."""
        checked = [r for r in self.rows if r.loo_agrees_with_max_weight is not None]
        if not checked:
            return None
        return sum(1 for r in checked if r.loo_agrees_with_max_weight) / len(checked)

    def to_artifact(self, *, label: str, git_commit: str, deployed_commit: Optional[str]) -> dict:
        return {
            "label": label,
            "git_commit": git_commit,
            "deployed_commit": deployed_commit,
            "config_fingerprint": config_fingerprint(),
            "n_input": self.total_input,
            "n_snapshots": self.scored,
            "n_skipped": sum(self.skipped.values()),
            "skipped_by_reason": dict(self.skipped),
            "repro_agreement": self.repro_agreement,
            "loo_weight_agreement": self.loo_weight_agreement,
            "rows": [asdict(r) for r in self.rows],
        }


async def scan(
    raw_records: Sequence[dict],
    aggregate: Aggregator,
    *,
    now: Optional[datetime] = None,
    loo: bool = True,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> ScanReport:
    """Score a whole dump. Every failure is counted and reported, never raised —
    one unparseable row must not cost the other 155 their measurement."""
    report = ScanReport(total_input=len(raw_records))
    for i, raw in enumerate(raw_records):
        rec, skip = parse_record(raw)
        if rec is None:
            report.skipped[skip or "unparseable"] = report.skipped.get(skip or "unparseable", 0) + 1
            continue
        try:
            row, skip = await score_record(rec, aggregate, now=now, loo=loo)
        except Exception as exc:  # noqa: BLE001 — a failing row is a result
            report.skipped[f"error:{type(exc).__name__}"] = (
                report.skipped.get(f"error:{type(exc).__name__}", 0) + 1
            )
            continue
        if row is None:
            report.skipped[skip or "unscorable"] = report.skipped.get(skip or "unscorable", 0) + 1
            continue
        report.rows.append(row)
        if on_progress:
            on_progress(i + 1, len(raw_records))
    return report


# ── Stratification ──────────────────────────────────────────────────────────


def split_by_settled(rows: Sequence[ScanRow]) -> dict[str, list[ScanRow]]:
    """Resolved vs unresolved. Only 21 of the corpus are resolved, so this split
    is about whether a signal is measurable on the outcome-bearing subset at
    all — not yet about calibration."""
    return {
        "resolved": [r for r in rows if r.resolved],
        "unresolved": [r for r in rows if not r.resolved],
    }


def split_by_corpus(rows: Sequence[ScanRow]) -> dict[str, list[ScanRow]]:
    """Pre- vs post-``CLEAN_CORPUS_START``. A signal that only fires on the
    pre-cutover half is a corpus artefact; one that fires on both is about the
    estimator."""
    return {
        f"post-{CLEAN_CORPUS_START}": [r for r in rows if r.s5_clean_corpus],
        f"pre-{CLEAN_CORPUS_START}": [r for r in rows if r.s5_clean_corpus is False],
    }
