"""Replay the article reduction over *persisted* claims (F1 item 3, retro#398).

``reduce_article`` is pure — claims plus configuration in, the article's five
fused scalars out, no globals — and that purity is the whole point of persisting
``claims_detail``: the same function can be re-run over stored rows. Until this
module, nothing did. The derivation ran at extraction time over the claims just
built, and no code ever read the persisted column back.

That gap is why retro#398 exists separately from retro#364. #364 delivers the
column and is explicitly scoped to *additive persistence only* — "the live
estimate must be bit-identical before and after" — so it can close, correctly
and completely, while every consumer that motivated it is still impossible:
retroactive backtesting, R1 claim-level fitting, the R6 shadow pool, and
measuring extractor instability rather than assuming it.

What a replay proves, and what it does not: agreement says a stored row still
reduces to the numbers it shipped with, i.e. the row is self-describing and can
be re-scored. Disagreement is a real finding either way round — the reduction
changed without the stored rows being migrated, or a scalar reads something the
claim layer does not keep. Both are silent today.

Deliberately **not** a fixture test: ``test_claims_detail.py`` already replays
this over in-memory extraction output. What has never been checked is whether
rows *written weeks ago by an older build* still reduce to what they carry.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

from .config import settings
from .forecaster import reduce_article
from .models import ClaimDetail

#: Scalars a stored row is expected to reproduce. Floats are compared at the
#: precision the wire actually carries (3dp) — the pool row stores rounded
#: values, so comparing raw floats would report drift that was never persisted.
_ROUNDED_3DP = ("stance", "certainty", "evidence_weight", "fact_signal")
COMPARED_FIELDS = _ROUNDED_3DP + (
    "evidence_class", "settled", "settlement_event_date", "quantitative_estimate",
    "claims", "event_actors", "event_target", "is_occurrence", "verified",
    "fact_signal_absent_reason", "facet",
    # retro#681. Rows written before the field existed carry neither column, and
    # `replay_row` skips a field the export does not carry — so adding these here
    # checks the new reduction on new rows without inventing drift on old ones.
    "reader_confidence_level", "reader_confidence_traps",
)


@dataclass(frozen=True)
class FieldDiff:
    field: str
    stored: Any
    replayed: Any


@dataclass(frozen=True)
class RowResult:
    """One stored row's verdict. ``error`` is set only for rows that could not
    be replayed at all — a row with no ``claims_detail`` is *skipped*, not
    failed, because forward-only population is expected (nothing was
    backfilled) and counting it as a mismatch would drown the real signal."""
    key: str
    skipped_reason: Optional[str] = None
    error: Optional[str] = None
    diffs: tuple[FieldDiff, ...] = ()

    @property
    def replayed_ok(self) -> bool:
        return self.error is None and self.skipped_reason is None and not self.diffs


def _norm(field: str, value: Any) -> Any:
    """Put a stored value and a replayed one on the same footing before comparing.

    ``settled`` is handled BEFORE the null guard, and that ordering is the whole
    point: the wire stores an unsettled article as ``None`` *or* ``False`` while
    the reduction always returns a bool, so a null guard that runs first reports
    ``stored=None replayed=False`` on every ordinary row. That is precisely what
    the first run against 268 real prod rows did — the fixtures never caught it,
    because a fixture built from the reduction always has a real bool there.
    """
    if field == "settled":
        return bool(value)
    if value is None:
        return None
    if field in _ROUNDED_3DP:
        return round(float(value), 3)
    if field in ("claims", "reader_confidence_traps"):
        return list(value)
    return value


def replay_row(row: dict) -> RowResult:
    """Replay one stored pool row. Never raises — a bad row is a result, not a crash."""
    key = str(row.get("url") or row.get("id") or "<unkeyed>")
    raw = row.get("claims_detail")
    if not raw:
        return RowResult(key=key, skipped_reason="no_claims_detail")
    try:
        claims = [ClaimDetail.model_validate(c) for c in raw]
        replayed = reduce_article(
            claims,
            settlement_min_stance=settings.settlement_min_claim_stance,
            settlement_min_certainty=settings.settlement_min_claim_certainty,
            class_weights=settings.evidence_class_weight,
            class_weight_default=settings.evidence_class_weight_default,
            class_weight_unclassified_cap=settings.evidence_class_weight_unclassified_cap,
        )
    except Exception as exc:  # noqa: BLE001 — any failure is a reportable result
        return RowResult(key=key, error=f"{type(exc).__name__}: {exc}")

    diffs = []
    for field in COMPARED_FIELDS:
        if field not in row:
            continue  # the export did not carry this column; not a mismatch
        stored = _norm(field, row.get(field))
        got = _norm(field, getattr(replayed, field))
        if stored != got:
            diffs.append(FieldDiff(field=field, stored=stored, replayed=got))
    return RowResult(key=key, diffs=tuple(diffs))


@dataclass(frozen=True)
class ReplayReport:
    total: int
    replayed: int
    agreed: int
    skipped: int
    errored: int
    by_field: dict
    results: tuple[RowResult, ...]

    @property
    def coverage(self) -> float:
        """Share of rows carrying a replayable claim layer at all."""
        return (self.replayed / self.total) if self.total else 0.0

    @property
    def agreement(self) -> float:
        """Share of *replayable* rows that reproduced. Deliberately not over
        ``total``: rows predating the column say nothing about the reduction."""
        return (self.agreed / self.replayed) if self.replayed else 0.0


def replay_rows(rows: Sequence[dict]) -> ReplayReport:
    results = [replay_row(r) for r in rows]
    by_field: dict = {}
    for r in results:
        for d in r.diffs:
            by_field[d.field] = by_field.get(d.field, 0) + 1
    skipped = sum(1 for r in results if r.skipped_reason)
    errored = sum(1 for r in results if r.error)
    replayed = len(results) - skipped - errored
    return ReplayReport(
        total=len(results),
        replayed=replayed,
        agreed=sum(1 for r in results if r.replayed_ok),
        skipped=skipped,
        errored=errored,
        by_field=by_field,
        results=tuple(results),
    )
