"""Unit tests for the retro#526 Stage A scorer.

The point of most of these is not that the arithmetic is right — the arithmetic
is the estimator's, imported not reimplemented — but that the *harness* cannot
silently report a non-result: a mapping slip, a half-frozen clock, or a scan
that measured nothing must all be visible rather than plausible.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from forecast_api import aggregation as aggregation_mod
from forecast_api import forecaster as forecaster_mod
from forecast_api.aggregation import settlement_vote_validity
from forecast_api.forecaster import run_pool_aggregate
from forecast_api.models import PoolAggregateRequest, PoolAggregateResponse, PoolSourceInput
from forecast_api.outlier_scan import (
    CLEAN_CORPUS_START,
    ScanRow,
    build_pool_sources,
    distributions,
    frozen_clock,
    parse_record,
    percentile,
    row_weights,
    scan,
    score_record,
    split_by_corpus,
    stance_to_percent,
    top_rows,
)

TODAY = datetime.now().strftime("%Y-%m-%d")
RECENT = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")


def enriched(**over) -> dict:
    """An `oracle_snapshot.sources[]` entry in daatan's own camelCase shape."""
    base = {
        "sourceId": "cmpoolrow123",
        "sourceName": "Reuters",
        "url": "https://example.com/a",
        "stance": 0.4,
        "certainty": 0.7,
        "credibilityWeight": 1.0,
        "relevanceScore": 0.7,
        "evidenceWeight": 0.6,
        "publishedAt": RECENT,
        "settled": False,
        "settlementEventDate": None,
        "outletName": "Reuters",
        "evidenceClass": "reporting",
        "claimsDetail": None,
        "carriedForward": False,
    }
    base.update(over)
    return base


def record(sources, **over) -> dict:
    snap = {
        "mean": 70.0,
        "std": 10.0,
        "ciLow": 55.0,
        "ciHigh": 85.0,
        "articlesUsed": len(sources),
        "settled": False,
        "sources": sources,
    }
    snap.update(over.pop("snapshot", {}))
    base = {
        "pid": "pred-1",
        "claim": "Something testable happens before the deadline",
        "status": "ACTIVE",
        "outcome_type": "BINARY",
        "resolved_at": None,
        "claim_created_at": "2026-08-01T00:00:00",
        "claim_direction": "ARRIVAL",
        "claim_deadline": "2026-12-31",
        "claim_archetype": "SCHEDULED",
        "confidence": 70,
        "ai_ci_low": 55,
        "ai_ci_high": 85,
        "snapshot_id": "snap-1",
        "snapshot_created_at": "2026-08-10T12:00:00",
        "kind": "evidence",
        "origin": "news-indexer",
        "oracle_snapshot": snap,
    }
    base.update(over)
    return base


# ── field mapping ───────────────────────────────────────────────────────────


def test_build_pool_sources_maps_the_recompute_wire_shape():
    pool, incomplete, invalid = build_pool_sources([enriched()])
    assert (incomplete, invalid) == (0, 0)
    s = pool[0]
    assert (s.stance, s.certainty, s.credibility_weight, s.relevance_score) == (0.4, 0.7, 1.0, 0.7)
    assert s.evidence_weight == 0.6
    assert s.published_date == RECENT
    assert s.outlet == "Reuters"


def test_source_id_is_never_sent():
    """The snapshot's `sourceId` is the pool-row cuid, not the leaderboard outlet
    id that `cap_source_mass` groups on — `recomputeFromPool` does not send it,
    and sending it would group every row into its own singleton bucket."""
    pool, _, _ = build_pool_sources([enriched(sourceId="cmpoolrow123")])
    assert pool[0].source_id is None


def test_rows_missing_a_required_scalar_are_dropped_like_daatan_drops_them():
    """daatan's `usable` filter requires stance/certainty/credibilityWeight/
    relevanceScore, so such a row was never in the published average either."""
    pool, incomplete, invalid = build_pool_sources(
        [enriched(), enriched(stance=None), enriched(relevanceScore=None)]
    )
    assert len(pool) == 1
    assert incomplete == 2
    assert invalid == 0


def test_a_row_the_wire_model_rejects_is_counted_not_raised():
    pool, incomplete, invalid = build_pool_sources([enriched(stance=7.5), enriched()])
    assert len(pool) == 1 and invalid == 1 and incomplete == 0


def test_claims_detail_validates_from_the_stored_camelcase_json():
    """daatan persists retro's own snake_case claim fields verbatim, so the
    stored blob must validate against `ClaimDetail` with no translation — if it
    ever stops, the settlement match gate and the clusterer both go quiet."""
    detail = [{"claim": "X happened", "stance": 0.9, "certainty": 0.8, "evidence_class": "reported_fact"}]
    pool, _, invalid = build_pool_sources([enriched(claimsDetail=detail)])
    assert invalid == 0
    assert pool[0].claims_detail and pool[0].claims_detail[0].claim == "X happened"


# ── parsing / skip accounting ───────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw,reason",
    [
        ({"oracle_snapshot": None}, "no_oracle_snapshot"),
        ({"oracle_snapshot": {"sources": [], "insufficient": True, "reason": "off_topic"}}, "abstained"),
        ({"oracle_snapshot": {"sources": [], "empty": True, "reason": "no_articles"}}, "abstained"),
        ({"oracle_snapshot": {"sources": []}}, "no_sources"),
    ],
)
def test_unscorable_records_report_a_reason(raw, reason):
    rec, skip = parse_record(raw)
    assert rec is None and skip == reason


def test_claim_direction_is_lowercased_and_anything_else_becomes_none():
    """daatan stores ARRIVAL/SURVIVAL; the Oracle accepts only the lowercase
    pair and treats anything else as absent (backtest_fact_signal_gate:127)."""
    assert parse_record(record([enriched()], claim_direction="ARRIVAL"))[0].claim_direction == "arrival"
    assert parse_record(record([enriched()], claim_direction="SIDEWAYS"))[0].claim_direction is None
    assert parse_record(record([enriched()], claim_direction=None))[0].claim_direction is None


# ── the clock ───────────────────────────────────────────────────────────────


def test_frozen_clock_patches_and_restores_both_namespaces():
    before = (aggregation_mod.datetime, forecaster_mod.datetime)
    with frozen_clock(datetime(2020, 1, 1)):
        assert aggregation_mod.datetime.now() == datetime(2020, 1, 1)
        assert forecaster_mod.datetime.now() == datetime(2020, 1, 1)
    assert (aggregation_mod.datetime, forecaster_mod.datetime) == before


async def test_the_forecaster_patch_is_load_bearing_for_recency():
    """`ref_date` for every row's recency decay is stamped in the FORECASTER
    namespace. Freeze it at publication and an article stops having aged."""
    req = PoolAggregateRequest(
        sources=[
            PoolSourceInput(stance=0.5, certainty=0.8, credibility_weight=1.0,
                            relevance_score=0.8, published_date="2026-01-01"),
            PoolSourceInput(stance=0.3, certainty=0.8, credibility_weight=1.0,
                            relevance_score=0.8, published_date="2026-01-02"),
        ]
    )
    now_res = await run_pool_aggregate(req)
    with frozen_clock(datetime(2026, 1, 3)):
        frozen_res = await run_pool_aggregate(req)
    assert frozen_res.evidence_mass > now_res.evidence_mass


def test_the_aggregation_patch_is_load_bearing_for_settlement_validity():
    """`settlement_vote_validity` is called from `aggregate_pool` WITHOUT a
    `today`, so it reads the clock in the AGGREGATION namespace. A non-occurrence
    vote is valid once the deadline has passed and demoted before it — patching
    only the forecaster would judge recency as-published while judging this
    vote today, a hybrid that is neither clock.
    """
    args = dict(
        stance=-0.95, settlement_event_date=None, published_date="2019-06-01",
        claim_direction="arrival", claim_deadline="2020-01-01",
        claim_created_at=None, claim_archetype=None,
    )
    assert settlement_vote_validity(**args) is None  # deadline long past: stands
    with frozen_clock(datetime(2019, 1, 1)):
        assert settlement_vote_validity(**args) is not None  # not yet: demoted


# ── weights ─────────────────────────────────────────────────────────────────


def test_row_weights_reproduce_the_estimators_own_per_source_product():
    from forecast_api.aggregation import recency_weight, relevance_weight
    from forecast_api.config import settings

    s = PoolSourceInput(stance=0.4, certainty=0.7, credibility_weight=1.2,
                        relevance_score=0.7, evidence_weight=0.6, published_date=RECENT)
    expected = (
        1.2 * 0.6
        * recency_weight(RECENT, TODAY, settings.recency_half_life_days, floor=settings.recency_floor)
        * relevance_weight(0.7)
    )
    assert row_weights([s], TODAY)[0] == pytest.approx(expected)


def test_a_row_with_no_evidence_weight_falls_back_to_capped_certainty():
    """Mirrors the estimator's own legacy fallback: an uncapped one would let
    the rows we know least about weigh most."""
    from forecast_api.config import settings

    cap = settings.evidence_class_weight_unclassified_cap
    s = PoolSourceInput(stance=0.4, certainty=1.0, credibility_weight=1.0,
                        relevance_score=1.0, evidence_weight=None, published_date=TODAY)
    assert row_weights([s], TODAY)[0] == pytest.approx(cap)


# ── scoring, against a stub aggregator ──────────────────────────────────────


def response(**over) -> PoolAggregateResponse:
    base = dict(mean=0.4, std=0.2, ci_low=0.1, ci_high=0.7, articles_used=3,
                settled=False, n_eff=2.4, evidence_mass=1.5, age_adjusted_mass=1.8)
    base.update(over)
    return PoolAggregateResponse(**base)


def stub(responses):
    """An aggregator that returns queued responses, then repeats the last."""
    queue = list(responses)

    async def _agg(req: PoolAggregateRequest) -> PoolAggregateResponse:
        return queue.pop(0) if len(queue) > 1 else queue[0]

    return _agg


async def test_score_record_populates_the_signal_columns():
    rec, _ = parse_record(record([enriched(), enriched(url="https://example.com/b", stance=-0.2)]))
    row, skip = await score_record(rec, stub([response()]), loo=False)
    assert skip is None
    assert row.n_scored == 2
    assert row.s2_stored_band_pct == 30  # ai_ci_high 85 - ai_ci_low 55
    assert row.s2_snapshot_band_pct == 30
    assert row.s2_recomputed_band_pct == pytest.approx((0.7 - 0.1) * 50)
    assert row.s3_median_p == pytest.approx(0.55)  # p(0.4)=0.7, p(-0.2)=0.4
    assert row.s3_centre_gap == pytest.approx(abs(0.70 - 0.55))  # vs the stored 70%
    assert row.s4_n_eff == 2.4
    assert row.s4_max_weight_share == pytest.approx(0.5)
    assert row.s5_clean_corpus is True


async def test_confidence_divergence_is_reported_not_treated_as_disagreement():
    """`predictions.confidence` is overwritten daily by the clock glide, so it
    is a glide measure — never a reproduction target."""
    rec, _ = parse_record(record([enriched()], confidence=58))
    row, _ = await score_record(rec, stub([response()]), loo=False)
    assert row.confidence_divergence_pct == pytest.approx(58 - 70)
    assert row.repro_agrees is not None  # reproduction is judged against the SNAPSHOT


async def test_s1_measures_the_gap_the_pin_created():
    """S1 is `|p_pinned - p_no_pin|`, where p_no_pin re-aggregates the same
    roster with every `settled` flag cleared — literally the pooled mean the
    pin discarded."""
    seen: list[PoolAggregateRequest] = []

    async def agg(req):
        seen.append(req)
        if len(seen) == 1:
            return response(mean=0.94, settled=True)   # today's rules: pinned
        if len(seen) == 2:
            return response(mean=0.94, settled=True)   # as-published repro
        return response(mean=-0.10, settled=False)     # the pool without its pin

    rec, _ = parse_record(record([enriched(settled=True, settlementEventDate="2026-08-05")]))
    row, _ = await score_record(rec, agg, loo=False)
    assert seen[-1].sources[0].settled is False, "the no-pin arm must clear the flags"
    assert row.s1_pooled_p_no_pin == pytest.approx(0.45)
    assert row.s1_pin_gap == pytest.approx(abs(0.97 - 0.45))
    assert row.s1b_pin_votes == 1


async def test_a_pool_that_abstains_without_its_pin_reports_that_instead_of_zero():
    """A gap of 0.0 here would bury the strongest possible version of the
    finding among the well-supported pins."""
    async def agg(req):
        if any(s.settled for s in req.sources):
            return response(mean=0.94, settled=True)
        return response(insufficient_data=True, reason="no_usable_weight")

    rec, _ = parse_record(record([enriched(settled=True, settlementEventDate="2026-08-05")]))
    row, _ = await score_record(rec, agg, loo=False)
    assert row.s1_pin_gap is None
    assert row.s1_no_pin_reason == "no_usable_weight"


async def test_an_insufficient_recompute_leaves_the_pooled_signals_null():
    rec, _ = parse_record(record([enriched()]))
    row, _ = await score_record(
        rec, stub([response(insufficient_data=True, reason="all_articles_off_topic")]), loo=False
    )
    assert row.recomputed_mean_pct is None
    assert row.recompute_reason == "all_articles_off_topic"
    assert row.s3_centre_gap is None and row.s2_recomputed_band_pct is None
    assert row.s5_clean_corpus is True  # covariates still describe the row


async def test_leave_one_out_names_the_row_that_moved_the_pool():
    calls: list[PoolAggregateRequest] = []

    async def agg(req):
        calls.append(req)
        if len(req.sources) == 2:
            return response(mean=0.40)
        # Dropping the first row swings the pool; dropping the second barely does.
        return response(mean=-0.40 if req.sources[0].url.endswith("/b") else 0.36)

    rec, _ = parse_record(record([enriched(), enriched(url="https://example.com/b")]))
    row, _ = await score_record(rec, agg, loo=True)
    assert row.loo_max_delta_url == "https://example.com/a"
    assert row.loo_max_abs_delta == pytest.approx(0.4)
    assert row.loo_agrees_with_max_weight is not None


# ── reporting ───────────────────────────────────────────────────────────────


def test_percentiles_interpolate():
    vals = [0.0, 1.0, 2.0, 3.0, 4.0]
    assert percentile(vals, 0.5) == 2.0
    assert percentile(vals, 0.1) == pytest.approx(0.4)
    assert percentile(vals, 1.0) == 4.0


def test_a_signals_n_counts_rows_where_it_is_defined_not_rows_scanned():
    """S1 exists only on pinned pools; averaging it over unpinned ones would
    report a pin gap made mostly of zeros nobody measured."""
    rows = [ScanRow("p1", "s1", "c", s1_pin_gap=0.5), ScanRow("p2", "s2", "c")]
    by_field = {d.field: d for d in distributions(rows)}
    assert by_field["s1_pin_gap"].n == 1
    assert by_field["s1_pin_gap"].mean == 0.5


def test_top_rows_ranks_by_the_signal():
    rows = [ScanRow(f"p{i}", f"s{i}", "c", s3_centre_gap=i / 10) for i in range(5)]
    assert [r.prediction_id for r in top_rows(rows, "s3_centre_gap", 2)] == ["p4", "p3"]


def test_corpus_split_stratifies_on_the_clean_corpus_boundary():
    rows = [ScanRow("p1", "s1", "c", s5_clean_corpus=True), ScanRow("p2", "s2", "c", s5_clean_corpus=False)]
    split = split_by_corpus(rows)
    assert [r.prediction_id for r in split[f"post-{CLEAN_CORPUS_START}"]] == ["p1"]
    assert [r.prediction_id for r in split[f"pre-{CLEAN_CORPUS_START}"]] == ["p2"]


def test_stance_to_percent_reproduces_daatans_publish_clamp():
    """An interval endpoint at stance +-0.99 rounds to 100 without the clamp,
    which is how 1,509 snapshots once published a literal certainty."""
    assert stance_to_percent(0.0) == 50
    assert stance_to_percent(0.99) == 99
    assert stance_to_percent(-0.99) == 1
    assert stance_to_percent(1.0) == 99


# ── the harness cannot report a non-result ──────────────────────────────────


async def test_a_scan_where_nothing_was_scorable_reports_zero_not_a_clean_pass():
    report = await scan(
        [{"oracle_snapshot": None}, {"oracle_snapshot": {"sources": [], "empty": True}}],
        stub([response()]),
    )
    assert report.scored == 0
    assert report.skipped == {"no_oracle_snapshot": 1, "abstained": 1}


async def test_one_failing_record_does_not_cost_the_others_their_measurement():
    calls = {"n": 0}

    async def agg(req):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        return response()

    report = await scan([record([enriched()]), record([enriched()], pid="pred-2")], agg, loo=False)
    assert report.scored == 1
    assert report.skipped == {"error:RuntimeError": 1}


async def test_a_roster_with_no_usable_row_is_skipped_with_a_reason():
    report = await scan([record([enriched(stance=None)])], stub([response()]), loo=False)
    assert report.scored == 0 and report.skipped == {"no_usable_sources": 1}


# ── V7: the golden payload, captured from prod ──────────────────────────────

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "outlier_scan_prod_snapshot.json").read_text(encoding="utf-8")
)


def test_the_frozen_roster_maps_to_the_body_daatan_actually_sends():
    """V7. Both halves of this fixture are real prod data for the same forecast:
    the stored ``oracle_snapshot.sources[]`` roster, and the ``/pool/aggregate``
    body ``recomputeFromPool`` builds from that forecast's pool rows.

    Stage A scores the roster; the live path scores the body. If the two ever
    produce different ``PoolSourceInput``s, every signal downstream shifts and
    the shift reads as an outlier rather than as the mapping bug it is. Nothing
    else in either repo asserts this — daatan's tests cover its own wire shape,
    retro's cover the estimator, and the join between them lived only in prose.

    ``added_at`` is excluded because retro ignores it (Pydantic drops unknown
    keys), and ``source_id`` because ``recomputeFromPool`` deliberately omits it.
    """
    pool, incomplete, invalid = build_pool_sources(FIXTURE["oracle_snapshot"]["sources"])
    assert (incomplete, invalid) == (0, 0)
    by_url = {s.url: s for s in pool}
    assert set(by_url) == {row["url"] for row in FIXTURE["daatan_recompute_body"]}

    for row in FIXTURE["daatan_recompute_body"]:
        got = by_url[row["url"]]
        assert got.source_id is None
        for wire_field in (
            "stance", "certainty", "credibility_weight", "relevance_score",
            "evidence_weight", "published_date", "settled", "settlement_event_date",
            "outlet",
        ):
            assert getattr(got, wire_field) == row[wire_field], f"{wire_field} on {row['url']}"
        stored = row["claims_detail"]
        assert (got.claims_detail is None) == (stored is None)
        if stored is not None:
            assert [c.claim for c in got.claims_detail] == [c["claim"] for c in stored]


def test_prod_stores_published_at_as_a_full_iso_timestamp():
    """Not `YYYY-MM-DD`. `recency_weight` truncates to the first 10 characters,
    so this passes through untouched — but a well-meaning reformat here would
    silently re-date every article in the corpus."""
    dates = [s["publishedAt"] for s in FIXTURE["oracle_snapshot"]["sources"] if s.get("publishedAt")]
    assert dates and any("T" in d for d in dates)
    pool, _, _ = build_pool_sources(FIXTURE["oracle_snapshot"]["sources"])
    assert {s.published_date for s in pool} == set(dates)


async def test_the_real_prod_snapshot_scores_end_to_end():
    rec, skip = parse_record({**FIXTURE["prediction"], "oracle_snapshot": FIXTURE["oracle_snapshot"]})
    assert skip is None
    row, skip = await score_record(rec, run_pool_aggregate, loo=True)
    assert skip is None and row.n_scored == len(FIXTURE["daatan_recompute_body"])
    assert row.recomputed_mean_pct is not None
    assert row.s4_n_eff is not None and row.s3_centre_gap is not None


# ── end to end, through the real estimator ──────────────────────────────────


async def test_scores_a_real_roster_through_the_real_aggregator():
    """The whole point of Stage A running in-process: this is the actual
    estimator, offline, with no HTTP and no rate limit."""
    sources = [
        enriched(url="https://a.example/1", stance=0.6),
        enriched(url="https://b.example/2", stance=0.2, sourceName="AP", outletName="AP"),
        enriched(url="https://c.example/3", stance=-0.1, sourceName="BBC", outletName="BBC"),
    ]
    report = await scan([record(sources)], run_pool_aggregate, loo=True)
    assert report.scored == 1
    row = report.rows[0]
    assert row.recomputed_mean_pct is not None
    assert 0 < row.s4_n_eff <= 3
    assert row.s4_max_weight_share == pytest.approx(1 / 3, abs=0.01)
    assert row.loo_max_abs_delta is not None
    assert report.repro_agreement is not None
    assert row.s5_claims_detail_coverage == 0.0


async def test_the_artifact_is_self_describing():
    report = await scan([record([enriched(), enriched(url="https://x.example/2")])],
                        run_pool_aggregate, loo=False)
    art = report.to_artifact(label="stage-a", git_commit="abc123", deployed_commit="abc123")
    assert art["n_snapshots"] == 1
    assert art["config_fingerprint"]["settlement_quality_floor"] == pytest.approx(0.20)
    assert art["git_commit"] == art["deployed_commit"]
    assert len(art["rows"]) == 1
