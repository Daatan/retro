"""Replaying the reduction over PERSISTED rows (retro#398).

``test_claims_detail.py`` already replays ``reduce_article`` over claims built
moments earlier in-process. What has never been checkable is the case
retroactive backtesting actually needs: a row written weeks ago by an older
build, read back out of storage, still reducing to the numbers it carries.

The distinction that matters most here is **skipped vs failed**. The column is
forward-only from 2026-08-02 with no backfill, so a row without
``claims_detail`` is expected and says nothing about the reduction. Counting
those as mismatches would bury the signal under history; counting them as
agreement would inflate it. They are their own bucket, and coverage is reported
next to agreement so neither can be read without the other.
"""
from __future__ import annotations

from forecast_api.claims_replay import replay_row, replay_rows


def _claim(**over) -> dict:
    base = {
        "claim": "The step was taken.", "quote": "Officials said the step was taken.",
        "stance": 0.8, "certainty": 0.9, "specificity": None, "prediction_type": None,
        "evidence_class": "reported_fact", "quantitative_estimate": None,
        "settled": False, "event_date": None, "fact_signal": None,
        "event_actors": None, "event_target": None, "is_occurrence": None,
        "verified": None,
    }
    base.update(over)
    return base


def _row_from_replay(claims: list[dict], **stored) -> dict:
    """Build a row whose stored scalars ARE what the reduction produces, then
    let the caller override individual ones to simulate drift.

    Derived through the real ``reduce_article`` rather than hand-written, so a
    change to the reduction cannot leave these fixtures asserting a stale
    expectation — the point under test is agreement, not any particular value.
    """
    from forecast_api.config import settings
    from forecast_api.forecaster import reduce_article
    from forecast_api.models import ClaimDetail

    row = {"url": "u", "claims_detail": claims}
    reduced = reduce_article(
        [ClaimDetail.model_validate(c) for c in claims],
        settlement_min_stance=settings.settlement_min_claim_stance,
        settlement_min_certainty=settings.settlement_min_claim_certainty,
        class_weights=settings.evidence_class_weight,
        class_weight_default=settings.evidence_class_weight_default,
        class_weight_unclassified_cap=settings.evidence_class_weight_unclassified_cap,
    )
    row.update({
        "stance": round(reduced.stance, 3),
        "certainty": round(reduced.certainty, 3),
        "evidence_weight": round(reduced.evidence_weight, 3),
        "evidence_class": reduced.evidence_class,
        "settled": reduced.settled,
        "claims": reduced.claims,
    })
    row.update(stored)
    return row


class TestAgreement:
    def test_a_faithful_row_replays_clean(self):
        row = _row_from_replay([_claim(), _claim(claim="Second.", stance=0.6, certainty=0.7)])
        assert replay_row(row).replayed_ok

    def test_drift_in_a_single_scalar_is_reported_with_both_values(self):
        row = _row_from_replay([_claim()], stance=0.123)
        result = replay_row(row)
        assert not result.replayed_ok
        assert [d.field for d in result.diffs] == ["stance"]
        assert result.diffs[0].stored == 0.123
        assert result.diffs[0].replayed != 0.123

    def test_an_unsettled_row_stored_as_null_is_not_drift(self):
        """The wire stores an unsettled article as None *or* False; the
        reduction always returns a bool. Found against 268 real prod rows,
        where it made every ordinary row report `stored=None replayed=False` —
        invisible to fixtures, whose stored value always comes from the
        reduction and is therefore already a bool."""
        row = _row_from_replay([_claim(settled=False)], settled=None)
        assert replay_row(row).replayed_ok

    def test_a_genuinely_settled_row_still_compares(self):
        """The null-collapse must not swallow real disagreement in the other
        direction — a row claiming settled=True whose claims do not settle."""
        row = _row_from_replay([_claim(settled=False)], settled=True)
        result = replay_row(row)
        assert [d.field for d in result.diffs] == ["settled"]

    def test_a_field_the_export_omitted_is_not_a_mismatch(self):
        """Exports carry whichever columns the query selected. An absent column
        must not read as drift, or every partial export reports false failures."""
        row = _row_from_replay([_claim()])
        del row["evidence_class"]
        assert replay_row(row).replayed_ok


class TestSkippedIsNotFailed:
    def test_a_row_without_claims_detail_is_skipped(self):
        result = replay_row({"url": "u", "claims_detail": None, "stance": 0.5})
        assert result.skipped_reason == "no_claims_detail"
        assert not result.diffs, "a pre-column row must not be reported as drift"
        assert not result.replayed_ok

    def test_skipped_rows_lower_coverage_but_never_agreement(self):
        faithful = _row_from_replay([_claim()])
        report = replay_rows([faithful] + [{"url": "old", "claims_detail": None}] * 9)
        assert (report.replayed, report.skipped) == (1, 9)
        assert report.coverage == 0.1
        assert report.agreement == 1.0, "agreement is over replayable rows, not all rows"


class TestFailureIsLoud:
    def test_an_unparseable_row_errors_rather_than_crashing(self):
        result = replay_row({"url": "u", "claims_detail": [{"claim": object()}]})
        assert result.error is not None
        assert not result.replayed_ok

    def test_one_bad_row_does_not_stop_the_others(self):
        report = replay_rows([
            _row_from_replay([_claim()]),
            {"url": "bad", "claims_detail": [{"stance": "not-a-number"}]},
        ])
        assert report.errored == 1
        assert report.agreed == 1

    def test_a_run_that_measured_nothing_is_distinguishable_from_a_clean_one(self):
        """The retro#395 trap: a fail-soft harness reporting zeros reads exactly
        like 'everything agreed'. Coverage is what tells them apart."""
        nothing = replay_rows([{"url": "old", "claims_detail": None}] * 5)
        assert nothing.agreed == 0 and nothing.replayed == 0
        assert nothing.coverage == 0.0
        assert nothing.agreement == 0.0, "no rows replayed must not read as 100% agreement"
