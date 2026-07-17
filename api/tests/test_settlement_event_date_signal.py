"""SourceSignal.settlement_event_date — the settlement anchor, threaded to callers.

Until now the per-claim ``event_date`` that justified a settlement was consumed
by the extraction guards and discarded: daatan's evidence pool stored a bare
``settled`` bit, so ``/pool/aggregate`` recomputes had nothing to re-validate a
stale settlement vote against (the 2026-07-16 audit's root defect D1/D3).
``derive_settlement_event_date`` collapses the article's settlement-grade
claims to one representative anchor date; the field rides on SourceSignal so
callers can persist it next to ``settled``.
"""
import pytest

from forecast_api import forecaster
from forecast_api.config import settings as api_settings
from forecast_api.forecaster import derive_settlement_event_date
from forecast_api.models import ArticleInput, ForecastRequest
from tm.models import ExtractionOutput, GatekeeperOutput, PredictionExtraction


def pred(stance: float, certainty: float = 0.95, event_date: str | None = None) -> PredictionExtraction:
    return PredictionExtraction(
        quote="q", claim="c", stance=stance, certainty=certainty,
        settled=True, event_date=event_date,
    )


class TestDeriveSettlementEventDate:
    def test_single_dated_settlement(self):
        assert derive_settlement_event_date([pred(1.0, event_date="2026-06-15")], 1.0) == "2026-06-15"

    def test_no_settled_claims(self):
        assert derive_settlement_event_date([], 0.7) is None

    def test_undated_settlement_yields_none(self):
        """A legitimately undated negative settlement (post-deadline expiry)."""
        assert derive_settlement_event_date([pred(-1.0)], -0.95) is None

    def test_highest_certainty_claim_wins(self):
        preds = [
            pred(1.0, certainty=0.9, event_date="2026-06-10"),
            pred(0.95, certainty=0.97, event_date="2026-06-12"),
        ]
        assert derive_settlement_event_date(preds, 0.97) == "2026-06-12"

    def test_certainty_tie_breaks_to_earliest_date(self):
        preds = [
            pred(1.0, certainty=0.95, event_date="2026-06-14"),
            pred(1.0, certainty=0.95, event_date="2026-06-11"),
        ]
        assert derive_settlement_event_date(preds, 1.0) == "2026-06-11"

    def test_only_claims_matching_the_article_direction_count(self):
        """A minority opposite-sign settled claim must not date the article's
        vote: the vote's direction is the collapsed stance's sign."""
        preds = [
            pred(-0.95, certainty=0.99, event_date="2026-06-01"),  # dissenting claim
            pred(1.0, certainty=0.9, event_date="2026-06-15"),
        ]
        assert derive_settlement_event_date(preds, 0.8) == "2026-06-15"

    def test_dated_beats_undated_regardless_of_certainty(self):
        preds = [
            pred(1.0, certainty=0.99, event_date=None),
            pred(1.0, certainty=0.9, event_date="2026-06-15"),
        ]
        assert derive_settlement_event_date(preds, 1.0) == "2026-06-15"


# ── end-to-end: the field rides on SourceSignal ───────────────────────────────

_TITLES = [
    "Champions crowned after decisive final game five",
    "Analysts weigh title odds ahead of pivotal matchup",
]


def _article(i: int) -> ArticleInput:
    return ArticleInput(
        url=f"https://source-{i}.example.com/story-{i}",
        title=_TITLES[i - 1],
        snippet=f"A snippet with enough length to pass the fallback minimum, variant {i}.",
        source=f"source-{i}",
        published_date="2026-06-16",
        text="A long enough prefetched article body about the event in question." * 3,
    )


def _patch_pipeline(monkeypatch, extractions_by_url):
    async def fake_gate(**_kwargs):
        return GatekeeperOutput(
            is_prediction=True, reason="on topic", prediction_count_estimate=2,
            relevance_score=1.0,
        ), {"total_tokens": 10}

    async def fake_extract(**kwargs):
        return ExtractionOutput(predictions=extractions_by_url[kwargs["source_name"]]), {"total_tokens": 20}

    monkeypatch.setattr(forecaster, "check_is_prediction", fake_gate)
    monkeypatch.setattr(forecaster, "extract_predictions", fake_extract)
    monkeypatch.setattr(api_settings, "cache_ttl_seconds", 0)


async def test_settlement_event_date_rides_on_the_source_signal(monkeypatch):
    _patch_pipeline(monkeypatch, {
        "source-1": [
            PredictionExtraction(
                quote="q", claim="settled claim", stance=1.0, certainty=0.95,
                settled=True, event_date="2026-06-15",
            ),
        ],
        "source-2": [
            PredictionExtraction(quote="q", claim="ordinary claim", stance=0.5, certainty=0.6),
        ],
    })
    resp = await forecaster.run_forecast(ForecastRequest(
        question="settlement anchor date on the wire — unique question",
        articles=[_article(1), _article(2)],
    ))

    by_id = {s.source_id: s for s in resp.sources}
    assert by_id["source-1.example.com"].settlement_event_date == "2026-06-15"
    assert by_id["source-1.example.com"].settled is True
    assert by_id["source-2.example.com"].settlement_event_date is None


def test_pool_aggregate_request_accepts_the_new_fields():
    """The recompute request mirrors the SourceSignal shape so callers can send
    back what they persisted."""
    from forecast_api.models import PoolAggregateRequest, PoolSourceInput
    req = PoolAggregateRequest(
        sources=[PoolSourceInput(
            stance=-0.98, certainty=0.93, credibility_weight=1.0,
            relevance_score=1.0, settled=True, settlement_event_date="2026-07-14",
        )],
        claim_created_at="2026-07-04",
        claim_archetype="scheduled",
    )
    assert req.sources[0].settlement_event_date == "2026-07-14"
    assert req.claim_archetype == "scheduled"


async def test_forecast_request_accepts_claim_window_metadata():
    """claim_created_at / claim_archetype are accepted (and currently unused) —
    the claim_direction precedent: additive, fail-open."""
    req = ForecastRequest(
        question="window metadata accepted — unique question",
        claim_created_at="2026-07-04T10:00:00.000Z",
        claim_archetype="scheduled",
    )
    assert req.claim_created_at.startswith("2026-07-04")
    assert req.claim_archetype == "scheduled"
