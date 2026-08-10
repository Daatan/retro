"""retro#449 Stage A — always-emit instrumentation for the settlement vote
weight/verified pairing.

retro#449's own 2026-08-09 investigation found that nothing in retro
captures "weight distribution of verified=null vs true/false settlement
votes" anywhere read-only — not the settlement logging (direction/counts/
reasons only), not daatan's persisted oracleSnapshot (drops `verified`),
not DebugInfo (debug-gated, no resolved weight). Any real threshold fix
(Stage B) needs that data to calibrate against rather than guess, the same
discipline retro#391's floor calibration required. This is Stage A only:
log every settlement-voting row unconditionally, mirroring
event=evidence_clusters (retro#412) — nothing reads it yet.
"""
from __future__ import annotations

from forecast_api import forecaster
from forecast_api.models import ArticleInput, ForecastRequest
from tm.models import ExtractionOutput, GatekeeperOutput, PredictionExtraction

_BODY = (
    "Fixture article body for the settlement vote weight instrumentation "
    "suite. The gatekeeper and extractor are stubbed, so no model ever "
    "reads this text; it exists to clear the pipeline's non-empty-body checks. "
) * 3


def _claim(**over):
    return PredictionExtraction(**{
        "quote": "Fixture quote.", "claim": "Fixture claim.",
        "stance": 0.4, "certainty": 0.6, **over,
    })


def _patch_single(monkeypatch, claims):
    """One article, one fixed claim set — every extract call gets it."""
    async def fake_gate(**kwargs):
        return (
            GatekeeperOutput(is_prediction=True, reason="fixture gate",
                              prediction_count_estimate=len(claims), relevance_score=1.0),
            {"total_tokens": 0},
        )

    async def fake_extract(**kwargs):
        return (ExtractionOutput(predictions=list(claims)), {"total_tokens": 0})

    monkeypatch.setattr(forecaster, "check_is_prediction", fake_gate)
    monkeypatch.setattr(forecaster, "extract_predictions", fake_extract)
    monkeypatch.setattr(forecaster, "get_credibility_weight", lambda sid: 0.8)


def _patch_by_url(monkeypatch, claims_by_url):
    """Multiple articles — each URL's stubbed claims differ, keyed off the
    article_text the fixture below makes distinguishable per URL."""
    async def fake_gate(**kwargs):
        n = max(len(v) for v in claims_by_url.values())
        return (
            GatekeeperOutput(is_prediction=True, reason="fixture gate",
                              prediction_count_estimate=n, relevance_score=1.0),
            {"total_tokens": 0},
        )

    async def fake_extract(*, article_text, **kwargs):
        claims = claims_by_url[article_text]
        return (ExtractionOutput(predictions=list(claims)), {"total_tokens": 0})

    monkeypatch.setattr(forecaster, "check_is_prediction", fake_gate)
    monkeypatch.setattr(forecaster, "extract_predictions", fake_extract)
    monkeypatch.setattr(forecaster, "get_credibility_weight", lambda sid: 0.8)


def _article(url, text=_BODY):
    return ArticleInput(
        url=url, title="Fixture dispatch",
        snippet="Fixture snippet, long enough to be usable by the pipeline.",
        source="fixture", published_date="2026-07-28", text=text,
    )


class TestSettlementVoteWeightInstrumentation:
    async def test_a_settled_row_logs_its_weight_and_verified_flag(self, monkeypatch, caplog):
        _patch_single(monkeypatch, [
            _claim(claim="The result was declared.", stance=1.0, certainty=0.95,
                   settled=True, event_date="2026-07-20", verified=True),
        ])
        with caplog.at_level("INFO"):
            resp = await forecaster.run_forecast(ForecastRequest(
                question="[F449-settled] Will the event occur?",
                articles=[_article("https://fixture.example.test/settled")],
            ))

        assert resp.sources[0].settled is True, "fixture must clear the settlement gates"
        # The fixture claim carries no fact_signal, so the UNRELATED fact-lane
        # rollup (SourceSignal.verified) is None here — proving the log below
        # reads the settlement claim's own verified flag, not that field.
        assert resp.sources[0].verified is None
        assert "event=settlement_vote_weight" in caplog.text
        assert "source=https://fixture.example.test/settled" in caplog.text
        assert "verified=True" in caplog.text

    async def test_an_unsettled_row_does_not_log(self, monkeypatch, caplog):
        _patch_single(monkeypatch, [
            _claim(claim="Ordinary reporting.", stance=0.2, certainty=0.6),
        ])
        with caplog.at_level("INFO"):
            resp = await forecaster.run_forecast(ForecastRequest(
                question="[F449-unsettled] Will the event occur?",
                articles=[_article("https://fixture.example.test/unsettled")],
            ))

        assert not resp.sources[0].settled
        assert "event=settlement_vote_weight" not in caplog.text

    async def test_only_the_settled_row_logs_in_a_mixed_pool(self, monkeypatch, caplog):
        """The loop must key off is_settled, not fire for every row in the
        pool regardless of outcome — a coordinated fabrication (retro#449's
        actual concern) would otherwise drown in ordinary-evidence noise."""
        settled_text = _BODY + " settled-variant"
        unsettled_text = _BODY + " unsettled-variant"
        _patch_by_url(monkeypatch, {
            settled_text: [
                _claim(claim="The result was declared.", stance=1.0, certainty=0.95,
                       settled=True, event_date="2026-07-20", verified=None),
            ],
            unsettled_text: [
                _claim(claim="Ordinary reporting.", stance=0.2, certainty=0.6),
            ],
        })
        with caplog.at_level("INFO"):
            resp = await forecaster.run_forecast(ForecastRequest(
                question="[F449-mixed] Will the event occur?",
                articles=[
                    _article("https://fixture.example.test/settled", settled_text),
                    _article("https://fixture.example.test/unsettled", unsettled_text),
                ],
            ))

        assert [s.settled for s in resp.sources] == [True, None]
        assert caplog.text.count("event=settlement_vote_weight") == 1
        assert "source=https://fixture.example.test/settled" in caplog.text
        assert "source=https://fixture.example.test/unsettled" not in caplog.text
        assert "verified=None" in caplog.text
