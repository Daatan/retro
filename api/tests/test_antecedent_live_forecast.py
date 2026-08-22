"""retro#583 — antecedent filtering wired into the LIVE /forecast path.

retro#582 (retro#573 Option 1) added antecedent pool-splitting to
``run_pool_aggregate`` only — the recompute-over-an-already-extracted-pool path.
``run_forecast`` does its own search + extraction and builds several PARALLEL
arrays (stance, weight, relevance, ...) in lockstep inside one per-article loop,
so this issue is really "prove the mask (``antecedent_keep_mask``, shared with
#582) gets applied to every one of those arrays together, not just to
``source_signals``" — a wrong array left unfiltered would silently misalign
weights against sources.

Runs the real ``run_forecast``, not a lower-level helper: the filter lives
between the per-article loop and ``aggregate_pool()``, deep inside
``_run_forecast_inner``, so nothing shallower would exercise it. Supplies
articles directly (the ``req.articles`` branch) so no search-provider mocking
is needed, following ``TestSuppliedVerdictAllowlist`` in
test_reuse_supplied_verdict.py. The gatekeeper and extractor are stubbed —
deterministic, no LLM/network call.
"""

import itertools
from types import SimpleNamespace
from unittest.mock import AsyncMock

from forecast_api import forecaster
from forecast_api.models import ArticleInput, ForecastRequest
from tm.models import GatekeeperOutput, PredictionExtraction

_SEQ = itertools.count()

_BODY_TEMPLATE = (
    "Fixture article body #{n} for the retro#583 antecedent-wiring suite. The "
    "gatekeeper and extractor are stubbed, so no model ever reads this text; it "
    "exists only to clear the pipeline's non-empty-body checks. "
) * 3


def _article(n: int, url: str) -> ArticleInput:
    return ArticleInput(
        url=url,
        title=f"Fixture headline {n}",
        snippet="A snippet long enough to clear the twenty-char fallback guard.",
        source="fixture",
        published_date="2026-07-28",
        text=_BODY_TEMPLATE.format(n=n),
    )


def _prediction(*, claim: str, is_conditional=None, antecedent_text_en=None, antecedent_polarity=None):
    return PredictionExtraction(
        quote="q", claim=claim, stance=0.6, certainty=0.8, settled=None,
        is_conditional=is_conditional, antecedent_text_en=antecedent_text_en,
        antecedent_polarity=antecedent_polarity,
    )


def _install_stubs(monkeypatch, article_to_predictions: dict):
    """``article_to_predictions`` keys on the article body text (what
    ``extract_predictions`` actually receives, per-article), same technique
    ``test_reuse_supplied_verdict.py`` uses to differentiate a per-article stub."""
    gk = AsyncMock(return_value=(
        GatekeeperOutput(is_prediction=True, reason="judged", relevance_score=0.8), {},
    ))

    async def _extract(*, article_text, **kwargs):
        preds = article_to_predictions[article_text]
        return SimpleNamespace(predictions=preds, author_lean=None, author_lean_certainty=None), {}

    monkeypatch.setattr(forecaster, "check_is_prediction", gk)
    monkeypatch.setattr(forecaster, "extract_predictions", _extract)
    monkeypatch.setattr(forecaster, "get_credibility_weight", lambda sid: 1.0)
    return gk


class TestAntecedentFiltersTheLiveForecastPath:
    async def test_none_query_keeps_every_article_unchanged(self, monkeypatch):
        # Regression guard: the field is new and optional — every existing caller
        # (antecedent_query absent) must see byte-identical behavior.
        a1, a2 = _article(1, "https://fixture.example.test/583/a"), _article(2, "https://fixture.example.test/583/b")
        _install_stubs(monkeypatch, {
            a1.text: [_prediction(claim="c1", is_conditional=True, antecedent_text_en="the ceasefire holds", antecedent_polarity=True)],
            a2.text: [_prediction(claim="c2", is_conditional=True, antecedent_text_en="China invades Taiwan", antecedent_polarity=True)],
        })
        resp = await forecaster.run_forecast(ForecastRequest(
            question=f"[583-none-{next(_SEQ)}] Will the consequent happen?",
            articles=[a1, a2],
        ))
        assert resp.insufficient_data is False
        assert resp.articles_used == 2

    async def test_matching_antecedent_source_is_kept_mismatched_is_dropped(self, monkeypatch):
        a1, a2 = _article(1, "https://fixture.example.test/583/match"), _article(2, "https://fixture.example.test/583/other")
        _install_stubs(monkeypatch, {
            a1.text: [_prediction(claim="c1", is_conditional=True, antecedent_text_en="the ceasefire holds", antecedent_polarity=True)],
            a2.text: [_prediction(claim="c2", is_conditional=True, antecedent_text_en="China invades Taiwan", antecedent_polarity=True)],
        })
        resp = await forecaster.run_forecast(ForecastRequest(
            question=f"[583-match-{next(_SEQ)}] Will the consequent happen?",
            articles=[a1, a2],
            antecedent_query="the ceasefire holds",
        ))
        assert resp.insufficient_data is False
        assert resp.articles_used == 1
        assert resp.sources[0].url == a1.url

    async def test_unconditional_source_survives_any_antecedent_filter(self, monkeypatch):
        a1, a2 = _article(1, "https://fixture.example.test/583/uncond"), _article(2, "https://fixture.example.test/583/other2")
        _install_stubs(monkeypatch, {
            a1.text: [_prediction(claim="c1", is_conditional=False)],
            a2.text: [_prediction(claim="c2", is_conditional=True, antecedent_text_en="China invades Taiwan", antecedent_polarity=True)],
        })
        resp = await forecaster.run_forecast(ForecastRequest(
            question=f"[583-uncond-{next(_SEQ)}] Will the consequent happen?",
            articles=[a1, a2],
            antecedent_query="the ceasefire holds",
        ))
        assert resp.articles_used == 1
        assert resp.sources[0].url == a1.url

    async def test_no_source_matches_returns_insufficient_data(self, monkeypatch):
        a1, a2 = _article(1, "https://fixture.example.test/583/none1"), _article(2, "https://fixture.example.test/583/none2")
        _install_stubs(monkeypatch, {
            a1.text: [_prediction(claim="c1", is_conditional=True, antecedent_text_en="China invades Taiwan", antecedent_polarity=True)],
            a2.text: [_prediction(claim="c2", is_conditional=True, antecedent_text_en="Iran strikes Israel", antecedent_polarity=True)],
        })
        resp = await forecaster.run_forecast(ForecastRequest(
            question=f"[583-none-match-{next(_SEQ)}] Will the consequent happen?",
            articles=[a1, a2],
            antecedent_query="the ceasefire holds",
        ))
        assert resp.insufficient_data is True
        assert resp.reason == "no_matching_antecedent"
        assert resp.articles_used == 0
        assert resp.sources == []

    async def test_polarity_mismatch_is_dropped(self, monkeypatch):
        a1 = _article(1, "https://fixture.example.test/583/negated")
        _install_stubs(monkeypatch, {
            a1.text: [_prediction(claim="c1", is_conditional=True, antecedent_text_en="the ceasefire holds", antecedent_polarity=False)],
        })
        resp = await forecaster.run_forecast(ForecastRequest(
            question=f"[583-polarity-{next(_SEQ)}] Will the consequent happen?",
            articles=[a1],
            antecedent_query="the ceasefire holds",
            antecedent_query_polarity=True,
        ))
        assert resp.insufficient_data is True
        assert resp.reason == "no_matching_antecedent"

    async def test_weights_and_sources_stay_aligned_after_filtering(self, monkeypatch):
        # The actual risk this issue calls out: filtering source_signals but leaving
        # one of the nine parallel arrays unfiltered would misalign weights against
        # sources rather than raising — this pins articles_used and n_eff/evidence_mass
        # to reflect ONLY the surviving (matching) source, not the original two.
        a1, a2 = _article(1, "https://fixture.example.test/583/align1"), _article(2, "https://fixture.example.test/583/align2")
        _install_stubs(monkeypatch, {
            a1.text: [_prediction(claim="c1", is_conditional=True, antecedent_text_en="the ceasefire holds", antecedent_polarity=True)],
            a2.text: [_prediction(claim="c2", is_conditional=True, antecedent_text_en="China invades Taiwan", antecedent_polarity=True)],
        })
        resp = await forecaster.run_forecast(ForecastRequest(
            question=f"[583-align-{next(_SEQ)}] Will the consequent happen?",
            articles=[a1, a2],
            antecedent_query="the ceasefire holds",
        ))
        assert len(resp.sources) == 1
        assert resp.articles_used == 1
        assert resp.n_eff > 0
        assert resp.evidence_mass > 0
