"""Endpoint tests for /relevance.

The endpoint is a thin wrapper over `tm.gatekeeper.check_is_prediction` — the SAME
screener `/forecast` already runs. These tests pin the contract that makes it reusable
by news-indexer's rescue path (which persists the verdict, so the shape must be stable),
and the invariant that matters most: it must not invent its own judgment. The LLM is
mocked — no live model call in CI, per repo convention.
"""

from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from forecast_api.main import app
from tm.models import GatekeeperOutput

client = TestClient(app)

HEADERS = {"x-api-key": "test-key"}
BODY = {
    "claim": "Benjamin Netanyahu will form the next government by 2026-12-31",
    "article_text": "Likud approved eight reserved slots for Netanyahu on its Knesset list.",
    "source_name": "Maariv",
    "article_date": "2026-07-13",
}


def _verdict(is_prediction: bool = True, relevance: float = 0.8) -> GatekeeperOutput:
    return GatekeeperOutput(
        is_prediction=is_prediction,
        reason="coalition list mechanics bear directly on who forms the government",
        prediction_count_estimate=2,
        relevance_score=relevance,
    )


def test_relevance_returns_the_gatekeepers_verdict_verbatim():
    # The endpoint must pass the gatekeeper's judgment through untouched — it exists to
    # expose that judgment, not to add a second opinion on top of it.
    with patch("forecast_api.main.check_is_prediction", new=AsyncMock(return_value=(_verdict(), {}))):
        r = client.post("/relevance", json=BODY, headers=HEADERS)
    assert r.status_code == 200
    body = r.json()
    assert body["is_prediction"] is True
    assert body["relevance_score"] == 0.8
    assert body["prediction_count_estimate"] == 2
    assert body["reason"]


def test_relevance_reports_the_judging_model():
    # news-indexer persists `model` alongside each verdict, so a stored judgment can
    # always be traced to what produced it (a model swap must be visible in the data).
    with patch("forecast_api.main.check_is_prediction", new=AsyncMock(return_value=(_verdict(), {}))):
        r = client.post("/relevance", json=BODY, headers=HEADERS)
    assert r.json()["model"] == "bedrock/us.amazon.nova-micro-v1:0"


def test_relevance_passes_the_claim_through_as_the_event_name():
    # The whole point is that the judgment is CLAIM-AWARE: the same article is relevant
    # to one forecast and not another. If the claim didn't reach the prompt, the verdict
    # would be claim-agnostic and the rescue path would be worthless.
    spy = AsyncMock(return_value=(_verdict(), {}))
    with patch("forecast_api.main.check_is_prediction", new=spy):
        client.post("/relevance", json=BODY, headers=HEADERS)
    assert spy.await_args.kwargs["event_name"] == BODY["claim"]
    assert spy.await_args.kwargs["article_text"] == BODY["article_text"]


def test_relevance_surfaces_a_negative_verdict():
    # An irrelevant article must come back as such rather than defaulting to relevant —
    # a rescue path that can't hear "no" would push everything it looked at.
    with patch(
        "forecast_api.main.check_is_prediction",
        new=AsyncMock(return_value=(_verdict(is_prediction=False, relevance=0.1), {})),
    ):
        r = client.post("/relevance", json=BODY, headers=HEADERS)
    assert r.status_code == 200
    assert r.json()["is_prediction"] is False
    assert r.json()["relevance_score"] == 0.1


def test_relevance_returns_502_when_the_model_call_fails():
    # A Bedrock failure must be a distinguishable error, never a silent "not relevant" —
    # the caller has to be able to tell "judged irrelevant" from "never got judged".
    with patch("forecast_api.main.check_is_prediction", new=AsyncMock(side_effect=RuntimeError("bedrock down"))):
        r = client.post("/relevance", json=BODY, headers=HEADERS)
    assert r.status_code == 502


def test_relevance_requires_the_api_key():
    with patch("forecast_api.main.check_is_prediction", new=AsyncMock(return_value=(_verdict(), {}))) as gk:
        r = client.post("/relevance", json=BODY)
    assert r.status_code == 422  # missing required header
    gk.assert_not_awaited()

    with patch("forecast_api.main.check_is_prediction", new=AsyncMock(return_value=(_verdict(), {}))) as gk:
        r = client.post("/relevance", json=BODY, headers={"x-api-key": "wrong"})
    assert r.status_code == 401
    gk.assert_not_awaited()


def test_short_form_is_off_by_default_so_forecast_is_untouched():
    """The safety property. The base prompt rejects anything "under ~200 meaningful words" — aimed at
    paywall stubs. Relaxing that for social posts must not relax it for /forecast, so the override is
    APPENDED and only when asked: with short_form off, the prompt text is byte-for-byte today's."""
    with patch("forecast_api.main.check_is_prediction", new=AsyncMock(return_value=(_verdict(), {}))) as gk:
        client.post("/relevance", json=BODY, headers=HEADERS)
    assert gk.await_args.kwargs["short_form"] is False


def test_short_form_is_passed_through_when_requested():
    with patch("forecast_api.main.check_is_prediction", new=AsyncMock(return_value=(_verdict(), {}))) as gk:
        r = client.post("/relevance", json={**BODY, "short_form": True}, headers=HEADERS)
    assert r.status_code == 200
    assert gk.await_args.kwargs["short_form"] is True


# ── Content-free input fails closed (retro#359) ────────────────────────────────────────────────
# These deliberately do NOT mock check_is_prediction: the guard lives inside it, and the property
# under test is that the endpoint as a whole never reaches the model for input carrying no
# proposition. Mocking the screener would test the mock. Only the LLM leg is stubbed.

URL_ONLY = {**BODY, "article_text": "https://www.c14.co.il/article/1641278", "short_form": True}


def test_url_only_article_is_rejected_without_reaching_the_model():
    """The measured incident: this exact 37-char body was endorsed for 76 of 127 forecasts at
    relevance 0.7-1.0 with invented justifications. /relevance is a public-ish surface and its
    verdict gets persisted and reused downstream (reuse_supplied_relevance), so the judge itself
    has to be the layer that fails closed — a caller-side guard in news-indexer is not enough."""
    with patch("tm.gatekeeper.complete_structured", new=AsyncMock()) as llm:
        r = client.post("/relevance", json=URL_ONLY, headers=HEADERS)
    llm.assert_not_awaited()
    assert r.status_code == 200
    body = r.json()
    assert body["is_prediction"] is False
    assert body["relevance_score"] == 0.0
    assert body["prediction_count_estimate"] == 0


def test_rejection_is_a_verdict_not_an_error():
    """200 with is_prediction=false, never 502. The caller distinguishes "judged irrelevant" from
    "never got judged" and retries the latter — a content-free post must land in the first bucket
    permanently, or news-indexer re-asks about the same bare URL forever."""
    with patch("tm.gatekeeper.complete_structured", new=AsyncMock()):
        r = client.post("/relevance", json=URL_ONLY, headers=HEADERS)
    assert r.status_code == 200
    assert r.json()["reason"]


# ── article_text length cap (retro#433) ─────────────────────────────────────
# article_text had no max_length while /relevance sits at 120/minute (12x /forecast's
# 10/minute) -- an uncapped-length payload at a high rate limit is an unmetered-cost
# vector. The cap matches retro#432's LlmMessage.content cap (32,000 chars).

def test_article_text_over_the_cap_is_rejected_without_reaching_the_model():
    too_long = {**BODY, "article_text": "x" * 32_001}
    with patch("forecast_api.main.check_is_prediction", new=AsyncMock(return_value=(_verdict(), {}))) as gk:
        r = client.post("/relevance", json=too_long, headers=HEADERS)
    assert r.status_code == 422
    gk.assert_not_awaited()


def test_article_text_at_the_cap_still_reaches_the_model():
    at_cap = {**BODY, "article_text": "x" * 32_000}
    with patch("forecast_api.main.check_is_prediction", new=AsyncMock(return_value=(_verdict(), {}))) as gk:
        r = client.post("/relevance", json=at_cap, headers=HEADERS)
    assert r.status_code == 200
    gk.assert_awaited_once()


def test_a_real_short_post_still_reaches_the_model():
    """The guard is a floor, not a filter — terse posts from curated journalists are the entire
    reason this endpoint exists, and must be judged on content exactly as before."""
    real = {**BODY, "article_text": "שרן השכל התפטרה מתפקידה כסגנית שר החוץ", "short_form": True}
    with patch(
        "tm.gatekeeper.complete_structured",
        new=AsyncMock(return_value=(_verdict(), {})),
    ) as llm:
        r = client.post("/relevance", json=real, headers=HEADERS)
    llm.assert_awaited_once()
    assert r.status_code == 200
    assert r.json()["is_prediction"] is True
