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
    assert r.json()["model"] == "bedrock/amazon.nova-micro-v1:0"


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
