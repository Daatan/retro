import json

import httpx
import pytest

from oracle_client import OracleClient


def test_forecast_sends_question_and_omits_criteria_when_absent():
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"mean": 0.2, "insufficient_data": False})

    transport = httpx.MockTransport(handle)
    with OracleClient("relay-key", transport=transport) as client:
        result = client.forecast("Will X happen?")

    assert result == {"mean": 0.2, "insufficient_data": False}
    body = json.loads(requests[0].content)
    assert body == {"question": "Will X happen?"}
    assert requests[0].headers["x-api-key"] == "relay-key"


def test_forecast_includes_resolution_criteria_when_present():
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"mean": 0.0})

    transport = httpx.MockTransport(handle)
    with OracleClient("relay-key", transport=transport) as client:
        client.forecast("Will X happen?", resolution_criteria="Resolves YES if X occurs.")

    body = json.loads(requests[0].content)
    assert body == {
        "question": "Will X happen?",
        "resolution_criteria": "Resolves YES if X occurs.",
    }


def test_forecast_raises_on_error_status():
    transport = httpx.MockTransport(lambda r: httpx.Response(401, json={"detail": "bad key"}))
    with OracleClient("relay-key", transport=transport) as client:
        with pytest.raises(httpx.HTTPStatusError):
            client.forecast("Will X happen?")
