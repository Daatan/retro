import httpx
import pytest

from metaculus_client import MetaculusClient


def _handler(requests: list[httpx.Request], responses: dict):
    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        key = (request.method, request.url.path)
        return responses[key]

    return handle


def test_open_binary_questions_sends_expected_params():
    requests: list[httpx.Request] = []
    page = httpx.Response(200, json={"results": [{"id": 1}, {"id": 2}], "next": None})
    transport = httpx.MockTransport(_handler(requests, {("GET", "/api/posts/"): page}))

    with MetaculusClient("tok", transport=transport) as client:
        results = client.open_binary_questions("bot-testing-area", limit=10)

    assert [r["id"] for r in results] == [1, 2]
    assert len(requests) == 1
    req = requests[0]
    assert req.headers["authorization"] == "Token tok"
    params = dict(httpx.QueryParams(req.url.query))
    assert params["tournaments"] == "bot-testing-area"
    assert params["statuses"] == "open"
    assert params["forecast_type"] == "binary"


def test_open_binary_questions_paginates():
    requests: list[httpx.Request] = []
    responses = iter(
        [
            httpx.Response(200, json={"results": [{"id": 1}], "next": "http://x/?offset=1"}),
            httpx.Response(200, json={"results": [{"id": 2}], "next": None}),
        ]
    )

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return next(responses)

    transport = httpx.MockTransport(handle)
    with MetaculusClient("tok", transport=transport) as client:
        results = client.open_binary_questions("bot-testing-area", limit=10)

    assert [r["id"] for r in results] == [1, 2]
    assert len(requests) == 2


def test_submit_binary_forecast_clamps_extremes():
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(201, json={})

    transport = httpx.MockTransport(handle)
    with MetaculusClient("tok", transport=transport) as client:
        client.submit_binary_forecast(123, 0.999)

    import json

    body = json.loads(requests[0].content)
    assert body == [{"question": 123, "source": "api", "probability_yes": 0.97}]


def test_submit_binary_forecast_raises_on_error():
    transport = httpx.MockTransport(lambda r: httpx.Response(500, json={"detail": "boom"}))
    with MetaculusClient("tok", transport=transport) as client:
        with pytest.raises(httpx.HTTPStatusError):
            client.submit_binary_forecast(1, 0.5)


def test_post_comment_payload_shape():
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(201, json={})

    transport = httpx.MockTransport(handle)
    with MetaculusClient("tok", transport=transport) as client:
        client.post_comment(42, "hello", private=True)

    import json

    body = json.loads(requests[0].content)
    assert body == {
        "text": "hello",
        "parent": None,
        "included_forecast": True,
        "is_private": True,
        "on_post": 42,
    }


def test_list_tournaments_accepts_bare_list_and_paginated_shape():
    for payload in ([{"slug": "a"}], {"results": [{"slug": "a"}]}):
        requests: list[httpx.Request] = []
        transport = httpx.MockTransport(
            _handler(requests, {("GET", "/api/projects/tournaments/"): httpx.Response(200, json=payload)})
        )
        with MetaculusClient("tok", transport=transport) as client:
            assert [t["slug"] for t in client.list_tournaments()] == ["a"]
        assert requests[0].headers["authorization"] == "Token tok"
