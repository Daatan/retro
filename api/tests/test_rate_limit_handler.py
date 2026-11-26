"""The 429 body names the limit that actually tripped (retro#1109).

It used to say "max 10 requests/minute" on every route, whatever the route's real limit
(5–120/min), so a caller throttled to the wrong number. A private app + Limiter keeps the
shared ``forecast_api.limiter`` state out of this test; the handler is the real one.
"""

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from forecast_api.main import _rate_limit_handler


def _client(limit: str) -> TestClient:
    limiter = Limiter(key_func=get_remote_address)
    app = FastAPI()
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)

    @app.get("/ping")
    @limiter.limit(limit)
    async def ping(request: Request):
        return {"ok": True}

    return TestClient(app)


def test_429_names_the_route_limit():
    client = _client("2/minute")
    assert client.get("/ping").status_code == 200
    assert client.get("/ping").status_code == 200
    r = client.get("/ping")
    assert r.status_code == 429
    assert r.json()["detail"] == "Rate limit exceeded — max 2 per 1 minute"


def test_429_detail_follows_a_different_limit():
    client = _client("1/minute")
    client.get("/ping")
    r = client.get("/ping")
    assert r.status_code == 429
    assert "max 1 per 1 minute" in r.json()["detail"]
    assert "10 requests/minute" not in r.json()["detail"]


def test_429_keeps_cors_headers_for_allowed_origin():
    client = _client("1/minute")
    client.get("/ping")
    r = client.get("/ping", headers={"origin": "https://bayes.daatan.com"})
    assert r.status_code == 429
    assert r.headers["access-control-allow-origin"] == "https://bayes.daatan.com"
