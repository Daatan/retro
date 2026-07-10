"""Endpoint tests for /fetch-url — proves it fetches through the SSRF-guarded
``safe_get`` (not trafilatura.fetch_url, which re-resolves the host and reopened
a TOCTOU/DNS-rebinding window) and rejects non-public targets."""

from unittest.mock import MagicMock, patch

import httpx
from fastapi.testclient import TestClient

from forecast_api.main import app
from tm.net_guard import UnsafeURLError

client = TestClient(app)


def _resp(status: int = 200, text: str = "<html>body</html>") -> MagicMock:
    r = MagicMock(spec=httpx.Response)
    r.status_code = status
    r.text = text
    return r


def test_fetch_url_extracts_via_safe_get():
    md = MagicMock()
    md.title, md.date, md.sitename = "Title", "2026-01-02", "Src"
    with (
        patch("forecast_api.main.safe_get", return_value=_resp()) as sg,
        patch("trafilatura.extract", return_value="extracted text"),
        patch("trafilatura.extract_metadata", return_value=md),
    ):
        r = client.post("/fetch-url", json={"url": "http://93.184.216.34/a"})
    assert r.status_code == 200
    body = r.json()
    assert body["text"] == "extracted text"
    assert body["title"] == "Title"
    assert body["date"] == "2026-01-02"
    sg.assert_called_once()


def test_fetch_url_rejects_internal_before_fetch():
    # is_safe_url rejects loopback up front — safe_get is never reached.
    with patch("forecast_api.main.safe_get") as sg:
        r = client.post("/fetch-url", json={"url": "http://127.0.0.1/x"})
    assert r.status_code == 422
    sg.assert_not_called()


def test_fetch_url_blocks_when_safe_get_raises():
    # Public initial host passes is_safe_url, but safe_get blocks a redirect to
    # an internal address (or a rebind) -> 422, no leaked response.
    with patch("forecast_api.main.safe_get", side_effect=UnsafeURLError("blocked")):
        r = client.post("/fetch-url", json={"url": "http://93.184.216.34/a"})
    assert r.status_code == 422
