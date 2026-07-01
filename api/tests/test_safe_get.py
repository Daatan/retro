"""Tests for ``safe_get`` — the redirect-revalidating SSRF-safe fetch used by the
forecast pipeline. Uses ``httpx.MockTransport`` so no real network is touched;
IP-literal hosts keep ``is_safe_url`` deterministic offline."""

import httpx
import pytest

from forecast_api import net_guard
from forecast_api.net_guard import UnsafeURLError, safe_get


def _patch_transport(monkeypatch, handler):
    """Force safe_get's internal httpx.Client to use a MockTransport."""
    real_client = httpx.Client

    def _factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(net_guard.httpx, "Client", _factory)


def test_blocks_initial_internal_url(monkeypatch):
    _patch_transport(monkeypatch, lambda r: httpx.Response(200))
    with pytest.raises(UnsafeURLError):
        safe_get("http://127.0.0.1/secrets")


def test_blocks_metadata_url(monkeypatch):
    _patch_transport(monkeypatch, lambda r: httpx.Response(200))
    with pytest.raises(UnsafeURLError):
        safe_get("http://169.254.169.254/latest/meta-data/")


def test_blocks_redirect_to_internal(monkeypatch):
    # A public host that 302-redirects to the cloud metadata endpoint must be blocked.
    _patch_transport(
        monkeypatch,
        lambda r: httpx.Response(302, headers={"location": "http://169.254.169.254/"}),
    )
    with pytest.raises(UnsafeURLError):
        safe_get("http://93.184.216.34/start")


def test_follows_safe_redirect(monkeypatch):
    def handler(request):
        if request.url.path == "/start":
            return httpx.Response(302, headers={"location": "http://8.8.8.8/final"})
        return httpx.Response(200, text="ok")

    _patch_transport(monkeypatch, handler)
    resp = safe_get("http://93.184.216.34/start")
    assert resp.status_code == 200
    assert resp.text == "ok"


def test_returns_direct_response(monkeypatch):
    _patch_transport(monkeypatch, lambda r: httpx.Response(200, text="hi"))
    resp = safe_get("http://93.184.216.34/")
    assert resp.text == "hi"


def test_caps_redirects(monkeypatch):
    _patch_transport(
        monkeypatch,
        lambda r: httpx.Response(302, headers={"location": "http://8.8.8.8/loop"}),
    )
    with pytest.raises(UnsafeURLError):
        safe_get("http://8.8.8.8/loop", max_redirects=2)
