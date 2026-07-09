"""SSRF guard tests for the pipeline — ``is_safe_url`` classification and the
redirect re-validation in ``safe_get`` / ``safe_get_async``. ``httpx.MockTransport``
keeps them offline; IP-literal / ``localhost`` hosts resolve without network DNS."""

import httpx
import pytest

from tm import net_guard
from tm.net_guard import UnsafeURLError, is_safe_url, safe_get, safe_get_async


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://example.com/x",
        "example.com/no-scheme",
        "gopher://127.0.0.1:8001/",
        "",
        "http://169.254.169.254/latest/meta-data/",
        "http://127.0.0.1/",
        "http://localhost:8001/health",
        "http://10.0.0.5/",
        "http://192.168.1.1/",
        "http://172.16.0.1/",
        "http://0.0.0.0/",
    ],
)
def test_is_safe_url_rejects(url):
    assert not is_safe_url(url)


@pytest.mark.parametrize("url", ["http://93.184.216.34/", "https://8.8.8.8/path"])
def test_is_safe_url_accepts_public(url):
    assert is_safe_url(url)


def _sync_client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def _async_client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_safe_get_blocks_initial_internal():
    with _sync_client(lambda r: httpx.Response(200)) as c:
        with pytest.raises(UnsafeURLError):
            safe_get("http://127.0.0.1/secrets", client=c)


def test_safe_get_blocks_redirect_to_internal():
    def handler(request):
        return httpx.Response(302, headers={"location": "http://169.254.169.254/"})

    with _sync_client(handler) as c:
        with pytest.raises(UnsafeURLError):
            safe_get("http://93.184.216.34/start", client=c)


def test_safe_get_follows_safe_redirect():
    def handler(request):
        if request.url.path == "/start":
            return httpx.Response(302, headers={"location": "http://8.8.8.8/final"})
        return httpx.Response(200, text="ok")

    with _sync_client(handler) as c:
        assert safe_get("http://93.184.216.34/start", client=c).text == "ok"


def test_safe_get_returns_direct_response():
    with _sync_client(lambda r: httpx.Response(200, text="hi")) as c:
        assert safe_get("http://93.184.216.34/", client=c).text == "hi"


def test_safe_get_caps_redirects():
    # A public host looping through public hops never trips is_safe_url, so the
    # hop cap is the only thing that stops it.
    def handler(request):
        return httpx.Response(302, headers={"location": "http://8.8.8.8/loop"})

    with _sync_client(handler) as c:
        with pytest.raises(UnsafeURLError):
            safe_get("http://8.8.8.8/loop", client=c, max_redirects=2)


def test_safe_get_opens_its_own_client_when_none_given(monkeypatch):
    # forecast_api's /fetch-url calls safe_get(url) with no client; the guard has
    # to open (and close) one itself.
    real_client = httpx.Client

    def _factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(lambda r: httpx.Response(200, text="hi"))
        return real_client(*args, **kwargs)

    monkeypatch.setattr(net_guard.httpx, "Client", _factory)
    assert safe_get("http://93.184.216.34/").text == "hi"


async def test_safe_get_async_blocks_redirect_to_internal():
    def handler(request):
        return httpx.Response(302, headers={"location": "http://127.0.0.1/"})

    async with _async_client(handler) as c:
        with pytest.raises(UnsafeURLError):
            await safe_get_async(c, "http://93.184.216.34/start")


async def test_safe_get_async_follows_safe_redirect():
    def handler(request):
        if request.url.path == "/s":
            return httpx.Response(302, headers={"location": "http://8.8.8.8/f"})
        return httpx.Response(200, text="ok")

    async with _async_client(handler) as c:
        r = await safe_get_async(c, "http://93.184.216.34/s")
        assert r.text == "ok"
