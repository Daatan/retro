"""Unit tests for ``_is_safe_url`` — the SSRF guard on the ``/fetch-url`` proxy.

The endpoint is a public-facing fetch proxy, so it must refuse non-http(s)
schemes and any host that resolves to a non-public address (loopback, RFC1918
private ranges, link-local incl. the cloud metadata IP 169.254.169.254, etc.).

These tests use IP-literal hosts and ``localhost`` so they are deterministic
offline — ``socket.getaddrinfo`` parses numeric literals and reads ``localhost``
from the hosts file without network DNS.
"""

from forecast_api.main import _is_safe_url


class TestRejectsUnsafeSchemes:
    def test_file_scheme(self):
        assert not _is_safe_url("file:///etc/passwd")

    def test_ftp_scheme(self):
        assert not _is_safe_url("ftp://example.com/x")

    def test_gopher_scheme(self):
        assert not _is_safe_url("gopher://127.0.0.1:8001/")

    def test_no_scheme(self):
        assert not _is_safe_url("example.com/path")

    def test_empty(self):
        assert not _is_safe_url("")


class TestRejectsNonPublicHosts:
    def test_cloud_metadata_link_local(self):
        assert not _is_safe_url("http://169.254.169.254/latest/meta-data/")

    def test_loopback_ip(self):
        assert not _is_safe_url("http://127.0.0.1/")

    def test_localhost_name(self):
        assert not _is_safe_url("http://localhost:8001/health")

    def test_rfc1918_10(self):
        assert not _is_safe_url("http://10.0.0.5/admin")

    def test_rfc1918_192_168(self):
        assert not _is_safe_url("https://192.168.1.1/")

    def test_rfc1918_172_16(self):
        assert not _is_safe_url("http://172.16.4.4/")

    def test_unspecified(self):
        assert not _is_safe_url("http://0.0.0.0/")


class TestAllowsPublicHttp:
    def test_public_ip_literal(self):
        # 8.8.8.8 is a public address; IP literal needs no network DNS.
        assert _is_safe_url("http://8.8.8.8/")

    def test_public_https_ip_literal(self):
        assert _is_safe_url("https://1.1.1.1/article")
