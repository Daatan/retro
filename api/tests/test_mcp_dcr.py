"""Tests for the DCR façade (mcp_dcr.py) that lets Claude's connector — which
requires OAuth Dynamic Client Registration — log in against a Cognito pool that
has none. Covers the config gating, the advertised AS metadata, and the static
/register response. No network: everything is exercised against the module's
route builder with settings monkeypatched.
"""

from starlette.applications import Starlette
from starlette.testclient import TestClient

from forecast_api.config import settings
from forecast_api.mcp_auth import SCOPE_FORECAST, SCOPE_READ
from forecast_api.mcp_dcr import authorization_server_metadata, create_dcr_routes

HOSTED_UI = "https://daatan-oracle.auth.eu-central-1.amazoncognito.com"
CLAUDE_CLIENT = "claudeclient123"


def _enable_dcr(monkeypatch):
    monkeypatch.setattr(settings, "cognito_user_pool_id", "eu-central-1_TESTPOOL")
    monkeypatch.setattr(settings, "cognito_hosted_ui_domain", HOSTED_UI)
    monkeypatch.setattr(settings, "cognito_claude_client_id", CLAUDE_CLIENT)
    monkeypatch.setattr(settings, "mcp_resource_url", "https://oracle.daatan.com/mcp")


def _client(monkeypatch) -> TestClient:
    _enable_dcr(monkeypatch)
    return TestClient(Starlette(routes=create_dcr_routes(settings)))


# ── config gating ─────────────────────────────────────────────────────────────

class TestDcrConfig:
    def test_enabled_when_all_present(self, monkeypatch):
        _enable_dcr(monkeypatch)
        assert settings.dcr_enabled is True
        assert settings.mcp_as_issuer == "https://oracle.daatan.com"
        assert settings.cognito_authorize_endpoint == f"{HOSTED_UI}/oauth2/authorize"
        assert settings.cognito_token_endpoint == f"{HOSTED_UI}/oauth2/token"

    def test_disabled_without_client_id(self, monkeypatch):
        _enable_dcr(monkeypatch)
        monkeypatch.setattr(settings, "cognito_claude_client_id", None)
        assert settings.dcr_enabled is False

    def test_disabled_without_hosted_ui(self, monkeypatch):
        _enable_dcr(monkeypatch)
        monkeypatch.setattr(settings, "cognito_hosted_ui_domain", None)
        assert settings.dcr_enabled is False

    def test_disabled_when_mcp_off(self, monkeypatch):
        # No pool id → RS not enabled → façade must stay off even if the rest is set.
        monkeypatch.setattr(settings, "cognito_user_pool_id", None)
        monkeypatch.setattr(settings, "cognito_hosted_ui_domain", HOSTED_UI)
        monkeypatch.setattr(settings, "cognito_claude_client_id", CLAUDE_CLIENT)
        assert settings.dcr_enabled is False

    def test_hosted_ui_trailing_slash_stripped(self, monkeypatch):
        monkeypatch.setattr(settings, "cognito_hosted_ui_domain", HOSTED_UI + "/")
        assert settings.cognito_authorize_endpoint == f"{HOSTED_UI}/oauth2/authorize"


# ── advertised authorization-server metadata ─────────────────────────────────

class TestAsMetadata:
    def test_metadata_shape(self, monkeypatch):
        _enable_dcr(monkeypatch)
        m = authorization_server_metadata(settings)
        # issuer is OUR origin (what the client fetched the doc from), not Cognito.
        assert m["issuer"] == "https://oracle.daatan.com"
        assert m["registration_endpoint"] == "https://oracle.daatan.com/register"
        # authorize/token point at OUR proxy (which strips offline_access before
        # forwarding to Cognito), not straight at Cognito.
        assert m["authorization_endpoint"] == "https://oracle.daatan.com/oauth2/authorize"
        assert m["token_endpoint"] == "https://oracle.daatan.com/oauth2/token"
        assert m["jwks_uri"] == settings.cognito_jwks_url
        # public PKCE client requirements Claude checks for.
        assert m["code_challenge_methods_supported"] == ["S256"]
        assert "none" in m["token_endpoint_auth_methods_supported"]
        assert m["response_types_supported"] == ["code"]
        # offline_access must NOT be advertised — Cognito rejects that scope name.
        assert "offline_access" not in m["scopes_supported"]
        assert SCOPE_READ in m["scopes_supported"]
        assert SCOPE_FORECAST in m["scopes_supported"]

    def test_metadata_served_at_well_known(self, monkeypatch):
        c = _client(monkeypatch)
        r = c.get("/.well-known/oauth-authorization-server")
        assert r.status_code == 200
        assert r.json()["registration_endpoint"] == "https://oracle.daatan.com/register"

    def test_openid_configuration_serves_same_doc(self, monkeypatch):
        c = _client(monkeypatch)
        a = c.get("/.well-known/oauth-authorization-server").json()
        b = c.get("/.well-known/openid-configuration").json()
        assert a == b


# ── /register façade ─────────────────────────────────────────────────────────

class TestRegister:
    def test_returns_static_public_client(self, monkeypatch):
        c = _client(monkeypatch)
        r = c.post(
            "/register",
            json={
                "redirect_uris": ["https://claude.ai/api/mcp/auth_callback"],
                "client_name": "Claude",
                "token_endpoint_auth_method": "none",
                "grant_types": ["authorization_code"],
            },
        )
        assert r.status_code == 201
        body = r.json()
        assert body["client_id"] == CLAUDE_CLIENT
        assert body["token_endpoint_auth_method"] == "none"
        assert body["redirect_uris"] == ["https://claude.ai/api/mcp/auth_callback"]
        # public client — never hand back a secret.
        assert "client_secret" not in body or body["client_secret"] in (None, "")

    def test_same_client_id_every_call(self, monkeypatch):
        # Claude loops forever if handed a NEW client_id each registration; the
        # façade must be stable.
        c = _client(monkeypatch)
        first = c.post("/register", json={"redirect_uris": ["https://claude.ai/api/mcp/auth_callback"]}).json()
        second = c.post("/register", json={"redirect_uris": ["http://localhost:8080/callback"]}).json()
        assert first["client_id"] == second["client_id"] == CLAUDE_CLIENT

    def test_empty_body_still_returns_client(self, monkeypatch):
        c = _client(monkeypatch)
        r = c.post("/register")
        assert r.status_code == 201
        assert r.json()["client_id"] == CLAUDE_CLIENT


# ── /oauth2/authorize proxy (strips offline_access) ──────────────────────────

class TestAuthorizeProxy:
    def test_strips_offline_access_and_preserves_params(self, monkeypatch):
        c = _client(monkeypatch)
        r = c.get(
            "/oauth2/authorize",
            params={
                "client_id": CLAUDE_CLIENT,
                "response_type": "code",
                "scope": "openid email offline_access oracle-mcp/read oracle-mcp/forecast",
                "redirect_uri": "https://claude.ai/api/mcp/auth_callback",
                "code_challenge": "abc123",
                "code_challenge_method": "S256",
                "state": "xyz",
            },
            follow_redirects=False,
        )
        assert r.status_code == 302
        loc = r.headers["location"]
        # redirected to the REAL Cognito authorize endpoint
        assert loc.startswith(f"{HOSTED_UI}/oauth2/authorize?")
        # the poison scope is gone, the rest survives
        assert "offline_access" not in loc
        assert "state=xyz" in loc
        assert "code_challenge=abc123" in loc
        assert "openid" in loc and "oracle-mcp" in loc


# ── /oauth2/token proxy (forwards to Cognito, strips offline_access) ──────────

class TestTokenProxy:
    def test_forwards_and_strips_offline_access(self, monkeypatch):
        _enable_dcr(monkeypatch)
        import forecast_api.mcp_dcr as mod

        captured = {}

        class _Resp:
            content = b'{"access_token":"AT","refresh_token":"RT","token_type":"Bearer"}'
            status_code = 200
            headers = {"content-type": "application/json"}

        class _Client:
            def __init__(self, *a, **k):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, url, content=None, headers=None):
                captured["url"] = url
                captured["content"] = content
                return _Resp()

        monkeypatch.setattr(mod.httpx, "AsyncClient", _Client)
        c = TestClient(Starlette(routes=create_dcr_routes(settings)))
        r = c.post(
            "/oauth2/token",
            data={
                "grant_type": "refresh_token",
                "refresh_token": "RT",
                "client_id": CLAUDE_CLIENT,
                "scope": "openid offline_access oracle-mcp/read",
            },
        )
        assert r.status_code == 200
        assert r.json()["access_token"] == "AT"
        # forwarded to Cognito's real token endpoint, offline_access stripped
        assert captured["url"] == f"{HOSTED_UI}/oauth2/token"
        assert "offline_access" not in captured["content"]
        assert "grant_type=refresh_token" in captured["content"]
