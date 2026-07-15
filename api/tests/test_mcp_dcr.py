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
        # authorize/token stay on Cognito; jwks on the pool.
        assert m["authorization_endpoint"] == f"{HOSTED_UI}/oauth2/authorize"
        assert m["token_endpoint"] == f"{HOSTED_UI}/oauth2/token"
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
