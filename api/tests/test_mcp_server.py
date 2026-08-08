"""Tests for the MCP server: tool registration, response shaping, the
Polymarket edge logic, the Cognito token verifier, and endpoint auth.

No network/LLM: run_forecast and the Polymarket lookup are monkeypatched, and
the verifier's JWKS client is stubbed with a locally generated RSA key.
"""

import time
from types import SimpleNamespace

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from forecast_api import mcp_limiter, mcp_server
from forecast_api.config import settings
from forecast_api.mcp_auth import (
    SCOPE_FORECAST,
    SCOPE_READ,
    CognitoTokenVerifier,
    ScopeError,
    require_scope,
)
from forecast_api.models import ForecastResponse

ISSUER = "https://cognito-idp.eu-central-1.amazonaws.com/eu-central-1_TESTPOOL"
CLIENT_ID = "client-abc"


@pytest.fixture(autouse=True)
def _reset_mcp_rate_limits():
    """Every tool call in this module goes through the real mcp_limiter.enforce()
    (only auth is monkeypatched) — reset its in-memory state before each test so
    call counts from other tests/cases can never accumulate toward a limit."""
    mcp_limiter.reset()
    yield


# ── helpers ─────────────────────────────────────────────────────────────────

def _forecast_response(mean: float, **kw) -> ForecastResponse:
    defaults = dict(
        question="Will X happen by 2027?",
        mean=mean,
        std=0.1,
        ci_low=mean - 0.2,
        ci_high=mean + 0.2,
        articles_used=6,
        sources=[],
    )
    defaults.update(kw)
    return ForecastResponse(**defaults)


def _market(question: str, yes_price: float, outcomes='["Yes","No"]') -> dict:
    return {
        "id": "12345",
        "question": question,
        "slug": "will-x-happen",
        "outcomes": outcomes,
        "outcomePrices": f'["{yes_price}","{round(1 - yes_price, 4)}"]',
        "volumeNum": 100000,
        "endDate": "2027-01-01",
        "active": True,
        "closed": False,
    }


def _patch_scope_ok(monkeypatch):
    monkeypatch.setattr(
        mcp_server, "require_scope",
        lambda scope: SimpleNamespace(scopes=[SCOPE_READ, SCOPE_FORECAST]),
    )


def _patch_forecast(monkeypatch, resp):
    async def fake_run_forecast(req):
        return resp
    monkeypatch.setattr(mcp_server, "run_forecast", fake_run_forecast)


# ── tool registration ───────────────────────────────────────────────────────

class TestToolRegistration:
    @pytest.mark.asyncio
    async def test_seven_tools(self, monkeypatch):
        monkeypatch.setattr(settings, "cognito_user_pool_id", "eu-central-1_TESTPOOL")
        monkeypatch.setattr(settings, "cognito_allowed_client_ids", CLIENT_ID)
        mcp = mcp_server.build_mcp()
        assert mcp is not None
        names = {t.name for t in await mcp.list_tools()}
        assert names == {
            "forecast", "search_news", "fetch_article", "bayes_nodes",
            "source_leaderboard", "polymarket_market", "polymarket_edge",
        }

    def test_disabled_without_cognito(self, monkeypatch):
        monkeypatch.setattr(settings, "cognito_user_pool_id", None)
        assert mcp_server.build_mcp() is None

    @pytest.mark.asyncio
    async def test_transport_security_allows_resource_host(self, monkeypatch):
        # Regression: behind nginx the transport sees Host: oracle.daatan.com; the
        # mcp SDK's DNS-rebinding guard defaults to localhost-only and would 421
        # every authenticated call unless the public host is explicitly allowed.
        monkeypatch.setattr(settings, "cognito_user_pool_id", "eu-central-1_TESTPOOL")
        monkeypatch.setattr(settings, "cognito_allowed_client_ids", CLIENT_ID)
        monkeypatch.setattr(settings, "mcp_resource_url", "https://oracle.daatan.com/mcp")
        mcp = mcp_server.build_mcp()
        ts = mcp.settings.transport_security
        assert ts is not None and ts.enable_dns_rebinding_protection is True
        assert "oracle.daatan.com" in ts.allowed_hosts
        assert "oracle.daatan.com:*" in ts.allowed_hosts
        assert "localhost:*" in ts.allowed_hosts
        assert "https://oracle.daatan.com" in ts.allowed_origins


# ── forecast tool shaping ───────────────────────────────────────────────────

class TestForecastTool:
    @pytest.mark.asyncio
    async def test_adds_probability(self, monkeypatch):
        _patch_scope_ok(monkeypatch)
        _patch_forecast(monkeypatch, _forecast_response(0.5))
        out = await mcp_server.forecast("Will X happen by 2027?")
        assert out["probability"] == 0.75  # (0.5 + 1) / 2
        assert out["mean_stance"] == 0.5
        assert out["ci"] == [0.65, 0.85]  # stance CI [0.3,0.7] -> prob space

    @pytest.mark.asyncio
    async def test_insufficient_data_probability_none(self, monkeypatch):
        _patch_scope_ok(monkeypatch)
        _patch_forecast(monkeypatch, _forecast_response(
            0.0, insufficient_data=True, reason="no_search_results"))
        out = await mcp_server.forecast("Will X happen by 2027?")
        assert out["probability"] is None
        assert out["insufficient_data"] is True

    @pytest.mark.asyncio
    async def test_rate_limited_after_the_per_caller_cap(self, monkeypatch):
        """retro#431: the forecast tool calls run_forecast in-process, never
        through the REST route's @limiter.limit("10/minute") — this is the
        tool-level enforcement that replaces it."""
        _patch_scope_ok(monkeypatch)
        _patch_forecast(monkeypatch, _forecast_response(0.5))
        monkeypatch.setattr(
            "forecast_api.mcp_limiter.get_access_token",
            lambda: SimpleNamespace(subject="user-1", client_id="client-abc"),
        )
        monkeypatch.setattr(mcp_server, "FORECAST_LIMIT", mcp_limiter.parse("2/minute"))

        await mcp_server.forecast("Will X happen by 2027?")
        await mcp_server.forecast("Will X happen by 2027?")
        with pytest.raises(mcp_limiter.MCPRateLimitExceeded):
            await mcp_server.forecast("Will X happen by 2027?")


# ── polymarket_edge logic ───────────────────────────────────────────────────

class TestPolymarketEdge:
    async def _run(self, monkeypatch, oracle_mean, yes_price, **market_kw):
        _patch_scope_ok(monkeypatch)
        _patch_forecast(monkeypatch, _forecast_response(oracle_mean))

        async def fake_resolve(market):
            return _market("Will X happen by 2027?", yes_price, **market_kw)
        monkeypatch.setattr(mcp_server, "resolve_market", fake_resolve)
        return await mcp_server.polymarket_edge("https://polymarket.com/event/will-x-happen")

    @pytest.mark.asyncio
    async def test_buy_yes(self, monkeypatch):
        # oracle mean 0.5 -> prob 0.75; market 0.40 -> edge +0.35
        out = await self._run(monkeypatch, oracle_mean=0.5, yes_price=0.40)
        assert out["oracle_probability"] == 0.75
        assert out["market_probability"] == 0.40
        assert out["edge"] == 0.35
        assert out["suggested_side"] == "BUY YES"

    @pytest.mark.asyncio
    async def test_buy_no(self, monkeypatch):
        # oracle mean -0.6 -> prob 0.20; market 0.60 -> edge -0.40
        out = await self._run(monkeypatch, oracle_mean=-0.6, yes_price=0.60)
        assert out["suggested_side"] == "BUY NO"
        assert out["edge"] == -0.40

    @pytest.mark.asyncio
    async def test_no_edge(self, monkeypatch):
        # oracle mean 0.04 -> prob 0.52; market 0.50 -> edge 0.02 < 0.05 threshold
        out = await self._run(monkeypatch, oracle_mean=0.04, yes_price=0.50)
        assert out["suggested_side"] == "NO EDGE"

    @pytest.mark.asyncio
    async def test_non_binary_market(self, monkeypatch):
        out = await self._run(monkeypatch, oracle_mean=0.5, yes_price=0.4,
                              outcomes='["Candidate A","Candidate B","Candidate C"]')
        assert out["suggested_side"] == "NONE"
        assert "not a binary" in out["note"]

    @pytest.mark.asyncio
    async def test_insufficient_data(self, monkeypatch):
        _patch_scope_ok(monkeypatch)
        _patch_forecast(monkeypatch, _forecast_response(
            0.0, insufficient_data=True, reason="no_search_results"))

        async def fake_resolve(market):
            return _market("Will X happen by 2027?", 0.5)
        monkeypatch.setattr(mcp_server, "resolve_market", fake_resolve)
        out = await mcp_server.polymarket_edge("https://polymarket.com/event/x")
        assert out["suggested_side"] == "NONE"
        assert out["oracle_probability"] is None

    @pytest.mark.asyncio
    async def test_market_not_found(self, monkeypatch):
        _patch_scope_ok(monkeypatch)

        async def fake_resolve(market):
            return None
        monkeypatch.setattr(mcp_server, "resolve_market", fake_resolve)
        out = await mcp_server.polymarket_edge("nonexistent-market")
        assert out["suggested_side"] == "NONE"
        assert "No Polymarket market found" in out["note"]


# ── Cognito token verifier ──────────────────────────────────────────────────

@pytest.fixture(scope="module")
def rsa_keypair():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return key, key.public_key()


def _verifier(rsa_keypair):
    _priv, pub = rsa_keypair
    v = CognitoTokenVerifier(
        issuer=ISSUER, jwks_url="https://unused", allowed_client_ids={CLIENT_ID},
        resource_url="https://oracle.daatan.com/mcp",
    )
    v._jwk_client = SimpleNamespace(
        get_signing_key_from_jwt=lambda token: SimpleNamespace(key=pub)
    )
    return v


def _token(rsa_keypair, **overrides):
    priv, _pub = rsa_keypair
    claims = {
        "iss": ISSUER,
        "client_id": CLIENT_ID,
        "token_use": "access",
        "scope": f"{SCOPE_READ} {SCOPE_FORECAST}",
        "sub": "user-123",
        "exp": int(time.time()) + 3600,
    }
    claims.update(overrides)
    return jwt.encode(claims, priv, algorithm="RS256")


class TestCognitoVerifier:
    @pytest.mark.asyncio
    async def test_valid_token(self, rsa_keypair):
        tok = await _verifier(rsa_keypair).verify_token(_token(rsa_keypair))
        assert tok is not None
        assert tok.client_id == CLIENT_ID
        assert set(tok.scopes) == {SCOPE_READ, SCOPE_FORECAST}
        assert tok.subject == "user-123"

    @pytest.mark.asyncio
    async def test_wrong_issuer(self, rsa_keypair):
        tok = await _verifier(rsa_keypair).verify_token(
            _token(rsa_keypair, iss="https://evil.example.com"))
        assert tok is None

    @pytest.mark.asyncio
    async def test_disallowed_client_id(self, rsa_keypair):
        tok = await _verifier(rsa_keypair).verify_token(
            _token(rsa_keypair, client_id="someone-else"))
        assert tok is None

    @pytest.mark.asyncio
    async def test_id_token_rejected(self, rsa_keypair):
        # id tokens (token_use=id) must not authorize the API
        tok = await _verifier(rsa_keypair).verify_token(
            _token(rsa_keypair, token_use="id"))
        assert tok is None

    @pytest.mark.asyncio
    async def test_expired(self, rsa_keypair):
        tok = await _verifier(rsa_keypair).verify_token(
            _token(rsa_keypair, exp=int(time.time()) - 10))
        assert tok is None


# ── require_scope ───────────────────────────────────────────────────────────

class TestRequireScope:
    def test_has_scope(self, monkeypatch):
        monkeypatch.setattr(
            "forecast_api.mcp_auth.get_access_token",
            lambda: SimpleNamespace(scopes=[SCOPE_READ, SCOPE_FORECAST], expires_at=None),
        )
        tok = require_scope(SCOPE_FORECAST)
        assert SCOPE_FORECAST in tok.scopes

    def test_missing_scope(self, monkeypatch):
        monkeypatch.setattr(
            "forecast_api.mcp_auth.get_access_token",
            lambda: SimpleNamespace(scopes=[SCOPE_READ], expires_at=None),
        )
        with pytest.raises(ScopeError):
            require_scope(SCOPE_FORECAST)

    def test_no_token(self, monkeypatch):
        monkeypatch.setattr("forecast_api.mcp_auth.get_access_token", lambda: None)
        with pytest.raises(ScopeError):
            require_scope(SCOPE_READ)


# ── endpoint auth (mirrors main.py's mount) ─────────────────────────────────

class TestEndpointAuth:
    def _app(self, monkeypatch):
        from contextlib import asynccontextmanager

        from fastapi import FastAPI
        from mcp.server.auth.routes import create_protected_resource_routes

        monkeypatch.setattr(settings, "cognito_user_pool_id", "eu-central-1_TESTPOOL")
        monkeypatch.setattr(settings, "cognito_allowed_client_ids", CLIENT_ID)
        mcp = mcp_server.build_mcp()

        @asynccontextmanager
        async def lifespan(app):
            async with mcp.session_manager.run():
                yield

        app = FastAPI(lifespan=lifespan)
        for r in create_protected_resource_routes(
            resource_url=settings.mcp_resource_url,
            authorization_servers=[settings.cognito_issuer],
            scopes_supported=[SCOPE_READ, SCOPE_FORECAST],
        ):
            app.router.routes.append(r)
        app.mount("/mcp", mcp.streamable_http_app())
        return app

    def test_prm_metadata_at_root(self, monkeypatch):
        from starlette.testclient import TestClient
        with TestClient(self._app(monkeypatch)) as c:
            r = c.get("/.well-known/oauth-protected-resource/mcp")
            assert r.status_code == 200
            body = r.json()
            assert body["resource"] == settings.mcp_resource_url
            assert settings.cognito_issuer in body["authorization_servers"]

    def test_mcp_requires_token(self, monkeypatch):
        from starlette.testclient import TestClient
        with TestClient(self._app(monkeypatch)) as c:
            r = c.post(
                "/mcp",
                json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
                headers={"Accept": "application/json, text/event-stream"},
            )
            assert r.status_code == 401
            assert "resource_metadata" in r.headers.get("www-authenticate", "")


# ── the confidence bucket (F16, retro#365) ──────────────────────────────────
# _confidence is the only place in the repo where the CI *width* drives a
# decision rather than a display, and it is what the two Polymarket-facing
# tools report. F16 widens intervals, so the bucket boundaries need pinning:
# an untested threshold that quietly demotes every forecast is the failure
# mode this class exists to catch.

class TestConfidenceBucket:
    def test_settled_is_always_high_regardless_of_width(self):
        resp = _forecast_response(0.9, settled=True, ci_low=-0.9, ci_high=0.9)
        assert mcp_server._confidence(resp) == "high"

    def test_high_needs_both_five_articles_and_a_tight_band(self):
        # width 0.20 in probability space, 6 articles
        assert mcp_server._confidence(
            _forecast_response(0.5, ci_low=0.3, ci_high=0.7, articles_used=6)
        ) == "high"
        # same band, too few articles
        assert mcp_server._confidence(
            _forecast_response(0.5, ci_low=0.3, ci_high=0.7, articles_used=4)
        ) == "medium"

    def test_bucket_boundaries(self):
        assert mcp_server._confidence(
            _forecast_response(0.0, ci_low=-0.29, ci_high=0.29, articles_used=9)
        ) == "high"
        assert mcp_server._confidence(
            _forecast_response(0.0, ci_low=-0.31, ci_high=0.31, articles_used=9)
        ) == "medium"
        assert mcp_server._confidence(
            _forecast_response(0.0, ci_low=-0.49, ci_high=0.49, articles_used=9)
        ) == "medium"
        assert mcp_server._confidence(
            _forecast_response(0.0, ci_low=-0.51, ci_high=0.51, articles_used=9)
        ) == "low"

    def test_the_high_boundary_is_float_fragile(self):
        """Documented, not fixed — out of F16's scope, but a reader deserves to
        know. A stance CI of ±0.3 is a probability width of exactly 0.30 by
        arithmetic, but ``_prob_ci`` returns 0.35/0.65 and 0.65 - 0.35 evaluates
        to 0.30000000000000004, so the ``<= 0.30`` test fails and the forecast
        is bucketed *medium*. The 0.50 boundary is exact (0.75 - 0.25) and does
        not have this problem. Worth an epsilon if the buckets are ever retuned.
        """
        assert mcp_server._confidence(
            _forecast_response(0.0, ci_low=-0.3, ci_high=0.3, articles_used=9)
        ) == "medium"
        assert mcp_server._confidence(
            _forecast_response(0.0, ci_low=-0.5, ci_high=0.5, articles_used=9)
        ) == "medium"

    def test_the_dispersion_floor_alone_can_never_demote_a_bucket(self):
        """pool_dispersion_floor's widest possible contribution is 2·1.96·σ in
        probability space. At σ = 0.05 that is 0.196, comfortably inside the
        0.30 high/medium boundary — so a unanimous pool cannot be demoted by
        the floor. This test is the tripwire on raising σ: anything above
        0.0765 silently downgrades every unanimous pool for every trader."""
        widest_prob = 2.0 * 1.96 * settings.pool_dispersion_floor
        assert widest_prob < 0.30
        resp = _forecast_response(
            0.0,
            ci_low=-widest_prob,      # stance width 2× the prob width…
            ci_high=widest_prob,      # …i.e. the floor at its very widest
            articles_used=6,
        )
        assert mcp_server._confidence(resp) == "high"
