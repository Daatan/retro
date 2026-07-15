"""DCR façade for the MCP OAuth flow (Cognito has no Dynamic Client Registration).

Claude's MCP connector will not use a static client_id: it discovers the
authorization server from our protected-resource metadata, fetches that AS's
`/.well-known/oauth-authorization-server`, and REQUIRES a `registration_endpoint`
there (RFC 7591) — it hard-fails discovery with "does not support dynamic client
registration" otherwise. Cognito publishes no such endpoint and can't be made to.

So the Oracle origin advertises *itself* as the authorization server in metadata
only. We serve AS metadata that mirrors Cognito's real /authorize and /token
endpoints but points `registration_endpoint` back at us, and our /register
returns the one pre-provisioned Cognito public client on every call — no client
is ever created (no CreateUserPoolClient, so no IAM grant, no reaping, no abuse
vector). The browser login and token exchange still happen against Cognito, so
the access token is minted and signed by Cognito (iss = the pool). The
Resource-Server verifier (mcp_auth.CognitoTokenVerifier) is therefore unchanged;
it only needs the Claude public client's id in COGNITO_ALLOWED_CLIENT_IDS.

Gated on settings.dcr_enabled (hosted-UI domain + claude client id present).
When off, the protected-resource metadata keeps pointing straight at the Cognito
issuer and none of these routes are added — the M2M-only deployment is untouched.
"""

import logging
import time

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from .config import ApiSettings
from .config import settings as _settings
from .mcp_auth import SCOPE_FORECAST, SCOPE_READ

logger = logging.getLogger(__name__)

# Scopes the façade advertises and echoes back. Deliberately omits
# `offline_access`: Cognito does not recognise that scope name and would reject
# the token request, so we never invite the client to ask for it (refresh tokens
# come from the app client's own refresh config, not a scope).
_ADVERTISED_SCOPES = ["openid", "email", SCOPE_READ, SCOPE_FORECAST]


def authorization_server_metadata(settings: ApiSettings) -> dict:
    """RFC 8414 authorization-server metadata served at the Oracle origin.

    `issuer` is our own origin (the URL clients fetched this document from, as
    RFC 8414 requires); authorization/token/jwks stay on Cognito;
    `registration_endpoint` is ours (the façade below).
    """
    issuer = settings.mcp_as_issuer
    return {
        "issuer": issuer,
        "authorization_endpoint": settings.cognito_authorize_endpoint,
        "token_endpoint": settings.cognito_token_endpoint,
        "registration_endpoint": f"{issuer}/register",
        "jwks_uri": settings.cognito_jwks_url,
        "scopes_supported": _ADVERTISED_SCOPES,
        "response_types_supported": ["code"],
        "response_modes_supported": ["query"],
        "grant_types_supported": ["authorization_code", "refresh_token"],
        "token_endpoint_auth_methods_supported": ["none"],
        "code_challenge_methods_supported": ["S256"],
    }


async def _get_as_metadata(request: Request) -> JSONResponse:
    return JSONResponse(authorization_server_metadata(_settings))


async def _register(request: Request) -> JSONResponse:
    """RFC 7591 façade: ignore the request body, hand back the one static public
    client. Returns the SAME client_id on every call — Claude accepts a fixed id
    and only loops if handed a NEW one each time. No Cognito client is created.

    Cognito is the real gate: a redirect_uri not pre-registered on the static
    client is rejected at /authorize, so we do not re-validate it here — we just
    echo what was sent (RFC 7591 clients expect their redirect_uris back).
    """
    try:
        body = await request.json()
    except Exception:  # empty/malformed body — still return the static client
        body = {}
    if not isinstance(body, dict):
        body = {}

    redirect_uris = body.get("redirect_uris") or []
    resp = {
        "client_id": _settings.cognito_claude_client_id,
        "client_id_issued_at": int(time.time()),
        # No client_secret — public PKCE client (token_endpoint_auth_method=none).
        "redirect_uris": redirect_uris,
        "grant_types": body.get("grant_types") or ["authorization_code", "refresh_token"],
        "response_types": body.get("response_types") or ["code"],
        "token_endpoint_auth_method": "none",
        "client_name": body.get("client_name") or "mcp-client",
        "scope": " ".join(_ADVERTISED_SCOPES),
    }
    logger.info("DCR façade returned static client_id for redirect_uris=%s", redirect_uris)
    return JSONResponse(resp, status_code=201)


def create_dcr_routes(settings: ApiSettings) -> list[Route]:
    """AS-metadata + registration routes for the origin root.

    Added to the MAIN app (not the /mcp mount) only when settings.dcr_enabled, so
    the well-known documents sit where OAuth clients look. `settings` is accepted
    for symmetry/testability; the handlers read the module singleton so a live
    env change is picked up without rebuilding routes.
    """
    return [
        Route("/.well-known/oauth-authorization-server", _get_as_metadata, methods=["GET"]),
        # Some clients probe the OIDC discovery path instead; serve the same doc.
        Route("/.well-known/openid-configuration", _get_as_metadata, methods=["GET"]),
        Route("/register", _register, methods=["POST"]),
    ]
