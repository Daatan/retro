"""DCR façade for the MCP OAuth flow (Cognito has no Dynamic Client Registration).

Claude's MCP connector will not use a static client_id: it discovers the
authorization server from our protected-resource metadata, fetches that AS's
`/.well-known/oauth-authorization-server`, and REQUIRES a `registration_endpoint`
there (RFC 7591) — it hard-fails discovery with "does not support dynamic client
registration" otherwise. Cognito publishes no such endpoint and can't be made to.

So the Oracul origin advertises *itself* as the authorization server in metadata
only, and proxies the OAuth endpoints:

- `/register` (RFC 7591 façade) returns the one pre-provisioned Cognito public
  client on every call — no client is ever created (no CreateUserPoolClient, so
  no IAM grant, no reaping, no abuse vector).
- `/oauth2/authorize` and `/oauth2/token` proxy Cognito's real endpoints but
  **strip the `offline_access` scope**. Claude always requests `offline_access`
  (to get a refresh token), but Cognito does not recognise that scope name and
  rejects the whole request with `invalid_scope` — which bounced the login before
  the user could sign in. Cognito issues refresh tokens from the app client's own
  config regardless of that scope, so stripping it costs nothing: Claude still
  gets its refresh token, it just can't ask for it by name.

The access token is still minted and signed by Cognito (iss = the pool), so the
Resource-Server verifier (mcp_auth.CognitoTokenVerifier) is unchanged; it only
needs the Claude public client's id in COGNITO_ALLOWED_CLIENT_IDS.

Gated on settings.dcr_enabled (hosted-UI domain + claude client id present).
When off, the protected-resource metadata keeps pointing straight at the Cognito
issuer and none of these routes are added — the M2M-only deployment is untouched.
"""

import logging
import time
from urllib.parse import parse_qsl, urlencode

import httpx
from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse, Response
from starlette.routing import Route

from .config import ApiSettings
from .config import settings as _settings
from .mcp_auth import SCOPE_FORECAST, SCOPE_READ

logger = logging.getLogger(__name__)

# Scopes the façade advertises and echoes back. Deliberately omits
# `offline_access` (see module docstring) — we also strip it below in case the
# client sends it anyway (Claude does).
_ADVERTISED_SCOPES = ["openid", "email", SCOPE_READ, SCOPE_FORECAST]
_DROP_SCOPES = {"offline_access"}


def _clean_scope(scope: str | None) -> str | None:
    """Drop scope names Cognito rejects (offline_access). Returns None if the
    input was None (so we don't inject an empty scope param)."""
    if scope is None:
        return None
    kept = [s for s in scope.split() if s not in _DROP_SCOPES]
    return " ".join(kept)


def authorization_server_metadata(settings: ApiSettings) -> dict:
    """RFC 8414 authorization-server metadata served at the Oracul origin.

    `issuer` is our own origin (the URL clients fetched this document from, as
    RFC 8414 requires). authorization/token point at our own proxy endpoints
    (which strip offline_access before Cognito); jwks stays on Cognito;
    `registration_endpoint` is ours.
    """
    issuer = settings.mcp_as_issuer
    return {
        "issuer": issuer,
        "authorization_endpoint": f"{issuer}/oauth2/authorize",
        "token_endpoint": f"{issuer}/oauth2/token",
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


async def _authorize(request: Request) -> Response:
    """Proxy Cognito's /oauth2/authorize, stripping offline_access from `scope`.

    Pure redirect: the user's browser is 302'd to the real Cognito authorize with
    all params preserved except the cleaned scope. Cognito then handles login and
    redirects to the client's registered redirect_uri with the code.
    """
    params = dict(request.query_params)
    if "scope" in params:
        params["scope"] = _clean_scope(params["scope"])
    target = _settings.cognito_authorize_endpoint
    if not target:  # dcr disabled / misconfigured — nothing to proxy to
        return JSONResponse({"error": "authorization endpoint unavailable"}, status_code=503)
    return RedirectResponse(f"{target}?{urlencode(params)}", status_code=302)


async def _token(request: Request) -> Response:
    """Proxy Cognito's /oauth2/token, stripping offline_access from any `scope`.

    The auth-code exchange carries no scope, but the later refresh_token grant may
    include one — strip it there too so refreshes don't 400. Forwards the
    form-encoded body verbatim (minus offline_access) and returns Cognito's
    response untouched.
    """
    target = _settings.cognito_token_endpoint
    if not target:
        return JSONResponse({"error": "token endpoint unavailable"}, status_code=503)

    raw = (await request.body()).decode("utf-8", "replace")
    pairs = parse_qsl(raw, keep_blank_values=True)
    cleaned = [(k, _clean_scope(v) if k == "scope" else v) for k, v in pairs]
    forward_body = urlencode(cleaned)

    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json",
    }
    # A confidential client would authenticate with a Basic header; the public
    # PKCE client sends none, but forward it if present so we stay grant-agnostic.
    if "authorization" in request.headers:
        headers["Authorization"] = request.headers["authorization"]

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            upstream = await client.post(target, content=forward_body, headers=headers)
    except httpx.HTTPError as exc:
        logger.warning("DCR token proxy upstream error: %s", exc)
        return JSONResponse({"error": "server_error", "error_description": "token endpoint unreachable"},
                            status_code=502)
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type", "application/json"),
    )


def create_dcr_routes(settings: ApiSettings) -> list[Route]:
    """AS-metadata + registration + OAuth-proxy routes for the origin root.

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
        Route("/oauth2/authorize", _authorize, methods=["GET"]),
        Route("/oauth2/token", _token, methods=["POST"]),
    ]
