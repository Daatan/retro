"""OAuth 2.1 Resource-Server auth for the MCP endpoint.

The Oracul is a Resource Server: it does not log anyone in, it only verifies the
Bearer JWT access tokens minted by the Cognito user pool (the Authorization
Server) and maps their scopes to allowed tools.

Cognito access-token specifics handled here:
  - signed RS256, keys from the pool's JWKS (cached via PyJWKClient)
  - `token_use` == "access" (id tokens are rejected)
  - no `aud` claim — the app client is identified by `client_id`, which we check
    against COGNITO_ALLOWED_CLIENT_IDS (fail closed: empty allow-list rejects all)
  - `scope` is a space-delimited string, e.g. "oracle-mcp/read oracle-mcp/forecast"
"""

import asyncio
import logging
import time
from typing import Optional

import jwt
from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.auth.provider import AccessToken, TokenVerifier

from .config import settings

logger = logging.getLogger(__name__)

# Cognito resource-server scopes (`<resource-server-id>/<scope>`).
SCOPE_READ = "oracle-mcp/read"
SCOPE_FORECAST = "oracle-mcp/forecast"


class ScopeError(Exception):
    """A tool call carried a valid token that lacks the required scope."""


class CognitoTokenVerifier(TokenVerifier):
    """Verifies Cognito JWT access tokens against the pool's JWKS.

    Returns None on any failure — the SDK maps that to 401. Never raises out of
    verify_token, so a malformed token can't 500 the endpoint.
    """

    def __init__(
        self,
        issuer: str,
        jwks_url: str,
        allowed_client_ids: set[str],
        resource_url: Optional[str] = None,
    ):
        self._issuer = issuer
        self._allowed_client_ids = allowed_client_ids
        self._resource_url = resource_url
        # PyJWKClient caches keys and refreshes on an unknown kid (key rotation).
        self._jwk_client = jwt.PyJWKClient(jwks_url, cache_keys=True)

    def _verify_sync(self, token: str) -> Optional[AccessToken]:
        try:
            signing_key = self._jwk_client.get_signing_key_from_jwt(token)
            claims = jwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256"],
                issuer=self._issuer,
                # Cognito access tokens carry no `aud`; client_id is checked below.
                options={"verify_aud": False},
            )
        except Exception as exc:  # invalid signature / expired / bad issuer / malformed
            logger.info("mcp token rejected: %s", exc)
            return None

        if claims.get("token_use") != "access":
            logger.info("mcp token rejected: token_use=%s (expected access)", claims.get("token_use"))
            return None

        client_id = claims.get("client_id") or ""
        if client_id not in self._allowed_client_ids:
            logger.info("mcp token rejected: client_id %r not allowed", client_id)
            return None

        scopes = (claims.get("scope") or "").split()
        return AccessToken(
            token=token,
            client_id=client_id,
            scopes=scopes,
            expires_at=claims.get("exp"),
            resource=self._resource_url,
            subject=claims.get("sub"),
            claims=claims,
        )

    async def verify_token(self, token: str) -> Optional[AccessToken]:
        # JWKS fetch + RS256 verify are sync/CPU-ish; keep them off the loop.
        return await asyncio.to_thread(self._verify_sync, token)


def require_scope(scope: str) -> AccessToken:
    """Assert the current caller's token carries `scope`; return the token.

    Called at the top of the expensive tools (forecast, polymarket_edge). The
    transport already enforced the global `oracle-mcp/read` floor, so reaching a
    tool means an authenticated token exists; this adds the finer check.
    """
    token = get_access_token()
    if token is None:  # defensive — should be unreachable behind RequireAuthMiddleware
        raise ScopeError("authentication required")
    if scope not in token.scopes:
        raise ScopeError(
            f"this tool requires the '{scope}' scope; your token has: "
            f"{', '.join(token.scopes) or '(none)'}"
        )
    # Belt-and-suspenders expiry check (jwt.decode already enforced exp).
    if token.expires_at is not None and token.expires_at < time.time():
        raise ScopeError("token expired")
    return token
