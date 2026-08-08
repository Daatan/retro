"""Rate limiting for MCP tool calls (retro#431).

The REST routes in main.py are decorated with @limiter.limit(...) (slowapi),
keyed on remote address. MCP tools (mcp_server.py) call the same underlying
service functions (run_forecast, run_search, fetch_and_extract) directly, in
process — they never pass through those decorated routes, so an authenticated
MCP client could loop the most expensive tool (forecast, LLM-backed) with no
cap at all.

This mirrors the REST limits per tool, using the same `limits` package
slowapi wraps underneath (fixed-window strategy, matching slowapi's default),
but keyed on the caller's authenticated Cognito identity — `sub` for the
human PKCE flow, `client_id` for M2M — rather than remote address. MCP calls
don't carry a Starlette Request the REST decorator relies on, and identity is
the more meaningful key anyway: MCP traffic can arrive from a shared
client-side proxy where IP wouldn't discriminate between callers.

In-memory, per-process, same as the REST limiter's default storage — not
shared across gunicorn workers. Consistent with existing behavior, not a new
gap: two workers under a determined caller roughly double the effective cap,
same as it already does for REST today.
"""

from limits import RateLimitItem, parse
from limits.storage import MemoryStorage
from limits.strategies import FixedWindowRateLimiter

from .mcp_auth import get_access_token

_storage = MemoryStorage()
_strategy = FixedWindowRateLimiter(_storage)

# Mirrors each tool's REST equivalent: forecast->10/min, search_news->60/min
# (/search), fetch_article->30/min (/fetch-url), polymarket_market->30/min
# (/pm/markets). polymarket_edge runs a full forecast internally, so it gets
# forecast's limit rather than a separate one. bayes_nodes/source_leaderboard
# have no REST limiter either — left uncapped for parity.
FORECAST_LIMIT = parse("10/minute")
SEARCH_LIMIT = parse("60/minute")
FETCH_LIMIT = parse("30/minute")
MARKET_LIMIT = parse("30/minute")


class MCPRateLimitExceeded(Exception):
    """A tool call exceeded its per-caller rate limit. Raising propagates to
    the MCP client as a tool error, same as ScopeError does for auth."""


def _caller_key() -> str:
    token = get_access_token()
    # Defensive — should be unreachable behind RequireAuthMiddleware, same
    # premise as require_scope()'s check in mcp_auth.py.
    if token is None:
        return "anonymous"
    return token.subject or token.client_id


def enforce(item: RateLimitItem, tool_name: str) -> None:
    key = _caller_key()
    if not _strategy.hit(item, "mcp", tool_name, key):
        raise MCPRateLimitExceeded(
            f"rate limit exceeded for tool '{tool_name}': max {item.amount}/{item.GRANULARITY.name.lower()} per caller"
        )


def reset() -> None:
    """Test-only: clear all rate-limit state so test order/count can't leak
    hits between cases."""
    _storage.reset()
