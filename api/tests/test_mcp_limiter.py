"""Tests for mcp_limiter (retro#431): MCP tool calls bypassed every REST
@limiter.limit(...) entirely since they never pass through a decorated route.
"""

from types import SimpleNamespace

import pytest
from limits import parse

from forecast_api import mcp_limiter


@pytest.fixture(autouse=True)
def _reset():
    mcp_limiter.reset()
    yield


def _token(sub=None, client_id="client-abc"):
    return SimpleNamespace(subject=sub, client_id=client_id)


class TestEnforce:
    def test_allows_up_to_the_limit(self, monkeypatch):
        monkeypatch.setattr(mcp_limiter, "get_access_token", lambda: _token(sub="user-1"))
        item = parse("3/minute")
        for _ in range(3):
            mcp_limiter.enforce(item, "some_tool")  # must not raise

    def test_raises_once_the_limit_is_exceeded(self, monkeypatch):
        monkeypatch.setattr(mcp_limiter, "get_access_token", lambda: _token(sub="user-1"))
        item = parse("3/minute")
        for _ in range(3):
            mcp_limiter.enforce(item, "some_tool")
        with pytest.raises(mcp_limiter.MCPRateLimitExceeded):
            mcp_limiter.enforce(item, "some_tool")

    def test_different_callers_get_independent_budgets(self, monkeypatch):
        """The bug this fixes: a per-process cap with no caller key would let
        one determined MCP client exhaust the budget for everyone else."""
        item = parse("1/minute")

        monkeypatch.setattr(mcp_limiter, "get_access_token", lambda: _token(sub="user-1"))
        mcp_limiter.enforce(item, "some_tool")  # user-1's only hit this window

        monkeypatch.setattr(mcp_limiter, "get_access_token", lambda: _token(sub="user-2"))
        mcp_limiter.enforce(item, "some_tool")  # user-2 is unaffected by user-1's hit

    def test_different_tools_get_independent_budgets(self, monkeypatch):
        monkeypatch.setattr(mcp_limiter, "get_access_token", lambda: _token(sub="user-1"))
        item = parse("1/minute")
        mcp_limiter.enforce(item, "tool_a")
        mcp_limiter.enforce(item, "tool_b")  # different tool_name, not the same bucket

    def test_m2m_token_with_no_subject_keys_on_client_id(self, monkeypatch):
        """M2M (client-credentials) Cognito tokens carry no `sub` claim."""
        item = parse("1/minute")
        monkeypatch.setattr(mcp_limiter, "get_access_token", lambda: _token(sub=None, client_id="m2m-client-a"))
        mcp_limiter.enforce(item, "some_tool")
        with pytest.raises(mcp_limiter.MCPRateLimitExceeded):
            mcp_limiter.enforce(item, "some_tool")

        monkeypatch.setattr(mcp_limiter, "get_access_token", lambda: _token(sub=None, client_id="m2m-client-b"))
        mcp_limiter.enforce(item, "some_tool")  # a different client_id is a separate budget

    def test_no_token_falls_back_to_a_shared_anonymous_bucket(self, monkeypatch):
        """Defensive path — should be unreachable behind RequireAuthMiddleware
        in production, mirroring require_scope()'s equivalent check."""
        monkeypatch.setattr(mcp_limiter, "get_access_token", lambda: None)
        item = parse("1/minute")
        mcp_limiter.enforce(item, "some_tool")
        with pytest.raises(mcp_limiter.MCPRateLimitExceeded):
            mcp_limiter.enforce(item, "some_tool")
