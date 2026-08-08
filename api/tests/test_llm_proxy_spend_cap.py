"""retro#432: the /llm proxy bypassed the per-key max_articles spend cap
entirely, and LlmMessage.content/messages had no size limit at all — a
budget-capped key (or any key) could drive unmetered LLM cost straight
through this route, defeating docs#57 item 1's whole point.
"""

import json
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from forecast_api.config import settings as api_settings
from forecast_api.main import app

client = TestClient(app)
NAMED = json.dumps({"staging": {"key": "staging-key", "max_articles": 3}})


class TestLlmProxyDeniedToCappedClients:
    BODY = {"messages": [{"role": "user", "content": "hi"}]}

    def test_capped_key_is_denied(self, monkeypatch):
        monkeypatch.setattr(api_settings, "oracle_api_keys", NAMED)
        r = client.post("/llm", json=self.BODY, headers={"x-api-key": "staging-key"})
        assert r.status_code == 403

    def test_uncapped_default_key_still_works(self):
        with patch("forecast_api.main.complete_text_once_with_usage",
                   new=AsyncMock(return_value=("hello", {}))):
            r = client.post("/llm", json=self.BODY, headers={"x-api-key": "test-key"})
        assert r.status_code == 200


class TestLlmMessageSizeLimits:
    HEADERS = {"x-api-key": "test-key"}

    def test_oversized_content_is_rejected(self):
        body = {"messages": [{"role": "user", "content": "x" * 32_001}]}
        r = client.post("/llm", json=body, headers=self.HEADERS)
        assert r.status_code == 422

    def test_content_at_the_limit_is_accepted(self):
        body = {"messages": [{"role": "user", "content": "x" * 32_000}]}
        with patch("forecast_api.main.complete_text_once_with_usage",
                   new=AsyncMock(return_value=("hello", {}))):
            r = client.post("/llm", json=body, headers=self.HEADERS)
        assert r.status_code == 200

    def test_too_many_messages_is_rejected(self):
        body = {"messages": [{"role": "user", "content": "hi"}] * 21}
        r = client.post("/llm", json=body, headers=self.HEADERS)
        assert r.status_code == 422

    def test_message_count_at_the_limit_is_accepted(self):
        body = {"messages": [{"role": "user", "content": "hi"}] * 20}
        with patch("forecast_api.main.complete_text_once_with_usage",
                   new=AsyncMock(return_value=("hello", {}))):
            r = client.post("/llm", json=body, headers=self.HEADERS)
        assert r.status_code == 200
