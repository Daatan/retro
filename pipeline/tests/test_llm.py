"""Tests for the shared LLM dispatch helpers (tm.llm).

These pin the behaviour that gatekeeper / extractor / aggregator used to each
implement inline and now delegate to:
  * the rate-limit retry contract (backoff schedule, what retries vs. propagates),
  * apply_routing's settings-driven kwargs,
  * extract_usage's token parsing,
and that each caller delegates to complete_structured with its exact params
(the thing most likely to break when inlined kwargs became function args).
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tm import llm


# ── retry_on_rate_limit ──────────────────────────────────────────────────────

class TestRetryOnRateLimit:
    async def test_succeeds_after_transient_rate_limits(self, monkeypatch):
        sleeps: list = []
        async def fake_sleep(s):
            sleeps.append(s)
        monkeypatch.setattr(llm.asyncio, "sleep", fake_sleep)

        calls = {"n": 0}

        @llm.retry_on_rate_limit
        async def flaky():
            calls["n"] += 1
            if calls["n"] < 3:
                raise RuntimeError("Error: 429 rate limit exceeded")
            return "ok"

        assert await flaky() == "ok"
        assert calls["n"] == 3
        # First attempt has no wait; the two retries wait the first two backoffs.
        assert sleeps == llm.RATE_LIMIT_BACKOFF[:2]

    async def test_exhaustion_raises_last_rate_limit_error(self, monkeypatch):
        monkeypatch.setattr(llm.asyncio, "sleep", AsyncMock())

        @llm.retry_on_rate_limit
        async def always():
            raise RuntimeError("429 rate limited")

        with pytest.raises(RuntimeError, match="429"):
            await always()
        # One sleep per backoff entry (every attempt after the first).
        waited = [c.args[0] for c in llm.asyncio.sleep.await_args_list]
        assert waited == llm.RATE_LIMIT_BACKOFF

    async def test_non_rate_limit_error_propagates_immediately(self, monkeypatch):
        monkeypatch.setattr(llm.asyncio, "sleep", AsyncMock())
        calls = {"n": 0}

        @llm.retry_on_rate_limit
        async def boom():
            calls["n"] += 1
            raise ValueError("schema validation failed")

        with pytest.raises(ValueError):
            await boom()
        assert calls["n"] == 1
        llm.asyncio.sleep.assert_not_awaited()

    async def test_first_call_succeeds_no_sleep(self, monkeypatch):
        monkeypatch.setattr(llm.asyncio, "sleep", AsyncMock())

        @llm.retry_on_rate_limit
        async def fine():
            return 42

        assert await fine() == 42
        llm.asyncio.sleep.assert_not_awaited()


# ── is_rate_limit_error ──────────────────────────────────────────────────────

class TestIsRateLimitError:
    @pytest.mark.parametrize("msg", [
        "429 Too Many Requests",
        "Rate limit exceeded",
        "Bedrock is temporarily unavailable",
        "request limit reached",
    ])
    def test_transient_messages_are_retryable(self, msg):
        assert llm.is_rate_limit_error(Exception(msg)) is True

    @pytest.mark.parametrize("msg", [
        "schema validation error",
        "connection refused",
        "invalid model id",
    ])
    def test_other_messages_are_not_retryable(self, msg):
        assert llm.is_rate_limit_error(Exception(msg)) is False


# ── apply_routing ────────────────────────────────────────────────────────────

class TestApplyRouting:
    def test_bedrock_default_adds_only_region(self, monkeypatch):
        monkeypatch.setattr(llm.settings, "model_api_base", "")
        monkeypatch.setattr(llm.settings, "aws_region", "us-east-1")
        out = llm.apply_routing({"model": "bedrock/x"})
        assert out["aws_region_name"] == "us-east-1"
        assert "api_base" not in out

    def test_api_base_override_adds_base_and_key(self, monkeypatch):
        monkeypatch.setattr(llm.settings, "model_api_base", "http://localhost:11434")
        monkeypatch.setattr(llm.settings, "model_api_key", "secret")
        monkeypatch.setattr(llm.settings, "aws_region", "")
        out = llm.apply_routing({"model": "ollama/x"})
        assert out["api_base"] == "http://localhost:11434"
        assert out["api_key"] == "secret"
        assert "aws_region_name" not in out


# ── extract_usage ────────────────────────────────────────────────────────────

class TestExtractUsage:
    def test_parses_token_counts(self):
        completion = SimpleNamespace(usage=SimpleNamespace(
            prompt_tokens=10, completion_tokens=5, total_tokens=15))
        assert llm.extract_usage(completion) == {
            "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

    def test_none_completion_returns_empty(self):
        assert llm.extract_usage(None) == {}

    def test_completion_without_usage_returns_empty(self):
        assert llm.extract_usage(SimpleNamespace(usage=None)) == {}


# ── caller delegation (the inline-kwargs → args refactor risk) ───────────────

class TestCompleteText:
    @staticmethod
    def _resp(text):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])

    async def test_builds_messages_from_prompt_and_system(self, monkeypatch):
        acompletion = AsyncMock(return_value=self._resp("hello"))
        monkeypatch.setattr(llm.litellm, "acompletion", acompletion)

        out = await llm.complete_text_once("bedrock/x", "the prompt", system="be terse", max_tokens=10)
        assert out == "hello"
        kw = acompletion.await_args.kwargs
        assert kw["model"] == "bedrock/x"
        assert kw["messages"] == [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "the prompt"},
        ]
        assert kw["max_tokens"] == 10
        # Optional params omitted when not supplied.
        assert "temperature" not in kw and "response_format" not in kw

    async def test_passes_through_optional_params_and_messages(self, monkeypatch):
        acompletion = AsyncMock(return_value=self._resp("x"))
        monkeypatch.setattr(llm.litellm, "acompletion", acompletion)

        msgs = [{"role": "user", "content": "hi"}]
        await llm.complete_text_once(
            "bedrock/x", messages=msgs, max_tokens=5,
            temperature=0.15, response_format={"type": "json_object"}, timeout=30,
        )
        kw = acompletion.await_args.kwargs
        assert kw["messages"] == msgs
        assert kw["temperature"] == 0.15
        assert kw["response_format"] == {"type": "json_object"}
        assert kw["timeout"] == 30

    async def test_complete_text_once_does_not_retry(self, monkeypatch):
        acompletion = AsyncMock(side_effect=RuntimeError("429 rate limit"))
        monkeypatch.setattr(llm.litellm, "acompletion", acompletion)
        monkeypatch.setattr(llm.asyncio, "sleep", AsyncMock())

        with pytest.raises(RuntimeError, match="429"):
            await llm.complete_text_once("bedrock/x", "p", max_tokens=5)
        assert acompletion.await_count == 1  # no retry
        llm.asyncio.sleep.assert_not_awaited()

    async def test_complete_text_retries_on_rate_limit(self, monkeypatch):
        calls = {"n": 0}
        async def flaky(**_kw):
            calls["n"] += 1
            if calls["n"] < 2:
                raise RuntimeError("429 rate limit")
            return self._resp("recovered")
        monkeypatch.setattr(llm.litellm, "acompletion", flaky)
        monkeypatch.setattr(llm.asyncio, "sleep", AsyncMock())

        out = await llm.complete_text("bedrock/x", "p", max_tokens=5)
        assert out == "recovered"
        assert calls["n"] == 2  # retried once


class TestCallerDelegation:
    async def test_gatekeeper_delegates_with_exact_params(self, monkeypatch):
        from tm import gatekeeper
        from tm.models import GatekeeperOutput

        captured = {}
        async def fake(model, response_model, prompt, *, max_tokens, timeout):
            captured.update(model=model, response_model=response_model,
                            prompt=prompt, max_tokens=max_tokens, timeout=timeout)
            return ("OUT", {"total_tokens": 1})
        monkeypatch.setattr(gatekeeper, "complete_structured", fake)

        result = await gatekeeper.check_is_prediction("article", "src", "2024-01-01", "Some Event")
        assert result == ("OUT", {"total_tokens": 1})
        assert captured["model"] == gatekeeper.settings.gatekeeper_model
        assert captured["response_model"] is GatekeeperOutput
        assert captured["max_tokens"] == 200
        assert captured["timeout"] == 90
        assert "Some Event" in captured["prompt"]

    async def test_extractor_delegates_with_exact_params(self, monkeypatch):
        from tm import extractor
        from tm.models import ExtractionOutput

        captured = {}
        async def fake(model, response_model, prompt, *, max_tokens, timeout):
            captured.update(model=model, response_model=response_model,
                            max_tokens=max_tokens, timeout=timeout)
            return ("OUT", {})
        monkeypatch.setattr(extractor, "complete_structured", fake)

        await extractor.extract_predictions("article", "src", "2024-01-01", "Event", "desc")
        assert captured["model"] == extractor.settings.extractor_model
        assert captured["response_model"] is ExtractionOutput
        assert captured["max_tokens"] == 1200
        assert captured["timeout"] == 180

    async def test_aggregator_delegates_and_returns_output_only(self, monkeypatch):
        from tm import aggregator
        from tm.models import PredictionExtraction

        captured = {}
        async def fake(model, response_model, prompt, *, max_tokens, timeout):
            captured.update(model=model, response_model=response_model,
                            max_tokens=max_tokens, timeout=timeout)
            return ("THE_OUTPUT", {"total_tokens": 9})
        monkeypatch.setattr(aggregator, "complete_structured", fake)

        preds = [
            PredictionExtraction(quote="q1", claim="c1", stance=0.5, certainty=0.6),
            PredictionExtraction(quote="q2", claim="c2", stance=-0.3, certainty=0.4),
        ]
        out = await aggregator.aggregate_article_predictions(preds, "Event", "src", "2024-01-01")
        assert out == "THE_OUTPUT"  # usage is discarded by the aggregator
        assert captured["model"] == aggregator.settings.extractor_model
        assert captured["response_model"] is PredictionExtraction
        assert captured["max_tokens"] == 1000
        assert captured["timeout"] == 120
