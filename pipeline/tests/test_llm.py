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

import litellm
import pytest

from tm import llm


def _stalled_timeout(time_taken="0.001"):
    return litellm.Timeout(
        message=f"Connection timed out. Timeout passed=Timeout(timeout=90.0), "
                f"time taken={time_taken} seconds",
        model="bedrock/x", llm_provider="bedrock",
    )


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

    async def test_retries_stalled_event_loop_timeout(self, monkeypatch):
        """retro#600: the same shared retry gets structured-output callers
        (gatekeeper/extractor/aggregator) through this false-positive too."""
        monkeypatch.setattr(llm.asyncio, "sleep", AsyncMock())
        calls = {"n": 0}

        @llm.retry_on_rate_limit
        async def flaky():
            calls["n"] += 1
            if calls["n"] < 2:
                raise _stalled_timeout()
            return "ok"

        assert await flaky() == "ok"
        assert calls["n"] == 2


# ── is_stalled_event_loop_timeout / retry_once_on_stalled_timeout (retro#600) ─

class TestIsStalledEventLoopTimeout:
    def test_near_zero_litellm_timeout_is_stalled(self):
        assert llm.is_stalled_event_loop_timeout(_stalled_timeout("0.001")) is True

    def test_slow_litellm_timeout_is_a_real_timeout(self):
        # A genuine timeout takes close to the full configured duration.
        assert llm.is_stalled_event_loop_timeout(_stalled_timeout("89.9")) is False

    def test_other_exception_types_are_not_stalled_timeouts(self):
        assert llm.is_stalled_event_loop_timeout(RuntimeError("time taken=0.001 seconds")) is False

    def test_litellm_timeout_without_parseable_elapsed_is_not_stalled(self):
        exc = litellm.Timeout(message="boom", model="bedrock/x", llm_provider="bedrock")
        assert llm.is_stalled_event_loop_timeout(exc) is False


class TestRetryOnceOnStalledTimeout:
    async def test_retries_exactly_once_and_succeeds(self):
        calls = {"n": 0}

        @llm.retry_once_on_stalled_timeout
        async def flaky():
            calls["n"] += 1
            if calls["n"] == 1:
                raise _stalled_timeout()
            return "ok"

        assert await flaky() == "ok"
        assert calls["n"] == 2

    async def test_second_stalled_timeout_in_a_row_propagates(self):
        calls = {"n": 0}

        @llm.retry_once_on_stalled_timeout
        async def always_stalled():
            calls["n"] += 1
            raise _stalled_timeout()

        with pytest.raises(litellm.Timeout):
            await always_stalled()
        assert calls["n"] == 2  # one retry, then give up

    async def test_other_exceptions_propagate_immediately(self):
        calls = {"n": 0}

        @llm.retry_once_on_stalled_timeout
        async def boom():
            calls["n"] += 1
            raise ValueError("schema validation failed")

        with pytest.raises(ValueError):
            await boom()
        assert calls["n"] == 1


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
            "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
            "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}

    def test_parses_cache_token_counts_when_present(self):
        completion = SimpleNamespace(usage=SimpleNamespace(
            prompt_tokens=10, completion_tokens=5, total_tokens=15,
            cache_read_input_tokens=7, cache_creation_input_tokens=3))
        assert llm.extract_usage(completion) == {
            "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
            "cache_read_input_tokens": 7, "cache_creation_input_tokens": 3}

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

    async def test_with_usage_returns_text_and_usage(self, monkeypatch):
        # The Oracle API's token_usage response field (docs#57 item 3) rides on
        # this sibling: same call, but the usage is returned instead of dropped.
        resp = self._resp("hello")
        resp.usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        monkeypatch.setattr(llm.litellm, "acompletion", AsyncMock(return_value=resp))

        text, usage = await llm.complete_text_once_with_usage("bedrock/x", "p", max_tokens=5)
        assert text == "hello"
        assert usage["prompt_tokens"] == 10
        assert usage["completion_tokens"] == 5
        assert usage["total_tokens"] == 15

    async def test_with_usage_reports_empty_dict_when_backend_has_none(self, monkeypatch):
        monkeypatch.setattr(llm.litellm, "acompletion", AsyncMock(return_value=self._resp("x")))
        _text, usage = await llm.complete_text_once_with_usage("bedrock/x", "p", max_tokens=5)
        assert usage == {}

    async def test_with_usage_retries_once_on_stalled_timeout(self, monkeypatch):
        calls = {"n": 0}
        async def flaky(**_kw):
            calls["n"] += 1
            if calls["n"] == 1:
                raise _stalled_timeout()
            return self._resp("recovered")
        monkeypatch.setattr(llm.litellm, "acompletion", flaky)

        text, _usage = await llm.complete_text_once_with_usage("bedrock/x", "p", max_tokens=5)
        assert text == "recovered"
        assert calls["n"] == 2

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


class TestModelSupportsPromptCache:
    """retro#650 — allowlist, not denylist: an unrecognized model must default to
    unsupported, so a new model family added to the fleet degrades to full-price
    calls rather than to every call failing."""

    @pytest.mark.parametrize("model", [
        "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "bedrock/us.amazon.nova-micro-v1:0",
        "bedrock/us.amazon.nova-2-lite-v1:0",
    ])
    def test_known_supported_families(self, model):
        assert llm._model_supports_prompt_cache(model) is True

    @pytest.mark.parametrize("model", [
        "bedrock/qwen.qwen3-32b-v1:0",
        "bedrock/zai.glm-4.7-flash",
        "bedrock/us.meta.llama4-scout-17b-instruct-v1:0",
        "bedrock/us.deepseek.r1-v1:0",
    ])
    def test_unrecognized_families_default_to_unsupported(self, model):
        assert llm._model_supports_prompt_cache(model) is False


class TestCompleteStructuredTemperature:
    """`temperature` is optional on the wire, not just in the signature.

    Bedrock's newest Anthropic models reject the parameter outright
    ("`temperature` is deprecated for this model"), which fails 100% of calls
    to them — the same shape of breakage retro#650 hit with cache_control.
    Passing None must omit the key entirely; every existing caller passes
    nothing and must keep getting 0."""

    @staticmethod
    def _patch_client(monkeypatch):
        captured = {}
        async def fake_create(**kwargs):
            captured.update(kwargs)
            return ("OUT", SimpleNamespace(usage=None))
        monkeypatch.setattr(llm.client.chat.completions, "create_with_completion", fake_create)
        return captured

    async def test_default_still_sends_zero(self, monkeypatch):
        captured = self._patch_client(monkeypatch)
        await llm.complete_structured("bedrock/x", dict, "P", max_tokens=10, timeout=5)
        assert captured["temperature"] == 0

    async def test_explicit_value_is_forwarded(self, monkeypatch):
        captured = self._patch_client(monkeypatch)
        await llm.complete_structured("bedrock/x", dict, "P", max_tokens=10, timeout=5,
                                      temperature=0.7)
        assert captured["temperature"] == 0.7

    async def test_none_omits_the_parameter_entirely(self, monkeypatch):
        captured = self._patch_client(monkeypatch)
        await llm.complete_structured("bedrock/eu.anthropic.claude-sonnet-5", dict, "P",
                                      max_tokens=10, timeout=5, temperature=None)
        assert "temperature" not in captured


class TestCompleteStructuredPromptCaching:
    """cached_prefix wiring in complete_structured itself — the content shape sent
    to the underlying instructor/litellm client, not just that callers pass the
    right args (that's TestCallerDelegation below)."""

    @staticmethod
    def _patch_client(monkeypatch):
        captured = {}
        async def fake_create(**kwargs):
            captured.update(kwargs)
            return ("OUT", SimpleNamespace(usage=None))
        monkeypatch.setattr(llm.client.chat.completions, "create_with_completion", fake_create)
        return captured

    async def test_cache_disabled_sends_one_flat_concatenated_string(self, monkeypatch):
        """With the flag off (verified on by default, but this must still hold for any
        environment/rollback that disables it): must NOT silently drop cached_prefix —
        content has to be the full prefix+prompt text, byte-identical to the single
        PROMPT string this used to be before the split. This is the exact bug caught
        while writing this test: an earlier draft sent `prompt` alone here, which
        would have shipped gatekeeper/extractor calls missing all their instructions."""
        monkeypatch.setattr(llm.settings, "enable_prompt_cache", False)
        captured = self._patch_client(monkeypatch)

        await llm.complete_structured(
            "bedrock/x", dict, "SUFFIX", max_tokens=10, timeout=5, cached_prefix="PREFIX ",
        )
        assert captured["messages"] == [{"role": "user", "content": "PREFIX SUFFIX"}]

    async def test_cache_enabled_splits_into_cache_marked_content_blocks(self, monkeypatch):
        monkeypatch.setattr(llm.settings, "enable_prompt_cache", True)
        captured = self._patch_client(monkeypatch)

        await llm.complete_structured(
            "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0", dict, "SUFFIX",
            max_tokens=10, timeout=5, cached_prefix="PREFIX ",
        )
        assert captured["messages"] == [{"role": "user", "content": [
            {"type": "text", "text": "PREFIX ", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "SUFFIX"},
        ]}]

    async def test_cache_enabled_but_unsupported_model_sends_flat_string(self, monkeypatch):
        """retro#650: Bedrock hard-rejects the whole call when cache_control is sent
        to a model family that doesn't support it (found via Qwen3 32B: 0/50 calls
        succeeded until this check existed). Must degrade to the uncached flat
        string -- same as the flag being off -- never send the cache block on
        spec, and never drop cached_prefix's content either."""
        monkeypatch.setattr(llm.settings, "enable_prompt_cache", True)
        captured = self._patch_client(monkeypatch)

        await llm.complete_structured(
            "bedrock/qwen.qwen3-32b-v1:0", dict, "SUFFIX",
            max_tokens=10, timeout=5, cached_prefix="PREFIX ",
        )
        assert captured["messages"] == [{"role": "user", "content": "PREFIX SUFFIX"}]

    async def test_cache_enabled_but_no_prefix_given_sends_flat_string(self, monkeypatch):
        """Callers that never pass cached_prefix (aggregator today) are unaffected by
        the flag either way."""
        monkeypatch.setattr(llm.settings, "enable_prompt_cache", True)
        captured = self._patch_client(monkeypatch)

        await llm.complete_structured("bedrock/x", dict, "SUFFIX", max_tokens=10, timeout=5)
        assert captured["messages"] == [{"role": "user", "content": "SUFFIX"}]


class TestCallerDelegation:
    async def test_gatekeeper_delegates_with_exact_params(self, monkeypatch):
        from tm import gatekeeper
        from tm.models import GatekeeperOutput

        captured = {}
        async def fake(model, response_model, prompt, *, max_tokens, timeout, cached_prefix=None):
            captured.update(model=model, response_model=response_model,
                            prompt=prompt, max_tokens=max_tokens, timeout=timeout,
                            cached_prefix=cached_prefix)
            return ("OUT", {"total_tokens": 1})
        monkeypatch.setattr(gatekeeper, "complete_structured", fake)

        result = await gatekeeper.check_is_prediction("article", "src", "2024-01-01", "Some Event")
        assert result == ("OUT", {"total_tokens": 1})
        assert captured["model"] == gatekeeper.settings.gatekeeper_model
        assert captured["response_model"] is GatekeeperOutput
        assert captured["max_tokens"] == 200
        assert captured["timeout"] == 90
        assert "Some Event" in captured["prompt"]
        # Full article text reaches the gate (no front-trim): the old
        # article_text[200:2700] slice emptied short inputs like this one.
        assert "article" in captured["prompt"]
        # The fixed instructions go through as their own cacheable prefix, not
        # concatenated into `prompt` — that's the whole point of the split.
        assert captured["cached_prefix"] == gatekeeper.PROMPT_PREFIX

    async def test_extractor_delegates_with_exact_params(self, monkeypatch):
        from tm import extractor
        from tm.models import ExtractionOutput

        captured = {}
        async def fake(model, response_model, prompt, *, max_tokens, timeout, cached_prefix=None):
            captured.update(model=model, response_model=response_model,
                            max_tokens=max_tokens, timeout=timeout, cached_prefix=cached_prefix)
            return ("OUT", {})
        monkeypatch.setattr(extractor, "complete_structured", fake)

        await extractor.extract_predictions("article", "src", "2024-01-01", "Event", "desc")
        assert captured["model"] == extractor.settings.extractor_model
        assert captured["response_model"] is ExtractionOutput
        assert captured["max_tokens"] == 1200
        assert captured["timeout"] == 180
        assert captured["cached_prefix"] == extractor.PROMPT_PREFIX

    async def test_aggregator_delegates_and_returns_output_only(self, monkeypatch):
        from tm import aggregator
        from tm.models import PredictionExtraction

        captured = {}
        # A real model, not a string sentinel: since retro#681 the aggregator
        # carries `reader_confidence` across the collapse, so it reads the
        # object back. A sentinel that isn't a PredictionExtraction would only
        # be testing that this function never touches its own return value.
        sentinel = PredictionExtraction(quote="qA", claim="cA", stance=0.1, certainty=0.5)

        async def fake(model, response_model, prompt, *, max_tokens, timeout):
            captured.update(model=model, response_model=response_model,
                            max_tokens=max_tokens, timeout=timeout)
            return (sentinel, {"total_tokens": 9})
        monkeypatch.setattr(aggregator, "complete_structured", fake)

        preds = [
            PredictionExtraction(quote="q1", claim="c1", stance=0.5, certainty=0.6),
            PredictionExtraction(quote="q2", claim="c2", stance=-0.3, certainty=0.4),
        ]
        out = await aggregator.aggregate_article_predictions(preds, "Event", "src", "2024-01-01")
        assert out is sentinel  # usage is discarded by the aggregator
        assert captured["model"] == aggregator.settings.extractor_model
        assert captured["response_model"] is PredictionExtraction
        assert captured["max_tokens"] == 1000
        assert captured["timeout"] == 120
