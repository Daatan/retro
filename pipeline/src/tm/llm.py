"""Shared LLM dispatch helpers for the TruthMachine pipeline.

The structured-extraction callers — gatekeeper, extractor, aggregator — all talk
to the same litellm/instructor backend (Bedrock Nova by default). Previously each
module copy-pasted four identical pieces:

  * the import-time ``litellm.api_key`` + instructor client construction,
  * the ``model_api_base`` / ``aws_region`` kwargs-routing block,
  * the ``[30, 60, 120]`` rate-limit retry schedule + error classification,
  * the token-usage extraction off the completion object.

They now live here once. ``forecaster.py`` (in the API) also reuses
``apply_routing`` for its keyword-distillation call.
"""

import asyncio
import functools

import instructor
import litellm

from .config import settings

# litellm reads this global for OpenAI-compatible backends. For ``bedrock/*``
# models (the default) litellm ignores it and authenticates with AWS
# credentials, so on the normal path this is a no-op; it only matters when
# ``model_api_base`` points litellm at an OpenAI-compatible server (e.g. Ollama).
litellm.api_key = settings.openrouter_api_key

# Shared instructor-wrapped client. MD_JSON mode coaxes structured JSON out of
# models without native tool-calling (Nova).
client = instructor.from_litellm(litellm.acompletion, mode=instructor.Mode.MD_JSON)

# Retry schedule (seconds) for transient rate-limit / throttling errors.
RATE_LIMIT_BACKOFF = [30, 60, 120]


def apply_routing(kwargs: dict) -> dict:
    """Inject optional backend-routing settings into a litellm kwargs dict.

    Mutates and returns ``kwargs``. When ``model_api_base`` is set we route to an
    OpenAI-compatible server; otherwise the ``bedrock/*`` model in
    ``kwargs["model"]`` is dispatched to AWS Bedrock under ``aws_region_name``.
    """
    if settings.model_api_base:
        kwargs["api_base"] = settings.model_api_base
        kwargs["api_key"] = settings.model_api_key
    if settings.aws_region:
        kwargs["aws_region_name"] = settings.aws_region
    return kwargs


def is_rate_limit_error(exc: Exception) -> bool:
    """True if the exception looks like a transient rate-limit / throttle worth retrying."""
    err = str(exc).lower()
    return "rate" in err or "429" in err or "limit" in err or "temporarily" in err


def extract_usage(completion) -> dict:
    """Pull prompt/completion/total token counts off a litellm completion.

    Returns ``{}`` when the completion carries no usage data.
    """
    if completion and hasattr(completion, "usage") and completion.usage:
        u = completion.usage
        return {
            "prompt_tokens": getattr(u, "prompt_tokens", 0),
            "completion_tokens": getattr(u, "completion_tokens", 0),
            "total_tokens": getattr(u, "total_tokens", 0),
        }
    return {}


def retry_on_rate_limit(func):
    """Retry an async call on transient rate-limit / throttle errors.

    Tries once, then waits ``RATE_LIMIT_BACKOFF`` seconds between further
    attempts. Non-rate-limit exceptions propagate immediately; if every attempt
    is rate-limited the last such exception is raised.
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        last_exc: Exception = RuntimeError("no attempts")
        for wait in [0] + RATE_LIMIT_BACKOFF:
            if wait:
                await asyncio.sleep(wait)
            try:
                return await func(*args, **kwargs)
            except Exception as exc:
                if is_rate_limit_error(exc):
                    last_exc = exc
                    continue
                raise
        raise last_exc
    return wrapper


@retry_on_rate_limit
async def complete_structured(
    model: str,
    response_model,
    prompt: str,
    *,
    max_tokens: int,
    timeout: int,
):
    """Make one structured-output LLM call and return ``(output, usage)``.

    Centralises the kwargs assembly (``apply_routing``), the instructor call and
    token-usage extraction shared by the gatekeeper, extractor and aggregator.
    Wrapped in :func:`retry_on_rate_limit`, so callers get the shared backoff for
    free. ``usage`` is ``{}`` when the backend reports none.
    """
    kwargs = apply_routing(dict(
        model=model,
        response_model=response_model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=1,
    ))
    output, completion = await client.chat.completions.create_with_completion(**kwargs)
    return output, extract_usage(completion)


async def complete_text_once(
    model: str,
    prompt: str | None = None,
    *,
    messages: list[dict] | None = None,
    system: str | None = None,
    max_tokens: int,
    temperature: float | None = None,
    response_format: dict | None = None,
    timeout: int | None = None,
) -> str:
    """Make one plain-text LLM call (no structured-output schema) and return the text.

    This is the single dispatch point for free-text / JSON-in-text completions
    across retro — keyword extraction, edge calibration, the /llm proxy. It
    routes through litellm to Bedrock (via :func:`apply_routing`). Pass either a
    ``prompt`` (optionally with ``system``) or a pre-built ``messages`` list.

    No retry: use this on latency-bounded paths (the /forecast keyword distill,
    the interactive /llm endpoint). For batch/offline callers that should ride
    out Bedrock throttling, use :func:`complete_text`.
    """
    if messages is None:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
    kwargs = apply_routing(dict(model=model, messages=messages, max_tokens=max_tokens))
    if temperature is not None:
        kwargs["temperature"] = temperature
    if response_format is not None:
        kwargs["response_format"] = response_format
    if timeout is not None:
        kwargs["timeout"] = timeout
    resp = await litellm.acompletion(**kwargs)
    return resp.choices[0].message.content


# Retrying variant for batch/offline callers (keyword scripts, calibration).
complete_text = retry_on_rate_limit(complete_text_once)
