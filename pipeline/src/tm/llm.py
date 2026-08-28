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
import logging
import re

import instructor
import litellm

from .config import settings

logger = logging.getLogger(__name__)

# Bedrock prompt caching (the Anthropic-style `cache_control: {"type": "ephemeral"}`
# content block) is only honoured by some model families. Sending it to one that
# doesn't support it isn't a silent no-op -- Bedrock hard-rejects the whole call
# ("You invoked an unsupported model or your request did not allow prompt
# caching."), which fails 100% of requests to that model (retro#650, found via
# Qwen3 32B: 0/50 gatekeeper calls succeeded until this check). Confirmed
# supporting the cache_control block on this account: the Anthropic (Claude) and
# Amazon Nova families. Everything else falls back to the uncached flat-string
# path -- cached_prefix is still concatenated in, so correctness is unaffected,
# only the cache-read cost saving is skipped. Allowlist, not denylist: an
# unrecognized model is assumed NOT to support it, so a new model family added
# to the fleet degrades to "full price" rather than to "every call fails".
_PROMPT_CACHE_MODEL_MARKERS = (".anthropic.", ".amazon.nova")


def _model_supports_prompt_cache(model: str) -> bool:
    return any(marker in model for marker in _PROMPT_CACHE_MODEL_MARKERS)

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

# retro#600: under memory pressure on the Oracle box, an async coroutine's
# deadline can expire while the event loop is stalled (GC pause, swap I/O) and
# then fire the instant the loop resumes — logging near-zero elapsed time
# despite a much longer real wall-clock stall. A genuine network timeout takes
# close to the full configured duration, so anything under this threshold is
# the stalled-loop false positive, not a real Bedrock failure.
_STALLED_TIMEOUT_RE = re.compile(r"time taken=([\d.]+) seconds")
_STALLED_TIMEOUT_THRESHOLD_S = 1.0


def is_stalled_event_loop_timeout(exc: Exception) -> bool:
    """True for the retro#600 signature: a ``litellm.Timeout`` firing in a
    fraction of a second despite a much longer configured timeout."""
    if not isinstance(exc, litellm.Timeout):
        return False
    match = _STALLED_TIMEOUT_RE.search(str(exc))
    return bool(match) and float(match.group(1)) < _STALLED_TIMEOUT_THRESHOLD_S


def retry_once_on_stalled_timeout(func):
    """Retry an async call exactly once, immediately, on a stalled-event-loop
    false-positive timeout (:func:`is_stalled_event_loop_timeout`). Any other
    exception, or a second stalled-timeout in a row, propagates — this is a
    workaround for one specific false-failure signature, not a general retry.
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as exc:
            if is_stalled_event_loop_timeout(exc):
                return await func(*args, **kwargs)
            raise
    return wrapper


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

    Also pulls ``cache_read_input_tokens``/``cache_creation_input_tokens`` when the
    backend reports them (Bedrock prompt caching) — 0 on any call that didn't use a
    cached prefix, or on a model/backend that doesn't support it. Returns ``{}`` when
    the completion carries no usage data at all.
    """
    if completion and hasattr(completion, "usage") and completion.usage:
        u = completion.usage
        return {
            "prompt_tokens": getattr(u, "prompt_tokens", 0),
            "completion_tokens": getattr(u, "completion_tokens", 0),
            "total_tokens": getattr(u, "total_tokens", 0),
            "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
            "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0,
        }
    return {}


def retry_on_rate_limit(func):
    """Retry an async call on transient rate-limit / throttle errors, and on the
    retro#600 stalled-event-loop false-positive timeout (:func:`is_stalled_event_loop_timeout`).

    Tries once, then waits ``RATE_LIMIT_BACKOFF`` seconds between further
    attempts. All other exceptions propagate immediately; if every attempt
    is rate-limited (or stalled-timeout) the last such exception is raised.
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
                if is_rate_limit_error(exc) or is_stalled_event_loop_timeout(exc):
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
    cached_prefix: str | None = None,
    temperature: float | None = 0,
):
    """Make one structured-output LLM call and return ``(output, usage)``.

    Centralises the kwargs assembly (``apply_routing``), the instructor call and
    token-usage extraction shared by the gatekeeper, extractor and aggregator.
    Wrapped in :func:`retry_on_rate_limit`, so callers get the shared backoff for
    free. ``usage`` is ``{}`` when the backend reports none.

    ``temperature`` defaults to 0: every current caller is a classification/
    extraction/aggregation task where the same input should reliably produce the
    same output, not a creative-generation one. Pass a higher value explicitly if
    a future caller needs otherwise. Pass ``None`` to omit the parameter from the
    request entirely — the newest Anthropic models on Bedrock reject it outright
    (``BedrockException: `temperature` is deprecated for this model``), which
    fails 100% of calls to them, so a caller reaching for one of those has no way
    to say "leave it off" other than this.

    ``cached_prefix``, when given and ``settings.enable_prompt_cache`` is on, is sent
    as its own Bedrock/Anthropic cache-marked content block ahead of ``prompt`` — for
    text that's identical on every call (e.g. the extractor's fixed instructions),
    so Bedrock bills a cache-read rate instead of full price on every repeat. When
    the flag is off, no prefix is given, or ``model`` isn't a family confirmed to
    support Bedrock prompt caching (retro#650 — sending the cache block to one that
    doesn't hard-fails the whole call), ``content`` is the same flat string as
    before caching existed — behaviour is unchanged.
    """
    if cached_prefix and settings.enable_prompt_cache and _model_supports_prompt_cache(model):
        content = [
            {"type": "text", "text": cached_prefix, "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": prompt},
        ]
    else:
        if cached_prefix and settings.enable_prompt_cache:
            logger.info("event=prompt_cache_skipped_unsupported_model model=%s", model)
        # Caching off (default) or no prefix given: fall back to one flat string —
        # byte-identical to the single PROMPT this used to be before the
        # PROMPT_PREFIX/PROMPT_SUFFIX split. Dropping cached_prefix here instead of
        # concatenating it would silently ship every gatekeeper/extractor call
        # without its fixed instructions — this branch must never do that.
        content = (cached_prefix or "") + prompt
    call_kwargs = dict(
        model=model,
        response_model=response_model,
        messages=[{"role": "user", "content": content}],
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=1,
    )
    if temperature is not None:
        call_kwargs["temperature"] = temperature
    kwargs = apply_routing(call_kwargs)
    output, completion = await client.chat.completions.create_with_completion(**kwargs)
    return output, extract_usage(completion)


@retry_once_on_stalled_timeout
async def complete_text_once_with_usage(
    model: str,
    prompt: str | None = None,
    *,
    messages: list[dict] | None = None,
    system: str | None = None,
    max_tokens: int,
    temperature: float | None = None,
    response_format: dict | None = None,
    timeout: int | None = None,
) -> tuple[str, dict]:
    """Make one plain-text LLM call and return ``(text, usage)``.

    Same call as :func:`complete_text_once`, but also returns the token usage
    (:func:`extract_usage`; ``{}`` when the backend reports none) — for callers
    that surface spend, e.g. the Oracle API's ``token_usage`` response field
    (docs#57 item 3).
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
    return resp.choices[0].message.content, extract_usage(resp)


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
    Callers that need the token usage too use
    :func:`complete_text_once_with_usage`, which this delegates to.

    No rate-limit retry: use this on latency-bounded paths (the /forecast
    keyword distill, the interactive /llm endpoint). For batch/offline callers
    that should ride out Bedrock throttling, use :func:`complete_text`.
    ``complete_text_once_with_usage`` still gets one immediate retry on the
    retro#600 stalled-event-loop false-positive timeout — that costs no
    meaningful latency and isn't a "wait and hope" retry.
    """
    text, _usage = await complete_text_once_with_usage(
        model,
        prompt,
        messages=messages,
        system=system,
        max_tokens=max_tokens,
        temperature=temperature,
        response_format=response_format,
        timeout=timeout,
    )
    return text


# Retrying variant for batch/offline callers (keyword scripts, calibration).
complete_text = retry_on_rate_limit(complete_text_once)
