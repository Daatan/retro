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
import json
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

# Bedrock rejects the `temperature` parameter outright on the newest Anthropic
# models ("`temperature` is deprecated for this model" -- a ValidationException /
# BadRequestError), which fails 100% of calls to them. Every Oracul stage sends
# temperature=0 by default, so a `model` override (or a default bump) onto one of
# these used to show up as a silent abstain, not an error (retro#916: 10/10
# second opinions abstained on Sonnet 5.5). Probed directly with Converse on
# 2026-10-11 (Haiku 5.5: 2026-10-12, retro#921) -- rejecting: Sonnet 5 / 5.5,
# Opus 4.7 / 4.8 / 5 / 5.5, Haiku 5.5; accepting:
# Sonnet 4.5 / 4.6, Opus 4.5 / 4.6, Haiku 4.5, Nova. Substring markers so the
# `us.` / `eu.` / `global.` profile prefixes all match. This list is evidence,
# not a guess -- extend it only from a probe of the new model.
_TEMPERATURE_REJECTING_MARKERS = (
    "claude-sonnet-5",
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-opus-5",
    "claude-haiku-5",
)


def _model_rejects_temperature(model: str) -> bool:
    return any(marker in model for marker in _TEMPERATURE_REJECTING_MARKERS)


# litellm >= 1.108 marks Gemini 3.5+ as not supporting `temperature` in its own model
# map and raises UnsupportedParamsError client-side, before any request is sent. Vertex
# itself accepts the parameter: probed 2026-11-20 on litellm 1.109.1, gemini-3.5-flash-lite
# answered and the request's generationConfig carried temperature 0. The bump made every
# /relevance call 502 for ~41 h (retro#1090). These are not rejecting models (the list
# above). The parameter is allowed explicitly, so temperature=0 still reaches the provider.
_TEMPERATURE_ALLOWLISTED_MARKERS = ("gemini",)


def _allow_param(kwargs: dict, param: str) -> None:
    """Add ``param`` to litellm's ``allowed_openai_params`` without dropping existing entries."""
    allowed = list(kwargs.get("allowed_openai_params") or [])
    if param not in allowed:
        allowed.append(param)
    kwargs["allowed_openai_params"] = allowed


def _set_temperature(kwargs: dict, model: str, temperature: float | None) -> None:
    """Put ``temperature`` on ``kwargs`` unless it is None or ``model`` rejects it."""
    if temperature is None:
        return
    if _model_rejects_temperature(model):
        logger.debug("event=temperature_omitted_unsupported_model model=%s", model)
        return
    kwargs["temperature"] = temperature
    if any(marker in model for marker in _TEMPERATURE_ALLOWLISTED_MARKERS):
        _allow_param(kwargs, "temperature")

# litellm reads this global for OpenAI-compatible backends. For ``bedrock/*``
# models (the default) litellm ignores it and authenticates with AWS
# credentials, so on the normal path this is a no-op; it only matters when
# ``model_api_base`` points litellm at an OpenAI-compatible server (e.g. Ollama).
litellm.api_key = settings.openrouter_api_key

# Shared instructor-wrapped client. MD_JSON mode coaxes structured JSON out of
# models without native tool-calling (Nova).
client = instructor.from_litellm(litellm.acompletion, mode=instructor.Mode.MD_JSON)


def rendered_response_schema(response_model) -> str:
    """The response model's JSON schema **exactly as MD_JSON puts it in the prompt**.

    This is prompt text, not plumbing (retro#700). Under ``Mode.MD_JSON`` instructor
    serialises ``response_model.model_json_schema()`` into a system message — so every
    ``Field(description=...)``, every enum member, and every model docstring (Pydantic
    copies those into the schema's ``description``) is input the model reads, on every
    call, alongside the hand-written PROMPT_PREFIX/PROMPT_SUFFIX.

    It is reproduced here rather than approximated, because the two plausible
    approximations both lie:

    * ``sort_keys=True`` hides field **reordering** — Pydantic emits ``properties`` in
      declaration order, so moving a field changes what the model sees while leaving a
      sorted hash identical.
    * the compact form under-counts the cost by ~43% (ExtractionOutput: 14,209 chars
      compact vs 20,287 as actually rendered), which matters because the number this
      feeds is a size ratchet.

    Coupled to instructor's internals on purpose, and pinned by
    ``pipeline/tests/test_rendered_schema.py`` against the message instructor really
    builds. If an instructor upgrade changes this serialisation the pin fails — which is
    correct, because the model's input changed and the prompt hash should move with it.
    """
    return json.dumps(
        response_model.model_json_schema(), indent=2, ensure_ascii=False,
    )

# Retry schedule (seconds) for transient rate-limit / throttling errors.
RATE_LIMIT_BACKOFF = [30, 60, 120]

# retro#600: under memory pressure on the Oracul box, an async coroutine's
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


# An output truncated at max_tokens is deterministic — the same prompt overflows the same
# cap every time — so retrying it is pure cost. instructor phrases it "output is incomplete
# due to a max_tokens length limit", and "limit" alone used to read as a throttle: four
# attempts, 210 s of sleep and ~8 model calls per truncated article (retro#803/#805 A/B,
# 2026-09-07 — a live-path article in that state times out instead of failing fast).
_NOT_A_RATE_LIMIT = ("max_tokens", "length limit")


def is_rate_limit_error(exc: Exception) -> bool:
    """True if the exception looks like a transient rate-limit / throttle worth retrying."""
    err = str(exc).lower()
    if any(marker in err for marker in _NOT_A_RATE_LIMIT):
        return False
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


# retro#1098 — Vertex AI Flex PayGo: the same model at half the per-token price, answered from
# shared capacity (~7 s instead of ~1 s on 3.5 Flash-Lite, measured 2026-11-22). Selected per
# request by these two headers; the response's `usageMetadata.trafficType` says which tier was
# actually billed (`ON_DEMAND_FLEX` vs `ON_DEMAND`), and litellm copies it into hidden params.
VERTEX_FLEX_HEADERS = {
    "X-Vertex-AI-LLM-Shared-Request-Type": "flex",
    "X-Vertex-AI-LLM-Request-Type": "shared",
}
FLEX_TRAFFIC_TYPE = "ON_DEMAND_FLEX"
# One aggregate line per this many Flex attempts, so the Flex share is measurable from the log
# without a line per call (/relevance runs ~30k calls/day).
_FLEX_STATS_EVERY = 500


def _model_supports_flex(model: str) -> bool:
    return model.startswith("vertex_ai/")


def traffic_type(completion) -> str | None:
    """The Vertex billing tier litellm copied off ``usageMetadata.trafficType``, or None."""
    hidden = getattr(completion, "_hidden_params", None) or {}
    fields = hidden.get("provider_specific_fields") or {}
    value = fields.get("traffic_type")
    return str(value) if value else None


class _FlexStats:
    """Cumulative Flex outcomes since process start; logs one line every `_FLEX_STATS_EVERY`."""

    def __init__(self) -> None:
        self.attempts = self.flex = self.not_flex = self.fallback = 0
        self.unsupported_logged: set[str] = set()

    def record(self, outcome: str) -> None:
        self.attempts += 1
        setattr(self, outcome, getattr(self, outcome) + 1)
        if self.attempts % _FLEX_STATS_EVERY == 0:
            logger.info(
                "event=flex_stats attempts=%d flex=%d not_flex=%d fallback=%d",
                self.attempts, self.flex, self.not_flex, self.fallback,
            )


_flex_stats = _FlexStats()


async def _create_with_flex(kwargs: dict, model: str, flex_timeout: int | None):
    """One Flex attempt, then — on ANY failure of it — one standard call with the same kwargs.

    Any failure, not a 429 string match: instructor re-raises provider errors wrapped in its own
    retry exception, and whether a Flex capacity shortage arrives as 429 or 503 is not something
    we have observed. A Flex failure must cost money, never a verdict. The fallback sits here,
    INSIDE `complete_structured`, so a Flex 429 never reaches `retry_on_rate_limit`'s 30/60/120 s
    backoff; only a failure of the standard call does, as before.
    """
    flex_kwargs = {**kwargs, "extra_headers": dict(VERTEX_FLEX_HEADERS)}
    if flex_timeout:
        flex_kwargs["timeout"] = flex_timeout
    try:
        output, completion = await client.chat.completions.create_with_completion(**flex_kwargs)
    except Exception as exc:  # noqa: BLE001 - any Flex failure falls back to standard, logged
        _flex_stats.record("fallback")
        logger.warning(
            "event=flex_fallback model=%s err_type=%s err=%.200s", model, type(exc).__name__, exc,
        )
        return await client.chat.completions.create_with_completion(**kwargs)
    tier = traffic_type(completion)
    if tier == FLEX_TRAFFIC_TYPE:
        _flex_stats.record("flex")
    else:
        # The headers were sent but the call was billed (or reported) as something else — the
        # cost saving is not happening. Loud, because nothing else would show it.
        _flex_stats.record("not_flex")
        logger.warning("event=flex_not_honoured model=%s traffic_type=%s", model, tier)
    return output, completion


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
    thinking: bool = False,
    flex: bool = False,
    flex_timeout: int | None = None,
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
    fails 100% of calls to them. Models known to reject it
    (:data:`_TEMPERATURE_REJECTING_MARKERS`) get it omitted automatically whatever
    the caller passes; ``None`` is for a model not on that list.

    ``cached_prefix``, when given and ``settings.enable_prompt_cache`` is on, is sent
    as its own Bedrock/Anthropic cache-marked content block ahead of ``prompt`` — for
    text that's identical on every call (e.g. the extractor's fixed instructions),
    so Bedrock bills a cache-read rate instead of full price on every repeat. When
    the flag is off, no prefix is given, or ``model`` isn't a family confirmed to
    support Bedrock prompt caching (retro#650 — sending the cache block to one that
    doesn't hard-fails the whole call), ``content`` is the same flat string as
    before caching existed — behaviour is unchanged.

    ``thinking`` (Haiku 5.x only; ignored for other models) lets the model reason before it
    answers. Off by default because adaptive thinking spends ``max_tokens`` and a small cap
    returns an empty body, so only a caller that also raises its cap opts in. The extractor
    does (retro#921): without reasoning, 5.5 scores a negated question ("X will NOT happen")
    as if it were the inner event.

    ``flex`` (retro#1098) sends the call to Vertex AI Flex PayGo — half price, slower — with
    ``flex_timeout`` as that attempt's timeout, and falls back to one standard call on any Flex
    failure (:func:`_create_with_flex`). Only ``vertex_ai/`` models have a Flex tier; for any
    other model the flag is ignored (logged once per model) and the call is the standard one.
    Off by default, so every existing caller's request is unchanged.
    """
    if cached_prefix and settings.enable_prompt_cache and _model_supports_prompt_cache(model):
        cache_control = {"type": "ephemeral"}
        if settings.prompt_cache_ttl:
            cache_control["ttl"] = settings.prompt_cache_ttl
        content = [
            {"type": "text", "text": cached_prefix, "cache_control": cache_control},
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
    _set_temperature(call_kwargs, model, temperature)
    if "claude-haiku-5" in model and not thinking:
        # Haiku 5.5 thinks adaptively by default; on a long structured prompt the
        # thinking block ate max_tokens and returned an empty body (retro#921). A caller
        # with a cap sized for it passes thinking=True and the key is simply omitted.
        call_kwargs["thinking"] = {"type": "disabled"}
        # litellm validates `thinking` against its model cost map; when the remote map
        # can't be fetched it falls back to a bundled copy that predates Haiku 5.5 and
        # raises UnsupportedParamsError (seen live 2026-10-12). Allow it explicitly.
        _allow_param(call_kwargs, "thinking")
    kwargs = apply_routing(call_kwargs)
    if flex and _model_supports_flex(model):
        output, completion = await _create_with_flex(kwargs, model, flex_timeout)
        return output, extract_usage(completion)
    if flex and model not in _flex_stats.unsupported_logged:
        _flex_stats.unsupported_logged.add(model)
        logger.info("event=flex_skipped_unsupported_model model=%s", model)
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
    that surface spend, e.g. the Oracul API's ``token_usage`` response field
    (docs#57 item 3).
    """
    if messages is None:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
    kwargs = apply_routing(dict(model=model, messages=messages, max_tokens=max_tokens))
    _set_temperature(kwargs, model, temperature)
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
