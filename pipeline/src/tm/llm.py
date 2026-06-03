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
