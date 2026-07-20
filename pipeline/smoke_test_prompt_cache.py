"""Bedrock prompt-caching smoke test.

NOT a CI test — it calls Bedrock directly and costs real (tiny) money. Run manually
before flipping `enable_prompt_cache` on for live traffic, and again after any change
to `llm.py`'s cached_prefix wiring:

    cd pipeline && AWS_REGION=us-east-1 .venv/bin/python smoke_test_prompt_cache.py

Per-model support for Bedrock/Anthropic prompt caching is NOT assumed — litellm's own
pricing table has empty cache-cost entries for the Nova models but populated ones for
Claude Haiku 4.5 on Bedrock (see docs/PROMPT_CACHING.md), so this script checks Nova
Micro, Nova Lite, and the live Haiku override explicitly rather than trusting any one
of them by analogy to the others.

For each model, this calls `complete_structured` TWICE with the identical
`cached_prefix`, back to back:
  - Call 1 is expected to be a cache WRITE: cache_creation_input_tokens > 0,
    cache_read_input_tokens == 0.
  - Call 2 (within the ~5 min ephemeral TTL) is expected to be a cache READ:
    cache_read_input_tokens > 0 (roughly the prefix's token count),
    cache_creation_input_tokens == 0.

A model/region combination that doesn't support caching typically surfaces as either
(a) a Bedrock ValidationException — caught and reported explicitly below, not confused
with "no savings" — or (b) both calls reporting all-zero cache_* fields with no error,
which this script also flags rather than silently treating as a pass.
"""
import asyncio

from tm.config import settings
from tm.llm import complete_structured
from tm.models import GatekeeperOutput

MODELS = [
    "bedrock/amazon.nova-micro-v1:0",
    "bedrock/amazon.nova-lite-v1:0",
    settings.extractor_model,  # whatever's live — the Haiku override on oracle-api, or the default
]

# ~1,200 words of filler, comfortably over every publicly documented Bedrock/Anthropic
# minimum cacheable-prefix size — this only needs to be long and IDENTICAL across both
# calls, not realistic prompt content.
CACHED_PREFIX = (
    "You are evaluating whether the following article is relevant to a forecasting "
    "claim. Judge strictly on evidence, not keyword overlap. " * 40
)


async def _probe(model: str) -> None:
    print(f"\n=== {model} ===")
    prior_flag = settings.enable_prompt_cache
    settings.enable_prompt_cache = True
    try:
        for call_num in (1, 2):
            try:
                _, usage = await complete_structured(
                    model, GatekeeperOutput, "Article: a routine local council meeting.",
                    max_tokens=50, timeout=30, cached_prefix=CACHED_PREFIX,
                )
            except Exception as exc:  # noqa: BLE001 — report, don't let one model's failure stop the rest
                print(f"  call {call_num}: EXCEPTION — {type(exc).__name__}: {exc}")
                return
            read = usage.get("cache_read_input_tokens", 0)
            write = usage.get("cache_creation_input_tokens", 0)
            print(f"  call {call_num}: cache_read_input_tokens={read} cache_creation_input_tokens={write}")
    finally:
        settings.enable_prompt_cache = prior_flag


async def main() -> None:
    for model in MODELS:
        await _probe(model)
    print(
        "\nExpected: call 1 shows write>0/read==0, call 2 (same run, within the TTL) "
        "shows read>0/write==0. If BOTH calls show read==0 and write==0 with no "
        "exception, that model/region is silently not caching — treat as unsupported, "
        "not as 'no savings yet'."
    )


if __name__ == "__main__":
    asyncio.run(main())
