# Bedrock prompt caching — gatekeeper/extractor cost reduction

## Why

July 2026 cost analysis found the article extractor (`pipeline/src/tm/extractor.py`)
responsible for the majority of the account's AWS bill, driven by Claude Haiku 4.5
running on the live `oracle-api` service. Roughly 87% of the extractor's per-call
input tokens are its fixed instructional/few-shot block (`PROMPT_PREFIX`, ~6,700
tokens) — identical text on every single call, system-wide, regardless of article or
prediction. That's exactly what Bedrock/Anthropic prompt caching exists for: mark a
repeated prefix once, and every subsequent call within the cache TTL bills a
cache-read rate (roughly 10% of normal input price) instead of full price for that
portion.

This is a pure cost lever — same model, same prompt content, same output. It does
NOT change what gets asked or answered.

## What changed

- `pipeline/src/tm/gatekeeper.py` and `pipeline/src/tm/extractor.py`: each module's
  single `PROMPT` constant was split into `PROMPT_PREFIX` (fixed instructions, no
  `.format()` placeholders, unchanged text) and `PROMPT_SUFFIX` (article + per-call
  variable fields + output-format spec, still `.format()`-ed exactly as before).
  `PROMPT_PREFIX + PROMPT_SUFFIX` is byte-identical to the old single `PROMPT` string
  — nothing was reworded or reordered.
- `pipeline/src/tm/llm.py::complete_structured` gained an optional `cached_prefix`
  param. When given AND `settings.enable_prompt_cache` is on, the message content
  becomes two blocks — the prefix marked `cache_control: {"type": "ephemeral"}`, then
  the per-call prompt. When the flag is off (the default) or no prefix is given,
  content is the same flat concatenated string as before caching existed at all —
  this fallback matters: it's what keeps every call correct even with the flag off.
- `pipeline/src/tm/config.py`: new `enable_prompt_cache: bool = False` kill-switch.

## What did NOT change

- `extractor.py`'s output-format spec + trailing few-shot examples (`extractor.py`'s
  former tail, ~500-600 tokens) still sit AFTER the per-call variable fields, so they
  aren't part of the cached prefix in this first pass. Moving them earlier would
  capture a bit more, but `tests/test_extractor_prompt.py` and prior incident notes
  document that Nova Lite is measurably sensitive to whitespace/ordering changes in
  this exact prompt — reordering needs its own regression pass (widen
  `eval_gatekeeper.py`'s pattern to the extractor) before it's worth the small extra
  win. Not done here.
- `aggregator.py`'s prompt is short with no large shared fixed block — left uncached.
- The batch pipeline (`orchestrator.py`/`runner.py`) uses the same `complete_structured`
  call path, so it gets caching for free once the flag is on — no separate change
  needed there.

## Rollout

1. `enable_prompt_cache` ships `False`. No behavior change on merge.
2. Run `pipeline/smoke_test_prompt_cache.py` manually (not CI — it calls Bedrock) against
   Nova Micro, Nova Lite, and whichever model `settings.extractor_model` resolves to
   live (the Haiku override on `oracle-api`, or the Nova Lite default on
   `truthmachine.service`). **Do not assume Nova caching works by analogy to Haiku** —
   litellm's own pricing table (`model_prices_and_context_window_backup.json`) has
   empty `cache_read_input_tokens`/`cache_creation_input_tokens` cost entries for the
   Nova models but populated ones for Haiku 4.5 on Bedrock, which is a concrete signal
   (not proof) that Nova caching may be unverified in this litellm version. Watch for
   a Bedrock `ValidationException` specifically — that's a hard "unsupported", not
   the same thing as "ran fine but showed zero savings."
3. Flip `enable_prompt_cache = True` only for models the smoke test confirms actually
   cache (call 1: `cache_creation_input_tokens > 0`; call 2, within ~5 min:
   `cache_read_input_tokens > 0`).
4. Measure via the now-logged `cache_read_input_tokens`/`cache_creation_input_tokens`
   fields on every call (immediate, unambiguous — don't wait on AWS billing lag), plus
   a controlled multi-day before/after Cost Explorer comparison on the
   `Claude Haiku 4.5 (Amazon Bedrock Edition)` service line, holding `Invocations`
   volume roughly constant.

Expected impact (directional — confirm against real data, don't treat as a promise):
with the ~6,700-token prefix at ~87% of the extractor's input tokens, and a cache-read
price around 10% of normal input price, this is roughly a 60% reduction in extractor
cost per call.
