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
- `pipeline/src/tm/config.py`: new `enable_prompt_cache` kill-switch, shipped `False`
  then flipped to `True` once the smoke test (below) confirmed it live.

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

## Rollout — DONE, verified against live Bedrock

`enable_prompt_cache` shipped `False`, then was flipped to `True` after
`pipeline/smoke_test_prompt_cache.py` and additional targeted runs against live
Bedrock (not mocked) confirmed real cache accounting using each caller's ACTUAL
production prompt/schema, not a synthetic stand-in:

| Model | Role | Result |
|---|---|---|
| `bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0` | live `oracle-api` extractor | 4/4 clean runs. Call 1: `cache_creation_input_tokens=9394`. Calls 2-4: `cache_read_input_tokens=9394`, `write=0`. **9,394 of 10,135 total input tokens (~93%) landed in the cached prefix.** |
| `bedrock/amazon.nova-micro-v1:0` | gatekeeper (both `oracle-api` and `truthmachine.service`) | 4/4 clean runs with the real gatekeeper prompt. Call 1 write=1182, calls 2-4 read=1182 (~90% of that call's tokens). |
| `bedrock/amazon.nova-lite-v1:0` | batch `truthmachine.service` extractor default | Caching itself works when a call succeeds (`cache_read_input_tokens` populated correctly) — but see the finding below, which is unrelated to caching. |

**A real, pre-existing bug surfaced while verifying this, filed separately —
[retro#306](https://github.com/Daatan/retro/issues/306):** Nova Lite, under
`instructor.Mode.MD_JSON`, intermittently (~60% of attempts in a 5-run local sample)
wraps its JSON response in a spurious `{"properties": {...}}` envelope instead of the
flat schema shape, failing pydantic validation. **Confirmed this reproduces with
`enable_prompt_cache=False` too** — it is not caused by, or related to, this change.
Left out of scope here; filed for separate investigation since it's a real
production-reliability question for the batch pipeline's extractor calls.

litellm's own pricing table (`model_prices_and_context_window_backup.json`) has empty
`cache_read_input_tokens`/`cache_creation_input_tokens` cost entries for the Nova
models but populated ones for Haiku — this turned out to just mean litellm can't
*price* Nova's cache tokens, not that Nova doesn't support caching mechanically; the
raw usage accounting works fine for it regardless.

**Debugging note for future smoke-test runs:** an early version of
`smoke_test_prompt_cache.py` used `max_tokens=50`, which is too small for
`GatekeeperOutput`'s schema in MD_JSON mode (reasoning text + JSON needs more room) —
the resulting truncated generation surfaced as an `instructor.InstructorRetryException`
wrapping a `litellm.Timeout` with an implausible `time taken=0.001 seconds`, which
looks exactly like "this model/region doesn't support the cache breakpoint" but isn't.
Fixed to `max_tokens=200` in the committed script. If a future run of this script
shows that same instant-timeout shape, suspect `max_tokens` before suspecting caching
support.

## Measuring ongoing savings

Via the now-logged `cache_read_input_tokens`/`cache_creation_input_tokens` fields on
every call (immediate, unambiguous — don't wait on AWS billing lag), plus a controlled
multi-day before/after Cost Explorer comparison on the
`Claude Haiku 4.5 (Amazon Bedrock Edition)` service line, holding `Invocations` volume
roughly constant.

Expected impact (directional estimate ahead of a real multi-day comparison): with
~93% of the extractor's input tokens now cached at a read price roughly 10% of normal
input price, this should land close to a 60%+ reduction in extractor cost per call.
