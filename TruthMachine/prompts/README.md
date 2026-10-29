# TruthMachine: Prompt Library

> **Last updated:** 2026-03-17

> ⚠️ **Reference / design library — NOT the prompts the live pipeline runs.**
> The production pipeline uses **inline** prompt constants in `pipeline/src/tm/`
> against Bedrock models (Nova Micro gatekeeper, Claude Haiku extractor/aggregator). The files here (and the models/stages and
> `registry.json` below) describe the original OSNC design and are kept for
> reference; several stages were never wired into the running code. The running
> code is authoritative — see the mapping table.

Each prompt corresponds to a pipeline stage. They are designed to be composed sequentially — the output of each stage feeds into the next.

## Reference prompt → live code

| Reference file | Live counterpart | Live model | Notes |
|---|---|---|---|
| `01_gatekeeper.md` | `tm/gatekeeper.py` (`PROMPT_PREFIX` / `PROMPT_SUFFIX`) | `bedrock/us.amazon.nova-micro-v1:0` | Live prompt is a **topic-relevance** filter (softened from "is_prediction" in PR #47); reason-first schema since retro#1024 |
| `02_forensic_extraction.md` | `tm/extractor.py` (`PROMPT_PREFIX` / `PROMPT_SUFFIX`) | `bedrock/us.anthropic.claude-haiku-5-5` (adaptive thinking) | The live schema has grown well past the original 5 fields — `ExtractionOutput` in `tm/models.py` and `docs/PROMPT_VERSIONS.md` are authoritative |
| `02b_article_aggregator.md` | `tm/aggregator.py` (`AGGREGATOR_PROMPT`) | `aggregator_model`, Claude Haiku 4.5 | Article-level collapse of high-spread predictions |
| `03_consensus_meter.md` | — | — | **Reference only** — no live call-site |
| `03_ground_truth.md` | — | — | **Reference only** — outcomes are set by hand in `data/events/*.json` |
| `04_event_matching.md` | — | — | **Reference only** — live pipeline runs per (event, source) cell, no LLM matching |
| `05_contrarianism.md` | — | — | **Reference only** |
| `06_page_generation.md` | `tm/generate_pages.py` (no LLM) | — | Live page generation is deterministic JSON, not an LLM prompt |

> **`registry.json` is stale**: it lists `gemini`/`deepseek` models the live
> pipeline does not use (it uses Bedrock, per `pipeline/src/tm/config.py`).
> The model/cost columns in the tables below are likewise aspirational.

---

## Pipeline Flow

```
Article
  │
  ▼
01_gatekeeper        → is_prediction? (cheap model)
  │ yes
  ▼
02_forensic_extraction → extract all predictions + metrics (DeepSeek)
  │
  ▼
04_event_matching    → match each prediction to seed event(s) (cheap model)
  │
  ▼
05_contrarianism     → compute deviation from consensus (after batch collected)
  │
  ▼
[scoring]            → Brier Score + ELO update (computed, not LLM)

Ground truth (run once per event, not per article):
03_ground_truth      → binary outcome determination (DeepSeek / GPT-4o)

Page generation (run once per event, after all predictions scored):
06_page_generation   → human-readable retro page (DeepSeek / Claude Sonnet)
```

---

## Prompt Files

| File | Stage | Model | Cost |
|---|---|---|---|
| `01_gatekeeper.md` | Filter — is this a prediction? | Nemotron 3 Nano | ~free |
| `02_forensic_extraction.md` | Extract all predictions + metrics | DeepSeek V3.2 | ~$0.25/1M |
| `03_ground_truth.md` | Determine binary event outcome | DeepSeek V3.2 / GPT-4o | ~$0.25–$5/1M |
| `04_event_matching.md` | Match prediction to seed event | Nemotron / DeepSeek | ~free–$0.25/1M |
| `05_contrarianism.md` | Score deviation from consensus | Nemotron 3 Nano | ~free |
| `06_page_generation.md` | Generate per-event analysis page | DeepSeek / Claude Sonnet | ~$0.25–$3/1M |

---

## Key Metrics Extracted (Stage 2)

| Metric | Range | Description |
|---|---|---|
| `stance` | -1.0 to 1.0 | How strongly the prediction implies the event will occur |
| `certainty` | 0.0 to 1.0 | Linguistic confidence: 0 = heavily hedged, 1 = absolute |
| `claim` | string | One-sentence English summary |
| `quote` | string | Exact sentence(s) from article (original language) |
| `settled` | bool | True when the source reports the outcome as an accomplished fact, not a prediction |

> **Note (PR #102):** Nine additional metrics (`sentiment`, `specificity`, `hedge_ratio`,
> `conditionality`, `magnitude`, `time_horizon`, `time_horizon_days`, `prediction_type`,
> `source_authority`) were dropped from the extractor prompt to cut latency (~5× reduction in
> generation budget). They remain as Optional fields in older atlas entries.
