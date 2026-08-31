# TruthMachine: Design Plan & Implementation Roadmap

> **Working name:** TruthMachine | **Status:** MVP Design v5 (Finalized) | **Last updated:** 2026-03-18

---

## 0. Product Vision

TruthMachine is a **B2B oracle SaaS** built on a retroactive media accuracy audit. It has two distinct outputs:

1. **Oracul API** — probability estimates for future events (e.g., `"Chances of Netanyahu winning elections: 0.76"`), sold via API to corporate clients, hedge funds, Polymarket traders, and government agencies.
2. **Media Reputation Rankings** — journalist and outlet accuracy scores (Brier Score, custom ELO variant). Free tier shows public leaderboard; paid tier gives raw data and full API access.

### The Factum Atlas
The core of the product is **The Factum Atlas** (Latin *Factum* = "fact/done"). It is a definitive map of historical reality against which all media claims are measured. It operates on the principle of **Outcome-Supervised Narrative Calibration (OSNC)** — waiting for reality to happen, then using it to "teach" the model who to trust.
---

## 1. Repository & Infrastructure

### Precise File Structure
The data is stored in a tiered, content-addressable filesystem to ensure auditability and cost-efficiency.

```
data/
  vault/
    articles/
      {sha256_hash}.json      # Raw article text + ingestion metadata
    extractions/
      {sha256_hash}_{eid}_{model_v}.json  # Structured LLM forensics
  atlas/
    {event_id}/
      {source_id}/
        entry_{timestamp}.json # Soft-link to vault + source-specific metadata
  events/
    {event_id}.json           # Individual event metadata (Outcome, Dates, Keywords)
  sources/
    {source_id}.json          # Individual source metadata (URL, Type, Language)
  pages/
    {event_id}.json           # Pre-processed data for the Next.js UI
  atlas_v1.db                 # SQLite performance layer (synced from JSON)
```

### Storage Strategy (Hybrid Architecture)
...

To ensure both institutional auditability and high-performance querying, we use a two-layer storage strategy.

**Layer 1: The Immutable Record (JSON Filesystem)**
- **Purpose:** Source of truth, Git-trackable, human-readable.
- **Structure:** `data/atlas/{event_id}/{source_id}/entry_{timestamp}.json`
- **Pointer System:** To minimize LLM costs for syndicated content (e.g., Reuters wires), we use a Content-Addressable "Vault":
  - `data/vault/articles/{article_hash}.json`: Stores raw text once.
  - `data/vault/extractions/{article_hash}_{event_id}_{model_v}.json`: Stores LLM output once.
  - `data/atlas/...`: Tiny "Soft Link" files pointing to the vault.

**Layer 2: The Performance Layer (SQLite)**
- **Purpose:** Powering the Next.js UI and the ML training loop.
- **Sync:** A background script (`src/tm/sync_atlas.py`) periodically scans the JSON files and populates an indexed SQLite DB (`atlas_v1.db`).

### Compute (AWS)
- **EC2 t4g.small** (2 vCPUs, 2 GiB RAM) — free-tier eligible through 2026.
- Single Postgres instance for Phase 2; SQLite is sufficient for MVP (~3,000 cells).

---

## 2. Source List

**25 sources total: 15 Israeli + 6 international + Polymarket (auxiliary data)**

### Media Handling
- **TV & Radio:** MVP strictly processes **written web articles**.
- **Cross-Platform Authors:** Identities are tracked (e.g., Amit Segal), but each publication (Ynet, Telegram, etc.) is stored as a separate data point to preserve the "voice" of the outlet.
- **Language:** Hebrew and English are treated as separate records (even if the content is similar) to capture linguistic nuances and differences in "hedging" across cultures.

---

## 3. Event Seed List

**125 concrete binary events across 9 domains.** See `EVENTS.md` for the full list.

**Event selection criteria (MVP):**
- **Objective & Binary:** Only events that clearly "happened" or "didn't happen."
- **Exclusions:** Subjective or vague events are pruned from the MVP to ensure mathematical calibration.

---

## 4. Data Model: The Forensic Vector

Each article in **The Factum Atlas** stores a full vector of metadata and extraction metrics. We do **not** discard low-signal or conflicting data.

### Fields per extraction (`extraction__{id}_v1.json`)

| Field | Type | Description |
|---|---|---|
| `stance` | float (-1.0 to 1.0) | Directional outlook (Bullish/Bearish) |
| `certainty` | float (0.0 to 1.0) | Linguistic confidence |
| `specificity` | float (0.0 to 1.0) | Concreteness/falsifiability |
| `hedge_ratio` | float (0.0 to 1.0) | Density of "might/could/possibly" |
| `source_authority` | float (0.0 to 1.0) | Personal opinion vs. named insider sources |
| `model_version` | string | e.g., `DeepSeek-V3.2-Prompt-v4` |
| `claim_english` | string | One-sentence summary |

### Versioning
Multiple extractions for the same article (different models or refined prompts) are all stored. This allows for backtesting different "referee" logic.

---

## 5. LLM Pipeline & Prompt Strategy

### Stage 1: The Gatekeeper
- **Model:** Nemotron 3 Nano.
- **Task:** Filter for forward-looking predictions.

### Stage 2: Forensic Extraction
- **Model:** DeepSeek V3.2.
- **Context:** Provided with `event_name` and `event_description` for targeted extraction.

### Logic Refinements
- **Search Strategy:** "One Event, One Query" (targeted keyword searches) for initial efficiency.
- **No Manual Overrides:** Referee feedback is handled by refining prompts and models, not by manual human editing, ensuring scalability.
- **No Scorable Threshold:** All predictions (even vague ones) are passed to the ML model; let the weights handle the signal-to-noise ratio.

---

## 9. Oracul Model

Generates calibrated probability estimates from the full forensic feature vector.

- **MVP Architecture:** **LightGBM** (gradient-boosted trees) — best-in-class for structured/tabular data at small-to-medium scale.
- **Feature vector per prediction:**
  - 11 forensic metrics: `stance`, `certainty`, `specificity`, `hedge_ratio`, `conditionality`, `magnitude`, `time_horizon_days`, `source_authority`, `prediction_type`, `contrarianism`, `sentiment`
  - Source historical Brier score per domain
  - Time-to-event (days between article and event resolution)
  - Topic/domain tags
  - Polymarket probability at publication time (where available)
  - Source type (wire, newspaper, blog, TV)
- **Calibration:** Isotonic regression applied post-training to ensure outputs are true probabilities, not just rankings.
- **Explainability:** SHAP values expose which features drove each forecast ("Bloomberg's energy track record contributed +0.12 to this estimate").
- **Training (OSNC):** Trained on historical "Prediction Vector → Outcome" pairs from the Factum Atlas.
- **Phase 2:** Attention-based model once 1,000+ resolved events are available — learns cross-source interaction patterns automatically.

---

## 11. Integration with Daatan Infrastructure

TruthMachine is designed to run within the broader **Daatan Infrastructure** (see `daatan/infra`).

### Deployment Layer
- **Hosting:** TruthMachine services run on a dedicated AWS EC2 instance in `eu-central-1` (see `ARCHITECTURE.md` → Deployment).
- **Primary consumer:** The **daatan** Next.js app calls the TruthMachine Oracle API (`oracle.daatan.com`) from its `context` and `express/guess` flows to surface calibrated forecasts to users.

### Inference Layer
- **Current:** All LLM calls go directly to **AWS Bedrock** (Nova Micro for gatekeeper, Nova Lite for extractor) via `litellm` with `instructor` for structured output. OpenRouter quota was exhausted in early testing; Bedrock is the live backend.
- **Models in use:** `bedrock/amazon.nova-micro-v1:0` (gatekeeper), `bedrock/amazon.nova-lite-v1:0` (extractor + aggregator)
- **Oracul API:** Calls the same pipeline modules directly — no separate proxy layer needed.

---

## 12. Open Problems & Future Roadmap

| # | Problem | Status |
|---|---|---|
| 1 | Ingestion Provider Selection | **PENDING** |
| 2 | Data Hashing for Audit Integrity | **TODO (Post-MVP)** |
| 3 | Automated Author Identity Merging | **TODO (Phase 2)** |
| 4 | Legal Review | **CRITICAL** |

---

*The Factum Atlas: The past is our data. The future is our product.*
