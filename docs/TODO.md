# TruthMachine TODO

- **Grounded search + LLM-call unification** (future, not started) — evaluate
  Gemini grounding and/or the Google Custom Search JSON API as search backends,
  and converge on one way to call search and each LLM across retro + daatan.
  Design note: [docs/ROADMAP_grounded_search_and_llm_unification.md](ROADMAP_grounded_search_and_llm_unification.md).

- **Retrieval relevance defect** (confirmed 2026-06-16, not started) — a meaningful
  fraction of live forecasts are backed by off-topic sources, on **both** the
  Oracle and LLM-fallback paths. Worst confirmed cases (latest snapshot, prod):
  - "Houthis lose territory" (fallback, 15 sources) → Kenya oil refinery, a Bronx
    stadium demolition, Palestinian expulsion — none about Houthis.
  - "EU admits ≥2 new members" (oracle, 15 sources) → "Proud of India", an Indian
    Union Minister, a deportation deal — matched on the word "Union/EU".
  - "Russian budget deficit" (fallback) → a Hebrew Turkey/Israel economy ranking.
  - "Deni Avdija → All-NBA" (fallback) → Rising Stars coverage of *other* players.
  - "Esperanto Wikipedia > 400k articles" (fallback) → Esperanto the language, an
    obituary, hardware IP — nothing about article counts.

  The high-source-count failures (15 sources, ~0 on-topic) point to keyword
  matching with no semantic relevance gate. Oracle-path failures implicate the
  news-indexer pgvector search (or extraction reuse across unrelated events);
  fallback-path failures implicate the LLM web-search query construction. Fix:
  add a relevance threshold between retrieval and aggregation.

  **Status (2026-06-17): largely addressed for the Oracle path** by the graded
  gatekeeper `relevance_score` + convex (relevance²) down-weight + the
  `all_articles_off_topic` floor (#208) and the full-text gatekeeper window
  (#209). The *off-subject* variant (Bronx stadium in a Houthi forecast) is
  handled. The *on-subject / off-predicate* variant is not — see next item.

- **Subject-relevant but claim-irrelevant sources → Oracle emits a confident
  non-answer instead of deferring** (confirmed 2026-06-17, not started). The
  relevance gate scores topic relevance to the question's *subject*, so news
  about the main actor passes even when it says nothing about the specific
  *predicate*. The Oracle then pools low-certainty, near-cancelling stances into
  a confident-looking ~50% and **`analyze` (context-update) overwrites a
  well-calibrated base-rate estimate with that coin-flip.**

  Live example (prod): *"Elon Musk will tweet about Daatan by Dec 31, 2028."*
  - At creation (no source) → LLM base-rate **5%** (near the 2% human forecast). Correct.
  - After `analyze` → Oracle **52%** from 2 articles: CoinDesk (Musk/**bitcoin**,
    stance +0.22, certainty 0.3) and CNBC (Musk/**Larry Page**, stance −0.08,
    certainty 0.5) — neither about Daatan or a tweet. Pooled mean 0.032, CI
    [41, 62]. The good 5% was overwritten by a meaningless 52%.

  Fix sketch (layered, cheapest first):
  1. **Decisiveness floor (do first).** Treat a non-decisive pool as
     `insufficient_data` — e.g. when the summed certainty-weighted evidence mass
     (`Σ credibility·certainty·recency·relevance²`) is below a config floor
     (analog of #208's `relevance_weight_floor`, on the certainty/evidence axis).
     That fires the existing LLM fallback, so the base rate is kept. A genuine
     balanced ~50% has high evidence mass and won't trip — only thin/low-certainty
     pools do.
  2. **Predicate-aware relevance (deeper).** Sharpen the gatekeeper to score
     relevance to the specific outcome in the question, not just the main actor
     (a bitcoin article should score ~0.1 for "will Musk tweet about Daatan").
     Then #208's relevance floor catches it. Needs prompt tuning + validation.
  3. *(Optional, daatan-side)* don't let `analyze` overwrite a higher-confidence
     prior with a low-confidence Oracle result — made unnecessary by (1).

  Net: "LLM base-rate when the Oracle has no on-claim signal, and ignore
  subject-only sources" — but enforced by the Oracle *deferring*, not by daatan.
