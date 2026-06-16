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
