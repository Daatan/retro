# Roadmap: grounded search + LLM-call unification

**Status:** proposed / not implemented. This is a design note to keep in mind, not a task to start.

Two related future directions, plus the cross-cutting rule they must both respect.

1. **Grounded search via Gemini** — let a grounding LLM (Gemini with the built-in
   Google Search tool) answer/research a question directly, instead of (or before)
   the scrape-and-extract pipeline.
2. **Google Search JSON API** — add Google's Programmable Search / Custom Search
   JSON API (`customsearch/v1`) as a search provider.
3. **Unify API calls across projects** — wherever it makes sense, every project
   (retro, daatan, ibi, duel, …) should reach search and each LLM the *same* way.

---

## Current state (so whoever implements this starts from facts)

`ibi` and `duel` are subsystems, not separate repos: **ibi** = the daatan IBI
endpoints (`/api/ibi/{llm,search,fetch-url}`) which proxy to the Oracle API;
**duel** = the retro backtest/scoring harness (`tm.duel_report`, `duel.html`).
Everything lives in the two repos **retro** (Python: Oracle API + pipeline) and
**daatan** (TypeScript/Next.js: product).

### Search — already a single chokepoint ✅
There is one logical search path, and it is good:

```
daatan /api/ibi/search ──▶ Oracle POST /search ──▶ tm.web_search.search_articles()
   (oracleSearch.ts,           (api/searcher.py)        multi-provider fallback chain:
    oracleClient.ts)                                    GDELT → SerpAPI → Serper → Brave
                                                        → Tavily → BrightData → Nimbleway
                                                        → ScrapingBee → Newsdata
                                                        → DataForSEO → DuckDuckGo
```

- The retro forecast pipeline and the batch ingestors (`gdelt_ingest`,
  `gnews_ingest`, `site_search`, `web_search_ingest`) also go through
  `tm.web_search` (or sibling site scrapers).
- Article-body extraction is consolidated in `tm.article_text.extract_article_body`
  (+ the trafilatura path in `forecaster._fetch_article_text`).

**Implication:** any new search backend belongs **inside `tm.web_search`'s chain**,
behind the existing `/search` contract. Add it once there and daatan / ibi / duel /
pipeline all benefit with zero per-project change.

### LLM dispatch — fragmented ⚠️ (this is what unification targets)
Three different ways to call a model today, and Gemini is already reached two of them:

| # | Where | Mechanism | Models |
|---|-------|-----------|--------|
| A | daatan `ResilientLLMService` (`src/lib/llm/`) | provider chain via `LLMProvider` interface | **Gemini** (`@google/generative-ai`, `gemini-2.5-flash`) → Ollama → OpenRouter |
| B | retro `tm.llm` | litellm + instructor | Bedrock Nova (`bedrock/amazon.nova-{micro,lite}`) — gatekeeper / extractor / aggregator / forecaster keyword-distill |
| C | retro raw `httpx` → OpenRouter | direct REST | `/llm` IBI proxy (pass-through), `calibrate_edges.py` (`anthropic/claude-haiku-4.5`), `improve_keywords.py` (`openrouter/google/gemini-2.0-flash-001`) |

So **Gemini is already integrated in daatan** (native SDK, no grounding tool
enabled yet) and reached *separately* from retro (via OpenRouter). Grounding does
**not** exist anywhere yet (no `google_search` tool / `groundingMetadata` usage).

---

## Option 1 — Gemini grounding (Google Search built-in)

Enable Gemini's `google_search` grounding tool so a single call returns an answer
plus `groundingMetadata` citations, skipping our scrape→extract pipeline.

- **daatan:** `GeminiProvider` already exists; grounding is essentially a tool flag
  on the `getGenerativeModel(...)` call + reading `groundingMetadata`. Small change.
- **retro:** no native Gemini path — would need either a litellm grounding call or
  the `google-genai` SDK.

**Tension to resolve before adopting:** the credibility engine (the whole point of
TruthMachine — per-source stance/certainty → leaderboard → weighted aggregation)
needs **full article text per source** to run gatekeeper + extractor. Gemini
grounding returns *citations (URLs + snippets)*, not the forensic inputs. So
grounding likely **augments** (fast first-pass answer, or a search-provider that
returns the grounded citation URLs) rather than **replaces** the pipeline — unless
we accept losing per-source credibility weighting for grounded answers.

## Option 2 — Google Custom Search JSON API  ✅ implemented (inert pending credentials)

`customsearch/v1` with an API key + a Programmable Search Engine `cx` id. Returns
title/link/snippet like the other providers.

- **Drop-in** as another entry in `tm.web_search`'s fallback chain — the
  lowest-friction option, fits the existing architecture exactly.
- Caveats: 100 free queries/day, then paid; 10k/day hard cap; general-web (the `cx`
  can be scoped to news sites).

**Status:** the provider is built — `_search_google_cse` in `pipeline/src/tm/web_search.py`,
slotted at position **1c** (right after GDELT Doc, before SerpAPI, so it's the primary
keyed provider once enabled), with a `/search/health` entry. It is **inert** until both
credentials are set, so it ships with zero behavior change.

**To activate (once Google startup credits land):**
1. Create a Programmable Search Engine; set it to "search the entire web" (or scope to
   news sites). Note its `cx`.
2. Set both secrets (env vars on the box, or AWS Secrets Manager):
   `GOOGLE_CSE_API_KEY` (= `openclaw/google-cse-api-key`) and
   `GOOGLE_CSE_CX` (= `openclaw/google-cse-cx`). Restart/reload the Oracle.
3. Verify: `GET /search/health` shows `google_cse: ok`; a `/search` returns
   `provider: "google_cse"`.

**Known limitations (deliberate v1, see code comments):** ≤10 results/request
(`limit>10` under-delivers; paginate later); **no server-side date filtering** (relies
on the post-hoc `_filter_by_date`, to avoid a malformed-`sort` 400); response shape +
429 behaviour are assumed from the docs and **need one live call to confirm** before
they're trusted.

These options are not exclusive — CSE as a `tm.web_search` provider is independent
of whether daatan uses Gemini grounding for a fast research answer.

---

## The unification rule (applies to whatever we pick)

1. **Search stays single-chokepoint.** Add Google CSE (or a grounding-backed search)
   *inside `tm.web_search`*, behind Oracle `/search`. **Do not** add a parallel
   Gemini-grounding search in daatan that bypasses the Oracle — that re-fragments
   the one thing that is currently unified.
2. **One way to reach each LLM.** Decide a single convention and document the model
   IDs:
   - Pick how Gemini is called — native `@google/generative-ai` (daatan's way) vs
     litellm `gemini/…` vs OpenRouter `openrouter/google/…`. Today retro uses the
     OpenRouter route for Gemini while daatan uses the native SDK; converge.
   - New LLM features go through the existing abstractions — `tm.llm.complete_structured`
     (retro) and the `LLMProvider` chain (daatan) — never a fresh ad-hoc client.
   - Consider whether the retro raw-httpx OpenRouter calls (row C) should move
     behind `tm.llm` so retro has one dispatch layer (noted, deferred — the `/llm`
     proxy is intentionally a generic pass-through).

## Open decisions (do not implement yet)
- Grounding vs CSE vs both; and does grounding feed the credibility pipeline or only
  serve fast non-scored answers?
- Single Gemini access path across repos (native SDK vs litellm vs OpenRouter)?
- Cost ceilings / quota handling for CSE and grounding (mirror the existing
  per-provider `_QUOTA_EXHAUSTED` pattern in `tm.web_search`).

## Affected files (pointers)
- retro search: `pipeline/src/tm/web_search.py`, `api/src/forecast_api/searcher.py`,
  `pipeline/src/tm/article_text.py`
- retro LLM: `pipeline/src/tm/llm.py`, `pipeline/scripts/improve_keywords.py`,
  `bayesoracle/calibrate_edges.py`, `api/src/forecast_api/main.py` (`/llm`)
- daatan LLM: `src/lib/llm/{index,service,types}.ts`,
  `src/lib/llm/providers/{gemini,openrouter,ollama}.ts`
- daatan search: `src/lib/services/{oracleSearch,oracleClient}.ts`,
  `src/app/api/ibi/{search,llm,fetch-url}/route.ts`
