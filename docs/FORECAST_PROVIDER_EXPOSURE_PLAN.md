# Plan: expose the search provider in the `/forecast` response

> **Status: implemented.** Resolved decisions: field names **`provider` / `provider_chain`**
> (consistent with `/search`); **also expose `distilled_query`**; on a forecast-cache
> hit report the **original engine** (cached response kept as-is). Correction applied:
> the empty/insufficient response is built by the separate helper `_empty_response(...)`,
> so `provider`/`provider_chain`/`distilled_query` are **threaded into it** at the
> post-search call sites (not "rely on the default"). Required daatan follow-up (1 line,
> separate PR): reader → `data.provider ?? data.provider_chain?.join(', ') ?? null`.

## Context

Daatan (the Oracle's main client) now logs every Oracle call with the search engine
used (daatan PR #831, issue Daatan/daatan#832). For `/search` calls it reads the
response's `provider` field. For **`/forecast`** calls the engine is not in the
response body, so those rows log `searchEngine = null`.

Good news: `/forecast` **already captures** the winning provider and the fallback
chain — it threads them into `DebugInfo` (`debug.search_provider` /
`debug.search_provider_chain`), but only when `debug=true`. This change simply
promotes those values into the top-level `ForecastResponse` so every caller gets
them without `debug` mode.

## The change (retro)

FastAPI service, Python/pytest.

### 1. Model — `api/src/forecast_api/models.py`

Add two fields to `ForecastResponse` (class at line ~149), mirroring the names and
defaults already used on `SearchResponse` (lines 27–28):

```python
provider: str = Field(default="", description="Search provider that served the underlying article search (e.g. 'gdelt', 'brave', 'none')")
provider_chain: list[str] = Field(default_factory=list, description="Full search fallback chain attempted, in order")
```

Defaults matter: every `ForecastResponse(...)` construction that doesn't set them
still validates (they fall back to `""`/`[]`), so no return path can break.

### 2. Populate — `api/src/forecast_api/forecaster.py`

`search_provider` and `provider_chain` are already function-local (declared ~475,
set at ~495/510/529/534/538 and fed into `DebugInfo` at ~567, ~685, ~730). Add
`provider=search_provider, provider_chain=provider_chain` to the `ForecastResponse(...)`
constructions:

- The main success response at **~748**.
- The placeholder/insufficient-data response at **~807** (`placeholder=True`). At that
  point `search_provider` is in scope; pass it through (or rely on the `""` default if a
  path reaches there before search ran).

Because the model has defaults, this is additive and backward compatible — existing
clients ignore the new fields.

### 3. Tests — `api/tests/`

- `test_searcher.py` already proves provider attribution flows through the thread-local
  capture (`TestRunSearchProviderAttribution`). Mirror that for the forecaster: add a test
  asserting `run_forecast(...)` returns `provider == "gdelt"` and the expected
  `provider_chain` when search is stubbed (pattern: monkeypatch `search_articles` to set
  `_ws._provider_local.name/chain`, as in the searcher test).
- Update the `ForecastResponse` fixture/assertions in `test_cache.py` only if any test
  asserts the full response dict exactly (the defaults mean most won't need changes).

No API version bump required (additive). `version` stays `"0.1.0"`; daatan's
`EXPECTED_API_VERSION = '0.1'` prefix check still passes.

## Naming decision (resolve before coding)

Daatan PR #831's reader (`src/lib/services/oracle.ts`) currently reads
`data.search_engine ?? data.providers_used?.join(', ')`. Retro's established
convention (on `/search`) is `provider` / `provider_chain`. Pick one:

- **Recommended — use `provider` / `provider_chain`** (consistent across `/search` and
  `/forecast`). Requires a trivial follow-up in daatan: change the reader to
  `data.provider ?? data.provider_chain?.join(', ') ?? null`. One line; it's defensive
  (returns null today), so nothing breaks in the interim.
- Alternative — name them `search_engine` / `providers_used` to match the already-merged
  daatan reader exactly (zero daatan change), at the cost of `/forecast` and `/search`
  using different field names for the same concept.

The consistency win of the recommended option outweighs the one-line daatan edit.

## Verification

1. `pytest api/tests/test_searcher.py api/tests/test_cache.py -q` (and the new forecaster
   test) pass.
2. Manual: `POST /forecast` (without `debug`) returns top-level `provider` +
   `provider_chain` matching what `debug.search_provider` would show.
3. End-to-end after both sides ship: trigger a forecast-path Oracle call from daatan
   (e.g. a forecast context update), then Admin → Oracle shows a `FORECAST` row with a
   non-null search engine instead of `—`.

## Rollout

- Retro change is additive/backward-compatible — ship independently.
- If the recommended naming is chosen, land the one-line daatan reader change in a small
  follow-up PR; until then forecast rows keep logging engine `null` (no harm).
- PR-only workflow in the retro repo (no direct pushes to `main`).
