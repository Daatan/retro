# GDELT as a free, deterministic search for Duel / Bediavad — plan

> Goal: give the retro engines (Bediavad source-leaderboard + Duel) a **free,
> reproducible, historical** article-discovery layer, **without changing the
> Oracle `/search` API**. Date: 2026-07-11. Verified live from the Oracle host
> `i-00ac444b94c5ff9b2` (egress IP `3.120.185.111`).

---

## What the code already has (don't rebuild)

- **GDELT DOC API** leg — `web_search._search_gdelt` (free, no key, ~3-month rolling window, 1-req/N-sec IP rate limit, cross-process slot lock + 429 cooldown + circuit breaker).
- **GDELT BigQuery GKG** leg — `web_search._search_gdelt_bq` over `gdelt-bq.gdeltv2.gkg_partitioned`, partition-pruned by `_PARTITIONTIME`, entity match via `REGEXP_CONTAINS` on V2Persons/V2Locations/V2Organizations/AllNames, URL-slug-synthesised titles. `google-cloud-bigquery 3.41.0` is installed in the pipeline venv.
- **`/search` API already supports `date_from` / `date_to`** (`models.py:19-26`).
- **Historical routing already exists**: the chain runs BigQuery **first** when `date_from` is older than 90 days (`web_search.py:1563-1567`), and BQ is last-resort for recent queries.

## Status: unblocked (2026-07-11)

The BigQuery credential — the only blocker — is now provisioned and verified
**live** from the Oracle host. A fresh JSON key was minted for the pre-existing
SA `retro-search@daatan.iam.gserviceaccount.com` (project `daatan`, already
scoped `bigquery.jobUser`) and stored as Secrets Manager
`daatan/gcp-service-account-key`. `_get_bq_client()` builds and a live query
returned real Ynet Oct-2023 URLs. Remaining blockers below are code-level, not
credential-level.

1. ~~The BigQuery credential is absent.~~ **DONE** — see above.
2. **The free DOC API is 429-throttled from our IP.** Two probes from the Oracle
   host both returned `HTTP 429 — "one request every 5 seconds"`; production
   `/forecast` already spends that IP's GDELT budget. DOC is 3-month-only anyway,
   so it can't serve "3 years ago." → **DOC API is not the retro mechanism.**
3. **The BQ leg can't filter by outlet.** `_extract_bq_terms` strips
   `site:`/`domain:` and matches entities only (`orchestrator.py:179` confirms
   "BigQuery ignores site: filters"). → **Per-source Bediavad cells ("just ynet")
   aren't expressible** without a small pipeline patch.

---

## Answers to the six questions

**1. Better free mechanism, without changing the API?**
Yes — **GDELT BigQuery GKG** is the deterministic, free, full-history URL index.
Pipeline: `GDELT-BQ (find URLs, free) → scrape / Wayback (get text) → LLM extract
→ Atlas cell / Duel`. For the *recent* window use the **news-indexer pgvector**
(already hosted, keyed, deterministic). No API change: historical `/search`
already routes BQ-first, and the Bediavad batch calls `_search_gdelt_bq()`
in-process, bypassing the HTTP API (and its DOC-API IP contention) entirely.

**2. Do I need a key from you?**
**Yes — one GCP service-account JSON**, least-privilege (**BigQuery Job User** +
public-data read), stored as Secrets Manager `daatan/gcp-service-account-key`
(the name the code already reads). GDELT itself needs no key; the *BigQuery
billing project* does. Alternative keyless path = raw GDELT files (below), more
engineering.

**3. Tested from our server?** Done — findings above (DOC 429; BQ lib present; BQ
client fails on missing key; secret absent).

**4. Cost of recomputing the whole table / one event — close to 0?**
**Measured (dry-run + one live run, 2026-07-11): yes, literally $0.**
- **1 event** (27-day window) = **9.93 GB → $0.062**. Same 9.93 GB whether you add
  more entity terms *or* a `SourceCommonName` domain filter — predicates don't add
  scanned columns, so **domain filtering is free**.
- **Whole 70-event recompute** ≈ 70 × 9.93 GB ≈ **0.7 TB → fits under the free
  1 TB/month tier → $0.**
- Sensitivity: a full-year single-term window = 140 GB ($0.88).
- **Design rule:** query **once per event** and bucket sources locally. Do NOT run
  per-(event × source): 1400 queries × 9.93 GB ≈ 14 TB ≈ $80.
Guardrail: set `maximum_bytes_billed` (~50 GB) — an unpruned scan of the full
~15 TB GKG is the only way to run up a bill.

**5. New API flag for gdelt-only?**
**No.** Historical `/search` already prefers BQ; the batch bypasses the API. A
tiny additive `providers=[...]` filter would only matter if you later want to
force gdelt-only on a *recent* (<90d) query — not the retro case.

**6. More ideas / caveats** — see the two sections below.

---

## Plan (phased)

**Phase 0 — unblock (needs you): provision the GCP key.**
Create a least-privilege service account, put its JSON in
`daatan/gcp-service-account-key`. Then I re-run the dry-run from the Oracle host
to (a) prove the client builds and (b) report exact per-event bytes/$.

**Phase 1 — measure + verify (no code).**
Dry-run the real query shape for a couple of representative events; confirm GKG
coverage for the outlets we score (esp. Hebrew — sanity-check per-`SourceCommonName`
row counts before trusting a domain).

**Phase 2 — pipeline patch (retro repo, PR, no API change). ✅ IMPLEMENTED (this PR).**
- Added optional **`domains=` filter + `max_rows=`** to `_search_gdelt_bq`
  (`SourceCommonName IN (...)`), backward-compatible (live Oracle path unchanged),
  plus a `maximum_bytes_billed` cost fuse. Verified live: 622 rows across exactly
  the two whitelisted outlets, 0 off-whitelist, 9.64 GB ($0.06).
- New **`tm.gdelt_bq_ingest`**: ONE per-event GKG scan over tracked outlets →
  bucket URLs by `source_id` → **Wayback-first** fetch (live fallback opt-in) →
  `data/raw_ingest/{source_id}/{event_id}/`, consumed by the unchanged
  `orchestrator local_file` extraction. Free-only, with a per-event miss-log.
- Added the **anti-lookahead backstop at Atlas-write** (`create_atlas_link` refuses
  post-outcome articles via `predates_outcome`), covering the near-duplicate-reuse
  and cached-extraction paths.
- Added a **`--discover`** mode surfacing tracked-outlet coverage volume for a topic
  (NO-event candidate hunting).

**Phase 3 — use it.**
Backfill the 78.5% empty Atlas cells and — the actual validity gate — **hunt
NO-outcome events** (predicted-then-fizzled) to break the 69/70-YES imbalance.
Re-run `backtest.py`; the results are now reproducible run-to-run.

**Phase 3a — NO-event curation pass (retro#509). ✅ DONE (this PR).** `pipeline/scripts/seed_no_events_batch1.py`
had speculatively seeded 6 candidate NO-events (`A20`, `B14`, `C10`, `C11`, `C13`, `D07`)
"pending Oracle in-window validation ... events that return no in-window coverage
get pruned." Running that validation surfaced a tooling bug, not a coverage
problem:

- `--events A20 B14 C10 C11 C13 D07` reported **0 GKG rows for 4 of the 6**
  (`A20`, `B14`, `C11`, `D07`) — `_extract_bq_terms` (`web_search.py`) filters
  generic single words ("Israel", "Iran", "Gaza", …) as too-common, and those
  4 events' `search_keywords` had no *other* proper noun, so the BigQuery query
  never even ran (`RuntimeError: no extractable entity terms in query`,
  silently indistinguishable from "GDELT has no coverage" in the batch summary).
- Live `--discover` (bypasses the fetch step, reports raw GKG row counts) over
  each event's predictive window, seeded with a real named entity per event,
  showed **heavy genuine coverage for all 6** — hundreds of articles across a
  dozen-plus tracked outlets each (e.g. `C13` "Israel Hezbollah" Jan 2024:
  haaretz 202, ynet 171, maariv 165, …; `D07` "Amir Yaron" Dec 2024: ynet 253,
  maariv 121, jpost 73, …). None qualifies for pruning.
- **Fix**: added a `duel_keywords` entry with one real named entity to each of
  the 4 blocked events (`A20`→"Netanyahu", "Knesset"; `B14`→"Netanyahu",
  "Hamas"; `C11`→"Netanyahu", "Isfahan"; `D07`→"Amir Yaron", Bank of Israel's
  governor) — the same pattern already used by `A19`/`B05`/`B08`/`B09`/`B10`/
  `C07`/`C08`/`C09`. Verified locally: `_extract_bq_terms` now returns
  non-empty terms for all 4, and a full `--events` run (not just `--discover`)
  executes the BigQuery scan for all 6 (previously 4 errored before the query
  even ran). **Wayback fetch yield is still low** (2 articles saved across all
  6 events in this run) — most GKG-indexed URLs from this period either have
  no pre-outcome Wayback snapshot or fail `_MIN_TEXT_CHARS`; that is a
  fetch-layer problem for the real backfill (retro#508, `--allow-live` /
  wider `--limit` territory), separate from the curation question this issue
  answers.
- **Outcome: 6/6 kept, 0 pruned.** All 6 batch-1 NO-events are confirmed
  GDELT-queryable with real predictive-window coverage (via `--discover`,
  which counts raw GKG rows independent of fetch success) and ready to feed
  the actual backfill (retro#508). A regression test
  (`TestNoEventEntityExtractability` in `test_gdelt_bq_ingest.py`) now fails
  CI if a future NO-event is added without at least one extractable entity
  term, so this gap can't silently reopen.

---

## Caveats / risks

- **GKG stores the URL, not the article body** — scrape + Wayback still required;
  3-year-old outlet URLs are often dead/paywalled (Wayback is the fallback and a
  bonus for anti-lookahead).
- **Entity-only, recency-ranked, slug-titled** BQ results are low-relevance by
  design — fine for *discovery* feeding the gatekeeper/extractor, not a ranked
  answer.
- **Class imbalance (69/70 YES) is still THE blocker.** GDELT helps you *find*
  NO-events; it doesn't fix the imbalance by itself.
- **Determinism is the real prize**: BQ over a fixed table + fixed window is
  reproducible, unlike the drifting SERP chain — exactly what a scientific
  backtest needs.
- **Run BQ from the server** (SSM/systemd), never the DOC API at volume — BQ has
  no per-IP rate contention with prod.
- **Key hygiene**: least-privilege SA, stored in SM like the other keys, rotate;
  set `maximum_bytes_billed` as a cost fuse.

## Keyless alternative (if you'd rather not provision GCP)

Download raw GDELT files (`data.gdeltproject.org/gdeltv2/*.gkg.csv`, free, no
key), filter by date window + `SourceCommonName` locally, extract URLs. Same
downstream pipeline. Trade-off: full free, but terabytes of I/O and more code
than a single BigQuery query. Recommend BigQuery unless key provisioning is a
hard no.
