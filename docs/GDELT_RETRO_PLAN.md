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
