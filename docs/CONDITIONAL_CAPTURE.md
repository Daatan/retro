# Conditional Claims Extraction — Phase 1 Capture

**Status:** Phase 1 (capture) shipped 2026-08-09, PR #504  
**Phase 2:** Measurement (Brier delta analysis) — pending  
**Phase 4:** Attenuation (scoring integration) — pending  

---

## Overview

A **conditional claim** is a prediction that depends on an antecedent event or condition:
- "If the court rules X, then Y will happen"
- "Should Congress pass the bill, prices will rise"
- "Absent a major announcement, the deal will close"

For ~5–10% of news articles (15–25% for analyst statements), conditional language appears.
The extractor now records these claims with their antecedents, causal relations, and (optionally)
explicit conditional probabilities — on **shadow lane** with zero scoring impact yet.

---

## What Phase 1 Captures (9 Fields Per Claim)

All fields are **Optional, default None** — backward compatible.

### Core Fields

| Field | Type | Example | Purpose |
|---|---|---|---|
| `is_conditional` | bool | True | Gate for Phase 4 attenuation |
| `antecedent_text` | str | "the court rules in favor" | Verbatim "if"-clause, original language |
| `antecedent_text_en` | str | "the court rules in favor" | English canonical form (for embedding) |
| `antecedent_polarity` | bool | True | False if negated ("if NOT X") |

### Relation & Strength

| Field | Type | Values | Purpose |
|---|---|---|---|
| `relation` | str | raises / lowers / requires / precludes / unclear | How antecedent affects consequence |
| `strength` | str | certain / likely / possible / unlikely | Linguistic likelihood of the conditional |

### Metadata

| Field | Type | Example | Purpose |
|---|---|---|---|
| `stated_probability` | float | 0.75 | Explicit P(consequence\|antecedent) if cited |
| `is_counterfactual` | bool | False | True for "had X not happened..." |
| `speaker` | str | "Reuters" | Attribution (outlet or analyst) |

### Pre-Resolution Semantics

All 9 conditional fields are recorded **BEFORE** the `enforce_*` chain runs. This is
asymmetric from other claim fields (stance, certainty, etc.) which are post-resolution.

**Why?** Conditionals must capture the source's original framing (antecedent as stated)
before enforcement. A conditional claim's truth depends on whether *both* the antecedent
and consequence resolve correctly.

---

## How Extraction Works (v1.1: Single-Call)

### 1. Lexical Pre-Filter

The extractor checks for 12 keywords in the article text:
- **if, unless, should, provided, were, in the event, absent, barring, contingent, depends, assuming, so long as**

Regex: word-boundary check (`\b<keyword>\b`), case-insensitive.

### 2. Conditional Block Gate

**If lexicon matches:** Include a 180-line instruction block in the LLM prompt  
**If no match:** Omit block (model expected to null all 9 fields)

### 3. LLM Call (Single, No Round-Trip)

- Extractor processes the article with or without the conditional block
- All 9 fields always in the schema (nullable)
- Cost: ~0s when lexicon doesn't match (instruction block omitted)

### 4. Persist to ClaimDetail

All claims (conditional or not) are stored in `EvidencePoolArticle.claimsDetail` JSON
with their 9 conditional fields (populated or null).

---

## Design Decisions

### Why v1.1 (Single-Call)?

| Decision | Rationale |
|---|---|
| **All 9 fields always in schema** | No API churn; constants schema shape across articles |
| **Lexical pre-filter gates instruction block** | Cheap word check (regex) avoids 180-line block for 90% of articles |
| **No second LLM round-trip** | Saves ~1.5s per article; Phase 1 is measurement-only, not scoring |
| **Append-only prompt pattern** | Matches existing precedent (short_form, language_hint); zero breaking changes |

### Why Not v1.0 (Two-Call)?

v1.0 would:
1. First call: extract claim + stance + certainty (no conditional fields)
2. Second call: *if* article contains conditional language, extract the 9 conditional fields separately

**Costs:** ~1.5s additional latency per conditional article, doubled LLM calls. v1.1 avoids this
by gating the instruction block, not the schema.

### What's Deferred to Phase 2+

- **5% bypass probe:** Measures false-negative rate of lexical pre-filter (can be added post-merge)
- **Brier measurement:** Analyze Brier delta when conditionals excluded vs. included
- **Step 4 attenuation:** Only if Phase 2 confirms conditionals improve forecasts
- **Embedding & linking:** `antecedent_text_en` is captured for future but not yet consumed

---

## Safety: Settlement Gate Unaffected (§3.0)

The **settlement-match gate** (retro#388) is the highest-risk component because it reads from
`claimsDetail`. It uses only 4 fields:
- `claim`, `quote`, `event_date`, `settled`

The new 9 conditional fields live entirely outside this set. Test
`test_settlement_gate_unchanged_with_conditional_fields()` runs the gate on identical sources
twice (once with conditional fields, once without) and verifies verdicts are identical.

---

## Backward Compatibility

- ✅ All 9 fields Optional, default None
- ✅ Old articles missing fields parse fine (Pydantic fills with None)
- ✅ Scoring systems unchanged until Phase 4
- ✅ Settlement gate unaffected (critical test passes)
- ✅ 696 API + 679 pipeline tests pass

---

## Current State (Phase 1: Live)

### What's Recorded

Every article processed by the extractor now has conditional claims captured in
`EvidencePoolArticle.claimsDetail`:

```json
{
  "claim": "prices will rise",
  "quote": "Should Congress pass the bill, prices will rise.",
  "stance": 0.8,
  "certainty": 0.7,
  "is_conditional": true,
  "antecedent_text": "Congress passes the bill",
  "antecedent_text_en": "Congress passes the bill",
  "antecedent_polarity": true,
  "relation": "requires",
  "strength": "certain",
  "stated_probability": null,
  "is_counterfactual": false,
  "speaker": "Reuters"
}
```

### What's Queryable

In prod (Oracul database):

```sql
-- Conditional claims per relation
SELECT relation, COUNT(*) FROM (
  SELECT jsonb_array_elements(claimsDetail)->>'relation' as relation
  FROM evidence_pool_article
  WHERE claimsDetail::text LIKE '%"is_conditional":true%'
) t
WHERE relation IS NOT NULL
GROUP BY relation;

-- Articles with high conditional density
SELECT url, COUNT(*) as cond_count
FROM evidence_pool_article
WHERE claimsDetail::text LIKE '%"is_conditional":true%'
GROUP BY url
HAVING COUNT(*) > 3
ORDER BY cond_count DESC;
```

### What's NOT Yet Active

- ✅ Conditional fields captured, persisted, queryable
- ✅ Pool-split filtering live on `/pool/aggregate` (retro#573 Option 1, below)
- ❌ No scoring impact yet (Phase 4 pending)
- ❌ No attenuation gates active
- ❌ No antecedent linking (Phase 2+ work)

---

## Pool-Split Filtering (retro#573 Option 1)

The first real consumer of this data, outside Phase 1's own tests: `PoolAggregateRequest`
(`/pool/aggregate`) accepts an optional `antecedent_query` (+ `antecedent_query_polarity`).
When set, the pool is filtered — BEFORE the existing weight loop, no new search, no new
prompt — to sources whose claims are either unconditional or conditional on a matching
antecedent, using lexical shingle-Jaccard over `antecedent_text_en`/`antecedent_polarity`
(`forecast_api/antecedent.py`, same no-embedding rationale as `clustering.py`/retro#355).
Sources conditional on a *different* antecedent are dropped; a pool with no matching claims
returns `insufficient_data`/`reason=no_matching_antecedent` rather than silently falling back
to the unfiltered (flat) pool. `None` (every caller before this issue): no-op, byte-identical
to today's pooling. Option 2 from retro#573 (an antecedent-aware stance prompt — an extraction
change) stays deferred behind the usual shadow gate. Tests: `api/tests/test_antecedent.py`
(pure filter) and `api/tests/test_pool_aggregate.py::TestAntecedentPoolSplit` (integration).

**Live `/forecast` path (retro#583):** `ForecastRequest` carries the same two fields with the
same semantics, wired into `_run_forecast_inner` right before `aggregate_pool()` runs — filters
`source_signals` and every parallel array the per-article loop built (stance/weight/relevance/
settled/...) by the same keep/drop mask (`antecedent_keep_mask`, the primitive both this path
and `/pool/aggregate` now share) rather than a single list, since the live path builds several
arrays in lockstep instead of one list of pool rows. `build_claim_meta` folds both fields into
the forecast cache key (appended, not interpolated, so an antecedent-less request still hashes
exactly as it did before #583) — closing the exact gap #582 shipped without: pre-#583, two
different antecedents on the same consequent, or a conditional and an unconditional query on
the same question, would have collided on the cache and silently served each other's answer.
The MCP `forecast` tool does not yet expose these fields — a separate, deliberately deferred
product decision, not part of this fix. Tests: `api/tests/test_antecedent.py::TestAntecedentKeepMask`
(the shared mask primitive), `api/tests/test_forecaster_helpers.py::TestBuildClaimMetaAntecedent`
(cache-key discrimination), `api/tests/test_antecedent_live_forecast.py` (end-to-end through the
real `run_forecast`, gatekeeper/extractor stubbed).

---

## Phase 2: Measurement (When Ready)

Once Phase 1 data accumulates (4–6 weeks of new articles):

1. **Brier delta:** Compare forecasts that include conditionals vs. exclude them
2. **False-negative rate:** Run 5% bypass probe (force include block for 5% of non-matching articles)
3. **Relation distribution:** Which relation types (raises/lowers/requires/precludes) are most common?
4. **Strength calibration:** Does linguistic strength match realized conditional accuracy?

**Decision gate:** If Brier improves measurably, proceed to Phase 4 (attenuation). Otherwise,
keep recording but don't gate.

---

## Phase 4: Attenuation Scoring (If Phase 2 Green-Lights)

Attenuate claims where `is_conditional=True` by multiplying into certainty weight in
`forecaster.py:268` (per-claim weighting). Gate only activates if Phase 2 measurement shows
conditional handling improves forecast accuracy.

---

## Testing

**Phase 1 test coverage:**

| Test | Purpose | Status |
|---|---|---|
| `test_conditional_fields_are_optional_and_default_to_none` | Fields default to None | ✅ PASS |
| `test_conditional_fields_can_be_populated` | Fields accept values | ✅ PASS |
| `test_conditional_fields_round_trip_through_json` | JSON serialization works | ✅ PASS |
| `test_conditional_fields_in_claims_detail_list` | Mixed lists work | ✅ PASS |
| `test_settlement_gate_unchanged_with_conditional_fields` | Gate unaffected (CRITICAL) | ✅ PASS |

All tests in `api/tests/test_claims_detail.py::TestConditionalFields`.

Full suite: 696 API tests + 679 pipeline tests (1,375 total) all pass.

---

## File References

**Extraction logic:** `pipeline/src/tm/extractor.py`
- `CONDITIONAL_LEXICON` — 12-keyword frozenset
- `has_conditional_language(text)` — Pre-filter function
- `_CONDITIONAL_BLOCK` — Instruction block with 3 worked examples
- `extract_predictions(include_conditional_block)` — LLM interface

**Models:**
- `pipeline/src/tm/models.py` — `PredictionExtraction` with 9 conditional fields
- `api/src/forecast_api/models.py` — `ClaimDetail` with 9 conditional fields

**Tests:**
- `api/tests/test_claims_detail.py::TestConditionalFields` — 5 tests
- Pipeline integration: all extractor tests pass

**Documentation:**
- This file (conditional-capture.md)
- `docs/ARCHITECTURE.md` — "Conditional Claims (Phase 1 Capture)" section
- Design (external): conditional-capture-phase1.md (draft design document)

---

## FAQ

**Q: Why are conditional fields pre-resolution?**  
A: Conditionals must capture the source's original framing. Once we start enforcing,
we've already collapsed stance/certainty via rules that assume unconditional claims.
Splitting conditional capture to pre-resolution preserves auditability.

**Q: Why not gate on something other than is_conditional?**  
A: Phase 4 will use `is_conditional=True` as the attenuation trigger. If a claim isn't
marked conditional, we treat it as ordinary evidence (status quo). Once measurement
confirms conditionals matter, we attenuate only those marked true.

**Q: What if the lexical pre-filter misses a conditional?**  
A: Phase 2 will measure the false-negative rate via 5% bypass probe. If it's >5%,
we can tune the lexicon or add a secondary signal. For Phase 1 (measurement-only),
~99% recall is acceptable.

**Q: Can I query conditionals in prod right now?**  
A: Yes. They're in `EvidencePoolArticle.claimsDetail` JSON. Use LIKE or jsonb operators
(see "What's Queryable" above).

**Q: When does Phase 2 start?**  
A: When sufficient data accumulates (4–6 weeks) and measurement prioritization aligns.
Mark will decide.

---

## Timeline

| Date | Event | Status |
|---|---|---|
| 2026-08-09 | Phase 1 merged (PR #504) | ✅ DONE |
| 2026-08-09 | Documentation updated | ✅ DONE |
| 2026-09-15 (est.) | Sufficient data for measurement | ⏳ PENDING |
| TBD | Phase 2 measurement complete | ⏳ PENDING |
| TBD | Phase 4 attenuation shipped (if approved) | ⏳ PENDING |

---

## Contact / Questions

For questions about conditional extraction design or measurement results, see:
- Issue tracker: Daatan/retro #364 (parent issue)
- Design doc: conditional-capture-phase1.md
- Code: see "File References" above
