# Claim extraction: alternative designs (retro#984 item 3)

Exploration write-up requested by Mark on 2026-10-20: designs that are *fundamentally* different
from today's extractor, not another model swap. This is a recommendation, not an implementation.
Date: 2026-10-29.

## What we have today

One LLM call per (article, question) pair: Haiku 5.5 with adaptive thinking
(`pipeline/src/tm/extractor.py`), ~34k input tokens per call (about 30k of that is the cached
`PROMPT_PREFIX` plus the MD_JSON-rendered `ExtractionOutput` schema), returning up to 5
`PredictionExtraction`s with ~14 requested fields. Each field is either *extracted* (a quote,
checkable against the article) or *elicited* (stance, certainty, `fact_signal`, `event_date`:
judgements, GLOSSARY.md). Around the call sit three things: Nova Micro (gatekeeper) before it,
the Jev skip-gate and `evidence_class` corrector beside it, and deterministic guards after it
(evidence window, entity-dyad check, precursor cap).

Known pressure points, all measured:

- **49% of articles yield zero predictions** on 5.5 (31% on 4.5); whether that is correct
  abstention or lost recall is retro#984 item 1.
- **One prompt carries every rule.** Sections interfere with each other: retro#720 showed that
  removing *one* unrelated section took a threshold case from 8/20 to 18/20 on Nova Lite. Fixes
  aimed at one rater have repeatedly failed or regressed others (#683, #657).
- **A shadow field next to a load-bearing one contaminates it**: `grounds` nulled
  `evidence_class` (retro#774).
- **The extraction is question-conditioned**, so an article pushed to N forecasts is read N
  times, and none of those reads can be reused for a question that arrives later.
- **Some prompt text lives outside our code**: instructor's MD_JSON wrapper (retro#1012).

## Already settled — not candidates here

- Replacing Haiku with Jev outright. Jev generates no text and fails cross-sentence context
  (decisions.md 2026-09-23).
- Dropping the shadow fields to shrink output. They are Oracle 2.5/3 inputs and must not be
  dropped (see the Oracle 2.5/3 note).
- Per-claim routing between models, a Gemini extractor, and a per-article prefilter in
  news-indexer (ni#463). All were measured and lost.

## Candidates

### A. Question-agnostic claim store: extract once per article, bind per question

Split the call in two:

1. **Extract** every forward-looking or factual claim in the article *without* the question:
   claim text, quote, actors, `event_date`, `evidence_class`, `attributed_to`. Do this once per
   article and store the result keyed by `content_hash`.
2. **Bind** each stored claim to a question with a small, cheap call (or Jev noul plus a
   judge). This step decides relevance, stance-toward-this-question and `fact_signal`.

- **For:**
  - The extraction cost is amortised over every question the article ever meets, including
    future ones.
  - The claims become nodes, which is exactly the Oracle 2.x graph-engine substrate.
  - The per-question prompt gets much shorter, so the rules interfere less (the #720 class of
    problem).
  - "Zero predictions" splits into two questions we can answer separately: "no claims in the
    article" versus "claims, but none bear on this question".
- **Against:**
  - Recall conditioned on the question is lost. Today the question tells the model what to look
    for, so a question-free pass may skip a sentence that only matters to one question.
  - Stance-to-question needs the claim's context, so the bind step must still see the
    surrounding span, not just the claim.
  - It is the largest change of the set: a new table, a new provenance schema, and a migration
    of the live lane.
- **First measurement (cheap, read-only):** the fan-out. How many distinct questions does an
  article get extracted for over 30 days (`oracle_call_logs` / retro#833 sizing)? Near 1 means
  the amortisation argument is weak and A is only worth it for the graph engine. Well above 1
  means it also pays for itself.

### B. Quote-first, then judge (extraction and elicitation as separate steps)

1. Call 1 returns only *extracted* material: verbatim quotes that bear on the event, each with
   its actors. A deterministic check rejects any quote that is not a substring of the article.
2. Call 2 elicits stance, certainty, `evidence_class` and `fact_signal` per surviving quote.

- **For:**
  - It enforces the extraction-versus-elicitation rule by construction: an elicited value
    always has a checked quote behind it.
  - Hallucinated spans die before anything is weighted on them.
  - Each prompt is smaller and single-purpose, so its sections cannot interfere.
  - The judge step is where Jev already measured better than Haiku on enum fields, so the two
    can be combined there.
- **Against:**
  - Two calls instead of one. On Haiku 5.5 that roughly doubles a cheap line, unless call 2
    runs only on articles where call 1 returned quotes (about half of them).
  - The judge has to see enough context around each quote (the cross-sentence case Jev failed
    on).
  - `grounds` (#774) is a warning, but it does not apply directly: that was an extra field in
    the same call, while here the steps are separate calls.
- **First measurement:** offline, on the A/B corpus. Run call 1 alone and count how often today's
  claims have a quote that fails the substring check. That rate is the hallucination the design
  would remove. If it is ~0, B buys structure but no quality.

### C. Native tool-use output instead of MD_JSON

Pass `ExtractionOutput` as a Bedrock Converse `toolConfig` schema (forced tool choice) instead of
the instructor MD_JSON mode, which pastes the rendered schema plus instructor's own wrapper text
into the system prompt.

- **For:**
  - The schema leaves the prompt text, so retro#700's "the schema is ~27% of the prompt" stops
    being a prompt-engineering cost.
  - instructor's wrapper stops being our prompt, which removes the retro#1012 coupling.
  - Constrained decoding makes parse failures structurally rarer.
- **Against:**
  - It is not a new extraction *design*, only a transport change.
  - Adaptive thinking combined with forced tool choice has to be verified on Bedrock for
    Haiku 5.5; forced tool use has historically been incompatible with extended thinking, so
    this may simply not be available with thinking on.
  - Prompt caching behaves differently once the schema moves into `toolConfig`.
- **First measurement:** a single smoke call. If thinking plus forced tool choice is rejected,
  C is off the table for the live extractor.

### D. Self-consistency: N cheap samples, keep the agreeing claims

5.5 is about 10× cheaper per token than 4.5. Run k=3 extractions and keep only claims that
appear in at least 2 of them (matched on quote or actors), with stance as the median.

- **For:**
  - It turns Bedrock's residual temperature-0 noise (retro#532, the volatile cases in retro#757)
    into a confidence signal.
  - No prompt change at all.
- **Against:**
  - Costs about 3×, at a time the credit cliff (~12-01) makes every recurring line matter.
  - It reduces variance, not bias: a consistently wrong reading survives.
  - Matching claims across samples is its own heuristic.
- **First measurement:** free. The volatile cases' existing 15-run arms already show how often a
  claim appears in under 2 of 3 random draws.

## Recommendation

1. **Start with A's fan-out measurement** (one read-only query). A is the only candidate that
   changes the *shape* of the problem and lines up with the 2.x graph engine. Whether it also
   pays for itself depends entirely on that number.
2. **Run B's quote-substring check offline on the existing A/B corpus.** It costs nothing beyond
   a local script over stored outputs, after the rerun with thinking on, and it tells us whether
   B fixes a real error rate.
3. **Park C** behind a single smoke call (thinking plus tool choice on Haiku 5.5).
   **Do not pursue D** before the credit decision (docs#229).

None of these should start before retro#984 item 1 decides keep/tune/rollback for the current
extractor. A and B would both inherit its recall baseline.
