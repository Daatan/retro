# Extractor model survey (2026-08-24, ad-hoc)

Not a shipped eval, not gated in CI, no code changed. This documents a one-off
research pass answering "would the extractor get materially better if we used
a different/better model, and at what cost?" — run interactively, results kept
here because the raw scripts and data live under a session scratch dir
(`/tmp/...`) and will not survive. If this survey needs to be re-run for real,
promote it to a script under `pipeline/scripts/` first (see Non-goals).

## Question and method

Does swapping `extractor_model` (`pipeline/src/tm/config.py`) change what the
extractor produces enough to matter, and what would each candidate cost?

- **Cases**: the 50 most-recently-modified real extraction records in the S3
  atlas snapshot (`truthmachine-atlas-snapshots-272007598366/latest.tgz`,
  `vault2/extractions/` + matching `vault2/articles/` + `data/events/*.json`)
  — real articles and events from the actively-running batch pipeline, not
  synthetic/de-named cases like `eval_extractor_adjacent_events.py` uses.
- **Call site**: only `extract_predictions()` (`pipeline/src/tm/extractor.py`)
  — the main article → claims extraction step. The current prompt/schema was
  held fixed; only `settings.extractor_model` was varied per batch. See
  **Scope** below for what this does *not* cover.
- **Reference model**: Claude Haiku 4.5 (`bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0`)
  — what `oracle-api` already runs in prod (`infra/oracle-api.service.d/extractor-model.conf`).
  Every candidate is scored by how closely it matches Haiku's behavior on the
  same 50 cases, not by an independent correctness label (there isn't one).
- **Primary metric — abstention agreement**: on 47/49 usable cases, Haiku 4.5
  extracted **zero** predictions (tangentially-related historical articles
  that don't actually bear on the forecast question — correct abstention,
  confirmed by manually re-running and inspecting a case's raw output, not a
  bug). So "agreement with Haiku" mostly measures whether a candidate is
  *equally selective*, not whether it agrees on content. This produces a
  **floor effect**: any model that extracts something from nearly every
  article (Nova Lite/Micro/Pro, Llama 3.3 70B, Llama 3.1 8B) can only ever
  coincidentally match Haiku's abstention on the ~2 cases Haiku *did* extract
  from, landing at a spurious-looking but structurally meaningless ~4–6%
  floor. Distinguish that from genuine selectivity (Nova 2 Lite, the Claude
  family, Qwen3 32B) before reading the ranking as "worse."
- **Cost**: real per-case token usage from each run × per-token price,
  extrapolated to $/1000 articles. Nova/Llama/Qwen/GLM prices are AWS Pricing
  API on-demand (non-batch) SKUs for `AmazonBedrock` in `us-east-1`, fetched
  2026-08-24 (catalog version `20260820130114`). Claude Haiku/Sonnet/Opus 4.5
  are **not yet** in that account's bulk price-list snapshot (too new) — used
  Anthropic's/Bedrock's published per-token list prices instead ($1/$5,
  $3/$15, $5/$25 per million respectively), flagged separately from the
  catalog-confirmed figures.

## Models tested

| Model | Family | $/1000 articles (measured) | Notes |
|---|---|---:|---|
| claude-haiku-4.5 | Anthropic | $9.46 | reference; public list price |
| claude-sonnet-4.5 | Anthropic | $28.18 | public list price |
| claude-opus-4.5 | Anthropic | $46.14 | public list price |
| claude-3-haiku (legacy) | Anthropic | — | **not tested** — Bedrock use-case access not granted on this account |
| nova-lite | Amazon | $0.27 | current batch-pipeline default |
| nova-micro | Amazon | $0.15 | current gatekeeper model (different task, tested separately below) |
| nova-2-lite | Amazon | $2.54 | |
| nova-pro | Amazon | $7.74 | |
| llama3.3-70b | Meta | $5.93 | |
| llama4-scout-17b | Meta | $1.34 | |
| llama4-maverick-17b | Meta | $2.01 | 4/50 malformed-JSON schema failures |
| llama3.1-8b | Meta | $1.83 | 6/50 truncated (1200-token `max_tokens` cap too tight) |
| llama3.2-11b | Meta | — | **not tested** — fails structured-output parsing outright |
| qwen3-32b | Alibaba | $1.10 | 1/50 truncated |
| glm-4.7-flash | Zhipu | $0.66 | |
| deepseek-v3.2 | DeepSeek | — | **not tested full batch** — sanity sample priced ~$3.50/1000, over the $2 ceiling this round targeted |
| deepseek-r1 | DeepSeek | — | **not tested** — no separate catalog price on this account; known reasoning-token overhead pushes real cost well above sticker price |
| kimi-k2.5 (Moonshot), glm-4.7 (full), qwen3-coder-30b/next-80b | — | — | **not tested** — priced above the $2 ceiling or not a general-language fit (coder-specialized variants) |

OpenRouter was evaluated as an alternate gateway for reaching more of these;
the `OPENROUTER_API_KEY` in `pipeline/.env` is invalid/revoked (401 "User not
found") and no working key was found in `daatan/.env`, Secrets Manager, or
SSM Parameter Store. Not pursued further — this account's own Bedrock catalog
already carries Qwen/DeepSeek/GLM/Kimi directly, which covered the
"more model families" ask without needing it.

## Results

```
model                       errs  abstain%  agree_w/haiku%  n_overlap  mean|Δstance|  $/1000-articles
nova-lite                      0      2.0%            6.1%          2          0.225            0.266
claude-haiku-4.5 (REF)         1     95.9%          100.0%          2          0.000            9.457
nova-micro                     0      0.0%            4.1%          2          0.875            0.151
nova-2-lite                    1     77.1%           79.2%          1          0.650            2.538
nova-pro                       0      0.0%            4.1%          2          0.125            7.740
claude-sonnet-4.5              0     83.7%           87.8%          2          0.975           28.178
claude-opus-4.5                0     87.8%           91.8%          2          0.675           46.143
llama3.3-70b                   0      0.0%            4.1%          2          0.225            5.929
llama4-scout-17b               0     32.7%           36.7%          2          0.625            1.338
llama3.1-8b                    6      0.0%            4.7%          2          0.225            1.829
llama4-maverick-17b            4     46.7%           51.1%          2          0.325            2.010
glm-4.7-flash                  0     22.4%           26.5%          2          0.975            0.662
qwen3-32b                      1     58.3%           60.4%          1          0.200            1.105
```

**Ranked by absolute closeness to Haiku 4.5:**

1. claude-opus-4.5 — 91.8%, $46.14/1000
2. claude-sonnet-4.5 — 87.8%, $28.18/1000
3. **nova-2-lite — 79.2%, $2.54/1000**
4. qwen3-32b — 60.4%, $1.10/1000
5. llama4-maverick-17b — 51.1%, $2.01/1000 (unreliable, see below)
6. llama4-scout-17b — 36.7%, $1.34/1000
7. glm-4.7-flash — 26.5%, $0.66/1000
8. nova-lite, llama3.1-8b, nova-pro, llama3.3-70b, nova-micro — all in the
   4–6% floor-effect band, i.e. "extracts from everything," not "matches
   Haiku"

**Ranked by value (agreement-% per dollar per 1000 articles):**

1. **qwen3-32b — 0.55**
2. glm-4.7-flash — 0.40
3. nova-2-lite — 0.31
4. llama4-scout-17b, nova-micro — 0.27
5. llama4-maverick-17b — 0.25
6. claude-haiku-4.5 (reference) — 0.11
7. everything else — ≤0.03

## Reliability findings (not just cost/agreement)

Two candidates failed the `instructor` MD_JSON structured-output contract on
a non-trivial fraction of real cases, independent of the agreement/cost
numbers above:

- **Llama 3.1 8B**: 6/50 (12%) hit the 1200-token `max_tokens` cap mid-output
  and came back truncated.
- **Llama 4 Maverick 17B**: 4/50 (8%) returned malformed JSON that failed
  schema validation even after `instructor`'s retry.
- **Qwen3 32B**: 1/50 (2%) hit the same truncation as Llama 3.1 8B.
- **Llama 3.2 11B** (sanity-tested, not run at full batch): fails structured
  output outright — dropped before the 50-case run.

None of this shows up in the agreement/cost table on its own; a model that's
"good" on the cases it completes but silently drops 8–12% of them is a
different kind of risk than one that's merely less selective.

## Conclusion

**Nova 2 Lite remains the best cost/selectivity tradeoff among cheap models**
(79.2% agreement at $2.54/1000) — no candidate found across four rounds of
testing (Bedrock Nova/Claude/Llama family, then a $2-ceiling Llama sweep,
then the Chinese-origin catalog) beats it on absolute closeness to Haiku's
behavior. **Qwen3 32B is the one genuinely new finding**: not a drop-in
replacement (60.4% agreement, a 19-point gap, plus a truncation failure), but
the best $-per-agreement-point of anything tested, worth keeping in mind if
that tradeoff ever matters more than matching Haiku's ceiling. Nothing
approaches Haiku's own selectivity except Sonnet/Opus 4.5, both far more
expensive than Haiku itself and therefore not cost candidates for anything
Haiku already does.

No model swap is recommended from this survey alone. This is descriptive,
not a go/no-go call — see `eval_extractor_adjacent_events.py` for the actual
gated methodology (false-settlement rate, explicit GO/NO-GO thresholds) that
any real swap proposal should go through before shipping.

## Scope — what this did *not* test

All of the above covers **only** `extract_predictions()`. retro has at least
five more LLM call sites that produce numbers/semantic judgments:

| Task | Explanation | Model used |
|---|---|---|
| **Extractor** — *tested above* | Article → claims: extracts stance, certainty, settled, `quantitative_estimate`, `evidence_class` per prediction. Main "funnel" stage. | `extractor_model` — Nova Lite (batch default) / **Claude Haiku 4.5** (prod override, `oracle-api` systemd drop-in), plus `threshold_extractor_model` for threshold-shaped batch events (retro#688, off by default — see below) |
| **Gatekeeper** — *tested, see below* | Pre-filter: is this article even relevant to the forecast question? Produces `relevance_score` (0–1) and `prediction_count_estimate`. | `gatekeeper_model` — **Nova Micro** (own separate knob, no override anywhere) |
| **Aggregator** | Collapses multiple predictions from one article into one combined read. | Reuses `extractor_model` (no separate knob) |
| **Settlement verifier** — *tested, see below* | Vetoes false/premature "this claim is settled" calls before they pin the forecast. Built specifically for the false-settlement failure mode (retro#532). | `settlement_verifier_model` (config default `None`) → falls back to `extractor_model` |
| **Premise verifier** | Checks whether a claim's premise still holds before using it as evidence. | `premise_verifier_model` (config default `None`) → falls back to `extractor_model` |
| **Oracle 2.0 — decompose** | Experimental playground: breaks a compound question into sub-claims. Not in the live production path. | `decompose_model` (default `None`) → falls back to `extractor_model` |
| **Oracle 2.0 — anchor match** | Experimental playground: matches an article to the nearest anchor event. | Directly uses `extractor_model` (no separate knob) |
| **Oracle 2.0 — precursor match** | Experimental playground: matches an article to a precursor/related event. | `precursor_match_model` (default `None`) → falls back to `extractor_model` |

All of the above except the gatekeeper default to whatever `extractor_model`
is set to, so today's results are directionally applicable to them — but
aggregator, premise verifier, and the Oracle 2.0 playground steps were still
never measured on their own.

## Gatekeeper stage results (2026-08-24 follow-up)

Same 50 real cases, run through `check_is_prediction()` (`pipeline/src/tm/gatekeeper.py`)
instead of the extractor. Reference is again Claude Haiku 4.5; candidates are
Nova Micro (current prod `gatekeeper_model`), Nova 2 Lite, and Qwen3 32B.
Metric: agreement on the `is_prediction` boolean, plus mean absolute delta on
`relevance_score` and `prediction_count_estimate` where both models judged
the article a prediction.

```
model                          errs  is_pred agree%   mean|Δrelevance|   mean|Δcount_est|  $/1000-articles
nova-micro (current prod)         1           93.9%              0.039              0.184            0.064
claude-haiku-4.5 (REF)            0          100.0%              0.000              0.000            4.202
nova-2-lite                       0           98.0%              0.018              0.140            1.203
qwen3-32b                        50            n/a                n/a                n/a            n/a
```

- **Nova Micro (current default) already agrees with Haiku 93.9% of the time**
  at a fraction of the cost ($0.06 vs $4.20/1000) — the gatekeeper is a coarser
  binary judgment than the extractor's multi-field output, so cheap models do
  much better here than they did on extraction.
- **Nova 2 Lite is closer still (98.0%)** at $1.20/1000, if the gatekeeper's
  cost ever becomes worth optimizing further.
- **Qwen3 32B failed all 50 cases**, not from a quality gap but a hard
  incompatibility: `litellm.APIConnectionError: ... "unsupported model or your
  request did not allow prompt caching"`. The gatekeeper's cached prompt
  prefix is ~1,050 tokens (`PROMPT_PREFIX` in `gatekeeper.py`) versus the
  extractor's ~11,200 tokens, where Qwen3 32B worked fine in the earlier
  round — the likely explanation is a minimum-cacheable-prefix threshold this
  model enforces strictly (hard error) that Nova/Claude either clear or
  silently no-op below. Qwen3 32B is not usable as a gatekeeper model without
  a code change (skip `cached_prefix` for models that reject short cache
  blocks, or route it around prompt caching entirely) — untested whether
  quality would be competitive if that were fixed.

**Bottom line: no reason to move off Nova Micro for the gatekeeper.** It
already tracks Haiku closely on this coarser task, at 1/16th Nova 2 Lite's
cost and 1/65th Haiku's.

## Settlement verifier stage results (2026-08-24 follow-up)

Unlike the extractor and gatekeeper, this stage has **real ground truth** to
test against, not just agreement-with-Haiku: `api/scripts/replay_settlement_verifier.py`
already exists, built to replay `verify_settlement()` over every pin
production has actually published, scored against the prediction's real
resolution status (`RESOLVED_CORRECT` / `RESOLVED_WRONG` / still `ACTIVE`).

**Sourcing the data.** The script needs a `pins.json` export — "the settling
rows of the latest pinning snapshot per prediction" — which lives in
**daatan's** production Postgres (`predictions` + `context_snapshots`), not
retro's own storage. Pulled via a read-only SQL query over SSM
(`aws ssm send-command` → `docker exec daatan-postgres psql`, output too
large for SSM's 24KB cap so routed through S3), keying off
`context_snapshots.oracle_snapshot->>'settled' = 'true'` per prediction and
extracting the settling `claimsDetail` rows (`claim`/`quote`/`event_date`/
`settled`) from each source. Of 36 predictions with a settled snapshot, 16
had at least one source claim actually marked `settled=true` at that
snapshot (the other 20 are real coverage gaps — the replay script's own
docstring already flags a summary-only/no-quote replay as a *lower bound* on
live behavior; this pull additionally required a claim-level `settled=true`
flag, which not every settled snapshot's rows carry). Those 16 pins carry
191 settling claims, 100% with a quote. 6 of the 16 have a known resolution
(5 `RESOLVED_CORRECT`, 1 `RESOLVED_WRONG`); the rest are still `ACTIVE`.

**Results**, replaying each model over the same 16 pins:

| Model | Correct pins wrongly vetoed | Caught the 1 known-wrong pin? | ACTIVE-pin veto rate |
|---|---:|---|---:|
| **claude-haiku-4.5 (REF)** | **0/5 (0%)** | No (kept it) | 5/10 |
| nova-2-lite | 1/4 (25%; 1 errored) | No (kept it) | 4/9 (1 errored) |
| qwen3-32b | 2/5 (40%) | No (kept it) | 5/10 |

- **This is the one stage where the cheap-model pattern reverses.** On the
  extractor and gatekeeper, cheap models were mostly *under-selective*
  (missing things Haiku caught). Here, Nova 2 Lite and Qwen3 32B are
  **over-vetoing** — suppressing settlements that later resolved correctly.
  Qwen3 32B falsely vetoed the Brent-crude-oil and Saudi-nuclear-agreement
  pins, both of which resolved `RESOLVED_CORRECT`; Nova 2 Lite falsely
  vetoed the Saudi-nuclear-agreement pin too. Haiku vetoed none of the 5
  known-correct pins.
- **None of the three caught the one known-wrong pin** (the Israel–Iran
  conflict claim, published at 97% and later `RESOLVED_WRONG`) — all three
  kept it. Not every wrong resolution is wrong for a reason this particular
  semantic check exists to catch (see the module's own docstring on why it
  targets aspect/role/direction specifically), so this is expected, not a
  failure of the exercise.
- Nova 2 Lite threw 1–3 transient `BedrockException: ServiceUnavailable`
  errors across two runs (concurrency 4 → 3 errors; concurrency 1 → 1 error);
  the gate fails open on error (never suppresses), so these silently count
  as "keep" in production, not "veto" — a reliability wrinkle distinct from
  the quality numbers above.
- **Sample size is small (n=4–5 measurable per model)** — directional, not
  conclusive. But it's a real reversal of the pattern from every other stage
  tested, worth treating as a genuine signal that cost is not the only axis
  a model can fail on here.

**Bottom line: no reason to consider a cheaper model for the settlement
verifier.** The stage's own docstring already says cost isn't a concern at
the rate it fires (33 pins in the system's entire history); this replay adds
the first quality evidence that the premium may specifically be buying
something here — not over-vetoing genuinely correct settlements — that the
cheaper candidates that looked competitive elsewhere do not reliably provide.

## Non-goals

- Not a replacement for `eval_extractor_adjacent_events.py`'s go/no-go gate —
  that measures false-settlement rate against an explicit threshold; this
  measures agreement-with-reference and cost.
- Not a prompt A/B (`docs/AB_HARNESS.md`) — the prompt was held fixed
  throughout; only the model varied.
- The scripts and raw per-case JSON this doc summarizes are not committed —
  they lived in a session scratch directory. Re-running this survey means
  rewriting the harness (S3 snapshot pull / daatan DB pull → replay per
  model → diff against a Haiku baseline), not resuming from saved state.

## Threshold routing in the batch lane (retro#688, 2026-08-28)

A third extractor knob, alongside the config default and the `oracle-api` drop-in:
**`threshold_extractor_model`** routes threshold-shaped *batch* events — the class where a
number decides the stance — to a different model. Off unless set. The live lane is untouched
either way (it passes its own `model=` per request and is already on Haiku 4.5).

### Where the seam turned out to be

The issue asked to route `claim_archetype = threshold` questions. **The batch lane has no
`claim_archetype`, and retro has no code that derives one.** The field belongs to the live
Oracul's `ForecastRequest`, and its own description says so: *"Temporal archetype from the
caller's claim classifier"* — daatan classifies its claims and sends the label over the wire.
Batch has no caller: `orchestrator.process_article` builds an `(event, source, article)` triple
from `data/events/*.json`, whose schema carries id / name / outcome / outcome_date /
search_keywords / llm_referee_criteria / predictive_window_days / category / tags. No
archetype — and no question either. Batch events are curated retroactive *event descriptions*
("Shekel drops below 4.0 NIS/USD"), not binary questions. `runner.py` already records the same
asymmetry for `enforce_deadline_arithmetic`: *"retroactive events don't pose a single binary
question with a direction the way a live /forecast request does."*

So `tm/archetype.py` supplies the missing input, deliberately as a **bool, not a four-way
classifier**. Reproducing daatan's scheduled/diffuse/threshold/none taxonomy in retro would
create a second classifier free to drift from the one that actually labels claims, with nothing
comparing them. The bool is used for model routing only — never written to a row, never compared
against a live `claim_archetype`. A misclassification costs money or quality on one extraction;
it cannot corrupt data.

### What it selects, measured on the real corpus

Threshold-shaped requires **both** a magnitude (digits carrying a unit, currency, percentage or
scale word) **and** a comparison/attainment cue (`exceeds`, `drops below`, `reaches`,
`raises … to`, `top 5`, a trailing `+`). Requiring both is what separates the class from
"contains a digit" — 18 of the 91 live events contain a digit that decides nothing.

**12 of 91 events = 13.2%**, confirming the issue's "the share is small, so the cost is bounded":

| | |
|---|---|
| **Routed** | Shekel drops below 4.0 NIS/USD · Bank of Israel raises interest rate to 4.75% · Israeli unemployment reaches 4.5%+ · protest movement reaches 100K+ weekly · Nvidia market cap exceeds $1 trillion · Israeli VC investment drops 50%+ YoY · Israeli AI startup raises $100M+ round · Israeli tech layoffs exceed 15,000 · Israel drops out of top 5 startup ecosystems · Brent crude exceeds $100/barrel · Europe reduces Russian gas dependency below 15% · Global oil price drops below $70/barrel |
| **Not routed, though they contain digits** | `$23B`/`$32B` Wiz deal sizes (the number names the deal, it does not decide it) · "resigns after 45 days" and "operational after 1 year" (durations) · "(6th government)" (ordinal) · "GPT-4", "DeepSeek R1" (names) · every bare year |

`Gemini Ultra surpasses GPT-4 on benchmarks` is the case that proves the two halves are ANDed:
it has a cue *and* a digit, and is still correctly excluded because `4` is not a magnitude.

### Two things that had to be fixed for this to be measurable at all

Both were latent, and both would have made the acceptance criteria silently unverifiable:

1. **Per-row provenance wrote `settings.extractor_model`, not the model that ran.** The global
   stops being the truth the moment any row is routed, so acceptance criterion 1 ("provenance
   shows Haiku on threshold rows") could never have been checked. `PipelineResult` now carries
   `extractor_model`, set on every return path including the failure ones.
2. **`_negative_marker_is_current` compared a cached row against the global.** A Haiku-produced
   marker judged against Nova is stale on *every* cycle, so the pipeline would re-extract it
   forever — on the expensive model, and only on the rows the routing exists to protect. It now
   takes the row's own effective model.

### Known limitation, not fixed here

The extraction cache key is `(article_hash, event_id, prompt_version)` — **the model is not in
it.** A row already extracted under Nova is reused as-is; routing only affects fresh extractions
and re-extractions. That is cost-preserving and deliberate. Forcing routed events to re-extract
would mean a one-off Haiku pass over the existing threshold corpus, which is a spend decision,
not an implementation detail. The same applies to the near-duplicate reuse path, which was
already model-agnostic before this change.

### Turning it on

Batch config comes from `~/truthmachine/.env` on the Oracul box, which `infra/ec2_run.sh` sources
into the environment. One line, then let the loop pick it up (it re-execs on change):

```
THRESHOLD_EXTRACTOR_MODEL=bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0
```

Empty or unset = off, which is what merging retro#688 ships. That default is deliberate: the
batch tree self-syncs to `origin/main` and re-execs every cycle, so a non-empty default would be
a live, unmeasured cost change within ~5 minutes of merge.
