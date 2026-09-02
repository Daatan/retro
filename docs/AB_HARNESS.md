# Extractor prompt A/B harness (retro#470)

Every extractor prompt edit changes behavior on every prediction, live. Two
edits shipped this way already — PR#309 and PR#314, both by hand: a fixed
case sample, run against the live model before and after the edit, diffed on
the facets that matter, zero-regression required. `eval_extractor_adjacent_events.py`
formalized one narrow slice of that (false-settlement rate across models).
This harness formalizes the general shape, so the next extractor prompt edit
doesn't start from a blank page — starting with retro#352 (deadline-direction
blindness) and retro#353 (resolution rules never reach the live extractor),
both of which were blocked on this.

Code: `pipeline/src/tm/ab_harness.py` (pure, unit-tested, no network) +
`pipeline/scripts/ab_extractor_prompt.py` (the live-model driver — **not** a
CI test, it calls Bedrock).

## Running it

Baseline and patched are two different git checkouts, so this is two runs
plus one comparison:

```bash
cd pipeline

# 1. On the baseline checkout (e.g. origin/main):
AWS_REGION=us-east-1 .venv/bin/python scripts/ab_extractor_prompt.py \
    run scripts/ab_cases/deadline_and_resolution_rules.json \
    --out /tmp/baseline.json --label baseline

# 2. On your patched branch:
AWS_REGION=us-east-1 .venv/bin/python scripts/ab_extractor_prompt.py \
    run scripts/ab_cases/deadline_and_resolution_rules.json \
    --out /tmp/patched.json --label patched

# 3. Compare (pure, no network, either checkout):
.venv/bin/python scripts/ab_extractor_prompt.py compare /tmp/baseline.json /tmp/patched.json
```

The exit code of `compare` **is** the zero-regression gate: `0` means no
in-scope case lost a facet it used to satisfy. Improvements and no-change
cases print but never fail the gate — only regressions do.

### The same-prompt/same-model control arm

retro#353's fix (pass the real resolution rules as `event_description`
instead of the bare question) doesn't touch the prompt text at all — the bug
is entirely in what `forecaster.py` passes as an argument. For a case like
that, add a `control_event_description` and run with
`--use-control-description`:

```bash
.venv/bin/python scripts/ab_extractor_prompt.py \
    run scripts/ab_cases/deadline_and_resolution_rules.json \
    --out /tmp/control.json --label control --use-control-description

.venv/bin/python scripts/ab_extractor_prompt.py compare /tmp/baseline.json /tmp/control.json
```

Both runs use the exact same checkout, model, and prompt text — the only
variable is the input content, isolating the effect to exactly the thing
retro#353 changes. Every case's `event_description` should already be the
correct, patched-behavior text (this is what the live extractor gets today);
`control_event_description` holds the alternative you're comparing against.

### Temporal-leakage cases

A case whose `article_date` postdates its own `claim_deadline` is flagged
`[LEAKAGE]` and excluded from the gate by default — AVeriTeC prior art
(cited in retro#352): hindsight evidence can make a patched prompt look
better than it actually is at forecast time. Pass `--allow-leakage` to
include such cases in the gate anyway.

## How lenient the gate is — read this before trusting a pass

`unmet_facets` satisfies a facet if **any** prediction in **any** run matches. A model that gets
a case right once in five runs passes it exactly like a model that gets it right every time.

That is the right default for a *prompt* A/B — the question there is whether an edit destroyed a
capability — but it flatters an unstable model when the variable is the model. retro#664's run
hit this directly: on `threshold-near-boundary-satisfied` (8.99% against a 9% ceiling) Haiku
alternates `+0.60` and `-0.20` run to run and passes on the strength of the positive runs, while
Nova Lite sits at `0.00/-0.10` and fails. The gate's verdict — "the candidate lost a case the
baseline held" — is true and useful, but "the baseline held it" means "held it sometimes".

When the model is the variable, read the per-run values alongside the gate, and treat
`eval_extractor_stability.py`'s sign-flip rate as the companion statistic.

### A single 5-run FAIL is a prompt to re-measure, not a result (retro#757)

`complete_structured` runs at `temperature=0`, but Bedrock's residual non-determinism at zero is
measured, not theoretical (retro#532: "6 of the 13 questions the gate ever saw more than once
returned BOTH verdicts on an unchanged vote-set"). While gating retro#697, `RUNS_PER_CASE = 5`
returned four different verdicts on one case across prompt variants that turned out to be
indistinguishable — two independent 15-run samples of the *same unchanged baseline* gave 13/15
and then 8/15. Every diagnosis made from a 5-run reading in that PR was wrong; every one made
from 15 runs with the baseline re-measured alongside held. #683 found the same thing independently
("Any FAIL gets investigated at higher `--runs` before it is called noise") without it being
written down here, so the next prompt PR repeated the ~6 hours of chasing sampling noise.

Two mechanical guards, neither of which changes what the gate fails on:

- A case with measured history of this (a `volatile: true` field in its corpus JSON) is always
  run at `VOLATILE_MIN_RUNS` (15) regardless of `--runs-per-case`, in `ab_extractor_prompt.py`'s
  `run` command. The four cases marked so far: `threshold-at-or-below-satisfied`,
  `threshold-tone-negative-number-satisfies`, `decider-denial-regression-sentinel`,
  `stance-tone-conflation-hazard-persists-control`. Add the tag to any case a re-measurement
  later confirms is unstable at the default count.
- `compare` prints `LOW CONFIDENCE` beside any `REGRESSION` line measured at fewer than
  `CONFIDENT_MIN_RUNS` (15) — informational only, so the exit code is unchanged and a CI-style
  gate still fails the same way it always did. Treat that line as the prompt to re-run the
  affected case at `--runs-per-case 15` (or higher) with the baseline alongside it before
  concluding the regression is real.

### The strict counterpart: `no_predictions` (retro#775)

Every reader in `_FACET_READERS` reads a value off *a* prediction, so any of them can only ever
be satisfied by something being extracted. That makes "the correct extraction is nothing at all"
inexpressible: with zero predictions, every named facet is unmet by construction, so a baseline
that wrongly extracts something and a patched prompt that correctly extracts nothing look
identical — both `unmet`, reported as `no change` instead of `improved`, and a future regression
back to over-extraction would look like `no change` too and pass the gate silently.

`expect: {"no_predictions": true}` is the deliberate exception to the any-run leniency above: it
uses the opposite reduce, met only if **every** run produced zero predictions — one stray
extraction in one run unmets it. Use it alone (or alongside facets that only make sense if the
case is later found still extracting something, e.g. while iterating on a fix); it does not
combine meaningfully with a facet that requires a prediction to exist.

### An arm that never ran is not a clean arm

`run` catches every per-case exception so one bad case can't abort a sweep. The cost of that is
an arm where *every* call failed — an expired API key, revoked model access, the wrong region —
still writing a well-formed results file.

Scoring such a file is worse than useless. A dead arm meets no facets, so a dead **baseline**
cannot be regressed against: the gate passed and printed every case as `improved … fixed [...]`,
rendering a total outage as a *win*. (A dead `patched` failed honestly; both dead printed
`no change`.) The arm nobody re-checks is usually the baseline, which is the one that fails
silently.

Both ends now refuse instead:

- `run` exits **1** when any case produced 0 usable runs, and names them. The file is still
  written — the exceptions inside it are the diagnosis — but the exit code refuses to call the
  arm comparable.
- `compare` exits **2** (distinct from the gate's **1**) if either side has a case with 0 usable
  runs, naming the arm. Results files get reused and re-compared long after `run`'s exit code is
  gone, so the check has to live at the gate too.

Read those exit codes directly, not through a pipe — `cmd | tail` reports `tail`'s status, not
`cmd`'s.

## What the gate cannot tell you: article-level fields (retro#686)

`unmet_facets` scores per-**prediction** facets. Nothing on `ExtractionOutput`
itself is in `_FACET_READERS` and nothing can be — `author_lean`,
`author_lean_certainty` and `consensus_view` are one value per article, not per
claim, so a gate built on "did any prediction match" has no place to put them.

For most of this harness's life that meant an arm's results file recorded
nothing about them at all: you could add an article-level field, ship it, and
have no way to ask this harness whether the model ever filled it. `author_lean`
was invisible here from the day it landed.

`run` now records them per run alongside the predictions (`article_runs` in the
results file) and prints a fill summary at the end of the arm:

```
Article-level fill over 105 run(s):
  author_lean               61/105 ( 58%)
  author_lean_certainty     61/105 ( 58%)
  consensus_view            32/105 ( 30%)  expects_yes=19, divided=8, expects_no=5

Tokens: 1683402 over 105 call(s) (16032/call)
```

Read it as a **fill rate, not a gate**. It says whether the model answers and
how the answers distribute — the two questions a shadow field is harvested to
settle, and the two the zero-regression gate is silent on. A field that comes
back >90% one value has collapsed and is measuring nothing, which is a finding;
it just isn't one `compare` can express. `usage_runs` carries the per-call token
usage behind the totals line, so a prompt edit's cost shows up next to its
effect.

## The other thing the gate cannot tell you: `quantity` vs the threshold (retro#683)

`quantity` is not in `_FACET_READERS` either, and deliberately not: it is a
shadow field whose validator (exact match on (value, unit, comparator) over 100
hand-labelled claims, accuracy ≥ 0.9 **per rater**) has not been run. Gating
every future prompt PR on it would make it a requirement before it is a
measurement.

What `run` prints instead is a diagnostic, over the cases that carry a
`question_threshold` — the bar the QUESTION sets, hand-declared on the case
rather than parsed, because the live parse (`question_quantity`) is Oracle 1.5
Phase 2 and a second unmeasured parser here would just be something for Phase 2
to disagree with:

```
Quantity vs question threshold — rater: patched
  case                                              target       fill      exact   code=stance   code ok  stance ok
  threshold-at-or-below-satisfied                 5/5  100%    5/5  100%   5/5  100%   5/5  100%      5         5
  threshold-between-bounds-contradicted           4/5   80%    5/6   83%   4/5   80%   1/4   25%      4         1
  ...
  TOTAL target 44/50 (88%), of which right-number-wrong-comparator 5; fill 48/55 (87%), exact 41/48 (85%), code agrees with stance 31/44 (70%), code abstained 4; correct: code 44, stance 33, of 55 prediction(s)
```

**`target` is the validator's own number; `exact` is not — read `target` first.**
`target` is per RUN and asks the validator's question: did this rater get THIS
case's number, unit and comparator right. `exact` is per PREDICTION, so a run
that extracts the target correctly *and* correctly reports a second figure from
the same article ("the other party took 24 seats") scores 1/2 on `exact` and 1/1
on `target`. That is not a rounding difference. On the first two-rater run it
inverted the ranking outright — Haiku scored `exact` 70% against Nova Lite's
78% while getting every single case right, purely because Haiku quotes more
generously. `target_miscomparated` splits the remainder the way the issue cares
about: right value and unit under the wrong comparator is a different defect
from not finding the number at all.

Read the last two numbers together. **`code ok` against `stance ok` is the
finding**, not a pass/fail: the issue expects agreement to be high on Haiku and
low on Nova Lite, and the gap is the argument for moving the comparison out of
the model's head. A row where the model extracted the right number and still
scored the stance the wrong way is retro#664 reproduced in one line.

### What the first two-rater run found (prompt v9, 2026-08-30)

| rater | `target` | comparator wrong | verdict vs the ≥0.9 bar |
|---|---|---|---|
| Haiku 4.5 (`us.anthropic.claude-haiku-4-5`) | 50/50 (100%) | 0 | passes |
| Nova Lite (`us.amazon.nova-lite-v1:0`) | 120/150 (80%) | 27 (18%) | fails |

Every Nova Lite failure is `comparator`, and always the same substitution: a
verb of movement becomes a bound. "inflation accelerated to 4.1 percent" comes
back `> 4.1` in 15 runs of 15; "support collapsed to just 35 percent" comes back
`< 35`. The value and the unit are right in all of them. That is retro#664 one
level down — tone displacing the number, moved from `stance` into `comparator` —
and it is worse than a miss, because `< 35` against "at least 30 percent"
*contradicts* where the stated level satisfies.

Two prompt edits written against exactly this (v9b, v9c — an explicit
movement-verb rule, using verbs the corpus does not use) moved `target` 80% →
72% and `fill` 86% → 74%. Both were reverted. Record it before reaching for a
third: the corpus says this rater is not reachable from the prompt here, and the
per-rater gate is the answer the issue already prescribed.

Run a second rater with `--model`; an arm is one model, so its report *is* that
rater's row:

```bash
AWS_REGION=us-east-1 uv run python scripts/ab_extractor_prompt.py run \
  scripts/ab_cases/numeric_threshold_blindness.json --runs 15 \
  --model bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0 --label haiku
```

`code abstained` is its own column for a reason. The comparison answers by
interval containment, so a bounded report ("stayed below 5%") is decidable
whenever every value it allows lands on one side, and abstains when it straddles
the bar or when the units do not match. An abstention is not a wrong answer and
must not be counted as one — and it is a different failure from an empty
`quantity`, which is why `fill` is separate from it.

`expect_quantity` on a case is the validator's other half: what a correct reader
should have extracted from THIS article, i.e. the "PR#671 known values". The
comparison is pinned in `tests/test_threshold_compare.py` to get all ten
retro#664 cases right from those values — the arithmetic was never the hard part.

## The corpus

One file per defect the corpus was built to catch. A prompt edit should run
every file whose subject it could plausibly touch, not just the one it targets —
retro#720 was found because a numeric-threshold case moved when an unrelated
bracket section was cut.

| file | n | issue | what it holds down |
|---|---|---|---|
| `deadline_and_resolution_rules.json` | 5 | retro#351/#352/#353 | a fact that defers the event past the claim deadline bears *against* it; plus the decider-statement sentinel and a control-arm reference case |
| `numeric_threshold_blindness.json` | 10 | retro#664 | a number decides the stance, not the sentence's tone — at-or-below, strictly-above, between-bounds, and the exact/near boundaries; every case also carries a `question_threshold` + `expect_quantity` for the retro#683 diagnostic above |
| `poll_facet_neither.json` | 4 | retro#541 | a poll or seat projection is `facet: neither`, not an announcement or a denial |
| `stance_tone_conflation.json` | 2 | retro#545 | an alarmed tone about a hazard is not evidence the hazard occurred |
| `multi_stage_brackets.json` | 5 | retro#720 | winning one stage of a bracket, series, runoff or staged approval is weak support for winning the whole thing |

### What the bracket file measured on `main` (2026-08-29, 5 runs/case)

Committed as a baseline before the section it covers is touched, since a corpus
with no multi-stage cases scores deleting that section as a clean win.

- **Haiku 4.5** (the live extractor): **5/5 pass**. It follows the section
  closely enough to reproduce its worked examples — a round-of-16 favourite
  reads +0.30/0.30 against the prompt's stated +0.3/0.3, five runs out of five
  with no variance.
- **Nova Lite** (the batch extractor): **3/5 pass**, both failures on magnitude
  alone. The three it passes — a runoff first-round lead, an advisory-panel
  recommendation, a 3–1 series lead — are the situations with a near-verbatim
  worked example in the section, or none of its business. The two it fails are
  the ones governed only by the section's *prose*: a knockout-tie favourite
  reads +0.90/0.80 where the prompt says +0.3/0.3, and reaching the final reads
  +0.90 where the prompt says +0.6.

Read together: on the weaker rater the section's **worked examples are doing
essentially all of the work and its prose almost none** — the same mechanism
retro#720 identified, seen from the other side. Anything that rewrites those
examples should re-measure both raters, because the two lanes are not failing
the same way.

## Adding a case

Cases live in JSON files under `pipeline/scripts/ab_cases/`. Each case:

```json
{
  "id": "unique-slug",
  "event_name": "...",
  "event_description": "...",
  "claim_deadline": "2026-09-05",
  "article_date": "2026-08-20",
  "article_text": "...",
  "expect": {"fact_signal_sign": -1, "is_occurrence": false},
  "control_event_description": null,
  "tags": ["retro-352"]
}
```

Two optional keys are diagnostic-only and never reach the gate (retro#683):
`question_threshold` — `{"comparator": "<="|"<"|">="|">"|"="|"between", "value":
9, "unit": "percent", "value_hi": null}`, the bar the question sets — and
`expect_quantity`, the `{value, unit, comparator}` a correct reader should have
extracted from the article. Add both on a case where a number decides the event;
omit both otherwise, and the case is simply left out of the quantity report.

`expect` names the facets a correct extraction must get right — see
`_FACET_READERS` in `ab_harness.py` for the current set (`stance_sign`,
`stance_band`, `claim_strength_band`, `fact_signal_sign`, `fact_signal_null`,
`is_occurrence`, `facet`, `verified`, `settled`, `evidence_class`). A facet is
satisfied if **any** prediction, in **any** run, matches the expected value —
matching how one correct signal among several extracted predictions (or one
correct run among several, given LLM non-determinism) is enough. The one
exception is `no_predictions` (see "The strict counterpart" above), which
uses an all-runs reduce instead — it is not in `_FACET_READERS` because it
has no per-prediction value to read.

### Direction vs strength (retro#720)

`stance_sign` answers *which way*, which is all most of this corpus needs. But
some prompt sections exist to control *how strongly*, and for those a sign
reader is blind: every worked example in `## Multi-stage / bracket events` is
positive (+0.2 … +0.6), so deleting that section outright cannot move
`stance_sign` on a single bracket case. Scored on sign alone, the deletion
reports as a clean pass while tournaments, playoff series and multi-round
elections quietly start reading as near-certainties.

`stance_band` and `claim_strength_band` are the magnitude readers, deliberately
coarse — `none` (<0.15), `weak` (<0.5), `moderate` (<0.8), `strong`. Four
buckets rather than a threshold predicate because `unmet_facets` compares with
`==`; the boundaries are read off the section's own numbers so `weak` vs
`moderate` splits where the prompt tells the model to split. They are magnitude
only, so a case can assert sign and strength independently. Being coarse, they
report a real improvement that stays inside one bucket as `no change` — read
the per-run values alongside, as with any other facet here.

### Target facets — an expectation `main` does not yet meet

Ordinarily a case must pass on `main` before it is committed; one that already
fails proves nothing about a later edit. The exception is a facet where the
prompt states a rule the model demonstrably does not follow. Such a facet can
only ever report as `improved` (`regressions` is `baseline_met & patched_unmet`,
and it was never in `baseline_met`), so it cannot gate a merge falsely — it just
makes the gap visible instead of invisible. Two of the bracket cases carry one
on Nova Lite, tagged `rater-split`.

Only assert one where the target value is defensible from the prompt's own text.
Where the right magnitude is genuinely arguable, leave the facet out — or use
`expect: {}` — rather than freezing a premature numeric call into the corpus.

**`expect: {}` is a legitimate pattern** — a reference case with no automated
assertion, included so a genuinely disputed or nuanced scenario (e.g. "how
negative should this be, exactly?") is on hand to eyeball manually rather
than forcing a premature numeric call into the case file. It never
regresses and never gates.

Follow the house de-naming convention (`Country R`, `Group M`, `Company X` —
see `eval_extractor_adjacent_events.py` and `tests/test_extractor_prompt.py`)
for synthetic cases; real article bodies that reproduced a live incident
(PR#314's approach) can be pulled in at run time instead of committed
verbatim.

## Reading the output

```
  improved   deadline-direction-fact-after-deadline: fixed ['fact_signal_sign']
  pass       deadline-direction-fact-before-deadline-control
  REGRESSION decider-denial-regression-sentinel: lost ['fact_signal_sign', 'is_occurrence']
  no change  some-other-case: still failing ['stance_sign']

Gate: FAIL (1 in-scope regression(s), leakage cases excluded)
```

- **REGRESSION** — baseline satisfied this facet, patched no longer does.
  This is what blocks a merge.
- **improved** — baseline failed, patched now passes.
- **no change** — still failing the same facets on both sides (informational).
- **pass** — every named facet satisfied on both sides.

## Non-goals

Not the backtest harness over the replayed claim layer (retro#403) — that
one answers "does the estimator score well over history" and is blocked on
corpus/outcome overlap until ~Dec 2026. This is a prompt-diff harness: does
*this specific edit* change *these specific facets* in the intended
direction, with no regressions, before it ships.
