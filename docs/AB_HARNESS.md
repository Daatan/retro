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

`expect` names the facets a correct extraction must get right — see
`_FACET_READERS` in `ab_harness.py` for the current set (`stance_sign`,
`fact_signal_sign`, `fact_signal_null`, `is_occurrence`, `verified`,
`settled`, `evidence_class`). A facet is satisfied if **any** prediction, in
**any** run, matches the expected value — matching how one correct signal
among several extracted predictions (or one correct run among several,
given LLM non-determinism) is enough.

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
