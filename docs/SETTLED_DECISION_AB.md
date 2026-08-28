# The settled decision: what an isolated re-judgement measures (retro#691 / retro#697)

retro#691 established that **58% of `settled=true` claims in the evidence pool are
about a different event than the question** (215 ADJACENT of 371 decided pairs, over
52 questions), and that no downstream gate closes it: the shipped trio stops 7 of the
9 zero-support pins, `predicate_echo` adds nothing on top, and requiring the surviving
outlets to agree on one event stops all 9 only by killing 9 of 20 real pins.

That left one road: move the decision upstream, to where `settled` is assigned.
`scripts/ab_settled_decision.py` measures that road before anyone edits
`PROMPT_PREFIX` — which is the only responsible way to propose an edit to a
byte-identical cacheable prefix shared by every call system-wide.

## Setup

Re-decide `settled` for all 387 labelled pairs, one claim at a time, given the
question, deadline, claim and quote — the same blind inputs the labeller saw.

* **Model**: `claude-haiku-4-5`, the live extractor (`oracle-api.service.d/extractor-model.conf`,
  verified live on the box). Every one of these 387 rows was extracted by that same
  model through `/forecast`, so nothing here is confounded by Nova Lite.
* **Arm A (baseline)**: the `## SETTLED` and `## MATCH THE EVENT` sections sliced out
  of the live `PROMPT_PREFIX` **at runtime**. A control pasted into the script would
  drift the first time someone edits the prompt.
* **Arm B (retro#697)**: the same rules, with `### Buried facts` subordinated to
  MATCH THE EVENT instead of standing beside it, plus the WHO/WHAT/SCOPE
  decomposition emitted as output rather than left as an unobservable internal step.
  Two changes at once, deliberately — that is what retro#697 proposes — so a positive
  result would not attribute between them.

## Result 1 — the isolated judgement does not reproduce the failure

| run | ADJACENT kept settled | SETTLES kept settled |
|---|---|---|
| **production** | 215/215 (100%) | 156/156 (100%) |
| baseline | 39/215 (18%) | 131/156 (84%) |
| retro#697 arm | 24/215 (11%) | 99/156 (63%) |

Production settled every one of these claims. The **same model** reading the **same
rules** rejects 82% of the adjacent ones when simply asked the question directly.

So the adjacency failure is not a gap in the prompt's wording. The rules already say
what they need to say. What differs is that in production the settled judgement is
made inline — one of ~20 fields, emitted while generating the claim from a whole
article — rather than as a decision in its own right.

The founding retro#691 pool (13 settled claims across 7 outlets, all ADJACENT, spared
by all three shipped gates) is a clean example: the isolated baseline rejects **all
13**, leaving zero outlets standing.

## Result 2 — retro#697's rewrite moves no pins

| run | bad pins stopped | good kept | well-supported kept |
|---|---|---|---|
| baseline | 7/9 | 12/20 | 9/12 |
| retro#697 arm | 7/9 | 13/20 | 9/12 |

The rewrite drops 15 more adjacent claims and 32 more settling ones. At claim level
that is a bad trade (SETTLES recall 84% → 63%); at pin level it is not a trade at all,
because it stops the same pins. **Not worth shipping on this evidence.**

## Result 3 — do not add a per-claim settlement pass either

The obvious reading of Result 1 is "run the settled decision as its own call". Scored
against the shipped gates on the same 29 pinning pools:

| gate | bad pins stopped | good kept | well-supported kept |
|---|---|---|---|
| none (today) | 0/9 | 20/20 | 12/12 |
| **shipped trio** | **7/9** | **17/20** | **11/12** |
| isolated per-claim pass | 7/9 | 12/20 | 9/12 |
| trio + isolated pass | 8/9 | 9/20 | 8/12 |

The isolated pass alone is strictly worse than the trio: same benefit, five more good
pins lost. Combined, the one extra bad pin costs three more well-supported ones. The
shipped trio stands.

The single pin that survives every combination is an announcement-shaped pool — five
claims, all `facet=announcement`, all announcing a party ahead of an election that has
not happened. Only `gate_announcement_facet` catches it, and that gate is refuted:
it costs 10–11 of 29 pins (see `settlement_semantic.gate_announcement_facet`). 8/9 is
the practical ceiling and nothing on the table reaches it affordably.

## What this measurement is not

It is not the full extractor call. Prod stores no article text (only `snippet` and the
per-claim `quote`), so this re-decides `settled` from the isolated inputs above. A
**negative** result is therefore decisive — a rewrite that cannot separate adjacent
from settling claims even on the isolated judgement will not do better inside a longer
call — but a positive one would still have to go through `AB_HARNESS.md` on whole
articles before any prompt PR.

## Reproducing

```bash
cd api
uv run python scripts/ab_settled_decision.py \
    --candidates cand.jsonl --labels labels.jsonl --out decisions.jsonl
uv run python scripts/ab_settled_decision.py --score \
    --candidates cand.jsonl --labels labels.jsonl --out decisions.jsonl
uv run python scripts/ab_settled_decision.py --dump-arms   # exactly what each arm sent
```

`--candidates` is `scripts/settlement_backtest_export.sql`'s output flattened to one
row per settled claim; `--labels` is `scripts/label_settlement_candidates.py`'s output.
Both hold prod data and stay out of this repo. The run is resumable — every decision is
appended to `--out` as it lands and keyed by `(arm, model, claim)`.
