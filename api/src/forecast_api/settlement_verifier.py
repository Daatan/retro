"""The settlement match gate — does the settling fact ARE the claim's event?

**Enforcing since 2026-08-03.** ``settlement_verifier_enabled`` runs it and logs
the verdict; ``settlement_verifier_enforce`` — now on — lets that verdict demote
the settlement. It shipped shadow-first and was flipped once the replay over
every pin production had ever published scored 5/5 on the pins with a known
outcome and every veto on an active pin had been reviewed by hand
(``docs/ORACLE_VARIABLES.md`` §2.1). A veto is a **demotion, not a deletion**:
the vetoed rows keep voting as ordinary evidence.

Why it exists (retro#388, retro#360). Before publishing a ±0.94 pin the system
has to answer one question: *is the reported fact this claim's own event?* That
question has four slots — WHO, WHAT ACTION, WHAT ASPECT (done, or merely
announced / agreed / scheduled), WHAT SCOPE — and the extractor emits a
two-slot dyad, ``event_actors`` + ``event_target``. Both live incidents filled
those two slots **correctly** and were still wrong:

- **retro#388** — two rows carrying ``event_actors="United States"``,
  ``event_target="Ukraine"`` on the claim *"will the US grant Ukraine a licence
  to produce Patriot systems"*, both quoting Trump saying the US **will** grant
  it. Right who, right target, right action, wrong **aspect**: an announcement
  of a future act is a precursor to it. Published 97% against a pool of 44%.
- **retro#360** — articles reporting that the *other* team won, on a
  two-named-actor question. Right entities, wrong **role**.

So a gate keyed on actors/target intersection — the fix #388 originally
proposed — would have fired on neither. That was verified against the live rows
rather than assumed, and it is why this check is semantic rather than a field
comparison: one question that covers all four slots at once.

Two rules in the prompt exist because the first replay over production's own
pins found the gate missing without them — both found and fixed before this
shipped, which is what the shadow-first rollout is for:

- **Direction.** Asked only *"do these facts settle it?"*, the gate KEPT the
  France-World-Cup pin, and said exactly why: France's elimination settles the
  question — negatively — while the pin was publishing YES at 97%. Facts that
  decide a question *against* the answer being published are not proof of it,
  however clearly they decide it. The pin's direction is now part of what is
  asked.
- **The quote governs.** On retro#388's own incident the gate KEPT the pin,
  reasoning from the claim summary (*"The United States has agreed to grant
  Ukraine a license…"*) while the quote beneath it said *"Trump said the US
  **will** grant"*. The summary is a paraphrase produced by the same extraction
  step that may have misread the sentence; the quote is what the source said.

Cost is not a concern at the rate this fires: production has published **33**
pins in its entire history, so the call happens on a path that is rare by
construction, and never on the ordinary pooling path.

The end state is deterministic — per-claim aspect and role as extractor shadow
fields, enforced in code the way ``enforce_precursor_cap`` enforces F9's
precursor cap. This is the safety net that can be measured against known
ground truth today rather than after a month of accrual, and it is expected to
be *replaced* by those fields, not to live beside them forever.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Optional, Sequence

from tm.llm import complete_text_once

logger = logging.getLogger(__name__)


# Principle-first, no worked examples and no numerals — same house style as the
# extractor's own sections, and for the same reason: examples teach the shape of
# the example, and the failures this catches are all shaped differently.
PROMPT_PREFIX = """You check whether a claim has actually been settled.

You are given a QUESTION, the ANSWER a system is about to publish as already
determined, and the FACTS it is about to treat as proof. Your only job is to
decide whether those facts settle that question in that direction.

Decompose the question into who acts, what action or outcome, and within what
scope. The facts must match every part. A near miss on any one of them is
evidence about the question, but it is not the question's outcome.

Where a fact's summary and its quoted sentence disagree, believe the QUOTE. The
summary is a paraphrase made by the same reading that may have misread the
sentence; the quote is what the source actually said.

Answer NO when:
- the facts settle the question the OTHER way. Facts that decide the question
  against the answer being published are not proof of it, however clearly they
  decide it;
- a different party acts, or the action lands on a different target, even when
  the parties are the same ones the question names and only their roles are
  swapped;
- the action is similar but not the same one;
- the action has been announced, promised, agreed in principle, approved in
  advance, planned or scheduled, but not carried out. A statement by the very
  party whose act would settle the question is still only a statement, however
  senior the speaker, however definite the wording, and however precisely it is
  dated;
- the fact belongs to a different instance, timeframe or arena of a recurring
  event;
- the facts are consistent with the outcome without reporting it.

Answer YES only when the facts report the question's own outcome as something
that has happened.

Reply with one JSON object and nothing else:
{"settles": true or false, "reason": "one short sentence"}
"""


@dataclass(frozen=True)
class SettlementVote:
    """One settling claim, as the verifier sees it."""
    outlet: Optional[str]
    claim: str
    quote: Optional[str]
    event_date: Optional[str]


@dataclass(frozen=True)
class Verdict:
    settles: bool
    reason: str
    #: True when the model could not be reached or its answer could not be
    #: parsed. Fail-OPEN: an unavailable verifier must never suppress a pin by
    #: itself, or an LLM outage silently changes published numbers.
    errored: bool = False


def build_prompt(question: str, votes: Sequence[SettlementVote], *, answer: str = "YES") -> str:
    lines = [
        PROMPT_PREFIX, "",
        f"QUESTION: {question}",
        f"ANSWER ABOUT TO BE PUBLISHED: {answer}",
        "", "FACTS:",
    ]
    for v in votes:
        lines.append(f"- {v.claim}")
        if v.quote:
            lines.append(f'  reported as: "{v.quote}"')
        meta = ", ".join(
            part for part in (v.outlet, f"dated {v.event_date}" if v.event_date else None) if part
        )
        if meta:
            lines.append(f"  ({meta})")
    return "\n".join(lines)


def parse_verdict(text: str) -> Verdict:
    """Parse the model's reply, tolerating the usual wrappers.

    Anything unparseable is an *error*, not a NO — see ``Verdict.errored``.
    """
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not match:
        return Verdict(settles=True, reason="unparseable verifier reply", errored=True)
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return Verdict(settles=True, reason="unparseable verifier reply", errored=True)
    settles = data.get("settles")
    if not isinstance(settles, bool):
        return Verdict(settles=True, reason="verifier reply missing a boolean verdict", errored=True)
    reason = str(data.get("reason") or "")[:300]
    return Verdict(settles=settles, reason=reason)


async def verify_settlement(
    question: str,
    votes: Sequence[SettlementVote],
    *,
    model: str,
    timeout_s: int,
    answer: str = "YES",
) -> Verdict:
    """Ask whether ``votes`` settle ``question``. Never raises.

    Fail-open on every failure path: no model, no answer, bad answer, timeout —
    all return ``settles=True, errored=True``. A pin that the pooling rules
    already justified must not be suppressed because an LLM was unavailable.
    """
    if not votes:
        return Verdict(settles=True, reason="no settling votes to check", errored=True)
    try:
        raw = await complete_text_once(
            model,
            build_prompt(question, votes, answer=answer),
            max_tokens=200,
            timeout=timeout_s,
        )
    except Exception as exc:  # noqa: BLE001 - fail-open is the point
        logger.warning("event=settlement_verifier_error err=%r", exc)
        return Verdict(settles=True, reason=f"verifier call failed: {exc!r}"[:300], errored=True)
    return parse_verdict(raw)
