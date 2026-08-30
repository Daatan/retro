"""WHO/WHAT/SCOPE decomposition of a question, for injection into the extractor's
event_description input (retro#758).

Why input, not output: retro#697 tried emitting this same decomposition as an
extractor *output* field and it moved no pins (``docs/SETTLED_DECISION_AB.md``)
— asking the model to describe its own reasoning after the fact didn't change the
reasoning. retro#758's control run instead handed the untouched v10 prompt a
once-per-question decomposition already appended to the RELATED EVENT
description, i.e. as an input the MATCH THE EVENT step can read directly rather
than derive. That moved the weak rater's sentinel case from 11/15 to 15/15 with
zero prompt or schema change.

One call per question (not per article), via ``event_decomposition_store`` — see
that module's docstring for the caching rationale.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from tm.llm import complete_text_once

logger = logging.getLogger(__name__)

PROMPT = """Decompose the forecasting question below into WHO / WHAT / SCOPE — \
the same three slots a careful reader checks before deciding whether a news \
article is reporting on this exact event, or merely something adjacent to it.

- WHO: the actor(s) the question is about — not a bystander who might be \
mentioned near the event.
- WHAT: the action or state itself, in the form the question asks about — an \
announcement or a proposal of the action is a DIFFERENT what than the action \
having happened.
- SCOPE: the deadline, threshold or quantity the question sets, stated as \
given. Use "none stated" if the question sets none.

Answer with exactly one line, this shape and nothing else:
WHO: <actor(s)>. WHAT: <action or state>. SCOPE: <deadline/threshold, or "none stated">.

QUESTION: {question}
{criteria_line}"""

_LINE_RE = re.compile(r"WHO:.*?SCOPE:[^\n]*", re.DOTALL)

#: Keep the appended line short — this is a compact identity check for the
#: extractor to read, not a restatement of the whole question.
_MAX_CHARS = 400


def parse_decomposition(text: str) -> Optional[str]:
    """Extract the WHO/…/SCOPE line from the model's reply. Returns None (never
    raises) for anything unparseable — the caller treats that exactly like a
    call failure: skip injection, run byte-identical to today."""
    match = _LINE_RE.search(text or "")
    if not match:
        return None
    line = " ".join(match.group(0).split())  # collapse embedded newlines/whitespace
    return line[:_MAX_CHARS] or None


async def decompose_event(
    question: str,
    resolution_criteria: Optional[str],
    *,
    model: str,
    timeout_s: int = 15,
) -> Optional[str]:
    """Ask once for a question's WHO/WHAT/SCOPE decomposition. Never raises —
    fails open to None (the caller's cue to leave event_description unchanged)
    on any error, timeout, or unparseable reply, exactly like
    ``settlement_verifier.verify_settlement``'s fail-open contract."""
    if not question:
        return None
    criteria_line = f"RESOLUTION CRITERIA: {resolution_criteria}" if resolution_criteria else ""
    prompt = PROMPT.format(question=question, criteria_line=criteria_line)
    try:
        raw = await complete_text_once(model, prompt, max_tokens=150, timeout=timeout_s, temperature=0)
    except Exception as exc:  # noqa: BLE001 - fail-open is the point
        logger.warning("event=event_decomposition_error err=%r", exc)
        return None
    return parse_decomposition(raw)
