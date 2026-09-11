"""The premise verifier — is this question still live? (retro#575, slice 1)

**Shadow/log-only.** Nothing here changes a response. The verdict is logged
(``event=premise_verifier``); acting on it — turning ``dead=True`` into a new
``insufficient_data`` reason — is a follow-up once real trigger/precision
data comes back, the same shadow-first rollout retro#545 slice (ii) (PR #586)
and the Gate-0 evidence-window shadow (PR #558) used.

Why it exists. The pool prices whatever evidence the topical search returns,
but nothing ever asks whether the question itself is still open. A premise
that already resolved usually has *no* fresh coverage — once something
settles, news moves on — so the topical search either returns stale
pre-resolution articles that read as live, or nothing at all (falling
through to the generic ``no_search_results`` reason, which reads as
"couldn't find evidence," not "the premise itself is dead"). Either way the
caller gets a confident number, or an unhelpfully generic abstain, on a
question that was never live to begin with.

This reuses ``settlement_verifier``'s shape end to end: same
``tm.llm.complete_text_once`` call, same frozen ``Verdict`` dataclass, same
principle-first prompt style (no worked examples — the failures this is
meant to catch are all shaped differently), and the same fail-open
discipline — an unavailable or unparseable verifier must never itself claim
a premise is dead.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional, Sequence

from tm.llm import complete_text_once

logger = logging.getLogger(__name__)


PROMPT_PREFIX = """You check whether a forecasting question is still open.

You are given a QUESTION (optionally with its DEADLINE, and — when a
DEADLINE is present — a TODAY line stating today's date and whether that
deadline has already passed) and a list of RESULTS — recent search results
on the topic. Trust the TODAY line's stated fact about the deadline rather
than inferring it yourself from the deadline text or from RESULTS' dates —
it is computed, not a guess. Your only job is to decide whether the
question's premise is already dead: either the event it asks about has
already happened (or definitively not happened) as an accomplished fact, or
the question has become structurally impossible to resolve as asked (the
body/position it depends on no longer exists, the vote already occurred, the
deadline has passed with nothing having happened that would still let it
occur).

Answer NO (the premise is still open) when:
- the results only report the event as announced, planned, scheduled, or
  expected, not carried out;
- the results are silent on the outcome — no evidence either way is not
  evidence the premise is dead;
- the results discuss a similar but different instance of a recurring event,
  or a different scope than the question asks about;
- you are not citing a specific result for the claim.

Answer YES only when a specific result reports the question's own outcome as
an accomplished fact, or makes the premise structurally impossible to still
occur. Name which result you're relying on.

Reply with one JSON object and nothing else:
{"dead": true or false, "reason": "one short sentence", "citation": "which result, or null"}
"""


@dataclass(frozen=True)
class PremiseResult:
    """One search result, as the verifier sees it — title/snippet/date only,
    no fetch: this reuses whatever the topical search already returned."""
    title: Optional[str]
    snippet: Optional[str]
    published_date: Optional[str]
    source: Optional[str]


@dataclass(frozen=True)
class Verdict:
    dead: bool
    reason: str
    #: True when the model could not be reached or its answer could not be
    #: parsed. Fail-OPEN: an unavailable verifier must never claim a premise
    #: is dead by itself.
    errored: bool = False


def _parse_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value[:10]).date()
    except (ValueError, TypeError):
        return None


def _deadline_fact(claim_deadline: Optional[str], today: date) -> Optional[str]:
    """A deterministic "has this deadline passed" fact, computed here rather
    than left for the model to infer from the raw deadline string — the
    inference itself was the cause of 4/12 false positives in retro#601's
    adjudicated sample (retro#817)."""
    deadline = _parse_date(claim_deadline)
    if deadline is None:
        return None
    delta = (deadline - today).days
    if delta < 0:
        return f"TODAY: {today.isoformat()} (the deadline has already passed, {-delta} day(s) ago)"
    if delta == 0:
        return f"TODAY: {today.isoformat()} (the deadline is today)"
    return f"TODAY: {today.isoformat()} (the deadline has NOT passed yet, {delta} day(s) remain)"


_STALE_RESULT_DAYS = 180


def _is_stale(published_date: Optional[str], claim_deadline: Optional[str], today: date) -> bool:
    """Whether a result predates the question's own target window by enough
    that it can only be describing a different, earlier occurrence.

    A prompt bullet asking the model to discount an old result was tried
    first and measured to have no effect — the model's own training-data
    recall of a real past event outweighed an in-prompt instruction to
    disregard it. Filtering the result out before it ever reaches the model
    removes the confounding evidence instead of asking the model to
    disregard evidence it can already see (retro#817 pattern 3).

    Measured relative to ``min(claim_deadline, today)``: for a past deadline,
    that's the deadline itself, so a result published right at a long-past
    deadline isn't flagged stale — it's exactly the evidence a re-check of an
    old deadline needs. For a future deadline, using the deadline directly
    would flag anything published today as "stale" relative to a still-distant
    target date, so the reference is capped at today instead.
    """
    published = _parse_date(published_date)
    if published is None:
        return False
    deadline = _parse_date(claim_deadline)
    reference = min(deadline, today) if deadline else today
    return (reference - published).days > _STALE_RESULT_DAYS


def premise_check_triggered(
    claim_deadline: Optional[str],
    claim_archetype: Optional[str],
    *,
    today: Optional[str] = None,
) -> bool:
    """Whether this request is worth the extra LLM call.

    Every ``/forecast`` call reaches this point, so firing unconditionally
    would double LLM cost on every request. Scope to the population where a
    dead premise is actually plausible: a ``scheduled``/``threshold``
    archetype (elections, court dates — exactly the shape retro#575's own
    examples are), or a ``claim_deadline`` that has already passed. Missing
    metadata (older callers that don't classify claims) never triggers —
    additive and fail-open, same framing ``ForecastRequest.claim_archetype``
    already promises.
    """
    if claim_archetype in ("scheduled", "threshold"):
        return True
    deadline = _parse_date(claim_deadline)
    if deadline is None:
        return False
    ref = _parse_date(today) or datetime.now().date()
    return deadline <= ref


def build_prompt(
    question: str,
    claim_deadline: Optional[str],
    results: Sequence[PremiseResult],
    *,
    today: Optional[str] = None,
) -> str:
    lines = [PROMPT_PREFIX, "", f"QUESTION: {question}"]
    if claim_deadline:
        lines.append(f"DEADLINE: {claim_deadline}")
        fact = _deadline_fact(claim_deadline, _parse_date(today) or datetime.now().date())
        if fact:
            lines.append(fact)
    lines.append("")
    lines.append("RESULTS:")
    for r in results:
        lines.append(f"- {r.title or '(no title)'}")
        if r.snippet:
            lines.append(f'  "{r.snippet}"')
        meta = ", ".join(
            part for part in (r.source, f"published {r.published_date}" if r.published_date else None) if part
        )
        if meta:
            lines.append(f"  ({meta})")
    return "\n".join(lines)


def parse_verdict(text: str) -> Verdict:
    """Parse the model's reply, tolerating the usual wrappers.

    Anything unparseable is an *error*, not a live/dead call — see
    ``Verdict.errored``.
    """
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not match:
        return Verdict(dead=False, reason="unparseable verifier reply", errored=True)
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return Verdict(dead=False, reason="unparseable verifier reply", errored=True)
    dead = data.get("dead")
    if not isinstance(dead, bool):
        return Verdict(dead=False, reason="verifier reply missing a boolean verdict", errored=True)
    reason = str(data.get("reason") or "")[:300]
    citation = data.get("citation")
    if citation:
        reason = f"{reason} (cites: {str(citation)[:120]})"
    return Verdict(dead=dead, reason=reason)


async def verify_premise(
    question: str,
    claim_deadline: Optional[str],
    results: Sequence[PremiseResult],
    *,
    model: str,
    timeout_s: int,
    today: Optional[str] = None,
) -> Verdict:
    """Ask whether ``question``'s premise is already dead. Never raises.

    Fail-open on every failure path: no results, a timeout, a bad or
    unparseable reply — all return ``dead=False, errored=True``. A question
    the pool would otherwise price normally must not be flagged dead because
    an LLM was unavailable.

    ``today`` is test-only (mirrors ``premise_check_triggered``'s own
    parameter) — production callers never pass it, so the prompt's stated
    "TODAY" fact is always the real date.
    """
    if not results:
        return Verdict(dead=False, reason="no results to check", errored=True)
    ref_today = _parse_date(today) or datetime.now().date()
    fresh_results = [r for r in results if not _is_stale(r.published_date, claim_deadline, ref_today)]
    if len(fresh_results) < len(results):
        logger.debug(
            "event=premise_verifier_stale_filtered dropped=%d kept=%d",
            len(results) - len(fresh_results), len(fresh_results),
        )
    if not fresh_results:
        return Verdict(dead=False, reason="all results too old to evidence this question's premise", errored=True)
    try:
        raw = await complete_text_once(
            model,
            build_prompt(question, claim_deadline, fresh_results, today=today),
            max_tokens=200,
            timeout=timeout_s,
            temperature=0,
        )
    except Exception as exc:  # noqa: BLE001 - fail-open is the point
        logger.warning("event=premise_verifier_error err=%r", exc)
        return Verdict(dead=False, reason=f"verifier call failed: {exc!r}"[:300], errored=True)
    return parse_verdict(raw)
