"""Candidate deterministic gates on the settlement pin — retro#691, analysis only.

**Nothing here is wired into the pipeline.** These are pure functions so the
backtest harness (``scripts/backtest_settlement_semantic.py``) can score them
against the settlement verifier's own recorded verdicts before anyone proposes
enforcing one.

The problem they target: every deterministic settlement guard we ship is
*temporal* — ``enforce_settlement_event_date`` (is it dated? dated before the
article?), ``settlement_vote_validity`` (after ``claim_created_at``? before the
deadline?). None asks whether the settled fact **is the claim's own event**. That
question is answered only by ``settlement_verifier``, one LLM call that fails
open, and which is the sole objector on 106 of the 250 enforced blocks in the
production log.

``enforce_winner_entity_consistency`` is the shape to copy: it reads the
extractor's own ``event_actors``/``event_target``/``is_occurrence`` facets,
compares them to the question, and strips ``settled`` when they disagree. It
just cannot fire here — it obtains the claim-side operand by regex
(``_match_versus_question``, "X beats Y"), and a role/office claim has no such
shape.

Which is the real gap: the article side of the dyad is elicited on every row;
**the claim has no stored dyad to compare it against.** The intended production
source for that is the per-question classifier (``predictions.classifier_output``
already exists and runs once per forecast). Until it emits one,
:func:`claim_subject_from_question` derives a PROXY from the question text so the
gates can be measured today. Every number this module produces through that proxy
is a lower bound on what a real classifier-supplied subject would achieve — and
the proxy's failure modes are the documented ones of ``_extract_named_entities``
(it cannot tell an actor-shaped span from a topic-shaped one, retro#644).
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Iterable, Optional, Sequence

# The stem matcher and entity heuristics already exist and are shared with the
# enforcing guard — reuse them rather than growing a second dialect of "does this
# string name that entity".
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "pipeline" / "src"))
from tm.extractor import (  # noqa: E402
    _extract_actor_shaped_entities,
    _extract_named_entities,
    _mentions_entity_stem,
)

#: Claim shapes whose outcome is a *state re-evaluated at the deadline* rather
#: than an irreversible event. "Netanyahu will be PM ON December 31 2026" is true
#: or false only on that date; nothing in July can settle it. An elimination, a
#: signing, or a death is the opposite — irreversible, and legitimately settled
#: mid-window.
POINT_IN_TIME = "point_in_time_state"
IRREVERSIBLE = "irreversible_event"

_ON_DATE = re.compile(
    r"\bon\s+(?:january|february|march|april|may|june|july|august|september|"
    r"october|november|december)\s+\d{1,2},?\s+\d{4}\b",
    re.I,
)
_REMAIN = re.compile(r"\b(remain|still be|continue to (?:be|serve)|hold(?:s)? office)\b", re.I)

#: Words that carry no discriminating power when echo-matching a claim's event
#: nouns — every political claim contains them.
_ECHO_STOPWORDS = frozenset({
    "will", "the", "and", "for", "with", "from", "that", "this", "any", "all",
    "least", "than", "least", "more", "less", "over", "under", "into", "onto",
    "before", "after", "during", "until", "least", "part", "new", "next",
    "first", "second", "third", "year", "years", "date", "time", "times",
})
_ECHO_MIN_LEN = 5


@dataclass(frozen=True)
class SettlementCandidate:
    """One settled claim as the extractor emitted it, flattened from
    ``claims_detail``. Field names mirror the stored JSON keys."""

    claim: str
    stance: float
    certainty: float
    outlet: Optional[str] = None
    event_actors: Optional[str] = None
    event_target: Optional[str] = None
    event_date: Optional[str] = None
    is_occurrence: Optional[bool] = None
    facet: Optional[str] = None
    evidence_class: Optional[str] = None


@dataclass(frozen=True)
class ClaimSubject:
    """The claim side of the dyad — what a classifier should be storing.

    ``actors`` are the named entities the claim is about; ``event_terms`` are the
    content words naming the event itself (the office, the contest, the action).
    ``outcome_kind`` decides whether a mid-window fact may settle at all.
    """

    actors: tuple[str, ...] = ()
    event_terms: tuple[str, ...] = ()
    outcome_kind: str = IRREVERSIBLE
    proxy: bool = True


def _content_terms(question: str) -> tuple[str, ...]:
    terms = []
    for word in re.findall(r"[A-Za-z][A-Za-z'-]+", question):
        low = word.lower()
        if len(low) < _ECHO_MIN_LEN or low in _ECHO_STOPWORDS:
            continue
        if low not in terms:
            terms.append(low)
    return tuple(terms)


def claim_subject_from_question(question: str) -> ClaimSubject:
    """PROXY claim subject derived from the question text.

    Stand-in for the classifier field this gate actually wants
    (``claim_actor``/``claim_predicate``/``claim_target``/``outcome_kind`` on
    ``predictions``). Good enough to measure the gates' shape; not good enough to
    enforce on. ``proxy=True`` is carried through so the harness can never report
    a proxy-derived number as a production one.
    """
    actors = tuple(_extract_actor_shaped_entities(question) or _extract_named_entities(question))
    kind = POINT_IN_TIME if (_ON_DATE.search(question) or _REMAIN.search(question)) else IRREVERSIBLE
    # The actors' own words must not count as event terms, or "an article that
    # mentions Netanyahu" would satisfy "names the office and the election".
    actor_words = {w.lower() for a in actors for w in re.findall(r"[A-Za-z']+", a)}
    terms = tuple(t for t in _content_terms(question) if t not in actor_words)
    return ClaimSubject(actors=actors, event_terms=terms, outcome_kind=kind, proxy=True)


# ── gates ────────────────────────────────────────────────────────────────────
# Each returns a demotion reason, or None to leave the candidate alone. All fail
# OPEN on absent metadata, matching every sibling in extractor.py.

def gate_occurrence_consistency(subject: ClaimSubject, cand: SettlementCandidate,
                                *, deadline: Optional[date] = None) -> Optional[str]:
    """``settled`` and ``is_occurrence=false`` contradict each other.

    Measured tiny in prod (2 grade-passing rows) — recorded so it is not mistaken
    for a fix. An unjudged ``None`` is left alone.
    """
    return "settled_but_not_occurrence" if cand.is_occurrence is False else None


def gate_announcement_facet(subject: ClaimSubject, cand: SettlementCandidate,
                            *, deadline: Optional[date] = None) -> Optional[str]:
    """An announcement settles the announcement, never the outcome it describes.

    ``enforce_decider_intent_stance_cap`` already caps this shape in the stance
    lane but explicitly leaves ``settled`` alone, and fails open whenever the
    model marked ``is_occurrence=true`` — which is exactly when it matters.
    """
    return "settled_on_announcement" if cand.facet in ("announcement", "denial") else None


def gate_predicate_echo(subject: ClaimSubject, cand: SettlementCandidate,
                        *, deadline: Optional[date] = None) -> Optional[str]:
    """Does the settled fact name the claim's own event?

    Requires the candidate's dyad or claim text to echo (a) at least one actor the
    claim names AND (b) at least one of the claim's event terms. "Yair Netanyahu
    left Florida" echoes the actor and no event term; "Likud primaries final
    results" echoes neither the office nor the general election.

    Fails open when the claim side is empty (nothing to compare) or the candidate
    carries no dyad and no claim text.
    """
    if not subject.actors or not subject.event_terms:
        return None
    haystack = " ".join(x for x in (cand.event_actors, cand.event_target, cand.claim) if x)
    if not haystack.strip():
        return None
    if not any(_mentions_entity_stem(haystack, a) for a in subject.actors):
        return "settlement_actor_mismatch"
    low = haystack.lower()
    if not any(re.search(rf"\b{re.escape(t)}\w*", low) for t in subject.event_terms):
        return "settlement_event_mismatch"
    return None


def gate_point_in_time(subject: ClaimSubject, cand: SettlementCandidate,
                       *, deadline: Optional[date] = None) -> Optional[str]:
    """A reversible state evaluated at a date cannot be settled before that date.

    Deliberately narrow: only fires when the claim shape is
    ``point_in_time_state``. An ARRIVAL claim about an irreversible result
    ("wins the 2026 election") is legitimately settled mid-window and is
    :func:`gate_predicate_echo`'s problem, not this one.
    """
    if subject.outcome_kind != POINT_IN_TIME or deadline is None:
        return None
    event = _parse_date(cand.event_date)
    if event is None or event >= deadline:
        return None
    return "settlement_before_evaluation_date"


ALL_GATES = {
    "predicate_echo": gate_predicate_echo,
    "point_in_time": gate_point_in_time,
    "announcement_facet": gate_announcement_facet,
    "occurrence_consistency": gate_occurrence_consistency,
}


def _parse_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


@dataclass
class GateOutcome:
    kept: list[SettlementCandidate] = field(default_factory=list)
    demoted: list[tuple[SettlementCandidate, str]] = field(default_factory=list)

    @property
    def distinct_outlets(self) -> int:
        return len({c.outlet for c in self.kept if c.outlet})


def apply_gates(
    subject: ClaimSubject,
    candidates: Sequence[SettlementCandidate],
    *,
    gates: Iterable[str],
    deadline: Optional[str] = None,
) -> GateOutcome:
    """Run the named gates over ``candidates``. First gate to fire wins."""
    parsed = _parse_date(deadline)
    out = GateOutcome()
    chosen = [(name, ALL_GATES[name]) for name in gates]
    for cand in candidates:
        reason = None
        for name, fn in chosen:
            reason = fn(subject, cand, deadline=parsed)
            if reason:
                break
        if reason:
            out.demoted.append((cand, reason))
        else:
            out.kept.append(cand)
    return out


def pin_survives(outcome: GateOutcome, *, min_sources: int) -> bool:
    """Whether a pin would still fire after the gates demoted what they demoted.

    Mirrors ``settlement_min_sources``: the count is over *independent sources*,
    so several claims from one outlet cannot carry a pin between them.
    """
    return outcome.distinct_outlets >= min_sources
