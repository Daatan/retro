"""Subject card + subject gate (retro#805, slice 1 of the verified-article-card design).

The pool problem this answers, measured 2026-09-07 on prod (retro#545 phase 0): 38 of 310
strong, usable pool rows in 14 gated election forecasts came from articles that never name
the forecast's subject at all — Hendel 46%, Milwidsky/Silman 100%. The gatekeeper (Nova
Micro) passes "actor's party / coalition" as indirect evidence by design, so a topic-adjacent
article reaches the extractor, and Haiku 4.5 then rewrites the article's subject into the
question's subject (W3 15/15) and votes ±1.0. No stage checks whether the subject is even
present (Daatan/docs funnel.md §2.1); this is that check.

Three parts, two of them here:

1. **Article card** — the extractor now emits ``ExtractionOutput.article_card``: the actors
   the ARTICLE names, as verbatim spans, which ``tm.extractor.verify_article_card`` checks
   against the article text and prunes. A verified span is *extracted* in the glossary sense
   — checkable — which the elicited ``event_actors``/``claim_actor`` never were.
2. **Subject card** (this module) — who the QUESTION is about, derived ONCE per question
   from ``event_name`` + ``resolution_criteria`` with multilingual surface forms, and cached
   (``subject_card_store``). Option B of the 2026-09-07 dilemma: retro-only, nothing in
   daatan's schema, so it can ship and be measured in one repo; a daatan-side curated field
   is the deferred Option A.
3. **The gate** — ``evaluate_subject_gate``: the article must name AT LEAST ONE subject actor
   ("any listed actor present": a party-only mention satisfies a party+person card; the
   Gantz/Hendel rules name no party, so a stricter rule would drop legitimate party coverage).
   Two routes with different trust:
     - ``surface_form`` — deterministic: a subject surface form found in the normalised
       article text (``text_contains_alias``). Cross-script by construction: the card
       carries Hebrew/Russian/Arabic spellings, the article is matched in its own script.
     - ``gloss`` — a verified span whose model-given ``name_en`` stem-matches a subject
       actor. The gloss is NOT verified against anything, and W3 is exactly Haiku
       relabelling the wrong person as the subject, so this route is logged separately;
       whether it may clear the gate alone in enforce mode is a shadow-period decision
       (``settings.subject_gate_trust_gloss``).
   Both routes miss for every actor → ``fired``. Fail-open everywhere: no subject card, an
   empty one, no article card, or a card with zero verified spans → the gate is skipped and
   says so (``matched_via="skipped"``), never fired. Enforcement (drop the article with
   outcome ``subject_absent``) is the forecaster's call behind ``subject_gate_enforce``,
   shadow-logged first, per the issue's acceptance criteria.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

from tm.extractor import (
    _mentions_entity_stem,
    normalize_for_match,
    text_contains_alias,
    verify_article_card,
)
from tm.llm import complete_structured

logger = logging.getLogger(__name__)

#: Part of the cache key — bump when PROMPT or the SubjectCard schema changes meaning.
SUBJECT_CARD_PROMPT_VERSION = "v1"

_MAX_ACTORS = 6
_MAX_FORMS_PER_ACTOR = 20


class SubjectActor(BaseModel):
    name_en: str = Field(description="The actor's name in English.")
    type: Literal["person", "party", "company", "country", "institution", "other"] = Field(
        description="What kind of actor.",
    )
    surface_forms: list[str] = Field(
        default_factory=list,
        description="Every way a news article might write this name: full name, surname "
                    "alone, transliteration and spelling variants, acronyms, short names — "
                    "in English and in each requested language.",
    )


class SubjectCard(BaseModel):
    actors: list[SubjectActor] = Field(
        default_factory=list,
        description="The actors the question is ABOUT. Empty when the question names none.",
    )

    @model_validator(mode="after")
    def _tidy(self) -> "SubjectCard":
        kept: list[SubjectActor] = []
        for actor in self.actors:
            name = (actor.name_en or "").strip()
            if not name:
                continue
            forms: list[str] = []
            seen: set[str] = set()
            for form in [name, *actor.surface_forms]:
                f = (form or "").strip()
                key = normalize_for_match(f)
                if not key or key in seen:
                    continue
                seen.add(key)
                forms.append(f)
            actor.name_en = name
            actor.surface_forms = forms[:_MAX_FORMS_PER_ACTOR]
            kept.append(actor)
        self.actors = kept[:_MAX_ACTORS]
        return self


PROMPT = """You are given a forecasting question and its resolution criteria. List the SUBJECT \
actors: the people, parties, companies, countries or institutions the question is ABOUT — \
those whose own action or state resolves it.

Rules:
- A bystander, an opponent, or the arena is not a subject: a country where an election is \
held is not a subject unless the question is about that country's own act; a rival party \
named only as context is not a subject.
- When the question is about a person's party, or the criteria name the party the person \
leads or runs with, list BOTH the person and the party as separate actors.
- Return an empty list when the question names no specific actor (e.g. "the next prime \
minister serves less than a full term").

For each actor give name_en, type, and surface_forms: every way a news article might write \
the name, in English and in each of these languages: {languages}. Include the full name, the \
surname alone, common transliteration and spelling variants (for Hebrew, both plene and \
defective spellings), standard acronyms and short names. 4 to 12 forms per actor. Never list \
a generic word on its own (party, minister, government, list, coalition).

QUESTION: {question}
{criteria_line}"""


async def derive_subject_card(
    question: str,
    resolution_criteria: Optional[str],
    *,
    model: str,
    languages: str,
    timeout_s: int = 20,
) -> Optional[SubjectCard]:
    """Ask once for a question's subject card. Never raises — fails open to None (the
    caller's cue to skip the gate) on any error or timeout, like ``decompose_event``.
    An empty ``SubjectCard`` (question names nobody) is a real, cacheable answer, not a
    failure."""
    if not question:
        return None
    criteria_line = f"RESOLUTION CRITERIA: {resolution_criteria}" if resolution_criteria else ""
    prompt = PROMPT.format(question=question, criteria_line=criteria_line, languages=languages)
    try:
        card, _usage = await complete_structured(
            model, SubjectCard, prompt, max_tokens=900, timeout=timeout_s,
        )
    except Exception as exc:  # noqa: BLE001 - fail-open is the point
        logger.warning("event=subject_card_error err=%r", exc)
        return None
    return card


@dataclass
class SubjectGateResult:
    evaluated: bool
    fired: bool
    matched_via: str  # surface_form | gloss | none | skipped
    matched_actor: Optional[str] = None
    matched_form: Optional[str] = None
    skip_reason: Optional[str] = None
    verified_spans: list[str] = field(default_factory=list)
    dropped_spans: list[str] = field(default_factory=list)
    bears_on_question: Optional[bool] = None
    subjects: list[str] = field(default_factory=list)


def _skipped(reason: str, **kw) -> SubjectGateResult:
    return SubjectGateResult(evaluated=False, fired=False, matched_via="skipped", skip_reason=reason, **kw)


def evaluate_subject_gate(
    article_card,
    article_text: str,
    subject_card: Optional[SubjectCard],
    *,
    trust_gloss: bool = False,
) -> SubjectGateResult:
    """Pure: no I/O, no logging beyond the span-verification lines. See the module doc for
    the routes and the fail-open contract. ``trust_gloss=False`` means a gloss-only match is
    reported (``matched_via="gloss"``) but the gate still counts as fired — the shadow
    period's number for "how often is the gloss the only thing clearing it"."""
    if subject_card is None:
        return _skipped("no_subject_card")
    subjects = [a.name_en for a in subject_card.actors]
    if not subjects:
        return _skipped("subject_card_empty")
    if article_card is None:
        return _skipped("no_article_card", subjects=subjects)
    verified, dropped = verify_article_card(article_card, article_text)
    spans = [a.span for a in verified.named_actors]
    base = dict(
        verified_spans=spans, dropped_spans=dropped, subjects=subjects,
        bears_on_question=verified.bears_on_question,
    )
    if not spans:
        return _skipped("no_verified_spans", **base)

    norm_text = normalize_for_match(article_text)
    for actor in subject_card.actors:
        for form in actor.surface_forms:
            if text_contains_alias(norm_text, form):
                return SubjectGateResult(
                    evaluated=True, fired=False, matched_via="surface_form",
                    matched_actor=actor.name_en, matched_form=form, **base,
                )
    for span_actor in verified.named_actors:
        gloss = span_actor.name_en
        if not gloss:
            continue
        for actor in subject_card.actors:
            if _gloss_matches(gloss, actor):
                return SubjectGateResult(
                    evaluated=True, fired=not trust_gloss, matched_via="gloss",
                    matched_actor=actor.name_en, matched_form=gloss, **base,
                )
    return SubjectGateResult(evaluated=True, fired=True, matched_via="none", **base)


def _gloss_matches(gloss: str, actor: SubjectActor) -> bool:
    ng = normalize_for_match(gloss)
    if any(normalize_for_match(f) == ng for f in actor.surface_forms):
        return True
    return _mentions_entity_stem(gloss, actor.name_en) or _mentions_entity_stem(actor.name_en, gloss)
