"""Prompt-content invariants for the extractor.

The adjacent-events section was A/B-sampled against the 2026-07-11 Illouz/Likud
false-settlement incident (article about MKs leaving Likud scored settled=true
for "a party withdraws from the race"). Measured on the incident article at
temperature 0, n=10: nova-lite produces a settlement-grade claim 10/10 without
the section and 8/10 with it — hardening, not a full fix (the reliable lever is
a stronger extractor model; see docs/ORACLE_VARIABLES.md). nova-lite is
sensitive even to whitespace changes in this section, so keep its text stable
and re-run the A/B before rewording.
"""

from tm.extractor import PROMPT


def test_adjacent_events_section_present():
    assert "## THE EVENT ITSELF vs. ADJACENT EVENTS" in PROMPT
    assert "it is NEVER settled and never carries the full +-1.0" in PROMPT
    assert "could a fact-checker cite this article alone" in PROMPT


def test_adjacent_events_examples_present():
    assert "a member leaving a party is not a party leaving the race" in PROMPT
    assert "leadership change is not a market exit" in PROMPT


def test_single_winner_contest_section_present():
    """The stance-inversion class: "Spain beat France" / "Argentina stun England"
    extracted as +1 settled FOR "France/England will win" (6 prod rows,
    2026-07-16 audit). The prompt previously had no rule mapping a rival's win
    to a negative settlement for the subject."""
    assert "## Single-winner contests" in PROMPT
    assert "it settles the related event NEGATIVELY" in PROMPT
    assert "never read the excitement of a decisive result as support for" in PROMPT


def test_single_winner_contest_examples_present():
    assert "Spain beat France 2-0 in Tuesday's semi-final" in PROMPT
    assert "Argentina stun England with a late rally" in PROMPT
    assert "a non-terminal loss" in PROMPT


def test_negated_events_section_present():
    """The negated-claim sign-inversion class (2026-07-19 pool audit): a Kyiv
    Post escalation op-ed scored stance −0.529 on "a ceasefire will NOT be
    implemented" — the extractor scored the inner event (ceasefire happens)
    and left the negation to the reader; its own extracted claims supported
    the claim as written."""
    assert "## Negated events — score the claim AS WRITTEN" in PROMPT
    assert "never score the inner event and leave the negation to the reader" in PROMPT


def test_negated_events_examples_present():
    assert "ceasefire will NOT be implemented" in PROMPT
    assert 'escalation SUPPORTS "no ceasefire"' in PROMPT
    assert "the negated claim is settled FALSE" in PROMPT


def test_capability_vs_occurrence_section_present():
    """The capability-as-occurrence class (2026-07-19 audit of 44 prod evidence
    rows): one identical claim — "Ukraine has demonstrated the capability to
    destroy major bridges using upgraded drones..." — appeared on 30 forecast_match
    rows from 9 DIFFERENT articles (airfields, refineries, the Crimea power grid,
    troop supply routes; none about the Kerch Bridge) at avg stance +0.50,
    relevance 0.74, against "Ukraine will successfully strike the Kerch Bridge by
    August 6, 2026", which sat at 97%. The prompt had no rule separating "can do it
    / did it elsewhere" from "will do it to THIS target by THIS date" — the strings
    "capab" and "intent" appeared nowhere in it — and "INFER the implication"
    actively invited it.

    The |stance| <= 0.3 cap here is deliberately TIGHTER than the adjacent-events
    section's <= 0.5. Do not harmonize them."""
    assert "## Capability and intent are not occurrence" in PROMPT
    assert "a PRECONDITION of the related event, never the event itself" in PROMPT
    assert "|stance| <= 0.3, certainty <= 0.4) and is NEVER settled" in PROMPT
    assert "never let a capability, an intent, or a success against another target stand" in PROMPT


def test_capability_vs_occurrence_examples_present():
    assert "Ukraine has demonstrated the capability to destroy major bridges" in PROMPT
    assert "a different target — the skill is shared, the event is not" in PROMPT
    assert "stated intent, not an occurrence" in PROMPT
    assert "a capability milestone, not a commercial launch" in PROMPT


def test_capability_companion_clauses_present():
    """Two clauses that bound the sections which otherwise argue the other way:
    "INFER the implication" (What counts as a signal) told the model to do exactly
    what the capability section forbids, and "Buried facts" tells it to hoist any
    past-tense clause and mark it settled — which is the shape of "Ukraine HAS
    DEMONSTRATED the capability...". Buried facts sits later in the prompt, so
    without its clause recency may favour it."""
    assert "INFER the implication — but infer only the implication" in PROMPT
    assert "the report's own subject and target carry" in PROMPT
    assert "report of THIS event — see the capability section above" in PROMPT


def test_foreclosing_negative_date_rule_present():
    """Negative settlements are now dated by the foreclosing event (needed by
    aggregation-time revalidation); the old rule said to leave them undated."""
    assert "is dated by the FORECLOSING event" in PROMPT
    assert "leave event_date empty" in PROMPT


def test_prompt_placeholders_still_format():
    # Guards against unescaped braces sneaking into future prompt edits.
    PROMPT.format(
        article_text="a",
        source_name="s",
        journalist="j",
        article_date="d",
        event_name="e",
        event_description="x",
        claim_deadline="2026-07-15",
    )
