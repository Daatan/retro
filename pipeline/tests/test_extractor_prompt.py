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
