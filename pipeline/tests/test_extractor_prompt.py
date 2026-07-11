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


def test_prompt_placeholders_still_format():
    # Guards against unescaped braces sneaking into future prompt edits.
    PROMPT.format(
        article_text="a",
        source_name="s",
        journalist="j",
        article_date="d",
        event_name="e",
        event_description="x",
    )
