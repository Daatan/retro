"""Prompt-content invariants for the extractor.

The adjacent-events rule was A/B-sampled against the 2026-07-11 Illouz/Likud
false-settlement incident (article about MKs leaving Likud scored settled=true
for "a party withdraws from the race"). Measured on the incident article at
temperature 0, n=10: nova-lite produces a settlement-grade claim 10/10 without
the rule and 8/10 with it — hardening, not a full fix (the reliable lever is
a stronger extractor model; see docs/ORACLE_VARIABLES.md). nova-lite is
sensitive even to whitespace changes in this rule, so keep its text stable
and re-run the A/B before rewording (see eval_extractor_adjacent_events.py).

retro#300 (2026-07-27): the adjacent-events, wrong-belligerent, and
date-is-not-a-match sections were consolidated into one "## MATCH THE EVENT"
block (they were three variations on "don't credit a near-miss as the event",
scattered across the prompt) — every asserted string below is preserved
verbatim from the pre-consolidation sections, re-run through
eval_extractor_adjacent_events.py before/after to confirm no regression.
Process-evidence, Capability/intent, and Multi-stage were deliberately left
OUT of the consolidation: they sit earlier in the prompt and reordering them
risks flipping the recency relationship with "Buried facts" (see
test_capability_companion_clauses_present) that retro#279/A-date-is-not-a-match
were built to protect.
"""

from tm.extractor import PROMPT_PREFIX, PROMPT_SUFFIX

# PROMPT was split into PROMPT_PREFIX (fixed instructions, cacheable, no .format()
# placeholders) + PROMPT_SUFFIX (article/variable fields + output-format spec) so
# llm.py::complete_structured can mark PROMPT_PREFIX as a Bedrock/Anthropic cache
# breakpoint. All the content this file asserts on lives in PROMPT_PREFIX_PREFIX; the
# placeholder-formatting test below now exercises PROMPT_SUFFIX instead.


def test_match_the_event_master_section_present():
    """retro#300: the unifying frame for the three near-miss rules below."""
    assert "## MATCH THE EVENT — do not credit a near-miss as the event" in PROMPT_PREFIX
    assert "WITHIN WHAT SCOPE (threshold, deadline, arena)" in PROMPT_PREFIX


def test_adjacent_events_section_present():
    assert "### A different subject type, action, or arena is ADJACENT evidence" in PROMPT_PREFIX
    assert "it is NEVER settled and never carries the full +-1.0" in PROMPT_PREFIX
    assert "could a fact-checker cite this article alone" in PROMPT_PREFIX


def test_adjacent_events_examples_present():
    assert "a member leaving a party is not a party leaving the race" in PROMPT_PREFIX
    assert "leadership change is not a market exit" in PROMPT_PREFIX


def test_wrong_belligerent_section_present():
    assert "### A named-actor claim needs the NAMED actor and target, not just the same conflict" in PROMPT_PREFIX
    assert "Check the actor and target BY NAME" in PROMPT_PREFIX
    assert "This is NEVER settled" in PROMPT_PREFIX


def test_wrong_belligerent_examples_present():
    assert "Two US soldiers were killed in an Iranian attack on a base in Jordan" in PROMPT_PREFIX
    assert "the US and Jordan, not Israel" in PROMPT_PREFIX
    assert "IRGC missiles struck US targets in Kuwait and Bahrain overnight" in PROMPT_PREFIX
    assert "matching the claim exactly" in PROMPT_PREFIX


def test_single_winner_contest_section_present():
    """The stance-inversion class: "Spain beat France" / "Argentina stun England"
    extracted as +1 settled FOR "France/England will win" (6 prod rows,
    2026-07-16 audit). The prompt previously had no rule mapping a rival's win
    to a negative settlement for the subject."""
    assert "## Single-winner contests" in PROMPT_PREFIX
    assert "it settles the related event NEGATIVELY" in PROMPT_PREFIX
    assert "never read the excitement of a decisive result as support for" in PROMPT_PREFIX


def test_single_winner_contest_examples_present():
    assert "Spain beat France 2-0 in Tuesday's semi-final" in PROMPT_PREFIX
    assert "Argentina stun England with a late rally" in PROMPT_PREFIX
    assert "a non-terminal loss" in PROMPT_PREFIX


def test_unverified_interested_party_section_present():
    """retro#299 (2026-07-19/20 audit): an IRGC damage claim ("claims to have destroyed
    85 U.S. military targets in Bahrain") scored certainty 0.7-grade despite being an
    unconfirmed wartime claim by a belligerent — sign was right, magnitude was not.
    Mirrors the VERIFIED vs CLAIMED test already in FACT_SIGNAL, but applied to the
    live-facing `certainty` field rather than the shadow fact_signal facet."""
    assert "## Unverified claims by an interested party — cap certainty" in PROMPT_PREFIX
    assert "carries certainty no higher than 0.5" in PROMPT_PREFIX
    assert "UNLESS the article ALSO reports independent confirmation" in PROMPT_PREFIX


def test_unverified_interested_party_examples_present():
    assert "Iran's Islamic Revolutionary Guard Corps claims to have destroyed 85 U.S. military" in PROMPT_PREFIX
    assert "an interested party's own unconfirmed damage claim" in PROMPT_PREFIX
    assert "Satellite imagery confirms extensive damage" in PROMPT_PREFIX


def test_negated_events_section_present():
    """The negated-claim sign-inversion class (2026-07-19 pool audit): a Kyiv
    Post escalation op-ed scored stance −0.529 on "a ceasefire will NOT be
    implemented" — the extractor scored the inner event (ceasefire happens)
    and left the negation to the reader; its own extracted claims supported
    the claim as written."""
    assert "## Negated events — score the claim AS WRITTEN" in PROMPT_PREFIX
    assert "never score the inner event and leave the negation to the reader" in PROMPT_PREFIX


def test_negated_events_examples_present():
    assert "ceasefire will NOT be implemented" in PROMPT_PREFIX
    assert 'escalation SUPPORTS "no ceasefire"' in PROMPT_PREFIX
    assert "the negated claim is settled FALSE" in PROMPT_PREFIX


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
    assert "## Capability and intent are not occurrence" in PROMPT_PREFIX
    assert "a PRECONDITION of the related event, never the event itself" in PROMPT_PREFIX
    assert "|stance| <= 0.3, certainty <= 0.4) and is NEVER settled" in PROMPT_PREFIX
    assert "never let a capability, an intent, or a success against another target stand" in PROMPT_PREFIX


def test_capability_vs_occurrence_examples_present():
    """Examples are deliberately de-named (Force F / Bridge K / Company X), matching
    the house convention used by "Candidate A wins contest C" and "Company X exits the
    European market". Measured on Haiku 4.5, 10 cases x n=3: de-naming scored 20/30 vs
    23/30 for the original named version — the entire gap being the intent/vow example
    (stance +0.30 -> +0.60, 3/3 -> 0/3), which replicated across two variants. Shipped
    anyway for maintainability; if intent cases regress in a pool audit, restoring a
    written-out named vow example is the first thing to try. See PR #302."""
    assert "Force F has demonstrated the capability to destroy major bridges" in PROMPT_PREFIX
    assert "Kerch" not in PROMPT_PREFIX
    assert "a different target — the skill is shared, the event is not" in PROMPT_PREFIX
    assert "stated intent, not an occurrence" in PROMPT_PREFIX
    assert "a capability milestone, not a commercial launch" in PROMPT_PREFIX


def test_capability_aggregation_cap_section_present():
    """retro#304: the existing cap bound when an article read as *about* preparation but
    not when it was topically on-point and every extracted claim was nevertheless
    intent/expectation — the model treated many accumulating intent signals as an
    aggregate occurrence signal ("lots of smoke"), e.g. stance 0.6-0.7 instead of the
    documented |0.3| cap. This section makes the cap explicitly per-claim, independent
    of the article's overall framing or how many similar signals it contains."""
    assert "## The capability/intent cap applies PER CLAIM, not to the article's overall urgency" in PROMPT_PREFIX
    assert "their number or density does not aggregate into occurrence" in PROMPT_PREFIX
    assert "is not itself a signal that qualifies for a higher cap" in PROMPT_PREFIX


def test_capability_companion_clauses_present():
    """Two clauses that bound the sections which otherwise argue the other way:
    "INFER the implication" (What counts as a signal) told the model to do exactly
    what the capability section forbids, and "Buried facts" tells it to hoist any
    past-tense clause and mark it settled — which is the shape of "Ukraine HAS
    DEMONSTRATED the capability...". Buried facts sits later in the prompt, so
    without its clause recency may favour it."""
    assert "INFER the implication — but infer only the implication" in PROMPT_PREFIX
    assert "the report's own subject and target carry" in PROMPT_PREFIX
    assert "report of THIS event — see the capability section above" in PROMPT_PREFIX


def test_foreclosing_negative_date_rule_present():
    """Negative settlements are now dated by the foreclosing event (needed by
    aggregation-time revalidation); the old rule said to leave them undated."""
    assert "is dated by the FORECLOSING event" in PROMPT_PREFIX
    assert "leave event_date empty" in PROMPT_PREFIX


def test_date_is_not_a_match_section_present():
    """retro#279: a dated fact about a DIFFERENT (adjacent) event was passing
    the DATES section's event_date requirement and settling — the date floor
    was being treated as sufficient on its own, without re-checking adjacency
    first."""
    assert "### A date does not excuse a near-miss" in PROMPT_PREFIX
    assert "never a substitute for it" in PROMPT_PREFIX
    assert "a precisely dated adjacent fact is still adjacent, never settled" in PROMPT_PREFIX


def test_author_lean_section_present():
    """author_lean captures the BYLINE author's own forecast for later author-accuracy
    scoring — a separate concern from the event estimate (evidence pool quality work,
    2026-07-21). It is deliberately kept out of the estimate; the section stresses that
    a quoted third party's view is not the byline's."""
    assert "## AUTHOR_LEAN" in PROMPT_PREFIX
    assert "does NOT feed the event estimate" in PROMPT_PREFIX
    assert "is that person's position, not the byline's" in PROMPT_PREFIX
    assert "treat it as the source's and return null" in PROMPT_PREFIX
    # Concordant multiple quoted sources must not leak into author_lean — the n=10 A/B (2026-07-21)
    # showed a stack of agreeing third-party forecasts wrongly scored the byline author +0.60.
    assert "merely stacks concordant quoted forecasts has author_lean null" in PROMPT_PREFIX
    # Sentiment-vs-forecast separation — the 2026-07-24 wild analysis found evaluative op-eds
    # (e.g. a critical piece on an inevitable US-Saudi nuclear deal) leaked a NEGATIVE author_lean
    # while their own extracted claims affirmed the event; disapproval must not flip the sign.
    assert "not whether they welcome it" in PROMPT_PREFIX
    assert "Approval or alarm about an outcome is sentiment" in PROMPT_PREFIX


def test_author_lean_in_output_contract():
    assert '"author_lean"' in PROMPT_SUFFIX
    assert '"author_lean_certainty"' in PROMPT_SUFFIX


def test_fact_signal_section_present():
    """fact_signal isolates the fact-lane from the fused stance for a future estimator
    (Phase 2 of the author-scoring redesign, 2026-07-21). The section must carry the three
    disciplining tests — dyad, occurrence-vs-precursor, verified-vs-claimed — that the
    evidence_class-only shortcut was shown NOT to cover."""
    assert "## FACT_SIGNAL" in PROMPT_PREFIX
    assert "REPORTED FACTS on their own establish about the related event" in PROMPT_PREFIX
    assert "A precursor is capped at |0.3|" in PROMPT_PREFIX
    assert "is context only" in PROMPT_PREFIX
    assert "A claimed-but-unverified event is down-weighted" in PROMPT_PREFIX


def test_fact_signal_in_output_contract():
    assert "fact_signal" in PROMPT_SUFFIX
    assert "event_actors" in PROMPT_SUFFIX
    assert "event_target" in PROMPT_SUFFIX
    assert "is_occurrence" in PROMPT_SUFFIX
    assert "verified" in PROMPT_SUFFIX


def test_prompt_placeholders_still_format():
    # Guards against unescaped braces sneaking into future prompt edits.
    PROMPT_SUFFIX.format(
        article_text="a",
        source_name="s",
        journalist="j",
        article_date="d",
        event_name="e",
        event_description="x",
        claim_deadline="2026-07-15",
    )


def test_decider_statements_exception_present():
    """Signal Lanes WS5: a decider's on-record statement is an intent-fact, not opinion.

    Fixes the Medvedev-denial class: the fact lane dropped official denials as
    "opinion" while keeping opponents' assertions, a measured upward bias
    (negative-stance rows nulled 33.0% vs 22.6% for positive, fact-era pool).
    A/B'd 2026-07-29 on live Haiku (3 runs/side): denial enters as a capped
    negative precursor 3/3 (was 0/3), F-35 mirror case flips −0.40→≈0, the
    over-trigger control (Fed official on a market claim) stays null 3/3, and
    the 22-article mobilization regression DEFLATES further (+0.278→+0.149
    mean fact_signal) — no rumor inflation. The wording is numeral-free by
    design: magnitude policy lives in estimator config, prompts only classify.
    Wording is Haiku-sensitive (two iterations were needed — "authority
    including a senior official speaking for it" is what made Medvedev-class
    denials register); re-run the A/B kit before rewording.
    """
    assert "The one EXCEPTION — DECIDER STATEMENTS" in PROMPT_PREFIX
    assert "whose own act or announcement would itself resolve the claim" in PROMPT_PREFIX
    assert "including a senior official speaking for that authority" in PROMPT_PREFIX
    assert "must never be nulled as opinion" in PROMPT_PREFIX
    assert "a denial, refusal, or ruling-out is a negative precursor" in PROMPT_PREFIX
    assert "remains claimed-and-unverified at most" in PROMPT_PREFIX
    # the null rule the exception carves out of must stay intact
    assert (
        "Return null (omit fact_signal and its facets) when the prediction "
        "rests on opinion, advocacy, or expectation with no reported fact that bears on the event."
    ) in PROMPT_PREFIX
