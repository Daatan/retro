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
    assert "## Unverified claims by an interested party — cap claim_strength" in PROMPT_PREFIX
    assert "carries claim_strength no higher than 0.5" in PROMPT_PREFIX
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


def test_alarming_tone_section_present():
    """retro#545's sign-error class, cross-model evidence (2026-08-25 extractor
    model survey): the identical Hebrew sentence about Chernobyl's "elephant's
    foot" keeping a city uninhabitable for a century — same quote, same claim
    wording, both models flagged reporting/high-certainty — scored stance
    +1.0 on one model's run and -1.0 on another's, on content that is
    unambiguously affirmative. The prompt had no rule separating an alarming
    quote's emotional register from what its content actually asserts."""
    assert "## Alarming or critical tone is not stance direction" in PROMPT_PREFIX
    assert "content argues against the related event" in PROMPT_PREFIX


def test_alarming_tone_example_present():
    assert "Site S remains hazardous to human habitation" in PROMPT_PREFIX
    assert "do not score it negative because" in PROMPT_PREFIX


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
    assert "|stance| <= 0.3, claim_strength <= 0.4) and is NEVER settled" in PROMPT_PREFIX
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
    # retro#354 D1: the |0.3| numeral was deleted from the prompt once enforce_precursor_cap
    # (extractor.py) started enforcing the ceiling in code regardless of what the model
    # emits — a numeral in prose is policy the estimator, not the prompt, should carry.
    assert "A precursor never scores as the event occurring" in PROMPT_PREFIX
    assert "is context only" in PROMPT_PREFIX
    assert "A claimed-but-unverified event is down-weighted" in PROMPT_PREFIX


def test_fact_signal_in_output_contract():
    assert "fact_signal" in PROMPT_SUFFIX
    assert "event_actors" in PROMPT_SUFFIX
    assert "event_target" in PROMPT_SUFFIX
    assert "is_occurrence" in PROMPT_SUFFIX
    assert "verified" in PROMPT_SUFFIX


def test_fact_signal_absent_reason_section_present():
    """retro#471: the null itself must stay honest — a consumer needs to tell
    'nothing found' from 'something found that points the other way'."""
    assert "fact_signal_absent_reason" in PROMPT_PREFIX
    assert "no_fact_found" in PROMPT_PREFIX
    assert "contrary_below_anchor" in PROMPT_PREFIX
    assert "opinion" in PROMPT_PREFIX


def test_fact_signal_absent_reason_in_output_contract():
    assert "fact_signal_absent_reason" in PROMPT_SUFFIX
    assert "never omit both" in PROMPT_SUFFIX


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


def test_negative_precursor_ladder_present():
    """Signal Lanes WS5b: contrary reported facts are graded negative precursors, not null.

    Generalizes WS5's negative-precursor category beyond decider statements.
    The fact lane was a positive-evidence accumulator by construction: the
    extreme-negative anchor demands near-impossibility while the positive side
    has a graded precursor ladder, so negative-stance rows nulled out 1.5x more
    often (33.0% vs 22.6%, fact-era pool). A/B'd 2026-07-29 on live Haiku,
    12-row nulled-negative sample (3 runs/side): gain-expected rows produce a
    negative fact_signal in 16/18 patched runs vs 8/17 baseline (polls vs
    "Likud will win", deal-in-limbo vs "agreement signed", "IRGC reluctant"
    vs "Iran will initiate"); a pure-opinion control column stays null 3/3.
    Regressions: the 22-article mobilization set keeps deflating (mean stored
    +0.273 -> +0.182, zero rows inflated >+0.1) and the WS5 5-case set holds
    (denial negative 2/3, F-35 2/3, over-trigger control clean 3/3 — the
    decider rule never fires on the Fed/market claim). Known residual, both
    sides equally: precursor magnitudes can exceed the prompt's cap with
    is_occurrence=false — the estimator-side clamp is issue #354/D1.
    Numeral-free by design; re-run the A/B kit before rewording.
    """
    assert "NEGATIVE PRECURSORS — the graded scale runs in BOTH directions" in PROMPT_PREFIX
    assert "never nulled merely because it points against the claim" in PROMPT_PREFIX
    assert "Reserve the extreme negative for facts that establish the event cannot happen" in PROMPT_PREFIX
    assert "grade contrary facts with the same discipline as supporting ones" in PROMPT_PREFIX
    # WS5's decider exception must survive the addition intact
    assert "The one EXCEPTION — DECIDER STATEMENTS" in PROMPT_PREFIX


def test_deadline_deferred_intent_section_present():
    """retro#352: a fact that defers the event past the claim deadline bears
    against a by-deadline claim even when the deferral is anchored to another
    named milestone ("after the elections") rather than a calendar date the
    DATES section's arithmetic can compare directly — `enforce_deadline_arithmetic`
    only fires on a parseable `event_date`, and a milestone-relative deferral
    never produces one. A/B'd 2026-08-09 on live Haiku (harness = retro#470):
    a hard variant with no explicit date ("wait until after the parliamentary
    elections") consistently nulled fact_signal (5/5 baseline runs) rather
    than grading it negative; patched 5/5 negative, zero regression on the
    sibling explicit-date/no-harm/decider cases. See
    pipeline/scripts/ab_cases/deadline_and_resolution_rules.json,
    case `deadline-direction-named-event-hard`.
    """
    assert "DEADLINE-DEFERRED INTENT" in PROMPT_PREFIX
    assert "bears AGAINST a by-deadline claim" in PROMPT_PREFIX
    assert "You do not need a resolvable event_date to score this" in PROMPT_PREFIX
    # Must stay inside the fact_signal negative-precursor discipline, not a separate rule.
    assert "under the same discipline as any other negative precursor" in PROMPT_PREFIX


def test_quantitative_estimate_share_exclusion_present():
    """retro#362 (lane-soundness F5): shares must never enter quantitative_estimate.

    The qe section taught poll numbers and seat projections into
    quantitative_estimate ("the poll puts Candidate Y at 45%") and told the
    model to self-align stance = 2*qe-1, while the cited_share class definition
    said the same figure "is explicitly NOT a probability" and even sanctioned
    the collision ("use this even when the same figure would also populate
    quantitative_estimate"). The prompt contradicted itself and the code
    enforced the wrong side: measured on prod (2026-08-01), 47 of the 117
    qe-carrying pool rows were cited_share — Knesset seat shares rewritten to
    stance = 2*share-1 at certainty 0.9, bit-exact on single-claim rows. The
    code-side guard (resolve_stance_certainty rewrites cited_probability only)
    is the enforcement; these strings keep the prompt teaching the same rule
    so the model's own stance/qe emissions stop encoding the category error.

    A/B'd 2026-08-01 on live Haiku (Oracle box, 3 runs/side, 5 cases): the two
    real Likud poll articles (the Kantar flash behind the bit-exact prod row +
    a Channel 13 jpost piece) emitted qe for a seat count in 3/6 baseline runs
    and 0/6 patched runs while keeping cited_share classification; the
    genuine-probability controls (poll-aggregator model 22%, Polymarket 18%)
    kept qe with aligned stance 3/3 on BOTH sides; the no-figure momentum
    control stayed qe-null both sides. Zero regressions, one iteration.
    """
    assert "PROBABILITY OF THE RELATED EVENT ITSELF" in PROMPT_PREFIX
    assert "seat projection is NOT a probability of the event" in PROMPT_PREFIX
    assert "leave `quantitative_estimate` null for those" in PROMPT_PREFIX
    assert "classify them `cited_share` below" in PROMPT_PREFIX
    # the class definition must agree (the old text sanctioned the collision)
    # (the enum block keeps hard newlines/indentation — normalize before matching)
    flat = " ".join(PROMPT_PREFIX.split())
    assert "A share must NEVER populate `quantitative_estimate`" in flat
    assert "use this even when the same figure" not in flat.lower()
    # the genuine-probability anchor behavior must survive the narrowing
    assert "Set `stance` to match it" in PROMPT_PREFIX
    assert "a model gives Team X an 18.83% chance to win the tournament" in PROMPT_PREFIX
    # and the output contract must carry the same exclusion
    assert "never a vote share or seat count" in PROMPT_SUFFIX


def test_report_kind_section_present():
    """retro#686 (unparked from #673 §2): "the rate is 8.75%" and "the rate was
    cut to 8.75%" carry the same stance toward "above 8%?" and are not the same
    evidence — one restates a standing situation a prior article may already
    have supplied, the other is new movement. The disciplining test in the
    section is deliberately semantic ("what would the sentence still tell you a
    month later") rather than grammatical: verb tense reads "held the rate" as a
    change, which is exactly backwards."""
    assert "## REPORT_KIND" in PROMPT_PREFIX
    assert "level  — the standing situation as it is" in PROMPT_PREFIX
    assert "change — a movement in it" in PROMPT_PREFIX
    assert "not its verb tense" in PROMPT_PREFIX
    assert "holding is the absence of a step" in PROMPT_PREFIX.replace("\n", " ")
    # The omit case is the field's honesty valve: a pure expectation is neither
    # member, and forcing one would fill the field with noise on op-eds.
    assert "Omit report_kind when the quote is neither" in PROMPT_PREFIX


def test_consensus_view_section_present():
    """retro#686 (unparked from #673's "predicted consensus"). Its kill
    criterion is >20% of non-null rows carrying the MODEL's own view rather than
    the article's, so all three of the "this is NOT" clauses are load-bearing —
    each names one of the three things that get recorded here by mistake."""
    assert "## CONSENSUS_VIEW" in PROMPT_PREFIX
    assert "Once per article, not per prediction" in PROMPT_PREFIX
    assert "It is not YOUR view" in PROMPT_PREFIX
    assert "that is author_lean" in PROMPT_PREFIX
    assert "It is not the stance of the quotes you extracted" in PROMPT_PREFIX
    # The worked disagreement case: consensus and author_lean must be able to
    # point opposite ways, or the field is a copy of one we already have.
    assert "wishful thinking" in PROMPT_PREFIX
    assert "Omit consensus_view when the article does not say" in PROMPT_PREFIX


def test_both_shadow_fields_disclaim_any_effect_on_stance():
    """Both blocks say, in their own words, that they do not move `stance`.

    A shadow field is supposed to be free: it is read by nothing, so the only
    way it can cost anything is by changing an answer that IS read. Describing
    two new kinds of judgement right before the output contract invites the
    model to treat them as scoring dimensions — a `change` reading as stronger
    evidence than a `level`, a `divided` consensus damping a confident quote.
    The sentences are cheap and the failure they prevent is silent, so they are
    pinned rather than left to a reviewer's eye.
    """
    flat = PROMPT_PREFIX.replace("\n", " ")
    assert "report_kind never changes your stance" in flat
    assert "a level that satisfies the question is exactly as positive" in flat
    assert "Like report_kind, it never changes a stance" in flat
    assert "exactly as you would have without this field" in flat


def test_the_prompt_does_not_quote_its_own_ab_cases():
    """A prompt that quotes a case from its own test set is a broken instrument.

    v8's first draft illustrated `report_kind` with "the rate is 8.75%" / "the
    central bank held the rate at 8.75%" — the numbers and sentence shape of the
    A/B case `threshold-at-or-below-satisfied`, which the same PR then used to
    decide whether the edit regressed anything. A case the prompt has already
    answered cannot measure the prompt. (The length-control experiment later
    showed this collision was NOT what moved Nova Lite — the rewrite stands on
    the principle, not on that result.)
    """
    flat = PROMPT_PREFIX.replace("\n", " ")
    for needle in ("8.75", "8.99", "benchmark interest rate"):
        assert needle not in flat, (
            f"the prompt quotes {needle!r}, which appears in an ab_cases fixture; "
            f"illustrate the rule from a domain the case corpus does not use"
        )


def test_report_kind_and_consensus_view_in_output_contract():
    assert "report_kind" in PROMPT_SUFFIX
    assert '"consensus_view"' in PROMPT_SUFFIX
    # Both must name their enum members where the model is told to emit them,
    # not only in the prose block far above.
    assert "one of level / change" in PROMPT_SUFFIX
    assert "expects_yes / expects_no / divided" in PROMPT_SUFFIX


def _worked_prediction_blocks() -> list[str]:
    """The prediction objects inside PROMPT_SUFFIX's worked examples.

    Split on the quote key, which starts every one of them; the leading chunk
    before the first is the instruction text and is dropped.
    """
    return PROMPT_SUFFIX.split('"quote":')[1:]


def test_every_worked_prediction_carries_report_kind():
    """The v7 lesson, pinned rather than left to review (docs/PROMPT_VERSIONS.md).

    v6 added the fact block to a worked example but omitted `facet` from it —
    `facet` was specified only in the prefix section — and its fill went from
    68% to **0% on both models**. A worked example is read as the definitive
    enumeration of its block, so omission from one is a far stronger signal
    than presence in the prose. A new per-prediction field therefore has to
    appear on EVERY worked prediction, not on a representative one.
    """
    blocks = _worked_prediction_blocks()
    assert len(blocks) >= 3, "the worked examples themselves went missing"
    missing = [i for i, b in enumerate(blocks) if '"report_kind"' not in b]
    assert not missing, (
        f"worked prediction(s) {missing} omit report_kind; v6 zeroed `facet` "
        f"on both models exactly this way"
    )


def test_every_worked_example_carries_consensus_view():
    """Same rule one level up — consensus_view is article-level, so its unit of
    enumeration is the top-level example object, not the prediction."""
    examples = PROMPT_SUFFIX.split('"predictions": [\n')[1:]
    assert len(examples) >= 2, "the worked examples themselves went missing"
    assert all('"consensus_view"' in e for e in examples)


def test_the_worked_report_kinds_do_not_all_agree():
    """report_kind's own kill criterion is ">90% one value" — a field that
    answers the same way everywhere has measured nothing. Worked examples that
    all showed `change` would manufacture precisely the failure the A/B is run
    to detect, and the measurement would then be of the prompt, not the model.
    """
    blocks = _worked_prediction_blocks()
    values = [b.split('"report_kind": "')[1].split('"')[0] for b in blocks]
    assert set(values) == {"level", "change"}, (
        f"worked report_kind values are {values}; both members must be shown"
    )


def test_the_worked_consensus_views_do_not_all_agree():
    """Same argument, and sharper here: consensus_view is one value per
    article, so a model copying the examples has far less to average over."""
    examples = PROMPT_SUFFIX.split('"consensus_view": "')[1:]
    values = [e.split('"')[0] for e in examples]
    assert len(set(values)) > 1, f"worked consensus_view values are {values}"


def test_consensus_view_is_not_taught_as_the_authors_own_view():
    """The one confusion that would make the field worthless. The output
    contract has to keep the two apart where the model actually emits them —
    the prose block is 100 lines further up and, per the v7 finding, loses to
    a worked example that says otherwise."""
    assert "what the article says OTHERS expect" in PROMPT_SUFFIX
    assert "the byline author's OWN forecast" in PROMPT_SUFFIX


def test_quantity_section_present():
    """retro#683 (Oracle 1.5 Phase 1, item 1.1). PR#671 measured the reason this
    field exists: on ten synthetic numeric-threshold cases Nova Lite returned
    stance +0.00 on every between-bounds case and inverted both tone traps. Three
    prompt sections already tell it to compare the numbers; #664's P2 was resolved
    as *the field, not another prompt fix*.

    So the elicitation has to ask for the NUMBER and not the verdict, and these
    strings are what keep it that way — a later edit that rephrases the block as
    "does the figure satisfy the question" would turn the field back into the
    judgment it was created to replace, with nothing else in CI to notice.
    """
    assert "## QUANTITY" in PROMPT_PREFIX
    flat = " ".join(PROMPT_PREFIX.split())
    assert "Do NOT decide whether it satisfies the question" in flat
    assert "it is done in code against the question's own threshold" in flat
    # every key the schema offers must be elicited by name
    for key in ("value", "unit", "comparator", "value_hi", "as_of"):
        assert key in PROMPT_PREFIX, f"the QUANTITY block never names {key}"
    # all six comparators, so the model is not told to answer something the
    # code-side comparison cannot read
    for comparator in ('"="', '"<"', '"<="', '">"', '">="', '"between"'):
        assert comparator in flat, f"the QUANTITY block never shows {comparator}"


def test_quantity_records_the_articles_relation_not_the_questions():
    """The field's whole failure mode in one rule. If the model rewrites "exports
    stayed below 40 million tonnes" to ">" because the QUESTION asks about
    exceeding 40 million tonnes, the extracted comparator is a restatement of the
    question and the code-side comparison compares the question to itself."""
    flat = " ".join(PROMPT_PREFIX.split())
    assert "Record the ARTICLE's relation, never the question's" in flat
    assert "destroys the one thing this field carries" in flat


def test_quantity_is_kept_apart_from_quantitative_estimate():
    """retro#362 narrowed `quantitative_estimate` to a cited PROBABILITY and sent
    shares, counts and rates to `cited_share` with nowhere to put the number.
    This is that nowhere, and the two must not re-merge: a level written into
    `quantitative_estimate` is rewritten to stance = 2*qe-1 in code, which is the
    bit-exact prod bug #362 was filed for."""
    flat = " ".join(PROMPT_PREFIX.split())
    assert "`quantity` is not `quantitative_estimate`" in flat
    assert "A level, a share, a count, a rate or a tonnage is not a probability" in flat


def test_quantity_disclaims_any_effect_on_stance():
    """Same sentence report_kind and consensus_view carry, for the same reason —
    but load-bearing here in a way it is not for them. This field is elicited on
    exactly the claims whose stance is hardest, so a model that reads it as
    permission to re-score would move the numbers the A/B corpus watches."""
    flat = " ".join(PROMPT_PREFIX.split())
    assert "Like report_kind, quantity never changes your stance" in flat
    assert "score the stance exactly as you would have without this field" in flat


def test_quantity_in_output_contract():
    """The v7 lesson: a rule stated only in the prose block 100 lines above loses
    to the output contract where the model actually emits the field."""
    assert '"comparator"' in PROMPT_SUFFIX
    assert "one of = / < / <= / > / >= / between" in PROMPT_SUFFIX
    assert "never whether it satisfies the question" in PROMPT_SUFFIX
    assert "OMIT quantity when the quote states no figure" in PROMPT_SUFFIX


def test_at_least_one_worked_prediction_carries_quantity():
    """The v6/v7 finding, applied to an optional field.

    `facet` went from 68% fill to 0% on both models when v6 showed its block in a
    worked example without it — a worked example is read as the definitive
    enumeration. `quantity` is legitimately absent from most predictions, so the
    rule is not "on every one" the way report_kind's is; it is that at least one
    example must SHOW it, or a field the prose asks for appears in none of the
    JSON the model is copying and its fill rate measures the omission instead of
    the model.
    """
    blocks = _worked_prediction_blocks()
    carrying = [b for b in blocks if '"quantity"' in b]
    assert carrying, "no worked prediction shows quantity; its fill rate would measure the prompt"
    assert len(carrying) < len(blocks), (
        "every worked prediction carries quantity, which teaches that a figure is "
        "always available; the field is optional and the examples must show that"
    )


def test_the_quantity_examples_do_not_quote_the_ab_corpus():
    """`test_the_prompt_does_not_quote_its_own_ab_cases` pins three needles from
    the v8 collision. This one states the rule for the case corpus this field is
    measured on: the retro#664 numbers must not appear in the block that teaches
    the model how to extract them, or the validator scores recall of the prompt.
    """
    # Scoped to the text this issue adds. The sections that predate it keep their
    # own examples, and three of their needles are already pinned above.
    quantity_block = PROMPT_PREFIX.split("## QUANTITY")[1]
    example = PROMPT_SUFFIX.split('Example — related event: "Airline A')[1]
    for needle in ("8.75", "8.99", "9.00 percent", "33 seats", "36 seats", "31 seats",
                   "2.4 percent", "4.1 percent", "policy rate", "inflation"):
        assert needle not in quantity_block, f"the QUANTITY block quotes {needle!r} from the case corpus"
        assert needle not in example, f"the worked quantity example quotes {needle!r} from the case corpus"
