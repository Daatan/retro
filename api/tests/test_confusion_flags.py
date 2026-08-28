"""retro#687 — the three deterministic confusion rules.

Two things are being pinned here, and the second is the one that actually bites:

1. each rule fires on the combination it names, and on nothing adjacent to it;
2. **every rule yields no flag when its inputs are absent** — not a default, not a
   guess. Two of the three read shadow fields a pre-#681 row simply does not have,
   and rule 2's inputs do not exist in the schema at all yet, so "missing input"
   is the common case in production right now rather than an edge case.
"""

import logging

import pytest

from forecast_api.confusion_flags import (
    CONFUSION_RULES,
    RULE_STANCE_VS_QUANTITY,
    RULE_TRAPPED_STRONG_CLAIM,
    RULE_UNSURE_SETTLEMENT,
    counts_by_rule,
    flags_for_claim,
    flags_for_rows,
    log_confusion_flags,
)
from forecast_api.models import ClaimDetail, ReaderConfidence

BAR = 0.8


def _claim(**overrides) -> ClaimDetail:
    defaults = dict(claim="A claim.", stance=0.5, claim_strength=0.9)
    defaults.update(overrides)
    return ClaimDetail(**defaults)


def _rc(level="high", trap=None) -> ReaderConfidence:
    return ReaderConfidence(level=level, trap=trap)


def _flags(claim, **kw) -> list[str]:
    return flags_for_claim(claim, claim_strength_min=BAR, **kw)


class _Row:
    """The structural minimum `flags_for_rows` reads — see its docstring."""

    def __init__(self, claims_detail, url="https://example.com/a"):
        self.claims_detail = claims_detail
        self.url = url


# ── rule 1: flat source, stumbling reader ────────────────────────────────────
def test_a_strong_claim_with_a_trap_is_flagged():
    claim = _claim(claim_strength=0.9, reader_confidence=_rc(trap="negation"))
    assert _flags(claim) == [RULE_TRAPPED_STRONG_CLAIM]


def test_the_bar_is_inclusive():
    """0.8 is 'flat enough' — the issue's rule reads `>= 0.8`, and an exclusive
    bar would silently drop the modal value of a field that clusters on round
    numbers."""
    claim = _claim(claim_strength=BAR, reader_confidence=_rc(trap="negation"))
    assert _flags(claim) == [RULE_TRAPPED_STRONG_CLAIM]


def test_a_hedged_claim_with_a_trap_is_not_flagged():
    claim = _claim(claim_strength=0.4, reader_confidence=_rc(trap="negation"))
    assert _flags(claim) == []


def test_a_strong_claim_without_a_trap_is_not_flagged():
    claim = _claim(claim_strength=0.95, reader_confidence=_rc(level="low", trap=None))
    assert _flags(claim) == []


def test_rule_1_keys_on_trap_not_level():
    """A trap does not imply a low level (ReaderConfidence's own docstring), and
    this rule wants the NAMED trap — the part an independent detector can score —
    not the self-reported level."""
    claim = _claim(claim_strength=0.9, reader_confidence=_rc(level="high", trap="tone_vs_content"))
    assert RULE_TRAPPED_STRONG_CLAIM in _flags(claim)


def test_rule_1_is_null_safe_without_reader_confidence():
    """The pre-#681 row: strong claim, no reader_confidence at all."""
    assert _flags(_claim(claim_strength=1.0, reader_confidence=None)) == []


# ── rule 2: stance contradicts the arithmetic ────────────────────────────────
# `quantity` does not exist on ClaimDetail yet (retro#683). These drive the rule
# through a stand-in carrying the same attribute, which is exactly what the real
# field will be — the rule reads it via getattr precisely so it self-activates.
class _ClaimWithQuantity:
    def __init__(self, stance, quantity, claim_strength=0.9):
        self.stance = stance
        self.quantity = quantity
        self.claim_strength = claim_strength
        self.settled = None
        self.reader_confidence = None


@pytest.mark.parametrize("stance,quantity,comparison", [
    (0.8, 93.0, "at_least"),    # says positive, reports a number under the bar
    (-0.8, 110.0, "at_least"),  # says negative, reports a number over it
    (0.8, 110.0, "at_most"),    # "stays below 100" — 110 does not support yes
    (-0.8, 93.0, "at_most"),
])
def test_a_stance_contradicting_its_own_number_is_flagged(stance, quantity, comparison):
    claim = _ClaimWithQuantity(stance=stance, quantity=quantity)
    assert _flags(claim, question_quantity=100.0, comparison=comparison) == [
        RULE_STANCE_VS_QUANTITY
    ]


@pytest.mark.parametrize("stance,quantity,comparison", [
    (0.8, 110.0, "at_least"),
    (-0.8, 93.0, "at_least"),
    (0.8, 93.0, "at_most"),
])
def test_a_stance_agreeing_with_its_number_is_not_flagged(stance, quantity, comparison):
    claim = _ClaimWithQuantity(stance=stance, quantity=quantity)
    assert _flags(claim, question_quantity=100.0, comparison=comparison) == []


def test_a_number_exactly_on_the_threshold_decides_nothing():
    claim = _ClaimWithQuantity(stance=-0.9, quantity=100.0)
    assert _flags(claim, question_quantity=100.0, comparison="at_least") == []


def test_a_zero_stance_is_not_a_direction():
    """No sign to contradict — flagging it would report an inconsistency that
    isn't there."""
    claim = _ClaimWithQuantity(stance=0.0, quantity=93.0)
    assert _flags(claim, question_quantity=100.0, comparison="at_least") == []


def test_rule_2_is_inert_without_the_question_threshold():
    """Today's live path: the claim may carry a number, Phase 2's
    `question_quantity` does not exist, so there is nothing to compare against."""
    claim = _ClaimWithQuantity(stance=0.8, quantity=93.0)
    assert _flags(claim) == []
    assert _flags(claim, comparison="at_least") == []


def test_rule_2_is_inert_without_the_claim_quantity():
    """The real ClaimDetail today: no `quantity` attribute at all."""
    claim = _claim(stance=0.8)
    assert not hasattr(claim, "quantity")
    assert _flags(claim, question_quantity=100.0, comparison="at_least") == []


def test_quantitative_estimate_is_not_the_claim_quantity():
    """The one wrong field to reach for. `quantitative_estimate` is a probability
    in [0,1] cited FOR the event; a question threshold is a count or a price.
    Comparing them would flag nearly every claim, since 0.85 < 100 always."""
    claim = _claim(stance=0.8, quantitative_estimate=0.85)
    assert claim.quantitative_estimate == 0.85
    assert _flags(claim, question_quantity=100.0, comparison="at_least") == []


# ── rule 3: a settlement the reader was unsure of ────────────────────────────
@pytest.mark.parametrize("level", ["medium", "low"])
def test_a_settlement_below_high_confidence_is_flagged(level):
    claim = _claim(settled=True, claim_strength=0.5, reader_confidence=_rc(level=level))
    assert _flags(claim) == [RULE_UNSURE_SETTLEMENT]


def test_a_high_confidence_settlement_is_not_flagged():
    claim = _claim(settled=True, claim_strength=0.5, reader_confidence=_rc(level="high"))
    assert _flags(claim) == []


@pytest.mark.parametrize("settled", [False, None])
def test_a_non_settlement_is_not_flagged_however_unsure(settled):
    """`settled` is Optional[bool]; None means 'not marked', which is not the same
    as False and must not be read as a settlement either way."""
    claim = _claim(settled=settled, claim_strength=0.5, reader_confidence=_rc(level="low"))
    assert _flags(claim) == []


def test_rule_3_is_null_safe_without_reader_confidence():
    assert _flags(_claim(settled=True, claim_strength=0.5, reader_confidence=None)) == []


# ── composition ──────────────────────────────────────────────────────────────
def test_one_claim_can_trip_two_rules_and_reports_both():
    """They name different inconsistencies; collapsing them to 'flagged' would
    throw away the grouping the exit report needs."""
    claim = _claim(
        settled=True, claim_strength=0.9, reader_confidence=_rc(level="low", trap="negation"),
    )
    assert _flags(claim) == [RULE_TRAPPED_STRONG_CLAIM, RULE_UNSURE_SETTLEMENT]


def test_flags_are_reported_in_rule_order():
    claim = _claim(
        settled=True, claim_strength=0.9, reader_confidence=_rc(level="low", trap="negation"),
    )
    fired = _flags(claim)
    assert fired == [r for r in CONFUSION_RULES if r in fired]


def test_rows_report_the_claim_index_so_a_flag_points_at_one_claim():
    clean = _claim(reader_confidence=_rc(level="high"))
    dirty = _claim(claim_strength=0.9, reader_confidence=_rc(trap="negation"))
    flags = flags_for_rows([_Row([clean, dirty])], claim_strength_min=BAR)
    assert [(f.rule, f.claim_index) for f in flags] == [(RULE_TRAPPED_STRONG_CLAIM, 1)]


def test_a_row_without_claims_detail_yields_nothing():
    """Pre-#364 legacy rows carry no claim layer — unevaluable, not clean."""
    assert flags_for_rows([_Row(None), _Row([])], claim_strength_min=BAR) == []


def test_counts_include_every_rule_even_at_zero():
    """A rule that never fires must be distinguishable from a rule that was never
    evaluated — the same reason the summary line logs on every pool."""
    counts = counts_by_rule([])
    assert counts == {rule: 0 for rule in CONFUSION_RULES}


# ── logging ──────────────────────────────────────────────────────────────────
def test_the_summary_line_fires_even_when_nothing_is_flagged(caplog):
    """Without a denominator a zero cannot be told apart from a path that never
    ran (retro#412's rationale, and Gate-0's)."""
    with caplog.at_level(logging.INFO, logger="forecast_api.confusion_flags"):
        log_confusion_flags(
            [_Row([_claim(reader_confidence=_rc(level="high"))])],
            question_hash="abc123", extractor_model="bedrock/haiku", claim_strength_min=BAR,
        )
    summaries = [r.getMessage() for r in caplog.records if "event=confusion_flags " in r.getMessage()]
    assert len(summaries) == 1
    assert "rows=1 evaluable=1 claims=1 flagged=0" in summaries[0]
    for rule in CONFUSION_RULES:
        assert f"{rule}=0" in summaries[0]


def test_a_flagged_claim_logs_its_rater_and_correlation_ids(caplog):
    with caplog.at_level(logging.INFO, logger="forecast_api.confusion_flags"):
        flags = log_confusion_flags(
            [_Row([_claim(claim_strength=0.9, reader_confidence=_rc(trap="negation"))])],
            question_hash="abc123", extractor_model="bedrock/haiku",
            claim_strength_min=BAR, prediction_id="pred-1",
        )
    assert len(flags) == 1
    line = next(r.getMessage() for r in caplog.records if "event=confusion_flag " in r.getMessage())
    assert f"rule={RULE_TRAPPED_STRONG_CLAIM}" in line
    assert "extractor_model=bedrock/haiku" in line
    assert "prediction_id=pred-1" in line
    assert "question=abc123" in line


def test_rows_with_no_claim_layer_are_counted_apart_from_clean_ones(caplog):
    """`evaluable` is what separates 'nothing to flag' from 'nothing to look at'."""
    with caplog.at_level(logging.INFO, logger="forecast_api.confusion_flags"):
        log_confusion_flags(
            [_Row(None), _Row([_claim(reader_confidence=_rc(level="high"))])],
            question_hash="abc123", extractor_model="unknown", claim_strength_min=BAR,
        )
    summary = next(r.getMessage() for r in caplog.records if "event=confusion_flags " in r.getMessage())
    assert "rows=2 evaluable=1 claims=1 flagged=0" in summary


def test_an_empty_pool_still_logs_a_summary(caplog):
    with caplog.at_level(logging.INFO, logger="forecast_api.confusion_flags"):
        log_confusion_flags([], question_hash="abc123", extractor_model="unknown",
                            claim_strength_min=BAR)
    summary = next(r.getMessage() for r in caplog.records if "event=confusion_flags " in r.getMessage())
    assert "rows=0 evaluable=0 claims=0 flagged=0" in summary
