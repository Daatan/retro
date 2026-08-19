"""A settlement vote may not contradict its own fact lane (retro#545, slice i).

`fact_signal` is the fact-lane counterpart of `stance` on the same axis — "+1 the
facts establish the event happened, -1 the facts establish it will not or cannot".
A `settled` claim asserts an accomplished fact rather than a reading of one, so
the two must agree in sign. When they don't, one of them is mis-signed and nothing
deterministic can tell which, so the claim is neutralised rather than inverted —
the `enforce_winner_entity_consistency` (retro#401) precedent.

Prod audit 2026-08-19: 46 of the 230 settled rows carrying a fact_signal oppose
their own stance at |fact_signal| >= 0.5, across 3 ACTIVE forecasts — 41 of them
the "Andy Burnham will REMAIN Prime Minister" cluster, every row stance=-1.00
settled off articles reporting he *took office*.
"""
from tm.extractor import (
    _SETTLEMENT_FACT_SIGNAL_ANCHOR,
    enforce_settlement_fact_signal_agreement,
)
from tm.models import PredictionExtraction


def pred(
    stance: float,
    fact_signal: float | None,
    *,
    settled: bool | None = True,
    certainty: float = 0.9,
    claim: str = "Andy Burnham will remain Prime Minister until 2028",
):
    return PredictionExtraction(
        quote="q", claim=claim, stance=stance, certainty=certainty,
        settled=settled, fact_signal=fact_signal,
    )


# ── the neutralisation itself ─────────────────────────────────────────────────

def test_the_burnham_row_is_neutralised():
    """stance=-1.00 settled against fact_signal=+1.00 — 41 live rows of this shape."""
    preds = [pred(-1.0, 1.0)]
    out = enforce_settlement_fact_signal_agreement(preds)
    assert out[0].stance == 0.0
    assert out[0].settled is False


def test_neutralised_in_the_other_direction_too():
    preds = [pred(0.9, -0.8)]
    enforce_settlement_fact_signal_agreement(preds)
    assert preds[0].stance == 0.0
    assert preds[0].settled is False


def test_returns_the_same_list_object():
    preds = [pred(-1.0, 1.0)]
    assert enforce_settlement_fact_signal_agreement(preds) is preds


def test_certainty_and_fact_signal_survive_so_the_row_stays_auditable():
    """Only the sign-bearing fields move: the row keeps its weight and its evidence."""
    preds = [pred(-1.0, 1.0, certainty=0.95)]
    enforce_settlement_fact_signal_agreement(preds)
    assert preds[0].certainty == 0.95
    assert preds[0].fact_signal == 1.0


def test_logs_the_conflict(caplog):
    preds = [pred(-1.0, 1.0)]
    with caplog.at_level("WARNING", logger="tm.extractor"):
        enforce_settlement_fact_signal_agreement(preds)
    assert any(
        "event=settlement_fact_signal_conflict" in r.message for r in caplog.records
    )


# ── everything that must be left alone ────────────────────────────────────────

def test_agreeing_signs_are_untouched():
    preds = [pred(-1.0, -1.0)]
    enforce_settlement_fact_signal_agreement(preds)
    assert preds[0].stance == -1.0
    assert preds[0].settled is True


def test_an_unsettled_claim_is_untouched_even_when_the_lanes_disagree():
    """Outside settlement, stance blends assertion with fact — an official
    asserting X while the reported facts point the other way is a coherent row,
    not a contradiction."""
    preds = [pred(0.85, -0.9, settled=False)]
    enforce_settlement_fact_signal_agreement(preds)
    assert preds[0].stance == 0.85
    assert preds[0].settled is False


def test_an_unjudged_settled_is_not_a_settlement():
    preds = [pred(-1.0, 1.0, settled=None)]
    enforce_settlement_fact_signal_agreement(preds)
    assert preds[0].stance == -1.0
    assert preds[0].settled is None


def test_a_missing_fact_signal_leaves_the_claim_alone():
    """Legitimately omitted for opinion/advocacy rows — the null is not a zero."""
    preds = [pred(-1.0, None)]
    enforce_settlement_fact_signal_agreement(preds)
    assert preds[0].stance == -1.0
    assert preds[0].settled is True


def test_a_fact_signal_below_the_anchor_leaves_the_claim_alone():
    """Precursor rows arrive here already clamped to ±0.3 by enforce_precursor_cap;
    below the anchor a fact_signal bears on the event rather than establishing it."""
    preds = [pred(-1.0, _SETTLEMENT_FACT_SIGNAL_ANCHOR - 0.01)]
    enforce_settlement_fact_signal_agreement(preds)
    assert preds[0].stance == -1.0
    assert preds[0].settled is True


def test_exactly_at_the_anchor_fires():
    preds = [pred(-1.0, _SETTLEMENT_FACT_SIGNAL_ANCHOR)]
    enforce_settlement_fact_signal_agreement(preds)
    assert preds[0].stance == 0.0


def test_a_zero_stance_has_no_sign_to_contradict():
    preds = [pred(0.0, 1.0)]
    enforce_settlement_fact_signal_agreement(preds)
    assert preds[0].stance == 0.0
    assert preds[0].settled is True


def test_a_zero_fact_signal_is_below_the_anchor_and_never_fires():
    """fact_signal=0 means the facts point neither way — not a contradiction."""
    preds = [pred(-1.0, 0.0)]
    enforce_settlement_fact_signal_agreement(preds)
    assert preds[0].stance == -1.0
    assert preds[0].settled is True


def test_mixed_batch_touches_only_the_conflicting_rows():
    conflicting = pred(-1.0, 1.0)
    agreeing = pred(-1.0, -1.0)
    unsettled = pred(0.9, -0.9, settled=False)
    enforce_settlement_fact_signal_agreement([conflicting, agreeing, unsettled])
    assert (conflicting.stance, conflicting.settled) == (0.0, False)
    assert (agreeing.stance, agreeing.settled) == (-1.0, True)
    assert (unsettled.stance, unsettled.settled) == (0.9, False)


def test_empty_batch():
    assert enforce_settlement_fact_signal_agreement([]) == []
