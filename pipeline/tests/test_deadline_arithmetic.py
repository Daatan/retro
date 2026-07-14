"""Arithmetic, not the LLM, decides the sign of a confidently-dated deadline claim.

The incident these pin down: asked whether a parliament dissolving "on Friday" met a
July 15 deadline, the extractor returned stance +1.0 / certainty 0.95 on 5 of 5 runs —
once rendering the claim as "dissolved on Friday, July 15", snapping the weekday onto the
deadline. That Friday was July 17. Given an article that spelled out "July 17" in plain
text, the same model returned -1.0 on 4 of 4 runs.
"""
import pytest

from tm.extractor import enforce_deadline_arithmetic
from tm.models import PredictionExtraction

DEADLINE = "2026-07-15"


def pred(stance: float, event_date: str | None = None, settled: bool | None = None, certainty: float = 0.95):
    return PredictionExtraction(
        quote="q", claim="c", stance=stance, certainty=certainty,
        settled=settled, event_date=event_date,
    )


# ── the incident ──────────────────────────────────────────────────────────────


def test_the_knesset_case_confident_yes_on_a_date_after_the_deadline_is_flipped():
    """The Guardian article: dissolution on 2026-07-17, deadline 2026-07-15 → NO, not YES."""
    [out] = enforce_deadline_arithmetic([pred(1.0, "2026-07-17", settled=True)], DEADLINE, "arrival")
    assert out.stance == -1.0
    assert out.certainty == 0.95  # only the sign moves
    assert out.settled is True


def test_a_date_on_or_before_the_deadline_is_left_alone():
    [out] = enforce_deadline_arithmetic([pred(1.0, "2026-07-12", settled=True)], DEADLINE, "arrival")
    assert out.stance == 1.0


def test_the_deadline_day_itself_counts_as_within():
    """"by July 15" includes July 15 — an off-by-one here would invert a correct signal."""
    [out] = enforce_deadline_arithmetic([pred(1.0, DEADLINE, settled=True)], DEADLINE, "arrival")
    assert out.stance == 1.0


def test_a_confident_no_on_a_date_within_the_deadline_is_also_corrected():
    """The override is symmetric — it is arithmetic, not a thumb on the scale toward NO."""
    [out] = enforce_deadline_arithmetic([pred(-1.0, "2026-07-14", settled=True)], DEADLINE, "arrival")
    assert out.stance == 1.0


# ── survival claims mirror arrival ────────────────────────────────────────────


def test_survival_claim_event_after_deadline_supports_the_claim():
    """"X will NOT happen by D" + the event lands after D → the claim holds (+)."""
    [out] = enforce_deadline_arithmetic([pred(-1.0, "2026-07-17", settled=True)], DEADLINE, "survival")
    assert out.stance == 1.0


def test_survival_claim_event_within_deadline_contradicts_the_claim():
    [out] = enforce_deadline_arithmetic([pred(1.0, "2026-07-14", settled=True)], DEADLINE, "survival")
    assert out.stance == -1.0


# ── only confident signals are overridden ─────────────────────────────────────


def test_a_hedged_signal_is_never_overridden():
    """"might slip past the deadline" is a real judgement; we have no business flipping it."""
    [out] = enforce_deadline_arithmetic([pred(0.4, "2026-07-17")], DEADLINE, "arrival")
    assert out.stance == 0.4


def test_settled_makes_even_a_weak_stance_eligible():
    [out] = enforce_deadline_arithmetic([pred(0.3, "2026-07-17", settled=True)], DEADLINE, "arrival")
    assert out.stance == -0.3


@pytest.mark.parametrize("stance", [0.9, -0.9, 1.0, -1.0])
def test_the_confidence_threshold_is_inclusive_at_0_9(stance: float):
    expected_flip = stance > 0  # arrival + event after deadline → positives are the wrong sign
    [out] = enforce_deadline_arithmetic([pred(stance, "2026-07-17")], DEADLINE, "arrival")
    assert (out.stance == -stance) is expected_flip


# ── fail open, in every direction ─────────────────────────────────────────────


@pytest.mark.parametrize("deadline,direction", [
    (None, "arrival"),           # caller does not classify claims
    ("2026-07-15", None),        # unknown direction — cannot reason about the sign
    ("2026-07-15", "diffuse"),   # a direction we have no arithmetic for
    ("not-a-date", "arrival"),
])
def test_missing_or_unusable_metadata_leaves_predictions_untouched(deadline, direction):
    [out] = enforce_deadline_arithmetic([pred(1.0, "2026-07-17", settled=True)], deadline, direction)
    assert out.stance == 1.0


@pytest.mark.parametrize("event_date", [None, "", "Friday", "2026-13-99"])
def test_no_usable_event_date_leaves_the_prediction_untouched(event_date):
    """The model omitting or garbling the date must never itself change a stance."""
    [out] = enforce_deadline_arithmetic([pred(1.0, event_date, settled=True)], DEADLINE, "arrival")
    assert out.stance == 1.0


def test_a_full_timestamp_deadline_is_accepted():
    """daatan sends claim_deadline as an ISO timestamp, not a bare date."""
    [out] = enforce_deadline_arithmetic(
        [pred(1.0, "2026-07-17", settled=True)], "2026-07-15T23:59:59.999Z", "arrival",
    )
    assert out.stance == -1.0


def test_each_prediction_is_judged_on_its_own_date():
    out = enforce_deadline_arithmetic(
        [
            pred(1.0, "2026-07-17", settled=True),  # after  → flip
            pred(1.0, "2026-07-14", settled=True),  # within → keep
            pred(0.2, "2026-07-17"),                # hedged → keep
            pred(1.0, None, settled=True),          # no date → keep
        ],
        DEADLINE, "arrival",
    )
    assert [p.stance for p in out] == [-1.0, 1.0, 0.2, 1.0]
