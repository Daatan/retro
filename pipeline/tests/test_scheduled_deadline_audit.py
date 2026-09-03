"""Log-only: does this article cover a scheduled/threshold claim whose deadline
has already passed, published after that deadline, with no event_date
confirming the outcome (retro#590, proposal 3 of retro#575)?

Scoping research on the issue found the "has this scheduled date passed" check
already exists twice in the pipeline, and neither is what this audits:
`aggregation.settlement_direction_allowed`/`settlement_vote_validity` compare
`claim_deadline` against wall-clock "today" too, but only at pool-aggregation
time and only once a settlement vote already exists. daatan's
`temporal-clock.ts` pins the published number once the deadline passes, but
never touches extraction/stance. So this is the first place in the pipeline a
"deadline passed, still unconfirmed" fact is visible at all — and, per the
issue's own framing, whether that's worth anything is unmeasured, hence
shadow-first: this only logs, it never mutates stance/claim_strength/settled.
"""
from tm.extractor import audit_scheduled_deadline_unconfirmed
from tm.models import PredictionExtraction

DEADLINE = "2026-08-01"
ARTICLE_AFTER = "2026-08-15"
TODAY = "2026-09-03"


def pred(*, event_date: str | None = None, claim: str = "The vote will pass"):
    return PredictionExtraction(
        quote="q", claim=claim, stance=0.8, certainty=0.8, event_date=event_date,
    )


# ── the shape itself ────────────────────────────────────────────────────────

def test_passed_deadline_scheduled_archetype_unconfirmed_logs_a_warning(caplog):
    preds = [pred()]
    with caplog.at_level("WARNING", logger="tm.extractor"):
        audit_scheduled_deadline_unconfirmed(
            preds, ARTICLE_AFTER, DEADLINE, "scheduled", today=TODAY,
        )
    assert any(
        "event=scheduled_deadline_unconfirmed " in r.message for r in caplog.records
    )


def test_threshold_archetype_also_fires(caplog):
    preds = [pred()]
    with caplog.at_level("WARNING", logger="tm.extractor"):
        audit_scheduled_deadline_unconfirmed(
            preds, ARTICLE_AFTER, DEADLINE, "threshold", today=TODAY,
        )
    assert any(
        "event=scheduled_deadline_unconfirmed " in r.message for r in caplog.records
    )


def test_claim_carrying_an_event_date_does_not_log():
    """Already handled by enforce_deadline_arithmetic/enforce_settlement_event_date —
    flagging it here would just re-describe their output."""
    preds = [pred(event_date="2026-07-30")]
    out = audit_scheduled_deadline_unconfirmed(
        preds, ARTICLE_AFTER, DEADLINE, "scheduled", today=TODAY,
    )
    assert out is preds  # still returns the list unmodified either way


def test_claim_carrying_an_event_date_is_not_counted_as_fired(caplog):
    preds = [pred(event_date="2026-07-30")]
    with caplog.at_level("WARNING", logger="tm.extractor"):
        audit_scheduled_deadline_unconfirmed(
            preds, ARTICLE_AFTER, DEADLINE, "scheduled", today=TODAY,
        )
    assert not any(
        "event=scheduled_deadline_unconfirmed " in r.message for r in caplog.records
    )


# ── fail-open branches ──────────────────────────────────────────────────────

def test_non_scheduled_archetype_fails_open(caplog):
    preds = [pred()]
    with caplog.at_level("WARNING", logger="tm.extractor"):
        audit_scheduled_deadline_unconfirmed(
            preds, ARTICLE_AFTER, DEADLINE, "trend", today=TODAY,
        )
    assert not any(
        "event=scheduled_deadline_unconfirmed " in r.message for r in caplog.records
    )


def test_missing_archetype_fails_open(caplog):
    preds = [pred()]
    with caplog.at_level("WARNING", logger="tm.extractor"):
        audit_scheduled_deadline_unconfirmed(
            preds, ARTICLE_AFTER, DEADLINE, None, today=TODAY,
        )
    assert not any(
        "event=scheduled_deadline_unconfirmed " in r.message for r in caplog.records
    )


def test_missing_deadline_fails_open(caplog):
    preds = [pred()]
    with caplog.at_level("WARNING", logger="tm.extractor"):
        audit_scheduled_deadline_unconfirmed(
            preds, ARTICLE_AFTER, None, "scheduled", today=TODAY,
        )
    assert not any(
        "event=scheduled_deadline_unconfirmed " in r.message for r in caplog.records
    )


def test_unparseable_deadline_fails_open(caplog):
    preds = [pred()]
    with caplog.at_level("WARNING", logger="tm.extractor"):
        audit_scheduled_deadline_unconfirmed(
            preds, ARTICLE_AFTER, "not a date", "scheduled", today=TODAY,
        )
    assert not any(
        "event=scheduled_deadline_unconfirmed " in r.message for r in caplog.records
    )


def test_future_deadline_fails_open(caplog):
    """Deadline hasn't passed yet — this is a normal preview, not an
    unconfirmed-past-deadline claim."""
    preds = [pred()]
    with caplog.at_level("WARNING", logger="tm.extractor"):
        audit_scheduled_deadline_unconfirmed(
            preds, ARTICLE_AFTER, "2026-12-01", "scheduled", today=TODAY,
        )
    assert not any(
        "event=scheduled_deadline_unconfirmed " in r.message for r in caplog.records
    )


def test_article_published_before_the_deadline_fails_open(caplog):
    """An older preview article naturally has nothing to confirm yet — not the
    "coverage moved on" shape this targets."""
    preds = [pred()]
    with caplog.at_level("WARNING", logger="tm.extractor"):
        audit_scheduled_deadline_unconfirmed(
            preds, "2026-07-01", DEADLINE, "scheduled", today=TODAY,
        )
    assert not any(
        "event=scheduled_deadline_unconfirmed " in r.message for r in caplog.records
    )


def test_missing_article_date_fails_open(caplog):
    preds = [pred()]
    with caplog.at_level("WARNING", logger="tm.extractor"):
        audit_scheduled_deadline_unconfirmed(
            preds, None, DEADLINE, "scheduled", today=TODAY,
        )
    assert not any(
        "event=scheduled_deadline_unconfirmed " in r.message for r in caplog.records
    )


# ── summary log line ────────────────────────────────────────────────────────

def test_summary_log_line_reports_eligible_and_fired_counts(caplog):
    preds = [
        pred(event_date="2026-07-30"),  # eligible, no-op (already dated)
        pred(),  # eligible, fires
        pred(),  # eligible, fires
    ]
    with caplog.at_level("INFO", logger="tm.extractor"):
        audit_scheduled_deadline_unconfirmed(
            preds, ARTICLE_AFTER, DEADLINE, "scheduled", today=TODAY,
        )
    summary = [
        r.message for r in caplog.records
        if "event=scheduled_deadline_unconfirmed_shadow" in r.message
    ]
    assert len(summary) == 1
    assert "eligible=3" in summary[0]
    assert "fired=2" in summary[0]
    assert "n=3" in summary[0]


def test_summary_log_line_reports_zero_eligible_on_a_no_op_run(caplog):
    preds = [pred(), pred()]
    with caplog.at_level("INFO", logger="tm.extractor"):
        audit_scheduled_deadline_unconfirmed(
            preds, ARTICLE_AFTER, DEADLINE, "trend", today=TODAY,
        )
    summary = [
        r.message for r in caplog.records
        if "event=scheduled_deadline_unconfirmed_shadow" in r.message
    ]
    assert len(summary) == 1
    assert "eligible=0" in summary[0]
    assert "fired=0" in summary[0]
    assert "n=2" in summary[0]


# ── never mutates ───────────────────────────────────────────────────────────

def test_never_mutates_the_prediction():
    preds = [pred()]
    out = audit_scheduled_deadline_unconfirmed(
        preds, ARTICLE_AFTER, DEADLINE, "scheduled", today=TODAY,
    )
    assert out is preds
    assert out[0].stance == 0.8
    assert out[0].claim_strength == 0.8
    assert out[0].settled is None
    assert out[0].event_date is None
