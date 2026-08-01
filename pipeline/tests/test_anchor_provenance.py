"""cited_probability must name a source whose figure could be verified.

The class carries the largest weight in the table (4.0) and is the only one that
authorizes the stance rewrite, and nothing checks where the number came from — so
one sentence of "a market prices this at 80%" in any article we crawl buys the
strongest evidence class in the system. The prompt's own canonical example ("a
poll-aggregator model gives Likud a 22% chance") names nobody, which is exactly
the shape.

Prod audit (2026-08-01, retro#369) over all 16 cited_probability rows: ~10 name a
checkable source (Opta ×5, Kalshi ×4, Polymarket), ~6 are not cited probabilities
at all — price targets from Goldman Sachs and Citigroup, and two carrying no
figure whatsoever. One check does both jobs.

SHADOW by default: with anchor_provenance_enforced off the check logs what it
would demote and changes nothing. These tests drive both sides explicitly.
"""
import pytest

from tm.config import settings
from tm.extractor import enforce_anchor_provenance, _names_allowlisted_source
from tm.models import PredictionExtraction


@pytest.fixture
def enforced(monkeypatch):
    monkeypatch.setattr(settings, "anchor_provenance_enforced", True)


def pred(quote: str, evidence_class: str = "cited_probability", qe: float | None = 0.8,
         claim: str = "c"):
    return PredictionExtraction(
        quote=quote, claim=claim, stance=0.6, certainty=0.7,
        evidence_class=evidence_class, quantitative_estimate=qe,
    )


# ── real rows from the audited pool ───────────────────────────────────────────


@pytest.mark.parametrize("quote", [
    "Polymarket prices Sindarov at 1.52 decimal odds, implying Gukesh retains his title with approximately 34% probability",
    "Market odds on Kalshi for Strait of Hormuz traffic returning to normal by July 2027 have fallen to 47%",
    "Opta's supercomputer predicts France has a 53% chance of reaching the final",
    "Opta's model projects France's winning probability at approximately 18.7%",
])
def test_a_named_market_or_model_keeps_the_class(quote, enforced):
    [out] = enforce_anchor_provenance([pred(quote)])
    assert out.evidence_class == "cited_probability"


@pytest.mark.parametrize("quote", [
    # the attack the issue describes
    "A market prices this at 80%",
    "A poll-aggregator model gives Likud a 22% chance of winning more than 33 seats",
    "Prediction market traders assign a 49% probability that Bitcoin falls below $50,000",
    # audited rows that were never cited probabilities in the first place
    "Goldman Sachs projects Brent crude will exceed $110 per barrel by year-end 2026",
    "Citigroup's 12-month price target for Bitcoin is $82,000",
    "Fed rate hikes are increasingly likely in the near term",
])
def test_an_unattributed_figure_loses_the_premium(quote, enforced):
    [out] = enforce_anchor_provenance([pred(quote)])
    assert out.evidence_class == "reporting"


def test_the_prompts_own_example_is_deliberately_demoted(enforced):
    """The prompt teaches "a poll-aggregator model gives Likud a 22% chance" as
    cited_probability. The allowlist narrows the class relative to that on
    purpose: an unnamed aggregator is exactly the unverifiable shape. Recorded
    here so the divergence is a decision, not a surprise."""
    [out] = enforce_anchor_provenance([pred(
        "A poll-aggregator model gives Likud a 22% chance of winning more than 33 seats"
    )])
    assert out.evidence_class == "reporting"


# ── what it must not touch ────────────────────────────────────────────────────


@pytest.mark.parametrize("cls", ["reported_fact", "cited_share", "reporting", "opinion"])
def test_other_classes_are_never_touched(cls, enforced):
    [out] = enforce_anchor_provenance([pred("some unattributed 80% figure", evidence_class=cls)])
    assert out.evidence_class == cls


def test_an_unclassified_claim_is_never_touched(enforced):
    [out] = enforce_anchor_provenance([pred("an unattributed 80% figure", evidence_class=None)])
    assert out.evidence_class is None


def test_a_demoted_claim_keeps_stance_certainty_and_its_number(enforced):
    """It loses the premium, not its vote — it still counts as ordinary evidence."""
    [out] = enforce_anchor_provenance([pred("a market prices this at 80%")])
    assert (out.stance, out.certainty) == (0.6, 0.7)
    assert out.quantitative_estimate == 0.8


def test_an_empty_list_is_a_no_op(enforced):
    assert enforce_anchor_provenance([]) == []


# ── fail-closed, and the shadow default ───────────────────────────────────────


def test_a_missing_quote_falls_back_to_the_claim(enforced):
    p = pred("", claim="Kalshi traders estimate a 43% probability")
    [out] = enforce_anchor_provenance([p])
    assert out.evidence_class == "cited_probability"


def test_no_text_at_all_fails_closed(enforced):
    """Absence of provenance costs the premium — the same asymmetry
    enforce_settlement_event_date applies to an undated positive settlement.
    An unverifiable premium is the exposure itself."""
    [out] = enforce_anchor_provenance([pred("", claim="")])
    assert out.evidence_class == "reporting"


def test_shadow_is_the_default_and_changes_nothing():
    """Default config: the check runs, logs, and leaves the class alone — so
    merging this cannot move prod behaviour or the R8 snapshots."""
    assert settings.anchor_provenance_enforced is False
    [out] = enforce_anchor_provenance([pred("a market prices this at 80%")])
    assert out.evidence_class == "cited_probability"


def test_shadow_still_reports_what_it_would_do(caplog):
    with caplog.at_level("WARNING"):
        enforce_anchor_provenance([pred("a market prices this at 80%")])
    assert "event=anchor_provenance_unattributed" in caplog.text
    assert "enforced=False" in caplog.text


def test_the_demotion_target_is_config(monkeypatch, enforced):
    monkeypatch.setattr(settings, "unattributed_probability_class", "cited_share")
    [out] = enforce_anchor_provenance([pred("a market prices this at 80%")])
    assert out.evidence_class == "cited_share"


# ── the matcher itself ────────────────────────────────────────────────────────


def test_matching_is_case_insensitive():
    assert _names_allowlisted_source("odds on KALSHI are 47%") == "Kalshi"


def test_a_multiword_name_matches_across_a_line_break():
    assert _names_allowlisted_source("a Morning\nConsult poll") == "Morning Consult"


def test_matching_is_word_bounded():
    """Substring matching would let any URL or coined word buy the premium."""
    assert _names_allowlisted_source("the Pewter Report gives it 80%") is None
    assert _names_allowlisted_source("Optatively speaking, 80%") is None


def test_the_allowlist_is_config(monkeypatch):
    monkeypatch.setattr(settings, "cited_probability_source_allowlist", ["Betfair"])
    assert _names_allowlisted_source("Kalshi says 47%") is None
    assert _names_allowlisted_source("Betfair says 47%") == "Betfair"


def test_the_two_integrated_markets_are_allowlisted():
    """Polymarket and Kalshi are wired into the system and independently
    checkable — if either ever falls out of the list, that is a bug."""
    for name in ("Polymarket", "Kalshi"):
        assert name in settings.cited_probability_source_allowlist
