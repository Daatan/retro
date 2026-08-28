"""retro#688 — threshold-shape detection and the batch extractor routing it feeds.

The event strings below are the REAL batch corpus (`data/events/*.json`), not invented
examples. That matters: the hard part of this classifier is not recognising "exceeds
$100/barrel", it is refusing the 18 corpus events that contain a digit which decides
nothing — deal sizes, model names, durations, ordinals and bare years. Every one of
those is a live row a naive digit test would have routed to the expensive model.
"""

import pytest

from tm.archetype import is_threshold_shaped, select_extractor_model
from tm.config import Settings

HAIKU = "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"


# ── the class we want ────────────────────────────────────────────────────────
@pytest.mark.parametrize("name", [
    "Shekel drops below 4.0 NIS/USD",
    "Bank of Israel raises interest rate to 4.75%",
    "Israeli unemployment reaches 4.5%+ during war",
    "Judicial reform protest movement reaches 100K+ weekly",
    "Nvidia market cap exceeds $1 trillion USD",
    "Israeli VC investment drops 50%+ YoY (2022→2023)",
    "Israeli AI startup raises $100M+ round (multiple)",
    "Israeli tech layoffs exceed 15,000 cumulative (2023)",
    "Start-Up Nation Central: Israel drops out of top 5 startup ecosystems",
    "Brent crude oil exceeds $100/barrel (Russia-Ukraine)",
    "Europe reduces Russian gas dependency below 15% of supply",
    "Global oil price drops below $70/barrel (demand fears)",
])
def test_a_number_that_decides_the_event_is_threshold_shaped(name):
    assert is_threshold_shaped(name) is True


# ── the near-misses, which are the whole point ───────────────────────────────
@pytest.mark.parametrize("name,why", [
    ("Wiz rejects Google $23B acquisition offer",
     "a magnitude with no comparison cue — the number names the deal, it does not decide it"),
    ("Google agrees to acquire Wiz for $32B", "same shape, and the corpus has both"),
    ("UK PM Liz Truss resigns after 45 days", "a duration, not a threshold"),
    ("Hamas operational after 1 year (Oct 7, 2024)", "a duration plus a bare year"),
    ("Netanyahu forms new government (6th government)", "an ordinal identifier"),
    ("Israel strikes Iran's nuclear facilities (April 2024 retaliation)", "a date"),
    ("Google Gemini Ultra surpasses GPT-4 on benchmarks",
     "has a cue AND a digit, but '4' is a model name, not a magnitude — the case that "
     "proves the two halves are ANDed"),
    ("Trump wins 2024 US presidential election", "a bare year"),
    ("Budget 2024 passes Knesset", "a bare year"),
    ("Early Knesset elections called in 2024", "a bare year"),
    ("Hamas launches mass surprise attack (Oct 7)", "a bare day-of-month"),
    ("GPT-4 released publicly", "a model name"),
    ("DeepSeek R1 release shocks AI market", "a version number"),
])
def test_a_number_that_merely_labels_something_is_not(name, why):
    assert is_threshold_shaped(name) is False, why


@pytest.mark.parametrize("name", [
    "Assad regime falls in Syria",
    "Finland joins NATO",
    "Israel formally declares state of war",
    "",
    None,
])
def test_no_number_at_all_is_not_threshold_shaped(name):
    assert is_threshold_shaped(name) is False


# ── routing ──────────────────────────────────────────────────────────────────
def test_routing_is_off_until_the_model_is_configured():
    """The shipped default. Merging retro#688 must change nothing: the batch tree
    self-syncs to origin/main and re-execs every cycle, so a non-empty default here
    would be a live, unmeasured cost change within ~5 minutes of merge."""
    off = Settings(threshold_extractor_model="")
    assert select_extractor_model("Brent crude oil exceeds $100/barrel", off) is None


def test_a_threshold_event_routes_once_configured():
    on = Settings(threshold_extractor_model=HAIKU)
    assert select_extractor_model("Brent crude oil exceeds $100/barrel", on) == HAIKU


def test_everything_else_keeps_the_configured_default_even_when_on():
    on = Settings(threshold_extractor_model=HAIKU)
    assert select_extractor_model("Finland joins NATO", on) is None
    assert select_extractor_model("Google agrees to acquire Wiz for $32B", on) is None


def test_an_absent_event_name_routes_nowhere():
    on = Settings(threshold_extractor_model=HAIKU)
    assert select_extractor_model(None, on) is None
    assert select_extractor_model("", on) is None


def test_whitespace_only_configuration_counts_as_off():
    """A stray `THRESHOLD_EXTRACTOR_MODEL= ` in the box's .env must not be read as a
    model id and passed to litellm."""
    blank = Settings(threshold_extractor_model="   ")
    assert select_extractor_model("Brent crude oil exceeds $100/barrel", blank) is None


def test_unrouted_returns_None_rather_than_echoing_the_global():
    """`None` is what extract_predictions(model=...) already means — 'keep the
    configured global' (retro#652). Echoing settings.extractor_model here would give
    'unrouted' two spellings, and the caller threads this value straight through."""
    on = Settings(threshold_extractor_model=HAIKU)
    result = select_extractor_model("Finland joins NATO", on)
    assert result is None
    assert result != on.extractor_model
