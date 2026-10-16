import pytest

from tm.models import PredictionExtraction, quote_states_a_number


def _pred(quote, value=1e9):
    return PredictionExtraction.model_validate({
        "quote": quote, "claim": "c", "stance": 0.5, "certainty": 0.8,
        "quantity": {"value": value, "unit": "euros", "comparator": ">="},
    })


@pytest.mark.parametrize("quote", [
    "A few billion euros have been earmarked for flood defences.",
    "Hundreds of trucks have crossed into Region X.",
    "Several thousand residents have left District D.",
])
def test_vague_count_drops_quantity(quote):
    assert _pred(quote).quantity is None


@pytest.mark.parametrize("quote", [
    "Spending reached 3.5 billion euros.",
    "About 90,000 yuan a tonne.",
    "A hundred trucks crossed.",
    "Two million residents left.",
    "Half of voters back it.",
    "Inflation fell to 4%.",
])
def test_stated_number_keeps_quantity(quote):
    assert _pred(quote).quantity is not None


def test_no_quantity_stays_none():
    p = PredictionExtraction.model_validate({"quote": "Many left.", "claim": "c", "stance": 0.5, "certainty": 0.8})
    assert p.quantity is None


def test_helper():
    assert not quote_states_a_number("a few billion")
    assert quote_states_a_number("a billion")
