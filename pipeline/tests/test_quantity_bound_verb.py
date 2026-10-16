import pytest

from tm.models import PredictionExtraction, quote_asserts_lower_bound


def _pred(quote, comparator="=", value=90000, unit="yuan per tonne"):
    return PredictionExtraction.model_validate({
        "quote": quote, "claim": "c", "stance": 0.5, "certainty": 0.8,
        "quantity": {"value": value, "unit": unit, "comparator": comparator},
    })


@pytest.mark.parametrize("quote,value", [
    ("Lithium carbonate topped 90,000 yuan a tonne on Tuesday.", 90000),
    ("The jobless rate breached 7 percent.", 7),
    ("Oil crossed $100 a barrel.", 100),
    ("Debt exceeded about 3 million euros.", 3_000_000),
    ("Output passed 1.5 billion units.", 1.5e9),
])
def test_bound_verb_rewrites_equals_to_greater_than(quote, value):
    assert _pred(quote, value=value).quantity.comparator == ">"


@pytest.mark.parametrize("quote,value", [
    ("The price failed to top 90,000 yuan.", 90000),
    ("It did not breach 7 percent.", 7),
    ("Prices reached 90,000 yuan a tonne.", 90000),       # a level, not a bound
    ("Lithium topped 80,000 yuan a tonne.", 90000),        # different number
    ("Output is 90,000 tonnes, topped only by Chile.", 90000),
])
def test_no_rewrite(quote, value):
    assert _pred(quote, value=value).quantity.comparator == "="


def test_other_comparators_untouched():
    assert _pred("Prices topped 90,000 yuan.", comparator=">=").quantity.comparator == ">="
    assert _pred("Prices topped 90,000 yuan.", comparator="<").quantity.comparator == "<"


def test_helper_scale_words():
    assert quote_asserts_lower_bound("It surpassed 3 million containers", 3e6)
    assert not quote_asserts_lower_bound("It surpassed 3 million containers", 3e3)
