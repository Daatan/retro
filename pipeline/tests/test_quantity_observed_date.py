"""retro#935 pattern 3: a quantity dated after the article is a projection, not a figure."""
from tm.extractor import enforce_quantity_observed_date
from tm.models import PredictionExtraction


def _pred(as_of, value=110.0):
    q = {"value": value, "unit": "USD per barrel", "comparator": "="}
    if as_of is not None:
        q["as_of"] = as_of
    return PredictionExtraction.model_validate({
        "quote": f"UBS sees the benchmark at ${value:g} per barrel at the end of the current year.",
        "claim": "c", "stance": -0.4, "certainty": 0.6, "quantity": q,
    })


def test_future_as_of_drops_quantity():
    (p,) = enforce_quantity_observed_date([_pred("2026-12-31")], "2026-11-02")
    assert p.quantity is None
    assert p.stance == -0.4  # the claim itself is untouched


def test_same_day_and_past_keep_quantity():
    preds = enforce_quantity_observed_date([_pred("2026-11-02"), _pred("2026-10-27")], "2026-11-02")
    assert all(p.quantity is not None for p in preds)


def test_one_day_slack():
    (kept,) = enforce_quantity_observed_date([_pred("2026-11-01")], "2026-10-31")
    (dropped,) = enforce_quantity_observed_date([_pred("2026-11-02")], "2026-10-31")
    assert kept.quantity is not None
    assert dropped.quantity is None


def test_fails_open():
    assert enforce_quantity_observed_date([_pred("2026-12-31")], None)[0].quantity is not None
    assert enforce_quantity_observed_date([_pred("2026-12-31")], "garbage")[0].quantity is not None
    assert enforce_quantity_observed_date([_pred(None)], "2026-11-02")[0].quantity is not None
    assert enforce_quantity_observed_date([_pred("end of year")], "2026-11-02")[0].quantity is not None


def test_timestamp_article_date():
    (p,) = enforce_quantity_observed_date([_pred("2026-12-31")], "2026-11-02T08:15:00Z")
    assert p.quantity is None
