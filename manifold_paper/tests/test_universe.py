from datetime import timedelta

import pytest

from conftest import raw_market
from manifold_client import parse_market
from universe import UniverseParams, eligible, is_personal

P = UniverseParams()


def test_good_market_is_eligible(now):
    assert eligible(parse_market(raw_market()), P, now)
    assert eligible(parse_market(raw_market(close_in_days=5.5)), P, now)


@pytest.mark.parametrize("kw", [
    dict(bettors=3), dict(volume=10.0), dict(close_in_days=0.1), dict(close_in_days=4.0), dict(close_in_days=90),
    dict(resolved=True), dict(prob=2.0),
])
def test_filters(kw, now):
    assert not eligible(parse_market(raw_market(**kw)), P, now)


@pytest.mark.parametrize("q,personal", [
    ("Will I like the UATX High School Honors event? [SHORT FUSE]", True),
    ("Will my sourdough rise tomorrow?", True),
    ("Will this market reach 100 traders?", True),
    ("Will the coin flip land heads?", True),
    ("Daily coinflip #412", True),
    ("Daily: will it rain?", True),
    ("Will the ECB cut rates at its December 2026 meeting?", False),
    ("Will Likud win more than 30 seats?", False),
])
def test_personal_filter(q, personal):
    assert is_personal(q) is personal
