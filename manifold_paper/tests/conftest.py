"""Payload shapes follow docs.manifold.markets/api LiteMarket / FullMarket and a
live /v0/markets row (2026-10-11); times are epoch milliseconds."""
import copy
from datetime import datetime, timedelta, timezone

import pytest

# Wall-clock based, not a fixed date: `parse_market` decides `closed` against the real clock,
# so a fixed NOW made every fixture market with a near close time "closed" once the real date
# passed it (the 2026-10-11 pin broke CI on 2026-10-17).
NOW = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)


def ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def raw_market(mid="m1", *, prob=0.40, close_in_days=7.0, bettors=40, volume=2000.0,
               question="Will the ECB cut rates at its December 2026 meeting?",
               resolved=False, resolution=None, outcome_type="BINARY", creator="alice", full=False) -> dict:
    d = {
        "id": mid, "creatorId": "uid", "creatorUsername": creator, "creatorName": creator.title(),
        "createdTime": ms(NOW - timedelta(days=3)), "closeTime": ms(NOW + timedelta(days=close_in_days)),
        "question": question, "slug": f"slug-{mid}", "url": f"https://manifold.markets/{creator}/slug-{mid}",
        "pool": {"NO": 100, "YES": 100}, "probability": prob, "p": 0.5, "totalLiquidity": 100,
        "outcomeType": outcome_type, "mechanism": "cpmm-1", "volume": volume, "volume24Hours": 10,
        "isResolved": resolved, "uniqueBettorCount": bettors, "lastUpdatedTime": ms(NOW),
    }
    if resolved:
        d.update(resolution=resolution or "YES", resolutionTime=ms(NOW))
    if full:
        d["textDescription"] = "Resolves YES if the ECB announces a deposit-rate cut."
        d["description"] = {"type": "doc", "content": []}
    return d


@pytest.fixture
def now() -> datetime:
    return NOW


@pytest.fixture
def mk():
    return lambda *a, **k: copy.deepcopy(raw_market(*a, **k))
