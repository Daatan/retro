import json

import pytest

# Trimmed from a live Gamma /events?slug=israel-election-likud-of-seats response
# (2026-09-16) plus one placeholder market (no prices, as Gamma ships "Party A")
# and one closed+resolved market, so every branch of parse_market is covered by
# real payload shapes.
LIKUD_EVENT = {
    "slug": "israel-election-likud-of-seats",
    "title": "Israel Election: Likud # of seats?",
    "endDate": "2026-10-27T12:00:00Z",
    "markets": [
        {"id": "2110576", "slug": "likud-lt-20", "question": "Will Likud win fewer than 20 seats in the 2026 Israeli legislative election?",
         "description": "Legislative elections are expected to be held in Israel on October 27, 2026. Resolves YES if Likud wins fewer than 20 seats.",
         "outcomes": json.dumps(["Yes", "No"]), "outcomePrices": json.dumps(["0.153", "0.847"]),
         "closed": False, "endDate": "2026-10-27T12:00:00Z", "volumeNum": 50432.1},
        {"id": "2110577", "slug": "likud-20-24", "question": "Will Likud win 20-24 seats in the 2026 Israeli legislative election?",
         "description": "x", "outcomes": json.dumps(["Yes", "No"]), "outcomePrices": json.dumps(["0.415", "0.585"]),
         "closed": False, "endDate": "2026-10-27T12:00:00Z", "volumeNum": 20230},
        {"id": "9000001", "slug": "party-a", "question": "Will Party A win the most seats?",
         "description": "", "outcomes": json.dumps(["Yes", "No"]), "outcomePrices": "[]",
         "closed": False, "endDate": "2026-10-27T12:00:00Z", "volumeNum": 0},
        {"id": "9000002", "slug": "resolved-yes", "question": "Did Zionist Home and Yashar run on the same list?",
         "description": "", "outcomes": json.dumps(["Yes", "No"]), "outcomePrices": json.dumps(["1", "0"]),
         "closed": True, "endDate": "2026-09-08T12:00:00Z", "volumeNum": 24827},
        {"id": "9000003", "slug": "multi", "question": "Which party wins?",
         "description": "", "outcomes": json.dumps(["Likud", "Yashar"]), "outcomePrices": json.dumps(["0.4", "0.6"]),
         "closed": False, "endDate": None, "volumeNum": 1},
    ],
}


@pytest.fixture
def likud_event() -> dict:
    return json.loads(json.dumps(LIKUD_EVENT))
