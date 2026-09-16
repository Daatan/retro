import httpx

from gamma_client import GammaClient, parse_market, resolved_outcome


def test_parse_market_reads_json_string_prices(likud_event):
    m = parse_market(likud_event["markets"][0], "israel-election-likud-of-seats")
    assert m is not None
    assert m.market_id == "2110576"
    assert m.yes_price == 0.153
    assert m.has_price and not m.closed and m.resolved_outcome is None
    assert m.volume_usd == 50432.1
    assert m.event_slug == "israel-election-likud-of-seats"


def test_placeholder_market_has_no_price(likud_event):
    m = parse_market(likud_event["markets"][2], "e")
    assert m is not None and m.yes_price is None and not m.has_price


def test_closed_market_with_collapsed_price_is_resolved(likud_event):
    m = parse_market(likud_event["markets"][3], "e")
    assert m.closed and m.resolved_outcome == "YES"
    assert resolved_outcome({"closed": True, "outcomePrices": '["0", "1"]'}) == "NO"
    # open markets are never "resolved", whatever the price
    assert resolved_outcome({"closed": False, "outcomePrices": '["1", "0"]'}) is None
    # closed but price not collapsed (e.g. voided) → unknown
    assert resolved_outcome({"closed": True, "outcomePrices": '["0.5", "0.5"]'}) is None


def test_non_yesno_market_is_dropped(likud_event):
    assert parse_market(likud_event["markets"][4], "e") is None


def test_event_markets_sends_user_agent_and_filters(likud_event):
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        assert request.url.path == "/events"
        assert request.url.params["slug"] == "israel-election-likud-of-seats"
        return httpx.Response(200, json=[likud_event])

    with GammaClient(transport=httpx.MockTransport(handle)) as g:
        markets = g.event_markets("israel-election-likud-of-seats")
    assert [m.market_id for m in markets] == ["2110576", "2110577", "9000001", "9000002"]
    assert "TruthMachine" in seen[0].headers["user-agent"]


def test_unknown_event_is_empty():
    with GammaClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=[]))) as g:
        assert g.event_markets("nope") == []
