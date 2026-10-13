import json
from datetime import timedelta

import httpx
import pytest

from conftest import NOW, raw_market
from ledger import Ledger
from manifold_client import ManifoldClient, parse_market
from oracle_client import OracleClient
from paper import (Params, gather_universe, resolution_criteria, run, snapshot_row, stance_to_prob,
                   trade_row, trade_side, wants_oracle_call)
from universe import UniverseParams

P = Params()
UP = UniverseParams()


def mkm(mid="1", **kw):
    return parse_market(raw_market(mid, **kw))


class TestStanceToProb:
    def test_boundaries(self):
        assert [stance_to_prob(x) for x in (-1.0, 0.0, 1.0)] == [0.0, 0.5, 1.0]

    def test_percent_scale_is_rejected(self):
        with pytest.raises(ValueError):
            stance_to_prob(50.0)


class TestDecisions:
    def test_never_asked(self):
        assert wants_oracle_call(mkm(), None, P, NOW) is True

    def test_a_market_with_a_number_is_never_asked_again(self):
        answered = {"ts": (NOW - timedelta(days=30)).isoformat(), "oracle_yes": 0.3}
        assert wants_oracle_call(mkm(), answered, P, NOW) is False

    def test_no_answer_is_retried_only_after_the_stale_window(self):
        fresh = {"ts": (NOW - timedelta(hours=6)).isoformat(), "oracle_reason": "all_articles_off_topic"}
        stale = {"ts": (NOW - timedelta(hours=73)).isoformat(), "oracle_reason": "all_articles_off_topic"}
        assert wants_oracle_call(mkm(), fresh, P, NOW) is False
        assert wants_oracle_call(mkm(), stale, P, NOW) is True

    def test_too_close_to_resolution_cannot_be_scored_so_is_not_asked(self):
        assert wants_oracle_call(mkm(close_in_days=4.0), None, P, NOW) is False
        assert wants_oracle_call(mkm(close_in_days=5.5), None, P, NOW) is True

    def test_scoring_band_is_not_asked(self):
        assert wants_oracle_call(mkm(prob=0.05), None, P, NOW) is False
        assert wants_oracle_call(mkm(prob=0.95), None, P, NOW) is False

    def test_closed_or_longshot_is_never_asked(self):
        assert wants_oracle_call(mkm(resolved=True), None, P, NOW) is False
        assert wants_oracle_call(mkm(prob=0.001), None, P, NOW) is False

    def test_trade_side_threshold_and_band(self):
        assert trade_side(0.55, 0.40, P) == "YES"
        assert trade_side(0.25, 0.40, P) == "NO"
        assert trade_side(0.45, 0.40, P) is None
        assert trade_side(0.99, 0.97, P) is None  # outside the tradeable band

    def test_trade_row_is_mana_and_paper(self):
        t = trade_row(mkm(prob=0.20), "NO", 0.05, P, NOW, "r")
        assert t["entry_price"] == 0.8 and t["shares"] == 125.0
        assert t["paper"] is True and "notional_mana" in t and "notional_usd" not in t

    def test_snapshot_carries_manifold_fields(self):
        row = snapshot_row(mkm(resolved=True, resolution="MKT"), None, NOW, "r")
        assert row["resolution_raw"] == "MKT" and row["resolved_outcome"] is None
        assert row["unique_bettors"] == 40 and row["volume_mana"] == 2000.0 and row["creator"] == "alice"

    def test_resolution_criteria_trims_at_a_word(self):
        m = parse_market(raw_market(full=True))
        assert resolution_criteria(m).startswith("Resolves YES")
        assert resolution_criteria(mkm()) is None


def _client(search_rows, by_id=None):
    by_id = by_id or {}

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.method == "GET"
        if req.url.path == "/v0/search-markets":
            return httpx.Response(200, json=search_rows if req.url.params["offset"] == "0" else [])
        mid = req.url.path.rsplit("/", 1)[-1]
        return httpx.Response(200, json=by_id[mid]) if mid in by_id else httpx.Response(404, json={})

    return ManifoldClient(transport=httpx.MockTransport(handler))


def _oracle(handler):
    return OracleClient("k", transport=httpx.MockTransport(handler))


def _full(mid, **kw):
    return raw_market(mid, full=True, **kw)


def test_run_snapshots_candidates_filters_junk_and_trades_on_edge(tmp_path):
    rows = [raw_market("good", prob=0.30, close_in_days=8), raw_market("thin", bettors=2),
            raw_market("mine", question="Will I finish my novel?"), raw_market("poll", outcome_type="POLL")]
    asked = []

    def oracle_handler(req):
        asked.append(json.loads(req.content))
        return httpx.Response(200, json={"mean": 0.5, "ci_low": 0.3, "ci_high": 0.7, "articles_used": 5})

    led = Ledger(tmp_path)
    with _client(rows, {"good": _full("good", prob=0.30, close_in_days=8)}) as c, _oracle(oracle_handler) as o:
        stats = run(c, o, led, P, UP, now=NOW)
    assert stats["markets"] == 1 and stats["oracle_calls"] == 1 and stats["trades_opened"] == 1
    assert asked[0]["resolution_criteria"].startswith("Resolves YES")  # description fetched by id
    (s,) = list(led.snapshots())
    assert s["market_id"] == "good" and s["oracle_yes"] == 0.75 and s["edge"] == 0.45
    (t,) = list(led.trades())
    assert t["side"] == "YES" and t["entry_price"] == 0.30


def test_tracked_market_is_rechecked_by_id_and_resolution_recorded(tmp_path):
    led = Ledger(tmp_path)
    led.append_snapshot({"market_id": "t1", "ts": NOW.isoformat(), "oracle_yes": 0.6, "market_yes": 0.4,
                         "resolved_outcome": None, "resolution_raw": None})
    # it dropped out of the open search, but /v0/market/t1 now says resolved NO
    with _client([], {"t1": _full("t1", resolved=True, resolution="NO", prob=1.0)}) as c, \
            _oracle(lambda r: pytest.fail("a resolved market must not be forecast")) as o:
        run(c, o, led, P, UP, now=NOW + timedelta(days=2))
    last = led.latest_snapshot_by_market()["t1"]
    assert last["resolved_outcome"] == "NO" and last["oracle_yes"] is None


def test_resolved_market_is_no_longer_rechecked(tmp_path):
    led = Ledger(tmp_path)
    led.append_snapshot({"market_id": "done", "ts": NOW.isoformat(), "resolved_outcome": "YES", "resolution_raw": "YES"})
    seen = []

    def handler(req):
        seen.append(req.url.path)
        return httpx.Response(200, json=[])

    with ManifoldClient(transport=httpx.MockTransport(handler)) as c:
        gather_universe(c, led, UP, NOW, search_pages=1)
    assert seen == ["/v0/search-markets"]


def test_cancelled_market_stops_being_tracked_and_is_not_scored(tmp_path):
    led = Ledger(tmp_path)
    led.append_snapshot({"market_id": "c1", "ts": NOW.isoformat(), "oracle_yes": 0.5, "market_yes": 0.5})
    with _client([], {"c1": _full("c1", resolved=True, resolution="CANCEL")}) as c, _oracle(lambda r: None) as o:
        run(c, o, led, P, UP, now=NOW + timedelta(days=1))
    last = led.latest_snapshot_by_market()["c1"]
    assert last["resolution_raw"] == "CANCEL" and last["resolved_outcome"] is None


def test_oracle_cap_goes_to_soonest_closing(tmp_path):
    rows = [raw_market("far", close_in_days=20), raw_market("near", close_in_days=6), raw_market("mid", close_in_days=9)]
    order = []

    def oracle_handler(req):
        order.append(json.loads(req.content)["question"])
        return httpx.Response(200, json={"mean": 0.0, "ci_low": -0.1, "ci_high": 0.1, "articles_used": 1})

    led = Ledger(tmp_path)
    full = {r["id"]: dict(r, textDescription="d") for r in rows}
    with _client(rows, full) as c, _oracle(oracle_handler) as o:
        stats = run(c, o, led, Params(max_markets_per_run=1), UP, now=NOW)
    assert stats["oracle_calls"] == 1
    assert [s["market_id"] for s in led.snapshots() if s["oracle_yes"] is not None] == ["near"]


def test_oracle_error_does_not_abort_run(tmp_path):
    led = Ledger(tmp_path)
    rows = [raw_market("a"), raw_market("b")]
    with _client(rows, {r["id"]: dict(r, textDescription="d") for r in rows}) as c, \
            _oracle(lambda r: httpx.Response(500, json={})) as o:
        stats = run(c, o, led, P, UP, now=NOW)
    assert stats["errors"] == 2 and len(list(led.snapshots())) == 2


def test_dry_run_writes_nothing(tmp_path):
    led = Ledger(tmp_path)
    with _client([raw_market("a")]) as c, _oracle(lambda r: pytest.fail("no Oracle call in dry run")) as o:
        run(c, o, led, P, UP, now=NOW, dry_run=True)
    assert list(led.snapshots()) == [] and list(led.trades()) == []


def test_answered_market_is_not_re_asked_on_later_runs(tmp_path):
    """Regression (retro#918): a price-only row after an answer used to hide it, so every
    market was re-asked on every run and the cap never reached the later-closing ones."""
    led = Ledger(tmp_path)
    rows = [raw_market("a", close_in_days=8)]
    full = {"a": dict(rows[0], textDescription="d")}
    calls = []

    def oracle_handler(req):
        calls.append(1)
        return httpx.Response(200, json={"mean": 0.0, "ci_low": -0.1, "ci_high": 0.1, "articles_used": 3})

    for hours in (0, 6, 12, 18):
        with _client(rows, full) as c, _oracle(oracle_handler) as o:
            run(c, o, led, P, UP, now=NOW + timedelta(hours=hours))
    assert len(calls) == 1 and len(list(led.snapshots())) == 4


def test_cap_prefers_never_asked_over_retries(tmp_path):
    led = Ledger(tmp_path)
    rows = [raw_market("old", close_in_days=6), raw_market("new", close_in_days=25)]
    full = {r["id"]: dict(r, textDescription="d") for r in rows}
    led.append_snapshot({"ts": (NOW - timedelta(days=4)).isoformat(), "run_id": "r0", "market_id": "old",
                         "market_slug": "s", "question": "Q", "end_date": rows[0] and "x", "closed": False,
                         "resolved_outcome": None, "resolution_raw": None, "market_yes": 0.4,
                         "oracle_yes": None, "oracle_reason": "no_usable_predictions", "edge": None})
    asked = []

    def oracle_handler(req):
        asked.append(json.loads(req.content)["question"])
        return httpx.Response(200, json={"mean": 0.0, "ci_low": -0.1, "ci_high": 0.1, "articles_used": 1})

    with _client(rows, full) as c, _oracle(oracle_handler) as o:
        run(c, o, led, Params(max_markets_per_run=1), UP, now=NOW)
    answered = [s["market_id"] for s in led.snapshots() if s["oracle_yes"] is not None]
    assert answered == ["new"]  # a never-asked market beats a retry even though it closes later
