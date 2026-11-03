import json
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from gamma_client import GammaClient, Market
from ledger import Ledger
from oracle_client import OracleClient
from paper import (Params, coherent_yes, resolution_criteria, run, snapshot_row, stance_to_prob, trade_row,
                   trade_side, wants_oracle_call)

NOW = datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc)
P = Params()


def mk(mid="1", yes=0.40, closed=False, desc="d", resolved=None, neg_risk=False) -> Market:
    return Market(market_id=mid, event_slug="e", market_slug="s", question=f"Q{mid}?", description=desc,
                  yes_price=yes, closed=closed, end_date="2026-10-27T12:00:00Z", volume_usd=1000.0,
                  resolved_outcome=resolved, neg_risk=neg_risk)


class TestStanceToProb:
    def test_boundaries(self):
        assert stance_to_prob(-1.0) == 0.0
        assert stance_to_prob(0.0) == 0.5
        assert stance_to_prob(1.0) == 1.0

    def test_percent_scale_is_rejected(self):
        # oracleSnapshot.mean is a percent in daatan; the API's mean is not. A
        # percent leaking in here must fail loudly, not become p=25.5.
        with pytest.raises(ValueError):
            stance_to_prob(50.0)


class TestWantsOracleCall:
    def test_never_asked(self):
        assert wants_oracle_call(mk(), None, P, NOW) is True

    def test_recent_answer_is_not_stale(self):
        last = {"ts": (NOW - timedelta(hours=6)).isoformat(), "oracle_yes": 0.3}
        assert wants_oracle_call(mk(), last, P, NOW) is False

    def test_recent_insufficient_data_is_not_retried_early(self):
        last = {"ts": (NOW - timedelta(hours=6)).isoformat(), "oracle_yes": None, "oracle_reason": "no_search_results"}
        assert wants_oracle_call(mk(), last, P, NOW) is False

    def test_stale_answer_is_re_asked(self):
        last = {"ts": (NOW - timedelta(hours=21)).isoformat(), "oracle_yes": 0.3}
        assert wants_oracle_call(mk(), last, P, NOW) is True

    def test_closed_or_unpriced_or_longshot_is_never_asked(self):
        assert wants_oracle_call(mk(closed=True), None, P, NOW) is False
        assert wants_oracle_call(mk(yes=None), None, P, NOW) is False
        assert wants_oracle_call(mk(yes=0.0005), None, P, NOW) is False
        assert wants_oracle_call(mk(yes=0.995), None, P, NOW) is False


class TestTradeSide:
    def test_threshold(self):
        assert trade_side(0.55, 0.40, P) == "YES"
        assert trade_side(0.25, 0.40, P) == "NO"
        assert trade_side(0.49, 0.40, P) is None
        assert trade_side(0.50, 0.40, P) == "YES"  # exactly at threshold counts

    def test_band(self):
        assert trade_side(0.30, 0.03, P) is None
        assert trade_side(0.50, 0.97, P) is None

    def test_no_side_entry_price_is_one_minus_yes(self):
        t = trade_row(mk(yes=0.895), "NO", 0.60, P, NOW, "r")
        assert t["entry_price"] == pytest.approx(0.105)
        assert t["shares"] == pytest.approx(100 / 0.105, rel=1e-3)
        assert t["paper"] is True and t["side"] == "NO"


class TestSnapshotRow:
    def test_oracle_fields_are_converted_to_yes_probability(self):
        resp = {"mean": 0.2, "ci_low": -0.4, "ci_high": 0.8, "confidence": "low", "articles_used": 3,
                "articles_found": 12, "provider": "news_indexer", "fallback_path": "primary",
                "provenance": {"build": 812}}
        row = snapshot_row(mk(yes=0.40), resp, NOW, "r")
        assert row["oracle_yes"] == 0.6
        assert row["edge"] == pytest.approx(0.2)
        assert row["oracle_ci_yes"] == [0.3, 0.9]
        assert row["articles_used"] == 3 and row["oracle_build"] == 812

    def test_insufficient_data_records_reason_not_number(self):
        row = snapshot_row(mk(), {"insufficient_data": True, "reason": "no_search_results", "mean": 0.0}, NOW, "r")
        assert row["oracle_yes"] is None and row["oracle_reason"] == "no_search_results"

    def test_price_only(self):
        row = snapshot_row(mk(yes=0.0005), None, NOW, "r")
        assert row["oracle_yes"] is None and row["oracle_reason"] is None and row["market_yes"] == 0.0005


def test_resolution_criteria_is_trimmed_at_a_word():
    m = mk(desc="word " * 1000)
    c = resolution_criteria(m)
    assert c is not None and len(c) <= 1500 and not c.endswith(" wor")
    assert resolution_criteria(mk(desc="")) is None


# ── partition coherence (retro#831) ─────────────────────────────────────────

class TestCoherentYes:
    def test_normalises_a_negrisk_partition(self):
        ms = [mk("a", 0.2, neg_risk=True), mk("b", 0.5, neg_risk=True)]
        assert coherent_yes(ms, {"a": 0.7, "b": 0.5}, P) == ({"a": 0.5833, "b": 0.4167}, 1.2)

    def test_scales_up_with_one_formula_when_the_partition_is_complete(self):
        ms = [mk("a", 0.2, neg_risk=True), mk("b", 0.5, neg_risk=True)]
        assert coherent_yes(ms, {"a": 0.2, "b": 0.6}, P) == ({"a": 0.25, "b": 0.75}, 0.8)

    def test_same_event_is_not_enough_without_negrisk(self):
        # "which parties win a seat": one event, several YES outcomes, negRisk false
        ms = [mk("a", 0.2), mk("b", 0.5)]
        assert coherent_yes(ms, {"a": 0.7, "b": 0.5}, P) is None

    def test_a_single_market_event_is_left_alone(self):
        assert coherent_yes([mk("a", 0.2, neg_risk=True)], {"a": 0.7}, P) is None

    def test_a_bucket_in_band_with_no_number_keeps_the_raw_numbers(self):
        ms = [mk("a", 0.2, neg_risk=True), mk("b", 0.5, neg_risk=True)]
        assert coherent_yes(ms, {"a": 0.7}, P) is None

    def test_bucket_outside_the_band_uses_its_price_and_gets_no_number(self):
        ms = [mk("a", 0.2, neg_risk=True), mk("b", 0.5, neg_risk=True), mk("c", 0.01, neg_risk=True)]
        # a stale Oracle number on "c" must not be used now that it left the band
        values, total = coherent_yes(ms, {"a": 0.7, "b": 0.5, "c": 0.9}, P)
        assert total == 1.21 and set(values) == {"a", "b"} and values["a"] == round(0.7 / 1.21, 4)

    def test_unpriced_placeholder_is_ignored(self):
        ms = [mk("a", 0.2, neg_risk=True), mk("b", 0.5, neg_risk=True), mk("x", None, neg_risk=True)]
        assert coherent_yes(ms, {"a": 0.7, "b": 0.5}, P) == ({"a": 0.5833, "b": 0.4167}, 1.2)

    def test_resolved_no_sibling_counts_as_zero_not_as_its_stale_number(self):
        ms = [mk("a", 0.2, neg_risk=True), mk("b", 0.5, neg_risk=True),
              mk("c", 0.0, closed=True, resolved="NO", neg_risk=True)]
        assert coherent_yes(ms, {"a": 0.7, "b": 0.5, "c": 0.9}, P) == ({"a": 0.5833, "b": 0.4167}, 1.2)

    def test_resolved_yes_sibling_makes_open_buckets_zero(self):
        ms = [mk("a", 0.2, neg_risk=True), mk("c", 1.0, closed=True, resolved="YES", neg_risk=True)]
        assert coherent_yes(ms, {"a": 0.7}, P) == ({"a": 0.0}, 1.0)

    def test_closed_but_unresolved_sibling_keeps_the_raw_numbers(self):
        ms = [mk("a", 0.2, neg_risk=True), mk("c", 0.5, closed=True, neg_risk=True)]
        assert coherent_yes(ms, {"a": 0.7, "c": 0.4}, P) is None

    def test_fully_closed_partition_has_nothing_to_normalise(self):
        ms = [mk("a", 0.0, closed=True, resolved="NO", neg_risk=True),
              mk("b", 1.0, closed=True, resolved="YES", neg_risk=True)]
        assert coherent_yes(ms, {"a": 0.7, "b": 0.5}, P) is None

    def test_all_zero_numbers_do_not_divide_by_zero(self):
        ms = [mk("a", 0.2, neg_risk=True), mk("b", 0.5, neg_risk=True)]
        assert coherent_yes(ms, {"a": 0.0, "b": 0.0}, P) is None


class TestTradeRowBasis:
    def test_raw_by_default(self):
        t = trade_row(mk(yes=0.2), "YES", 0.7, P, NOW, "r")
        assert t["decision_basis"] == "raw" and t["oracle_yes_coherent"] is None and t["edge"] == 0.5

    def test_coherent_keeps_raw_and_reports_the_decision_edge(self):
        t = trade_row(mk(yes=0.2), "YES", 0.7, P, NOW, "r", coherent=0.5)
        assert t["decision_basis"] == "coherent" and t["oracle_yes"] == 0.7
        assert t["oracle_yes_coherent"] == 0.5 and t["edge"] == 0.3


# ── end-to-end run against mocked Gamma + Oracle ────────────────────────────

def _gamma(likud_event):
    return GammaClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=[likud_event])))


def _oracle(handler):
    return OracleClient("k", transport=httpx.MockTransport(handler))


def test_run_snapshots_everything_priced_and_trades_on_edge(tmp_path, likud_event):
    asked: list[dict] = []

    def oracle_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        asked.append(body)
        # market 2110576 priced 0.153: Oracle says 0.7 (mean 0.4) → BUY YES
        # market 2110577 priced 0.415: Oracle says 0.5 (mean 0.0) → no edge
        mean = 0.4 if "fewer than 20" in body["question"] else 0.0
        return httpx.Response(200, json={"mean": mean, "ci_low": mean - 0.2, "ci_high": mean + 0.2,
                                         "articles_used": 4, "articles_found": 9, "confidence": "medium"})

    led = Ledger(tmp_path)
    with _gamma(likud_event) as g, _oracle(oracle_handler) as o:
        stats = run(g, o, led, ("israel-election-likud-of-seats",), P, now=NOW)

    assert stats["oracle_calls"] == 2  # the placeholder (no price) and the closed market are not asked
    assert stats["skipped_no_price"] == 1
    assert stats["trades_opened"] == 1
    assert asked[0]["resolution_criteria"].startswith("Legislative elections")
    snaps = list(led.snapshots())
    assert {s["market_id"] for s in snaps} == {"2110576", "2110577", "9000002"}
    closed = next(s for s in snaps if s["market_id"] == "9000002")
    assert closed["closed"] is True and closed["resolved_outcome"] == "YES" and closed["oracle_yes"] is None
    trades = list(led.trades())
    assert len(trades) == 1 and trades[0]["market_id"] == "2110576" and trades[0]["side"] == "YES"
    assert trades[0]["entry_price"] == 0.153


def _partition_event(likud_event):
    # the two live buckets of the fixture, as a negRisk partition
    likud_event["markets"] = [dict(m, negRisk=True) for m in likud_event["markets"] if m["id"] in ("2110576", "2110577")]
    return likud_event


def _answer(mean):
    return httpx.Response(200, json={"mean": mean, "ci_low": mean - 0.2, "ci_high": mean + 0.2, "articles_used": 4,
                                     "articles_found": 9, "confidence": "medium"})


def test_partition_trades_on_the_coherent_number_and_keeps_the_raw_one(tmp_path, likud_event):
    # raw 0.7 + 0.7 → coherent 0.5 each. Against 0.415 the raw edge (0.285) would
    # trade and the coherent one (0.085) does not; against 0.153 both trade.
    led = Ledger(tmp_path)
    with _gamma(_partition_event(likud_event)) as g, _oracle(lambda r: _answer(0.4)) as o:
        stats = run(g, o, led, ("israel-election-likud-of-seats",), P, now=NOW)

    assert stats["coherent_events"] == 1 and stats["trades_opened"] == 1
    snaps = {s["market_id"]: s for s in led.snapshots()}
    assert snaps["2110577"]["oracle_yes"] == 0.7 and snaps["2110577"]["edge"] == 0.285
    assert snaps["2110577"]["oracle_yes_coherent"] == 0.5 and snaps["2110577"]["edge_coherent"] == 0.085
    assert snaps["2110577"]["coherent_sum"] == 1.4
    t = next(iter(led.trades()))
    assert t["market_id"] == "2110576" and t["decision_basis"] == "coherent"
    assert t["oracle_yes"] == 0.7 and t["oracle_yes_coherent"] == 0.5 and t["edge"] == 0.347


def test_partition_uses_a_sibling_number_from_an_earlier_run(tmp_path, likud_event):
    led = Ledger(tmp_path)
    led.append_snapshot({"ts": (NOW - timedelta(hours=6)).isoformat(), "market_id": "2110577", "oracle_yes": 0.7})
    led.append_snapshot({"ts": (NOW - timedelta(hours=1)).isoformat(), "market_id": "2110577", "oracle_yes": None,
                         "oracle_reason": None})  # the latest row is price-only
    asked: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        asked.append(json.loads(request.content)["question"])
        return _answer(0.4)

    p = Params(max_markets_per_run=1)
    with _gamma(_partition_event(likud_event)) as g, _oracle(handler) as o:
        run(g, o, led, ("israel-election-likud-of-seats",), p, now=NOW)

    assert len(asked) == 1 and "fewer than 20" in asked[0]
    snaps = list(led.snapshots())[-2:]
    assert [s["oracle_yes_coherent"] for s in snaps] == [0.5, 0.5]
    assert snaps[1]["oracle_yes"] is None  # price-only: no fresh number, so no trade on it
    assert [t["market_id"] for t in led.trades()] == ["2110576"]


def test_partition_with_an_unasked_sibling_stays_raw_and_says_so(tmp_path, likud_event):
    led = Ledger(tmp_path)
    with _gamma(_partition_event(likud_event)) as g, _oracle(lambda r: _answer(0.4)) as o:
        stats = run(g, o, led, ("israel-election-likud-of-seats",), Params(max_markets_per_run=1), now=NOW)
    assert stats["coherent_events"] == 0
    t = next(iter(led.trades()))
    assert t["decision_basis"] == "raw" and t["oracle_yes_coherent"] is None and t["edge"] == 0.547


def test_second_run_within_stale_window_makes_no_oracle_calls_and_no_duplicate_trade(tmp_path, likud_event):
    calls = 0

    def oracle_handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"mean": 0.4, "ci_low": 0.2, "ci_high": 0.6, "articles_used": 4})

    led = Ledger(tmp_path)
    with _gamma(likud_event) as g, _oracle(oracle_handler) as o:
        run(g, o, led, ("israel-election-likud-of-seats",), P, now=NOW)
        first_calls = calls
        stats = run(g, o, led, ("israel-election-likud-of-seats",), P, now=NOW + timedelta(hours=6))
    assert first_calls == 2 and calls == 2
    assert stats["oracle_calls"] == 0 and stats["trades_opened"] == 0
    assert len(list(led.trades())) == 2  # both markets had edge on run 1 (0.7 vs 0.153 and 0.415); none re-opened
    # price-only snapshots were still written on run 2 (the MTM series keeps going)
    assert len([s for s in led.snapshots() if s["run_id"].startswith("20260916T18")]) == 3


def test_markets_are_re_asked_after_the_stale_window_not_after_a_price_only_row(tmp_path, likud_event):
    # the 6h timer with a 20h window: asked at 0h, then not until 24h (retro#1047)
    calls_at: list[int] = []
    hour = 0

    def handler(request: httpx.Request) -> httpx.Response:
        calls_at.append(hour)
        return _answer(0.4)

    led = Ledger(tmp_path)
    with _gamma(likud_event) as g, _oracle(handler) as o:
        for hour in (0, 6, 12, 18, 24):
            run(g, o, led, ("israel-election-likud-of-seats",), P, now=NOW + timedelta(hours=hour))
    assert calls_at == [0, 0, 24, 24]  # two askable markets, each asked twice


def test_a_failed_call_is_retried_on_the_next_run(tmp_path, likud_event):
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500) if calls <= 2 else _answer(0.0)

    led = Ledger(tmp_path)
    with _gamma(likud_event) as g, _oracle(handler) as o:
        first = run(g, o, led, ("israel-election-likud-of-seats",), P, now=NOW)
        second = run(g, o, led, ("israel-election-likud-of-seats",), P, now=NOW + timedelta(hours=6))
    assert first["errors"] == 2 and second["oracle_calls"] == 2


def test_max_markets_per_run_caps_oracle_calls(tmp_path, likud_event):
    led = Ledger(tmp_path)
    handler = lambda r: httpx.Response(200, json={"mean": 0.0, "ci_low": -0.1, "ci_high": 0.1, "articles_used": 1})
    with _gamma(likud_event) as g, _oracle(handler) as o:
        stats = run(g, o, led, ("x",), Params(max_markets_per_run=1), now=NOW)
    assert stats["oracle_calls"] == 1


def test_oracle_error_does_not_abort_the_run(tmp_path, likud_event):
    led = Ledger(tmp_path)
    with _gamma(likud_event) as g, _oracle(lambda r: httpx.Response(500, json={"detail": "boom"})) as o:
        stats = run(g, o, led, ("x",), P, now=NOW)
    assert stats["errors"] == 2 and stats["oracle_calls"] == 0
    assert len(list(led.snapshots())) == 3  # price-only rows still recorded


def test_dry_run_writes_nothing(tmp_path, likud_event):
    led = Ledger(tmp_path)
    with _gamma(likud_event) as g, _oracle(lambda r: pytest.fail("must not call Oracle")) as o:
        run(g, o, led, ("x",), P, now=NOW, dry_run=True)
    assert list(led.snapshots()) == [] and list(led.trades()) == []
