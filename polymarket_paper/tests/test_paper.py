import json
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from gamma_client import GammaClient, Market
from ledger import Ledger
from oracle_client import OracleClient
from paper import (Params, resolution_criteria, run, snapshot_row, stance_to_prob, trade_row,
                   trade_side, wants_oracle_call)

NOW = datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc)
P = Params()


def mk(mid="1", yes=0.40, closed=False, desc="d", resolved=None) -> Market:
    return Market(market_id=mid, event_slug="e", market_slug="s", question=f"Q{mid}?", description=desc,
                  yes_price=yes, closed=closed, end_date="2026-10-27T12:00:00Z", volume_usd=1000.0,
                  resolved_outcome=resolved)


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

    def test_price_only_row_counts_as_never_asked(self):
        assert wants_oracle_call(mk(), {"ts": NOW.isoformat(), "oracle_yes": None, "oracle_reason": None}, P, NOW) is True

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
