from ledger import Ledger


def test_roundtrip_and_latest(tmp_path):
    led = Ledger(tmp_path / "ledger")
    led.append_snapshot({"ts": "2026-09-16T00:00:00+00:00", "market_id": "1", "market_yes": 0.4})
    led.append_snapshot({"ts": "2026-09-16T06:00:00+00:00", "market_id": "1", "market_yes": 0.5})
    led.append_snapshot({"ts": "2026-09-16T06:00:00+00:00", "market_id": "2", "market_yes": 0.9})
    led.append_trade({"market_id": "2", "side": "NO"})

    assert led.last_asked_by_market() == {}  # price-only rows are not answers
    assert led.open_trade_market_ids() == {"2"}
    assert len(list(led.snapshots())) == 3


def test_last_asked_skips_price_only_rows_and_counts_a_reason_as_an_answer(tmp_path):
    led = Ledger(tmp_path / "ledger")
    led.append_snapshot({"ts": "t0", "market_id": "1", "oracle_yes": 0.4, "oracle_reason": None})
    led.append_snapshot({"ts": "t1", "market_id": "1", "oracle_yes": None, "oracle_reason": None})
    led.append_snapshot({"ts": "t2", "market_id": "2", "oracle_yes": None, "oracle_reason": "no_search_results"})
    asked = led.last_asked_by_market()
    assert asked["1"]["ts"] == "t0" and asked["2"]["ts"] == "t2"


def test_empty_ledger_reads_cleanly(tmp_path):
    led = Ledger(tmp_path / "x")
    assert led.last_asked_by_market() == {}
    assert led.open_trade_market_ids() == set()
