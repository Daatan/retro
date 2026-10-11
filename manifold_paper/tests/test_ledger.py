from ledger import Ledger


def test_roundtrip_and_latest(tmp_path):
    led = Ledger(tmp_path / "ledger")
    led.append_snapshot({"ts": "2026-09-16T00:00:00+00:00", "market_id": "1", "market_yes": 0.4})
    led.append_snapshot({"ts": "2026-09-16T06:00:00+00:00", "market_id": "1", "market_yes": 0.5})
    led.append_snapshot({"ts": "2026-09-16T06:00:00+00:00", "market_id": "2", "market_yes": 0.9})
    led.append_trade({"market_id": "2", "side": "NO"})

    latest = led.latest_snapshot_by_market()
    assert latest["1"]["market_yes"] == 0.5
    assert set(latest) == {"1", "2"}
    assert led.open_trade_market_ids() == {"2"}
    assert len(list(led.snapshots())) == 3


def test_empty_ledger_reads_cleanly(tmp_path):
    led = Ledger(tmp_path / "x")
    assert led.latest_snapshot_by_market() == {}
    assert led.open_trade_market_ids() == set()
