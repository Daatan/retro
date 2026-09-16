"""GET /pm/paper's report builder (retro#620) — tested against the module, per
this suite's convention for ledger-backed reports (see test_settlement_pin_ledger.py)."""

from __future__ import annotations

import json
from pathlib import Path

from forecast_api.pm_paper import build_report, render_markdown


def _write(dir_: Path, name: str, rows: list[dict]) -> None:
    dir_.mkdir(parents=True, exist_ok=True)
    (dir_ / name).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _snap(mid, ts, yes, oracle=None, outcome=None, closed=False, **extra):
    return {"ts": ts, "run_id": ts, "market_id": mid, "event_slug": "ev", "question": f"Q{mid}", "end_date": None,
            "closed": closed, "resolved_outcome": outcome, "market_yes": yes, "oracle_yes": oracle,
            "oracle_reason": None, "edge": None if oracle is None else round(oracle - yes, 4), **extra}


def test_empty_ledger_dir(tmp_path):
    r = build_report(tmp_path / "missing")
    assert r["summary"]["n_markets"] == 0 and r["markets"] == [] and r["positions"] == []
    assert "scorecard" in render_markdown(r)


def test_brier_scored_from_the_last_oracle_snapshot_on_resolved_markets(tmp_path):
    _write(tmp_path, "snapshots.jsonl", [
        _snap("1", "2026-09-16T00:00:00+00:00", 0.40, oracle=0.70, articles_used=4),
        _snap("1", "2026-09-17T00:00:00+00:00", 0.55),                      # price-only row
        _snap("1", "2026-10-28T00:00:00+00:00", 1.0, outcome="YES", closed=True),
        _snap("2", "2026-09-16T00:00:00+00:00", 0.90, oracle=0.60),
        _snap("2", "2026-10-28T00:00:00+00:00", 1.0, outcome="YES", closed=True),
        _snap("3", "2026-09-16T00:00:00+00:00", 0.10),                      # never asked, open
    ])
    r = build_report(tmp_path)
    s = r["summary"]
    assert s["n_markets"] == 3 and s["n_with_oracle"] == 2 and s["n_resolved"] == 2 and s["n_scored"] == 2
    m1 = next(m for m in r["markets"] if m["market_id"] == "1")
    # Oracle 0.70 vs market 0.40 at the same snapshot, outcome YES
    assert m1["brier_oracle"] == round((0.70 - 1) ** 2, 4)
    assert m1["brier_market"] == round((0.40 - 1) ** 2, 4)
    assert m1["market_yes"] == 1.0 and m1["oracle_yes"] == 0.70 and m1["articles_used"] == 4
    assert s["brier_oracle"] == round(((0.3 ** 2) + (0.4 ** 2)) / 2, 4)
    assert s["brier_market"] == round(((0.6 ** 2) + (0.1 ** 2)) / 2, 4)
    assert s["oracle_calls"] == 2


def test_positions_mark_to_market_then_settle(tmp_path):
    _write(tmp_path, "snapshots.jsonl", [
        _snap("1", "2026-09-16T00:00:00+00:00", 0.40, oracle=0.70),
        _snap("1", "2026-09-17T00:00:00+00:00", 0.50),
        _snap("2", "2026-09-16T00:00:00+00:00", 0.80, oracle=0.50),
    ])
    _write(tmp_path, "trades.jsonl", [
        {"trade_id": "a", "ts": "2026-09-16T00:00:00+00:00", "market_id": "1", "question": "Q1", "side": "YES",
         "entry_price": 0.40, "notional_usd": 100.0, "shares": 250.0, "oracle_yes": 0.7, "market_yes": 0.4, "edge": 0.3},
        {"trade_id": "b", "ts": "2026-09-16T00:00:00+00:00", "market_id": "2", "question": "Q2", "side": "NO",
         "entry_price": 0.20, "notional_usd": 100.0, "shares": 500.0, "oracle_yes": 0.5, "market_yes": 0.8, "edge": -0.3},
    ])
    r = build_report(tmp_path)
    a, b = r["positions"]
    assert a["status"] == "open" and a["current_side_price"] == 0.5 and a["value_usd"] == 125.0 and a["pnl_usd"] == 25.0
    # NO side is valued at 1 − yes
    assert b["current_side_price"] == 0.2 and b["pnl_usd"] == 0.0
    assert r["summary"]["unrealized_pnl_usd"] == 25.0 and r["summary"]["realized_pnl_usd"] == 0.0

    # market 2 resolves NO → the NO position pays out 1/share
    with (tmp_path / "snapshots.jsonl").open("a") as fh:
        fh.write(json.dumps(_snap("2", "2026-10-28T00:00:00+00:00", 0.0, outcome="NO", closed=True)) + "\n")
    r = build_report(tmp_path)
    b = r["positions"][1]
    assert b["status"] == "won" and b["value_usd"] == 500.0 and b["pnl_usd"] == 400.0
    assert r["summary"]["n_won"] == 1 and r["summary"]["realized_pnl_usd"] == 400.0 and r["summary"]["n_open"] == 1
    md = render_markdown(r)
    assert "| Q2 | NO | 0.200 | 1.000 | $400.00 | won |" in md


def test_torn_last_line_is_ignored(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "snapshots.jsonl").write_text(json.dumps(_snap("1", "t", 0.4)) + "\n{\"market_id\": \"2\", \"ts\"", encoding="utf-8")
    assert build_report(tmp_path)["summary"]["n_markets"] == 1


def test_route_serves_json_and_markdown(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from forecast_api import main as api_main
    from forecast_api.config import settings

    _write(tmp_path, "snapshots.jsonl", [_snap("1", "2026-09-16T00:00:00+00:00", 0.40, oracle=0.70)])
    monkeypatch.setattr(type(settings), "resolved_polymarket_paper_ledger_dir", property(lambda self: tmp_path))
    with TestClient(api_main.app) as client:
        r = client.get("/pm/paper")
        assert r.status_code == 200 and r.json()["summary"]["n_markets"] == 1
        r = client.get("/pm/paper", params={"format": "md"})
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/markdown")
        assert "| Q1 | 0.400 | 0.700 | 0.300 | open |" in r.text
        assert client.get("/pm/paper", params={"format": "xml"}).status_code == 422
