"""GET /manifold/paper's report builder (retro#918) — module-level, same convention as test_pm_paper.py."""

from __future__ import annotations

import json
from pathlib import Path

from forecast_api.manifold_paper import MIN_SCORED, build_report, render_markdown


def _write(dir_: Path, name: str, rows: list[dict]) -> None:
    dir_.mkdir(parents=True, exist_ok=True)
    (dir_ / name).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _snap(mid, ts, yes, oracle=None, outcome=None, raw=None, closed=False, **extra):
    return {"ts": ts, "run_id": ts, "market_id": mid, "market_slug": f"s{mid}", "question": f"Q{mid}",
            "end_date": None, "closed": closed, "resolved_outcome": outcome, "resolution_raw": raw,
            "market_yes": yes, "oracle_yes": oracle, "oracle_reason": None,
            "edge": None if oracle is None else round(oracle - yes, 4), **extra}


def test_empty_ledger_dir(tmp_path):
    r = build_report(tmp_path / "missing")
    s = r["summary"]
    assert s["n_markets"] == 0 and s["coverage"] is None and s["n_scored"] == 0
    assert s["criterion"] == {"min_scored": MIN_SCORED, "pass_margin": 0.01, "enough_data": False, "passes": False}
    assert "calibration scorecard" in render_markdown(r)


def test_scores_last_oracle_snapshot_on_yes_no_only(tmp_path):
    _write(tmp_path, "snapshots.jsonl", [
        _snap("1", "2026-10-11T00:00:00+00:00", 0.40, oracle=0.70),
        _snap("1", "2026-10-12T00:00:00+00:00", 0.55),                       # price-only
        _snap("1", "2026-10-20T00:00:00+00:00", 1.0, outcome="YES", raw="YES", closed=True),
        _snap("2", "2026-10-11T00:00:00+00:00", 0.90, oracle=0.60),
        _snap("2", "2026-10-20T00:00:00+00:00", 0.0, outcome="NO", raw="NO", closed=True),
        _snap("3", "2026-10-11T00:00:00+00:00", 0.50, oracle=0.50),
        _snap("3", "2026-10-20T00:00:00+00:00", 0.50, raw="MKT", closed=True),    # void: not scored
        _snap("4", "2026-10-11T00:00:00+00:00", 0.50, oracle=0.50),
        _snap("4", "2026-10-20T00:00:00+00:00", 0.50, raw="CANCEL", closed=True),
        _snap("5", "2026-10-11T00:00:00+00:00", 0.10),                       # never asked
    ])
    s = build_report(tmp_path)["summary"]
    assert (s["n_markets"], s["n_with_oracle"], s["n_resolved"], s["n_void"], s["n_scored"]) == (5, 4, 2, 2, 2)
    assert s["coverage"] == 0.8
    assert s["brier_oracle"] == round((0.3 ** 2 + 0.6 ** 2) / 2, 4)
    assert s["brier_market"] == round((0.6 ** 2 + 0.9 ** 2) / 2, 4)
    assert s["brier_diff_mean"] == round(((0.09 - 0.36) + (0.36 - 0.81)) / 2, 4)
    assert s["logloss_oracle"] is not None and s["criterion"]["enough_data"] is False


def test_reliability_bins_and_bootstrap_are_deterministic(tmp_path):
    rows = []
    for i in range(40):
        outcome = "YES" if i % 4 else "NO"          # 75% YES
        rows += [_snap(str(i), "2026-10-11T00:00:00+00:00", 0.5, oracle=0.75),
                 _snap(str(i), "2026-10-20T00:00:00+00:00", 1.0 if outcome == "YES" else 0.0,
                       outcome=outcome, raw=outcome, closed=True)]
    _write(tmp_path, "snapshots.jsonl", rows)
    a, b = build_report(tmp_path), build_report(tmp_path)
    assert a["summary"]["brier_diff_ci95"] == b["summary"]["brier_diff_ci95"]
    top = next(x for x in a["reliability"] if x["n"])
    assert top["bin"] == "0.6-0.8" and top["n"] == 40 and top["mean_predicted"] == 0.75 and top["observed_yes"] == 0.75
    lo, hi = a["summary"]["brier_diff_ci95"]
    assert lo <= a["summary"]["brier_diff_mean"] <= hi


def test_pass_bar_requires_enough_scored_markets(tmp_path):
    rows = []
    for i in range(MIN_SCORED):
        rows += [_snap(str(i), "2026-10-11T00:00:00+00:00", 0.5, oracle=0.5),
                 _snap(str(i), "2026-10-20T00:00:00+00:00", 1.0, outcome="YES", raw="YES", closed=True)]
    _write(tmp_path, "snapshots.jsonl", rows)
    c = build_report(tmp_path)["summary"]["criterion"]
    assert c["enough_data"] is True and c["passes"] is True  # tie with the market is within margin


def test_positions_settle_void_and_mark_to_market(tmp_path):
    _write(tmp_path, "snapshots.jsonl", [
        _snap("1", "2026-10-11T00:00:00+00:00", 0.40, oracle=0.70),
        _snap("1", "2026-10-20T00:00:00+00:00", 1.0, outcome="YES", raw="YES", closed=True),
        _snap("2", "2026-10-11T00:00:00+00:00", 0.80, oracle=0.50),
        _snap("2", "2026-10-20T00:00:00+00:00", 0.5, raw="CANCEL", closed=True),
        _snap("3", "2026-10-11T00:00:00+00:00", 0.60, oracle=0.30),
    ])
    base = {"ts": "2026-10-11T00:00:00+00:00", "oracle_yes": 0.5, "market_yes": 0.5, "edge": 0.0}
    _write(tmp_path, "trades.jsonl", [
        {**base, "trade_id": "a", "market_id": "1", "question": "Q1", "side": "YES", "entry_price": 0.4,
         "notional_mana": 100.0, "shares": 250.0},
        {**base, "trade_id": "b", "market_id": "2", "question": "Q2", "side": "NO", "entry_price": 0.2,
         "notional_mana": 100.0, "shares": 500.0},
        {**base, "trade_id": "c", "market_id": "3", "question": "Q3", "side": "NO", "entry_price": 0.4,
         "notional_mana": 100.0, "shares": 250.0},
    ])
    r = build_report(tmp_path)
    pos = {p["trade_id"]: p for p in r["positions"]}
    assert pos["a"]["status"] == "won" and pos["a"]["pnl_mana"] == 150.0
    assert pos["b"]["status"] == "void" and pos["b"]["pnl_mana"] is None
    assert pos["c"]["status"] == "open" and pos["c"]["current_side_price"] == 0.4 and pos["c"]["pnl_mana"] == 0.0
    s = r["summary"]
    assert (s["n_won"], s["n_lost"], s["n_open"], s["realized_pnl_mana"]) == (1, 0, 1, 150.0)


def test_torn_last_line_does_not_break_the_report(tmp_path):
    _write(tmp_path, "snapshots.jsonl", [_snap("1", "2026-10-11T00:00:00+00:00", 0.4, oracle=0.6)])
    with (tmp_path / "snapshots.jsonl").open("a") as fh:
        fh.write('{"market_id": "2", "ts": ')
    assert build_report(tmp_path)["summary"]["n_markets"] == 1


def test_route_serves_json_and_markdown(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from forecast_api import main as api_main
    from forecast_api.config import settings

    _write(tmp_path, "snapshots.jsonl", [_snap("1", "2026-10-11T00:00:00+00:00", 0.40, oracle=0.70)])
    monkeypatch.setattr(type(settings), "resolved_manifold_paper_ledger_dir", property(lambda self: tmp_path))
    with TestClient(api_main.app) as client:
        r = client.get("/manifold/paper")
        assert r.status_code == 200 and r.json()["summary"]["n_markets"] == 1
        r = client.get("/manifold/paper", params={"format": "md"})
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/markdown")
        assert "| Q1 | 0.400 | 0.700 | 0.300 | open |" in r.text
        assert client.get("/manifold/paper", params={"format": "xml"}).status_code == 422
