"""retro#955 — Jev-weighted pool shadow: pass-1 `max_noul` is persisted per (url, q8) and
`/pool/aggregate` logs what the mean would be with it as relevance. Log-only: the response is
identical with the flag on or off."""
import logging

import pytest

from forecast_api import jev_value_store as jvs
from forecast_api.config import settings as api_settings
from forecast_api.forecaster import _log_jev_gate, run_pool_aggregate
from forecast_api.jev_shadow import question_fingerprint
from forecast_api.models import PoolAggregateRequest, PoolSourceInput

Q = "Will the coalition survive until January?"


def _src(url, stance, rel):
    return PoolSourceInput(stance=stance, certainty=0.8, credibility_weight=1.0,
                           relevance_score=rel, published_date="2026-10-01", url=url)


def _req(n=4):
    return PoolAggregateRequest(
        question=Q,
        sources=[_src(f"https://x/{i}", 0.6 if i % 2 else -0.4, 0.8) for i in range(n)],
    )


@pytest.fixture
def shadow(tmp_path, monkeypatch):
    monkeypatch.setattr(api_settings, "pool_shadow_a_enabled", True)
    monkeypatch.setattr(api_settings, "pool_shadow_a_path", tmp_path / "jev")
    return tmp_path / "jev"


def test_store_roundtrip_and_noops(tmp_path):
    p = tmp_path / "s"
    jvs.put_max_noul(p, "https://x/1", "abcd1234", 0.42)
    assert jvs.get_max_noul(p, "https://x/1", "abcd1234") == pytest.approx(0.42)
    assert jvs.get_max_noul(p, "https://x/1", "other000") is None
    jvs.put_max_noul(p, None, "abcd1234", 0.5)       # no url: no-op
    jvs.put_max_noul(p, "https://x/2", "", 0.5)      # no q8: no-op
    jvs.put_max_noul(p, "https://x/3", "abcd1234", None)
    assert jvs.get_max_noul(p, "https://x/3", "abcd1234") is None


def test_store_fails_open(monkeypatch, tmp_path):
    monkeypatch.setattr(jvs, "_get_store", lambda p: (_ for _ in ()).throw(OSError("disk")))
    jvs.put_max_noul(tmp_path, "u", "q", 0.1)        # must not raise
    assert jvs.get_max_noul(tmp_path, "u", "q") is None


def test_jev_gate_log_writes_value_only_when_enabled(shadow, monkeypatch):
    kw = dict(would_skip=False, n_preds=1, text="t", language="en", url="https://x/0",
              prediction_id=None, question=Q)
    monkeypatch.setattr(api_settings, "pool_shadow_a_enabled", False)
    _log_jev_gate({"max_noul": 0.3}, **kw)
    assert jvs.get_max_noul(shadow, "https://x/0", question_fingerprint(Q)) is None
    monkeypatch.setattr(api_settings, "pool_shadow_a_enabled", True)
    _log_jev_gate({"max_noul": 0.3}, **kw)
    assert jvs.get_max_noul(shadow, "https://x/0", question_fingerprint(Q)) == pytest.approx(0.3)


async def test_shadow_logs_and_leaves_response_unchanged(shadow, caplog, monkeypatch):
    q8 = question_fingerprint(Q)
    for i, v in enumerate((0.05, 0.9)):
        jvs.put_max_noul(shadow, f"https://x/{i}", q8, v)
    with caplog.at_level(logging.INFO, logger="forecast_api.forecaster"):
        on = await run_pool_aggregate(_req())
    line = next(r.getMessage() for r in caplog.records if "event=pool_shadow_a " in r.getMessage())
    assert f"q8={q8}" in line and "n_cov=2" in line and "n=4" in line
    assert "a=none" not in line and "a_x=none" not in line
    live = float(line.split("live=")[1].split()[0])
    a = float(line.split(" a=")[1].split()[0])
    assert a != pytest.approx(live, abs=1e-4)   # Jev changed the weights, so the mean moved
    monkeypatch.setattr(api_settings, "pool_shadow_a_enabled", False)
    off = await run_pool_aggregate(_req())
    assert on.model_dump() == off.model_dump()


async def test_shadow_with_no_coverage_logs_zero(shadow, caplog):
    with caplog.at_level(logging.INFO, logger="forecast_api.forecaster"):
        await run_pool_aggregate(_req())
    assert any("event=pool_shadow_a " in r.getMessage() and "n_cov=0" in r.getMessage()
               for r in caplog.records)


async def test_shadow_off_logs_nothing_and_needs_a_question(shadow, caplog, monkeypatch):
    monkeypatch.setattr(api_settings, "pool_shadow_a_enabled", False)
    with caplog.at_level(logging.INFO, logger="forecast_api.forecaster"):
        await run_pool_aggregate(_req())
    assert not any("pool_shadow_a" in r.getMessage() for r in caplog.records)
    monkeypatch.setattr(api_settings, "pool_shadow_a_enabled", True)
    caplog.clear()
    req = _req(); req.question = None
    with caplog.at_level(logging.INFO, logger="forecast_api.forecaster"):
        await run_pool_aggregate(req)
    assert not any("pool_shadow_a" in r.getMessage() for r in caplog.records)
