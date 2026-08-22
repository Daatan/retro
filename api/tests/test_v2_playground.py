"""retro#595 — the v2 playground: job lifecycle through the API, and the
engine's non-negotiables (an unpriced edge is never elicited; pruning keeps
priced edges only; the combination locks anchors and leaves the root free)."""

import time
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from forecast_api import v2_playground as pg
from forecast_api.main import app
from forecast_api.models import ForecastResponse, PoolAggregateResponse, SourceSignal

client = TestClient(app)
HEADERS = {"x-api-key": "test-key"}


def _src(stance: float, claim: str) -> SourceSignal:
    return SourceSignal(
        source_id="s", source_name="S", url="https://x/1", stance=stance, certainty=0.8, credibility_weight=1.0,
        claims=[claim], published_date="2026-08-01", recency_weight=1.0, relevance_score=0.9,
    )


def _forecast(mean: float, *, sources=None, insufficient=False) -> ForecastResponse:
    return ForecastResponse(
        question="q", mean=mean, std=0.2, ci_low=mean - 0.2, ci_high=mean + 0.2, articles_used=len(sources or []),
        sources=sources or [], placeholder=insufficient, insufficient_data=insufficient, articles_found=0,
    )


def _pool(mean: float, *, insufficient=False) -> PoolAggregateResponse:
    return PoolAggregateResponse(
        mean=mean, std=0.1, ci_low=mean - 0.1, ci_high=mean + 0.1, articles_used=0 if insufficient else 3,
        settled=False, insufficient_data=insufficient, reason="no_matching_antecedent" if insufficient else None,
    )


@pytest.fixture(autouse=True)
def _tmp_store(tmp_path, monkeypatch):
    monkeypatch.setattr(pg.settings, "data_dir", tmp_path)
    pg._stores.clear()


def test_requires_api_key():
    r = client.post("/v2/forecast", json={"question": "Will it rain tomorrow in Tel Aviv?"}, headers={"x-api-key": "nope"})
    assert r.status_code == 401


def test_depth_bounds():
    r = client.post("/v2/forecast", json={"question": "Will it rain tomorrow in Tel Aviv?", "depth": 7}, headers=HEADERS)
    assert r.status_code == 422


def test_unknown_job_is_404():
    assert client.get("/v2/jobs/doesnotexist", headers=HEADERS).status_code == 404


async def test_end_to_end_trace_unpriced_edge_is_never_elicited(monkeypatch):
    """Root priced, one precursor priced, the pool cannot split on it on the
    NO side → the edge is unpriced, the child is pruned, the flat number stands."""
    root_sources = [_src(0.4, "if the ceasefire holds, Israel withdraws")]
    monkeypatch.setattr(pg, "run_forecast", AsyncMock(side_effect=[_forecast(0.4, sources=root_sources), _forecast(0.2)]))
    monkeypatch.setattr(pg, "run_pool_aggregate", AsyncMock(side_effect=[_pool(0.6), _pool(0.0, insufficient=True)]))
    monkeypatch.setattr(
        pg, "complete_text_once_with_usage",
        AsyncMock(return_value=('{"precursors":[{"text":"The ceasefire holds through October","why":"w","effect":"raises"}]}', {"total_tokens": 10})),
    )
    monkeypatch.setattr(pg, "_anchor", AsyncMock(return_value=None))

    req = pg.V2ForecastRequest(question="Will Israel withdraw from southern Lebanon by Dec 31?", depth=1)
    job = pg.new_job(req)
    await pg.run_job(job["id"], req)
    job = pg.get_job(job["id"])

    assert job["status"] == "done"
    assert [n["text"] for n in job["nodes"]] == [req.question, "The ceasefire holds through October"]
    assert job["nodes"][0]["flat"]["p"] == 0.7
    edge = job["edges"][0]
    assert edge["method"] == "unpriced" and edge["p_given_yes"] is None
    assert job["nodes"][1]["pruned"] is True
    assert job["result"]["propagated_p"] is None and job["result"]["flat_p"] == 0.7
    assert job["prompts"][0]["step"] == "decompose" and "ceasefire" in job["prompts"][0]["response"]
    assert job["calls"] == {"forecast": 2, "pool_aggregate": 2, "llm": 1, "polymarket": 0}


async def test_priced_edge_propagates_and_anchor_is_locked(monkeypatch):
    root_sources = [_src(0.0, "depends on the ceasefire")]
    monkeypatch.setattr(pg, "run_forecast", AsyncMock(side_effect=[_forecast(0.0, sources=root_sources), _forecast(0.6)]))
    monkeypatch.setattr(pg, "run_pool_aggregate", AsyncMock(side_effect=[_pool(0.8), _pool(-0.8)]))
    monkeypatch.setattr(
        pg, "complete_text_once_with_usage",
        AsyncMock(return_value=('{"precursors":[{"text":"The ceasefire holds","why":"w","effect":"raises"}]}', {})),
    )

    async def fake_anchor(job, node):
        if node["depth"] == 1:
            node["anchor"] = {"kind": "polymarket", "p": 0.9, "market": {"slug": "ceasefire"}}
            node["status"] = "anchored"

    monkeypatch.setattr(pg, "_anchor", fake_anchor)

    req = pg.V2ForecastRequest(question="Will Israel withdraw from southern Lebanon by Dec 31?", depth=3)
    job = pg.new_job(req)
    await pg.run_job(job["id"], req)
    job = pg.get_job(job["id"])

    assert job["status"] == "done", job["error"]
    edge = job["edges"][0]
    assert edge["method"] == "pool_split" and (edge["p_given_yes"], edge["p_given_no"]) == (0.9, 0.1)
    assert job["nodes"][1]["pruned"] is False and job["nodes"][1]["status"] == "anchored"
    assert job["result"]["locked"] == {"n2": 0.9}
    assert job["result"]["edges_used"] == 1
    # the anchored child is locked, so propagation pulls the root toward P(yes)
    assert job["result"]["propagated_p"] > job["result"]["flat_p"]
    # anchors end the recursion: depth 3 requested, only one decomposition happened
    assert job["calls"]["llm"] == 1


def test_api_lifecycle(monkeypatch):
    async def fake_run(job_id, req):
        job = pg.get_job(job_id)
        job["status"] = "done"
        job["result"] = {"flat_p": 0.5, "propagated_p": 0.5}
        pg._save(job)

    monkeypatch.setattr(pg, "run_job", fake_run)
    with TestClient(app) as c:  # keeps the event loop alive so the task runs
        r = c.post("/v2/forecast", json={"question": "Will the Knesset dissolve by October 2026?"}, headers=HEADERS)
        assert r.status_code == 202, r.text
        job_id = r.json()["job_id"]
        for _ in range(100):
            g = c.get(f"/v2/jobs/{job_id}", headers=HEADERS)
            assert g.status_code == 200
            if g.json()["status"] == "done":
                break
            time.sleep(0.02)
    assert g.json()["status"] == "done"
    assert g.json()["params"]["depth"] == 2
