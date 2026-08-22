"""retro#595 — the v2 playground: job lifecycle through the API, and the
engine's non-negotiables (an unpriced edge is never elicited; pruning keeps
priced edges only; the combination locks anchors and leaves the root free)."""

import time
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from forecast_api import v2_playground as pg
from forecast_api.main import app
from forecast_api.models import ClaimDetail, ForecastResponse, PoolAggregateResponse, SourceSignal

client = TestClient(app)
HEADERS = {"x-api-key": "test-key"}


def _src(stance: float, claim: str, antecedent: str | None = None) -> SourceSignal:
    detail = [ClaimDetail(claim=claim, stance=stance, certainty=0.8, antecedent_text_en=antecedent)] if antecedent else None
    return SourceSignal(
        source_id="s", source_name="S", url="https://x/1", stance=stance, certainty=0.8, credibility_weight=1.0,
        claims=[claim], published_date="2026-08-01", recency_weight=1.0, relevance_score=0.9, claims_detail=detail,
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
    root_sources = [_src(0.4, "if the ceasefire holds, Israel withdraws", antecedent="the ceasefire holds through October")]
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
    root_sources = [_src(0.0, "depends on the ceasefire", antecedent="the ceasefire holds")]
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


async def test_unconditional_pool_gives_no_edge(monkeypatch):
    """The antecedent filter keeps unconditional sources on both sides, so a
    pool with no conditional claim would "split" into two identical numbers.
    That is not an edge: pool_aggregate must not even be called."""
    root_sources = [_src(0.4, "Israel will withdraw"), _src(-0.2, "talks stall")]
    monkeypatch.setattr(pg, "run_forecast", AsyncMock(side_effect=[_forecast(0.1, sources=root_sources), _forecast(0.2)]))
    agg = AsyncMock(side_effect=[_pool(0.1), _pool(0.1)])
    monkeypatch.setattr(pg, "run_pool_aggregate", agg)
    monkeypatch.setattr(
        pg, "complete_text_once_with_usage",
        AsyncMock(return_value=('{"precursors":[{"text":"The ceasefire holds","why":"w","effect":"raises"}]}', {})),
    )
    monkeypatch.setattr(pg, "_anchor", AsyncMock(return_value=None))

    req = pg.V2ForecastRequest(question="Will Israel withdraw from southern Lebanon by Dec 31?", depth=1)
    job = pg.new_job(req)
    await pg.run_job(job["id"], req)
    job = pg.get_job(job["id"])

    edge = job["edges"][0]
    assert edge["method"] == "unpriced" and edge["conditional_hits"] == 0 and "conditions on" in edge["reason"]
    assert agg.await_count == 0
    assert job["nodes"][1]["pruned"] is True and job["result"]["propagated_p"] is None


async def test_anchor_needs_same_question_verdict(monkeypatch):
    """A Gamma keyword hit is a same-TOPIC market, not necessarily the same
    question. Only an LLM 'same' verdict locks it; otherwise it is kept as a
    rejected candidate and the node stays free."""
    market = {"question": "Will the next US-Iran meeting be in the US by Sep 30?", "slug": "meeting", "outcomes": ["Yes", "No"]}
    monkeypatch.setattr("forecast_api.polymarket_live.resolve_market", AsyncMock(return_value=market))
    monkeypatch.setattr("forecast_api.polymarket_live.is_binary_yesno", lambda m: True)
    monkeypatch.setattr("forecast_api.polymarket_live.current_yes_price", lambda m: 0.003)
    monkeypatch.setattr("forecast_api.polymarket_live.summarize_market", lambda m: dict(m))
    llm = AsyncMock(return_value=('{"same": false, "why": "a meeting is not an agreement"}', {}))
    monkeypatch.setattr(pg, "complete_text_once_with_usage", llm)

    job = pg.new_job(pg.V2ForecastRequest(question="Will Iran and the US sign a nuclear agreement by Dec 31?"))
    node = {"id": "n1", "text": job["question"], "depth": 0, "status": "priced", "anchor": None, "anchor_candidate": None}
    job["nodes"].append(node)
    await pg._anchor(job, node)
    assert node["anchor"] is None and node["status"] != "anchored"
    assert node["anchor_candidate"]["p"] == 0.003 and node["anchor_candidate"]["verdict"]["same"] is False
    assert job["prompts"][-1]["step"] == "anchor_match"

    llm.return_value = ('{"same": true, "why": "identical"}', {})
    await pg._anchor(job, node)
    assert node["status"] == "anchored" and node["anchor"]["p"] == 0.003

    # no verdict (LLM error) fails closed
    node2 = dict(node, id="n9", anchor=None, anchor_candidate=None, status="priced")
    llm.side_effect = RuntimeError("boom")
    await pg._anchor(job, node2)
    assert node2["anchor"] is None and node2["anchor_candidate"]["verdict"] is None
