"""retro#595 — the v2 playground: job lifecycle through the API, and the
engine's non-negotiables (an unpriced edge is never elicited; pruning keeps
priced edges only; the combination locks anchors and leaves the root free)."""

import asyncio
import time
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from forecast_api import daatan_client, v2_playground as pg
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
    assert job["calls"] == {"forecast": 2, "pool_aggregate": 2, "llm": 1, "polymarket": 0, "daatan": 0}


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


class TestPrecursorMatchShadow:
    """retro#608 — shadow-only precursor candidate-match. Flag-off must be
    byte-identical to today (the pre-existing e2e tests above already prove
    this: precursor_match_enabled defaults False and both pass unmodified).
    Flag-on assertions only ever touch the new node["precursor_match"] field —
    never node["status"]/node["anchor"]/pricing/pruning/_combine's output."""

    def _root_node(self, job) -> dict:
        node = {
            "id": "n1", "text": job["question"], "depth": 0, "status": "priced", "flat": None,
            "anchor": None, "anchor_candidate": None, "precursor_match": None,
        }
        job["nodes"].append(node)
        return node

    async def test_flag_off_never_fires_match_or_calls_daatan(self, monkeypatch):
        monkeypatch.setattr(pg.settings, "precursor_match_enabled", False)
        match_called = AsyncMock()
        monkeypatch.setattr(pg, "_precursor_match", match_called)
        root_sources = [_src(0.4, "if the ceasefire holds, Israel withdraws", antecedent="the ceasefire holds through October")]
        monkeypatch.setattr(pg, "run_forecast", AsyncMock(side_effect=[_forecast(0.4, sources=root_sources), _forecast(0.2)]))
        monkeypatch.setattr(pg, "run_pool_aggregate", AsyncMock(side_effect=[_pool(0.6), _pool(0.0, insufficient=True)]))
        monkeypatch.setattr(
            pg, "complete_text_once_with_usage",
            AsyncMock(return_value=('{"precursors":[{"text":"The ceasefire holds through October","why":"w","effect":"raises"}]}', {})),
        )
        monkeypatch.setattr(pg, "_anchor", AsyncMock(return_value=None))

        req = pg.V2ForecastRequest(question="Will Israel withdraw from southern Lebanon by Dec 31?", depth=1)
        job = pg.new_job(req)
        await pg.run_job(job["id"], req)
        job = pg.get_job(job["id"])

        assert match_called.await_count == 0
        assert job["calls"]["daatan"] == 0
        assert all(n["precursor_match"] is None for n in job["nodes"])

    async def test_daatan_lookup_error_is_distinguishable_from_not_found(self, monkeypatch):
        job = pg.new_job(pg.V2ForecastRequest(question="Will X happen by Dec 31?"))
        node = self._root_node(job)

        async def raising(*args, **kwargs):
            raise daatan_client.DaatanLookupError("HTTP 500")

        monkeypatch.setattr(daatan_client, "find_similar_forecasts", raising)
        error_result = await pg._match_daatan(job, node)

        monkeypatch.setattr(daatan_client, "find_similar_forecasts", AsyncMock(return_value=[]))
        not_found_result = await pg._match_daatan(job, node)

        assert error_result["status"] == "error" and "HTTP 500" in error_result["detail"]
        assert not_found_result["status"] == "not_found"
        assert error_result["status"] != not_found_result["status"]

    async def test_relation_classifier_failure_keeps_ok_status_with_none_relation(self, monkeypatch):
        """Distinguishable from both not_found (no candidate) and error (lookup
        itself failed): candidate found, but the LLM verdict was unparseable."""
        job = pg.new_job(pg.V2ForecastRequest(question="Will X happen by Dec 31?"))
        node = self._root_node(job)

        monkeypatch.setattr(
            pg, "_match_daatan",
            AsyncMock(return_value={"status": "ok", "candidate": {"id": "p1", "claimText": "X", "score": 0.5}}),
        )
        monkeypatch.setattr(pg, "_match_polymarket", AsyncMock(return_value={"status": "not_found"}))
        monkeypatch.setattr(pg, "complete_text_once_with_usage", AsyncMock(return_value=("not json at all", {})))

        await pg._precursor_match(job, node)

        assert node["precursor_match"]["daatan"]["status"] == "ok"
        assert node["precursor_match"]["daatan"]["relation"] is None
        assert node["precursor_match"]["polymarket"]["status"] == "not_found"

    async def test_alias_match_does_not_change_status_or_anchor(self, monkeypatch):
        """A confirmed alias verdict in node["precursor_match"] must never leak
        into node["anchor"], node["status"], or pricing — that's the entire
        shadow-only contract this slice exists to prove out first."""
        monkeypatch.setattr(pg.settings, "precursor_match_enabled", True)
        root_sources = [_src(0.4, "if the ceasefire holds, Israel withdraws", antecedent="the ceasefire holds through October")]
        monkeypatch.setattr(pg, "run_forecast", AsyncMock(side_effect=[_forecast(0.4, sources=root_sources), _forecast(0.2)]))
        monkeypatch.setattr(pg, "run_pool_aggregate", AsyncMock(side_effect=[_pool(0.6), _pool(0.0, insufficient=True)]))

        async def fake_llm(model, *, messages, max_tokens, temperature, timeout):
            content = messages[0]["content"]
            if "<question>" in content:  # DECOMPOSE_PROMPT
                return '{"precursors":[{"text":"The ceasefire holds through October","why":"w","effect":"raises"}]}', {}
            return '{"relation_type":"alias","direction":"a_to_b","polarity":"same","reason":"same event"}', {}

        monkeypatch.setattr(pg, "complete_text_once_with_usage", fake_llm)
        monkeypatch.setattr(pg, "_anchor", AsyncMock(return_value=None))
        monkeypatch.setattr(pg, "_match_polymarket", AsyncMock(return_value={"status": "not_found"}))
        monkeypatch.setattr(
            pg, "_match_daatan",
            AsyncMock(return_value={"status": "ok", "candidate": {"id": "pred1", "claimText": "The ceasefire holds through October", "score": 0.95}}),
        )

        req = pg.V2ForecastRequest(question="Will Israel withdraw from southern Lebanon by Dec 31?", depth=1)
        job = pg.new_job(req)
        await pg.run_job(job["id"], req)
        job = pg.get_job(job["id"])

        child = job["nodes"][1]
        assert child["precursor_match"]["daatan"]["status"] == "ok"
        assert child["precursor_match"]["daatan"]["relation"]["relation_type"] == "alias"
        assert child["anchor"] is None
        assert child["status"] == "priced"
        assert job["status"] == "done", job["error"]

    async def test_match_task_crash_does_not_break_sibling_pricing(self, monkeypatch):
        """A bug in the new shadow-only code must never propagate through
        _expand's gather and crash pricing for this node or its siblings —
        regression test for _await_match_task_safely."""
        monkeypatch.setattr(pg.settings, "precursor_match_enabled", True)
        monkeypatch.setattr(pg, "_precursor_match", AsyncMock(side_effect=RuntimeError("boom")))
        root_sources = [_src(0.4, "if the ceasefire holds, Israel withdraws", antecedent="the ceasefire holds through October")]
        monkeypatch.setattr(pg, "run_forecast", AsyncMock(side_effect=[_forecast(0.4, sources=root_sources), _forecast(0.2)]))
        monkeypatch.setattr(pg, "run_pool_aggregate", AsyncMock(side_effect=[_pool(0.6), _pool(0.0, insufficient=True)]))
        monkeypatch.setattr(
            pg, "complete_text_once_with_usage",
            AsyncMock(return_value=('{"precursors":[{"text":"The ceasefire holds through October","why":"w","effect":"raises"}]}', {})),
        )
        monkeypatch.setattr(pg, "_anchor", AsyncMock(return_value=None))

        req = pg.V2ForecastRequest(question="Will Israel withdraw from southern Lebanon by Dec 31?", depth=1)
        job = pg.new_job(req)
        await pg.run_job(job["id"], req)
        job = pg.get_job(job["id"])

        assert job["status"] == "done", job["error"]
        assert job["nodes"][1]["status"] == "priced"

    async def test_polymarket_fetched_once_per_node_when_shadow_enabled(self, monkeypatch):
        """retro#608's cache-reuse fix: enabling the shadow step must not double
        Gamma HTTP volume — one resolve_market call per node, shared between
        _match_polymarket and _anchor via node["_polymarket_cache"]."""
        market = {"question": "some market", "slug": "m", "outcomes": ["Yes", "No"], "outcomePrices": ["0.5", "0.5"]}
        resolve = AsyncMock(return_value=market)
        monkeypatch.setattr("forecast_api.polymarket_live.resolve_market", resolve)
        monkeypatch.setattr(pg, "_match_daatan", AsyncMock(return_value={"status": "not_found"}))
        monkeypatch.setattr(pg, "complete_text_once_with_usage", AsyncMock(return_value=('{"same": true, "why": "x"}', {})))

        job = pg.new_job(pg.V2ForecastRequest(question="will X happen"))
        node = self._root_node(job)

        match_task = asyncio.create_task(pg._precursor_match(job, node))
        await pg._await_match_task_safely(job, node, match_task)
        await pg._anchor(job, node)

        assert resolve.await_count == 1
        assert job["calls"]["polymarket"] == 1
        assert node["anchor"] is not None
        assert "_polymarket_cache" not in node  # popped by _anchor, never left to be persisted

    async def test_match_fires_even_when_price_flat_hits_call_cap(self, monkeypatch):
        """retro#608's original point: the candidate-match check must not wait
        on pricing succeeding — it must fire regardless of the forecast-call cap."""
        monkeypatch.setattr(pg.settings, "precursor_match_enabled", True)
        match_called = AsyncMock()
        monkeypatch.setattr(pg, "_precursor_match", match_called)
        monkeypatch.setattr(
            pg, "complete_text_once_with_usage",
            AsyncMock(return_value=('{"precursors":[{"text":"The ceasefire holds","why":"w","effect":"raises"}]}', {})),
        )

        req = pg.V2ForecastRequest(question="Will Israel withdraw from southern Lebanon by Dec 31?", max_forecast_calls=1)
        job = pg.new_job(req)
        root = self._root_node(job)
        job["calls"]["forecast"] = 1  # already at the cap before _expand even starts

        await pg._expand(job, root, None, req)

        assert match_called.await_count == 1
        child = job["nodes"][1]
        assert child["status"] == "unpriced" and child["flat"]["reason"] == "call_cap"


class TestPrecursorMatchShippedDefaults:
    """Both flags default off — this slice is shadow-only by construction, and a
    silent flip to on is exactly the change that should never pass review
    unnoticed (mirrors TestShippedDefaults in test_premise_verifier.py)."""

    def test_precursor_match_ships_disabled(self):
        from forecast_api.config import ApiSettings
        assert ApiSettings.model_fields["precursor_match_enabled"].default is False

    def test_enforce_ships_disabled_and_is_unread_this_slice(self):
        from forecast_api.config import ApiSettings
        assert ApiSettings.model_fields["precursor_match_enforce"].default is False
