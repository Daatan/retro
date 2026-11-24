"""Golden snapshots of every outgoing LLM request, per role (retro#1102, PR 0 of platform#60).

The model-registry refactor (platform#60) moves model ids and call parameters out of the call
sites into one registry. That refactor must change nothing on the wire. These goldens were
captured from the code as it stood before the refactor, and they are the proof that it did not.

**What is captured.** Each covered role is driven through the code line that picks its model. The
test never re-implements a fallback chain such as ``settings.x_model or settings.judge_model``,
because that line is exactly what the refactor moves. Capture happens at the lowest point where
kwargs leave our code:

* the structured path: ``tm.llm.client.chat.completions.create_with_completion``, i.e. the
  instructor client. ``instructor_mode`` is recorded next to the kwargs.
* the text path: ``litellm.acompletion``.
* Jev: the httpx request, through ``httpx.MockTransport``.

**Every kwarg is recorded, not an allowlist**, so a kwarg the refactor adds or drops shows up
as a diff. Three things are normalised:

* ``messages``: roles, content block types and ``cache_control`` only. Prompt text is pinned
  elsewhere (``pipeline/tests/test_rendered_schema.py`` and the prompt-version locks), and some
  prompts embed today's date.
* ``response_model``: the class name.
* ``api_key``: whether it is set, never its value.
* ``temperature``: written as a float. Call sites mix ``0`` and ``0.0``, which is the same request.

The number of calls per role is pinned too (the length of each list).

**Model ids.** Repo defaults, except the two the prod box overrides through committed systemd
drop-ins (``infra/oracle-api.service.d/``). Those are ``EXTRACTOR_MODEL`` (equal to the repo
default today) and ``RELEVANCE_MODEL``. ``VERTEXAI_LOCATION`` is read by litellm from the
environment and never passes through our kwargs, so no golden shows it. Every setting a request
reads is reset to its class default first, so a developer's shell or ``.env`` cannot leak in.

**Not covered here.** The extractor shadow call (``extractor_shadow_model``, off by default).
Jev requests other than pass 1: the per-sentence scoring and date requests in ``run_jev_shadow``
and the evidence_class corrector in ``jev_class.py`` (off by default). They share ``_ask``/``_post``
with pass 1, so URL, headers and model are pinned; their question shapes are not. Offline A/B,
eval, labelling and smoke scripts (``api/scripts/ab_settled_decision.py``,
``api/scripts/label_settlement_candidates.py``, ``pipeline/eval_*.py``,
``pipeline/smoke_test_prompt_cache.py``, ``pipeline/scripts/ab_extractor_prompt.py``). The batch
orchestrator itself: the threshold-routing goldens start at ``run_article``, fed by the same
``select_extractor_model`` call the orchestrator makes. The two lane drivers fail on any
LLM call made outside the stages they spy on, so a new unlabelled call does not go unseen.

**Regenerating.** A golden is written only by ``UPDATE_GOLDENS=1 uv run pytest
tests/test_model_request_goldens.py``. Without the flag, a missing golden fails rather than
being created. Regenerate deliberately: when a request change is the point of the PR, say so in
the PR and show the golden diff. A refactor PR that claims to change nothing must not touch
``tests/fixtures/model_requests/``.
"""
from __future__ import annotations

import asyncio
import contextvars
import difflib
import functools
import importlib.util
import json
import logging
import os
import re
import sys
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from forecast_api import forecaster
from forecast_api import jev_shadow as js
from forecast_api import v2_playground as pg
from forecast_api.config import settings as api_settings
from forecast_api.limiter import limiter
from forecast_api.main import app
from forecast_api.models import ArticleInput, ForecastRequest
from forecast_api.settlement_verifier import SettlementVote
from forecast_api.subject_card import SubjectCard
from tm import llm
from tm import runner as tm_runner
from tm.aggregator import aggregate_article_predictions
from tm.archetype import is_threshold_shaped, select_extractor_model
from tm.config import settings as tm_settings
from tm.models import ExtractionOutput, GatekeeperOutput, PredictionExtraction

GOLDEN_DIR = Path(__file__).parent / "fixtures" / "model_requests"
UPDATE = os.environ.get("UPDATE_GOLDENS") == "1"
REPO = Path(__file__).resolve().parents[2]

# Live prod overrides (committed drop-ins extractor-model.conf, relevance-model-vertex.conf).
PROD_EXTRACTOR_MODEL = "bedrock/us.anthropic.claude-haiku-5-5"
PROD_RELEVANCE_MODEL = "vertex_ai/gemini-3.5-flash-lite"
# The value the threshold_extractor_model comment in tm/config.py documents for turning it on.
THRESHOLD_EXTRACTOR_EXAMPLE = "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"

HEADERS = {"x-api-key": "test-key"}
QUESTION = "Will the central bank cut its policy rate before the end of 2026?"
CRITERIA = "Resolves Yes if the central bank announces a policy-rate cut before 2027-01-01."
THRESHOLD_EVENT = "Inflation exceeds 4% by December 2026"
ARTICLE = (
    "The central bank's governor said on Monday that inflation had eased for a third month. "
    "Two members of the monetary committee told reporters they expect a rate cut in December. "
    "Analysts at three banks now forecast the policy rate falling by a quarter point this year. "
) * 3
_FRESH = (date.today() - timedelta(days=1)).isoformat()

# ── capture ──────────────────────────────────────────────────────────────────

_ROLE: contextvars.ContextVar[str | None] = contextvars.ContextVar("golden_role", default=None)
_TEXT_REPLY = '{"settles": true, "dead": false, "reason": "fixture reply"}'


def _canned(response_model):
    """A valid instance of whatever the caller asked for, so every caller's post-processing runs."""
    name = response_model.__name__
    if name == "GatekeeperOutput":
        return GatekeeperOutput(is_prediction=True, reason="fixture", relevance_score=1.0,
                                prediction_count_estimate=1)
    if name == "ExtractionOutput":
        return ExtractionOutput(predictions=[])
    if name == "PredictionExtraction":
        return _claim(0.5)
    if name == "SubjectCard":
        return SubjectCard(actors=[])
    raise AssertionError(f"no canned output for {name}: add one")


def _claim(stance: float) -> PredictionExtraction:
    return PredictionExtraction(
        quote="Two members of the monetary committee told reporters they expect a rate cut.",
        claim="The central bank will cut its policy rate in December.",
        stance=stance, certainty=0.6, settled=False,
    )


def _shape_messages(messages) -> list[dict]:
    shaped = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            blocks: object = "str"
        else:
            blocks = [
                {"type": b.get("type"), **({"cache_control": b["cache_control"]} if "cache_control" in b else {})}
                for b in content
            ]
        shaped.append({"role": m.get("role"), "content": blocks})
    return shaped


def _normalise(path: str, kwargs: dict) -> dict:
    rec: dict = {"path": path}
    for key, value in kwargs.items():
        if key == "messages":
            rec[key] = _shape_messages(value)
        elif key == "response_model":
            rec[key] = value.__name__
        elif key == "api_key":
            rec[key] = "<set>" if value else value
        elif key == "temperature" and isinstance(value, int):
            rec[key] = float(value)  # call sites mix 0 and 0.0; the request is the same
        else:
            rec[key] = value
    if path == "structured":
        rec["instructor_mode"] = llm.client.mode.value
    # Snapshot now: a caller may reuse and mutate its kwargs dict after the call.
    return json.loads(json.dumps(rec, default=repr))


class Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str | None, dict]] = []
        self.fail_flex = False

    def role(self, role: str | None) -> list[dict]:
        return [rec for r, rec in self.calls if r == role]

    def all(self) -> list[dict]:
        return [rec for _, rec in self.calls]

    def take(self) -> list[dict]:
        out = self.all()
        self.calls.clear()
        return out


@pytest.fixture
def rec(monkeypatch) -> Recorder:
    recorder = Recorder()

    async def fake_create(**kwargs):
        recorder.calls.append((_ROLE.get(), _normalise("structured", kwargs)))
        flex = bool(kwargs.get("extra_headers"))
        if flex and recorder.fail_flex:
            raise RuntimeError("fixture: Flex capacity unavailable")
        completion = SimpleNamespace(usage=None, _hidden_params={
            "provider_specific_fields": {"traffic_type": "ON_DEMAND_FLEX" if flex else "ON_DEMAND"}})
        return _canned(kwargs["response_model"]), completion

    async def fake_acompletion(**kwargs):
        recorder.calls.append((_ROLE.get(), _normalise("text", kwargs)))
        message = SimpleNamespace(content=_TEXT_REPLY)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)

    monkeypatch.setattr(llm.client.chat.completions, "create_with_completion", fake_create)
    monkeypatch.setattr(llm.litellm, "acompletion", fake_acompletion)
    return recorder


def _spy(role: str, fn):
    """Pass-through wrapper that labels every LLM call made underneath it with ``role``."""
    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        token = _ROLE.set(role)
        try:
            return await fn(*args, **kwargs)
        finally:
            _ROLE.reset(token)
    return wrapper


# ── settings: repo defaults, then the two prod overrides ─────────────────────

_TM_FIELDS = re.compile(r"_model$|^extractor_|^enable_prompt_cache$|^prompt_cache_ttl$|^model_api_|^aws_region$")
_API_FIELDS = re.compile(
    r"_model$|_timeout_seconds$|^relevance_flex_|^jev_provider$|^jev_shadow_api_url$"
    r"|^settlement_verifier_votes$|^subject_gate_languages$"
)


@pytest.fixture(autouse=True)
def _hermetic_settings(monkeypatch):
    for obj, pattern in ((tm_settings, _TM_FIELDS), (api_settings, _API_FIELDS)):
        for name, field in type(obj).model_fields.items():
            if pattern.search(name):
                monkeypatch.setattr(obj, name, field.get_default(call_default_factory=True))
    monkeypatch.setattr(tm_settings, "extractor_model", PROD_EXTRACTOR_MODEL)
    monkeypatch.setattr(api_settings, "relevance_model", PROD_RELEVANCE_MODEL)
    monkeypatch.setattr(limiter, "enabled", False)
    forecaster.forecast_cache.clear()


# ── golden files ─────────────────────────────────────────────────────────────

def _render(payload) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _assert_golden(name: str, payload: dict) -> None:
    """Compare ``payload`` with ``fixtures/model_requests/<name>.json``; write it under UPDATE_GOLDENS=1."""
    empty = [k for k, v in payload.items() if isinstance(v, list) and not v]
    assert not empty, f"{name}: no LLM call captured for {empty}; refusing to pin an empty golden"
    path = GOLDEN_DIR / f"{name}.json"
    text = _render(payload)
    if UPDATE:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return
    if not path.exists():
        pytest.fail(f"missing golden {path.relative_to(REPO)}. Goldens are written only with "
                    "UPDATE_GOLDENS=1, deliberately; see this module's docstring.", pytrace=False)
    expected = path.read_text(encoding="utf-8")
    if text != expected:
        diff = "".join(difflib.unified_diff(
            expected.splitlines(keepends=True), text.splitlines(keepends=True),
            fromfile=f"golden {name}.json", tofile="captured now",
        ))
        pytest.fail(f"the outgoing LLM request for {name!r} changed:\n{diff}\nIf that is the intent of "
                    "this change, regenerate with UPDATE_GOLDENS=1 and explain the diff in the PR.",
                    pytrace=False)


# ── drivers ──────────────────────────────────────────────────────────────────

async def _batch_lane(monkeypatch, rec, event_name: str = QUESTION) -> None:
    """The batch pipeline from ``run_article`` down. The routing argument comes from the same
    ``select_extractor_model`` call the orchestrator makes; the orchestrator itself is not driven."""
    monkeypatch.setattr(tm_runner, "update_cell", lambda *a, **kw: None)
    monkeypatch.setattr(tm_runner, "check_is_prediction", _spy("gatekeeper", tm_runner.check_is_prediction))
    monkeypatch.setattr(tm_runner, "extract_predictions", _spy("extractor", tm_runner.extract_predictions))
    article = tm_runner.ArticleInput(
        text=ARTICLE, source_id="src-fixture", source_name="Fixture Times", article_date="2026-11-01",
        event_id="E-fixture", event_name=event_name, event_description=CRITERIA,
        article_url="https://news.example.test/story",
    )
    result = await tm_runner.run_article(article, extractor_model=select_extractor_model(event_name))
    assert result.error is None, result.error
    assert not rec.role(None), f"LLM call outside every spied stage: {rec.role(None)}"


async def _forecast_lane(monkeypatch, rec, **request_fields) -> None:
    """One live /forecast request over a supplied article, with the stage functions spied."""
    for name, role in (("check_is_prediction", "gatekeeper"), ("extract_predictions", "extractor"),
                       ("derive_subject_card", "subject_card"), ("decompose_event", "event_decomposition"),
                       ("verify_premise", "premise_verifier")):
        monkeypatch.setattr(forecaster, name, _spy(role, getattr(forecaster, name)))
    await forecaster.run_forecast(ForecastRequest(
        question=QUESTION, resolution_criteria=CRITERIA,
        articles=[ArticleInput(
            url="https://news.example.test/story", title="Central bank signals a cut",
            snippet="Two committee members expect a rate cut in December.",
            source="news.example.test", published_date=_FRESH, text=ARTICLE,
        )],
        **request_fields,
    ))
    assert not rec.role(None), f"LLM call outside every spied stage: {rec.role(None)}"


def _load_script(path: Path, name: str, monkeypatch):
    """Import a standalone script without letting its import-time side effects outlive the test."""
    monkeypatch.delenv("CALIBRATE_EDGES_MODEL", raising=False)
    env_before, path_before = dict(os.environ), list(sys.path)
    ws_logger = logging.getLogger("tm.web_search")
    level_before = ws_logger.level
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        os.environ.clear()
        os.environ.update(env_before)
        sys.path[:] = path_before
        ws_logger.setLevel(level_before)
    return module


# ── roles ────────────────────────────────────────────────────────────────────

async def test_gatekeeper(rec, monkeypatch):
    await _batch_lane(monkeypatch, rec)
    batch = rec.role("gatekeeper")
    rec.calls.clear()
    await _forecast_lane(monkeypatch, rec)
    _assert_golden("gatekeeper", {"batch_lane": batch, "forecast_lane": rec.role("gatekeeper")})


async def test_extractor(rec, monkeypatch):
    await _batch_lane(monkeypatch, rec)
    batch_on = rec.role("extractor")
    rec.calls.clear()
    await _forecast_lane(monkeypatch, rec)
    forecast_on = rec.role("extractor")
    rec.calls.clear()
    monkeypatch.setattr(tm_settings, "extractor_thinking", False)
    await _batch_lane(monkeypatch, rec)
    _assert_golden("extractor", {
        "batch_lane_thinking_on": batch_on,
        "forecast_lane_thinking_on": forecast_on,
        "batch_lane_thinking_off": rec.role("extractor"),
    })


async def test_threshold_extractor(rec, monkeypatch):
    """retro#688 routing: off by default (the event falls through to the default extractor)."""
    assert is_threshold_shaped(THRESHOLD_EVENT)
    await _batch_lane(monkeypatch, rec, THRESHOLD_EVENT)
    off = rec.role("extractor")
    rec.calls.clear()
    monkeypatch.setattr(tm_settings, "threshold_extractor_model", THRESHOLD_EXTRACTOR_EXAMPLE)
    await _batch_lane(monkeypatch, rec, THRESHOLD_EVENT)
    _assert_golden("threshold_extractor", {
        "off_repo_default": off,
        "on_documented_value": rec.role("extractor"),
    })


async def test_aggregator(rec):
    await aggregate_article_predictions(
        [_claim(0.9), _claim(-0.9)], QUESTION, "Fixture Times", "2026-11-01",
    )
    _assert_golden("aggregator", {"calls": rec.take()})


async def test_relevance(rec, monkeypatch):
    """/relevance via the endpoint: Flex on/off x short_form, plus a failed Flex attempt."""
    client = TestClient(app)
    body = {"claim": QUESTION, "article_text": ARTICLE, "source_name": "Fixture Times",
            "article_date": "2026-11-01"}
    payload: dict = {}
    for flex in (False, True):
        monkeypatch.setattr(api_settings, "relevance_flex_enabled", flex)
        for short_form in (False, True):
            r = client.post("/relevance", json={**body, "short_form": short_form}, headers=HEADERS)
            assert r.status_code == 200, r.text
            payload[f"flex_{'on' if flex else 'off'}_short_form_{str(short_form).lower()}"] = rec.take()
    rec.fail_flex = True
    r = client.post("/relevance", json={**body, "short_form": True}, headers=HEADERS)
    assert r.status_code == 200, r.text
    payload["flex_on_short_form_true_flex_attempt_fails"] = rec.take()
    _assert_golden("relevance", payload)


async def test_llm_proxy_default(rec):
    """The raw /llm proxy with no model and no temperature in the body."""
    r = TestClient(app).post("/llm", json={"messages": [{"role": "user", "content": "Say hi."}]},
                             headers=HEADERS)
    assert r.status_code == 200, r.text
    _assert_golden("llm_proxy", {"default": rec.take()})


async def test_query_distill(rec):
    await forecaster._distill_query(QUESTION)
    _assert_golden("query_distill", {"calls": rec.take()})


async def test_settlement_verifier(rec, monkeypatch):
    monkeypatch.setattr(api_settings, "settlement_verifier_enabled", True)
    monkeypatch.setattr(api_settings, "settlement_verdict_cache_enabled", False)
    vote = SettlementVote(outlet="news.example.test", claim="The central bank cut its policy rate.",
                          quote="The central bank cut its policy rate by a quarter point.",
                          event_date="2026-11-01")
    agg = SimpleNamespace(settled=True, settlement_vote_indices=[0], mean=1.0)
    await forecaster._apply_settlement_match_gate(
        agg, question=QUESTION, votes_for_index=lambda i: [vote], rerun=lambda flags: None,
        settled_flags=[True],
    )
    _assert_golden("settlement_verifier", {"calls": rec.take()})


async def test_subject_card(rec, monkeypatch):
    monkeypatch.setattr(api_settings, "subject_gate_enabled", True)
    monkeypatch.setattr(api_settings, "subject_gate_cache_enabled", False)
    await _forecast_lane(monkeypatch, rec)
    _assert_golden("subject_card", {"calls": rec.role("subject_card")})


async def test_premise_verifier(rec, monkeypatch):
    monkeypatch.setattr(api_settings, "premise_verifier_enabled", True)
    await _forecast_lane(monkeypatch, rec, claim_archetype="scheduled")
    _assert_golden("premise_verifier", {"calls": rec.role("premise_verifier")})


async def test_event_decomposition(rec, monkeypatch):
    monkeypatch.setattr(api_settings, "inject_event_decomposition", True)
    monkeypatch.setattr(api_settings, "event_decomposition_cache_enabled", False)
    await _forecast_lane(monkeypatch, rec)
    _assert_golden("event_decomposition", {"calls": rec.role("event_decomposition")})


async def test_jev_ab_question_rewrite(rec, monkeypatch):
    """retro#849: the Jev gate A/B rewrite. Its 15 s timeout is an outer asyncio.wait_for."""
    js._SIMPLIFIED.clear()

    async def no_second_pass(**kwargs):
        return None
    monkeypatch.setattr(forecaster, "jev_pass1", no_second_pass)
    forecaster._fire_jev_gate_ab(text=ARTICLE, question=QUESTION, url=None, n_preds=0, live_max_noul=None)
    await asyncio.gather(*list(forecaster._AB_TASKS))
    js._SIMPLIFIED.clear()
    _assert_golden("jev_ab_question_rewrite", {"calls": rec.take()})


async def test_v2_playground(rec, monkeypatch):
    monkeypatch.setattr(pg, "_save", lambda job: None)
    job = {"id": "golden", "prompts": [], "calls": {"llm": 0}}
    node = {"id": "n0", "text": QUESTION}
    payload = {}
    await pg._decompose(job, node, pg.V2ForecastRequest(question=QUESTION))
    payload["decompose"] = rec.take()
    await pg._same_question(job, node, "Will the central bank cut rates in 2026?")
    payload["anchor_match"] = rec.take()
    await pg._classify_relation(job, node, "Will inflation fall below 2% in 2026?")
    payload["precursor_match"] = rec.take()
    _assert_golden("v2_playground", payload)


def test_calibrate_edges(rec, monkeypatch):
    """bayesoracle's offline edge calibration (sync script: asyncio.run inside)."""
    module = _load_script(REPO / "bayesoracle" / "calibrate_edges.py", "golden_calibrate_edges", monkeypatch)
    module._llm("Estimate pY and pN for the fixture edge.")
    _assert_golden("calibrate_edges", {"calls": rec.take()})


async def test_improve_keywords(rec, monkeypatch):
    module = _load_script(REPO / "pipeline" / "scripts" / "improve_keywords.py", "golden_improve_keywords",
                          monkeypatch)
    await module.generate_keywords({"id": "A01", "name": QUESTION, "outcome_date": "2026-12-31"})
    _assert_golden("improve_keywords", {"calls": rec.take()})


# ── llm.py quirk branches, per model id ──────────────────────────────────────

QUIRK_MODELS = [
    # Haiku 5.5: rejects temperature, thinking disabled unless opted in, prompt cache.
    "bedrock/us.anthropic.claude-haiku-5-5",
    "bedrock/eu.anthropic.claude-haiku-5-5",
    "bedrock/global.anthropic.claude-haiku-5-5",
    "bedrock/anthropic.claude-haiku-5-5",      # no profile prefix: the cache marker needs ".anthropic."
    "us.anthropic.claude-haiku-5-5",           # no provider prefix
    "openrouter/anthropic/claude-haiku-5-5",
    # Controls and the other temperature-rejecting families.
    "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "bedrock/us.anthropic.claude-sonnet-4-6",
    "bedrock/us.anthropic.claude-sonnet-5-5",
    "bedrock/global.anthropic.claude-opus-4-8",
    "bedrock/us.anthropic.claude-opus-5",
    # Nova: prompt cache with the profile prefix only.
    "bedrock/us.amazon.nova-micro-v1:0",
    "bedrock/amazon.nova-micro-v1:0",
    # Gemini: temperature allowlisted; Flex only under vertex_ai/.
    "vertex_ai/gemini-3.5-flash-lite",
    "gemini/gemini-3.5-flash-lite",
    "openrouter/google/gemini-3.5-flash-lite",
    # Neither cache nor any quirk (retro#650).
    "bedrock/qwen.qwen3-32b-v1:0",
]


def _slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9.-]+", "_", model)


@pytest.mark.parametrize("model", QUIRK_MODELS)
async def test_llm_quirk_matrix(rec, model):
    payload = {}
    for name, extra in (("structured", {}), ("structured_thinking", {"thinking": True}),
                        ("structured_flex", {"flex": True, "flex_timeout": 25})):
        await llm.complete_structured(model, GatekeeperOutput, "fixture prompt", max_tokens=100, timeout=30,
                                      cached_prefix="fixture prefix", **extra)
        payload[name] = rec.take()
    await llm.complete_text_once_with_usage(model, "fixture prompt", system="fixture system", max_tokens=100,
                                            temperature=0, timeout=30)
    payload["text"] = rec.take()
    _assert_golden(f"quirks/{_slug(model)}", payload)


# ── Jev (raw httpx, outside litellm) ─────────────────────────────────────────

def _question_shape(questions: dict) -> dict:
    """Question ids with their keys; the per-sentence selection ids collapse to one ``s<i>``."""
    return {re.sub(r"^s\d+$", "s<i>", qid): sorted(q) for qid, q in questions.items()}


JEV_CASES = ["typesafe", "openrouter", "clef-flash", "clef", "legacy", "legacy_openrouter_dropin"]
# Headers our code sets. httpx adds its own (user-agent, accept-encoding, ...), which an httpx bump
# may change with no change of ours, so those are left out.
_JEV_OWN_HEADERS = {"authorization", "content-type"}


def test_jev_cases_cover_every_provider():
    """A new PROVIDERS entry must come with a golden."""
    assert set(JEV_CASES) - {"legacy", "legacy_openrouter_dropin"} == set(js.PROVIDERS)


@pytest.mark.parametrize("provider", JEV_CASES)
async def test_jev_request(monkeypatch, provider):
    """One pass-1 call per provider: the selection request plus the per-question negation request."""
    monkeypatch.setattr(api_settings, "jev_provider_ssm_name", "")
    if provider == "legacy_openrouter_dropin":  # infra/oracle-api.service.d/jev-openrouter.conf
        monkeypatch.setattr(api_settings, "jev_shadow_api_url", "https://openrouter.ai/api/v1/systemone")
        monkeypatch.setattr(api_settings, "jev_model", "typesafe/jev-1.13")
    elif provider != "legacy":
        monkeypatch.setattr(api_settings, "jev_provider", provider)
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct-fixture")

    def no_ssm(name):
        raise AssertionError(f"the golden must not read SSM ({name})")
    monkeypatch.setattr(js, "_read_ssm", no_ssm)
    js._NEG_CACHE.clear()
    js._SERVED.clear()
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append({
            "method": request.method,
            "url": f"{request.url.scheme}://{request.url.host}{request.url.path}",
            "header_names": sorted(h for h in request.headers.keys() if h in _JEV_OWN_HEADERS),
            "json_keys": sorted(body),
            "model": body.get("model"),
            "state_keys": sorted(body.get("state") or {}),
            "questions": _question_shape(body.get("questions") or {}),
        })
        answers = {k: {"type": "noul", "noul": 0.5} for k in body.get("questions") or {}}
        return httpx.Response(200, json={"model": body.get("model"), "answers": answers,
                                         "usage": {"input_tokens": 1}})

    out = await js.jev_pass1(text=ARTICLE, question=QUESTION, api_key="k", transport=httpx.MockTransport(handler))
    js._NEG_CACHE.clear()
    assert "err" not in out and "skip" not in out, out
    _assert_golden(f"jev/{provider}", {"requests": sorted(seen, key=lambda s: json.dumps(s, sort_keys=True))})
