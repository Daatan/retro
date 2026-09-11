"""The premise verifier (retro#575 slice 1) — shadow/log-only.

Three things are under test.

**The trigger gate must stay narrow.** Every ``/forecast`` call reaches this
point, so firing unconditionally would double LLM cost on every ordinary
request. Only a ``scheduled``/``threshold`` archetype or an already-past
``claim_deadline`` should fire it — missing metadata, a future deadline, or a
``diffuse``/``none`` archetype must not.

**Parsing fails open.** An unreachable model, a timeout, or an unparseable
reply must never itself mark a premise dead.

**Shadow means shadow.** With the feature on and triggered, the verifier is
called, but nothing about the response changes — no field, no reason, no
mutation — regardless of the verdict. That is the whole safety contract for
this slice; enforcement is a follow-up once real trigger/precision data
justifies it.
"""
from __future__ import annotations

from datetime import date, timedelta

import pytest

from forecast_api import forecaster
from forecast_api.config import settings as api_settings
from forecast_api.models import ArticleInput, ForecastRequest
from forecast_api import premise_verifier
from forecast_api.premise_verifier import (
    PremiseResult,
    Verdict,
    build_prompt,
    parse_verdict,
    premise_check_triggered,
    verify_premise,
)
from tm.models import ExtractionOutput, GatekeeperOutput, PredictionExtraction


_FRESH = (date.today() - timedelta(days=1)).isoformat()
_PAST_DEADLINE = (date.today() - timedelta(days=3)).isoformat()
_FUTURE_DEADLINE = (date.today() + timedelta(days=30)).isoformat()
_FAR_FUTURE_DEADLINE = (date.today() + timedelta(days=400)).isoformat()
_BODY = (
    "Fixture article body for the premise-verifier suite; the gatekeeper and "
    "extractor are stubbed, so no model reads it. "
) * 3
_TITLES = [
    "Harbour quartz meridian dispatch",
    "Lantern cobalt thicket bulletin",
]


class TestPrompt:
    def test_deadline_and_results_are_in_the_payload(self):
        prompt = build_prompt(
            "Will the assembly still be sitting?",
            "2026-08-01",
            [PremiseResult(
                title="Assembly dissolved after no-confidence vote",
                snippet="The chamber formally dissolved on Tuesday.",
                published_date="2026-08-02",
                source="wire-service",
            )],
        )
        assert "DEADLINE: 2026-08-01" in prompt
        assert "Assembly dissolved after no-confidence vote" in prompt
        assert "formally dissolved on Tuesday" in prompt
        assert "wire-service" in prompt

    def test_no_deadline_is_simply_absent(self):
        prompt = build_prompt("Will it happen?", None, [
            PremiseResult(title="Something", snippet=None, published_date=None, source=None),
        ])
        assert "DEADLINE:" not in prompt
        assert "TODAY:" not in prompt

    def test_future_deadline_states_it_has_not_passed(self):
        """retro#817: without a stated fact, the model had to infer
        deadline-vs-today itself, which produced 4/12 false positives in
        retro#601's sample — all future deadlines misread as already past."""
        prompt = build_prompt(
            "Will the assembly still be sitting?", _FUTURE_DEADLINE,
            [PremiseResult(title="x", snippet=None, published_date=None, source=None)],
            today=date.today().isoformat(),
        )
        assert "has NOT passed yet" in prompt

    def test_past_deadline_states_it_has_passed(self):
        prompt = build_prompt(
            "Will the assembly still be sitting?", _PAST_DEADLINE,
            [PremiseResult(title="x", snippet=None, published_date=None, source=None)],
            today=date.today().isoformat(),
        )
        assert "already passed" in prompt

    def test_unparseable_deadline_states_no_fact(self):
        prompt = build_prompt(
            "Will it happen?", "not-a-date",
            [PremiseResult(title="x", snippet=None, published_date=None, source=None)],
        )
        assert "DEADLINE: not-a-date" in prompt
        assert "TODAY:" not in prompt


class TestStaleResultFiltering:
    """retro#817 pattern 3: a real, live-model-confirmed bug. A prompt bullet
    asking the model to discount an old result was tried first and measured
    to have zero effect — the model's own training-data recall of a real
    past event outweighs an in-prompt instruction to disregard it. Filtering
    the result out before `build_prompt()` ever sees it is what worked."""

    async def test_a_stale_result_never_reaches_the_prompt(self, monkeypatch):
        seen_prompts: list[str] = []

        async def fake_complete(model, prompt, **kwargs):
            seen_prompts.append(prompt)
            return '{"dead": false, "reason": "fixture", "citation": null}'

        monkeypatch.setattr(premise_verifier, "complete_text_once", fake_complete)
        await verify_premise(
            "Will Naftali Bennett be sworn in as Prime Minister?",
            _FUTURE_DEADLINE,
            [
                PremiseResult(
                    title="Stale 2021 swearing-in article",
                    snippet="Bennett was sworn in as PM.",
                    published_date="2021-06-13",
                    source="wire-service",
                ),
                PremiseResult(
                    title="Fresh coalition talks update",
                    snippet="Coalition talks continue ahead of the vote.",
                    published_date=_FRESH,
                    source="wire-service",
                ),
            ],
            model="fixture-model", timeout_s=30,
        )
        assert len(seen_prompts) == 1
        assert "Stale 2021 swearing-in article" not in seen_prompts[0]
        assert "Fresh coalition talks update" in seen_prompts[0]

    async def test_all_results_stale_short_circuits_without_calling_the_model(self, monkeypatch):
        async def fail_if_called(*args, **kwargs):
            raise AssertionError("the model must not be called when every result is stale")

        monkeypatch.setattr(premise_verifier, "complete_text_once", fail_if_called)
        verdict = await verify_premise(
            "Will Naftali Bennett be sworn in as Prime Minister?",
            _FUTURE_DEADLINE,
            [PremiseResult(
                title="Stale 2021 swearing-in article",
                snippet="Bennett was sworn in as PM.",
                published_date="2021-06-13",
                source="wire-service",
            )],
            model="fixture-model", timeout_s=30,
        )
        assert verdict.dead is False
        assert verdict.errored is True

    async def test_result_shortly_before_a_long_past_deadline_is_not_filtered(self, monkeypatch):
        """An old article is only noise relative to a *recent* deadline. One
        published just before a deadline that itself is long past is exactly
        the evidence a re-check of an old question needs — it must not be
        thrown away just because it's old relative to *today*."""
        seen_prompts: list[str] = []

        async def fake_complete(model, prompt, **kwargs):
            seen_prompts.append(prompt)
            return '{"dead": true, "reason": "fixture", "citation": "result 1"}'

        monkeypatch.setattr(premise_verifier, "complete_text_once", fake_complete)
        await verify_premise(
            "Will the Knesset pass the 2022 budget by its statutory deadline?",
            "2021-11-14",
            [PremiseResult(
                title="Knesset approves 2022 budget just before deadline",
                snippet="Lawmakers passed the budget before the deadline.",
                published_date="2021-11-05",
                source="wire-service",
            )],
            model="fixture-model", timeout_s=30,
        )
        assert len(seen_prompts) == 1
        assert "Knesset approves 2022 budget just before deadline" in seen_prompts[0]

    async def test_fresh_result_not_filtered_for_a_far_future_deadline(self, monkeypatch):
        """Staleness is measured against min(deadline, today), not the raw
        deadline. A deadline a year out must not make a result published
        today read as ~a-year-old and get dropped — that would silently
        exclude every far-future-deadline question from shadow data."""
        seen_prompts: list[str] = []

        async def fake_complete(model, prompt, **kwargs):
            seen_prompts.append(prompt)
            return '{"dead": false, "reason": "fixture", "citation": null}'

        monkeypatch.setattr(premise_verifier, "complete_text_once", fake_complete)
        await verify_premise(
            "Will BTC reach $200k by end of 2027?",
            _FAR_FUTURE_DEADLINE,
            [PremiseResult(
                title="BTC price update",
                snippet="Bitcoin trades near recent highs.",
                published_date=_FRESH,
                source="wire-service",
            )],
            model="fixture-model", timeout_s=30,
        )
        assert len(seen_prompts) == 1
        assert "BTC price update" in seen_prompts[0]


class TestParsing:
    def test_clean_json(self):
        v = parse_verdict('{"dead": true, "reason": "already occurred", "citation": "result 1"}')
        assert (v.dead, v.errored) == (True, False)
        assert "already occurred" in v.reason
        assert "result 1" in v.reason

    def test_json_wrapped_in_prose_or_fences(self):
        v = parse_verdict('Here:\n```json\n{"dead": false, "reason": "still open"}\n```')
        assert (v.dead, v.errored) == (False, False)

    @pytest.mark.parametrize("text", ["", "no idea", '{"dead": "true"}', "{not json}"])
    def test_anything_unparseable_fails_open(self, text):
        v = parse_verdict(text)
        assert v.dead is False, "an unreadable verdict must never read as dead"
        assert v.errored is True


class TestTriggerGate:
    def test_scheduled_archetype_always_fires(self):
        assert premise_check_triggered(None, "scheduled") is True

    def test_threshold_archetype_always_fires(self):
        assert premise_check_triggered(_FUTURE_DEADLINE, "threshold") is True

    def test_diffuse_archetype_with_no_deadline_never_fires(self):
        assert premise_check_triggered(None, "diffuse") is False

    def test_no_metadata_at_all_never_fires(self):
        assert premise_check_triggered(None, None) is False

    def test_future_deadline_without_archetype_does_not_fire(self):
        assert premise_check_triggered(_FUTURE_DEADLINE, None) is False

    def test_past_deadline_without_archetype_fires(self):
        assert premise_check_triggered(_PAST_DEADLINE, None) is True

    def test_past_deadline_with_diffuse_archetype_still_fires(self):
        """The deadline check is independent of archetype — a diffuse claim
        whose deadline has passed is exactly the population this exists for."""
        assert premise_check_triggered(_PAST_DEADLINE, "diffuse") is True

    def test_unparseable_deadline_does_not_fire(self):
        assert premise_check_triggered("not-a-date", None) is False


def _patch(monkeypatch, *, verdict: Verdict | None = None):
    async def fake_gate(**kwargs):
        return (GatekeeperOutput(
            is_prediction=True, reason="fixture gate",
            prediction_count_estimate=1, relevance_score=1.0,
        ), {"total_tokens": 0})

    async def fake_extract(**kwargs):
        return (ExtractionOutput(predictions=[PredictionExtraction(
            quote="A neutral fixture quote.",
            claim="A neutral fixture claim.",
            stance=0.4, certainty=0.5, settled=False,
        )]), {"total_tokens": 0})

    monkeypatch.setattr(forecaster, "check_is_prediction", fake_gate)
    monkeypatch.setattr(forecaster, "extract_predictions", fake_extract)
    monkeypatch.setattr(forecaster, "get_credibility_weight", lambda sid: 1.0)
    calls: list[tuple] = []
    if verdict is not None:
        async def fake_verify(question, claim_deadline, results, **kwargs):
            calls.append((question, claim_deadline, list(results)))
            return verdict
        monkeypatch.setattr(forecaster, "verify_premise", fake_verify)
    return calls


async def _run(monkeypatch, question: str, *, verdict: Verdict | None = None,
                enabled: bool = True, claim_deadline: str | None = None,
                claim_archetype: str | None = None):
    calls = _patch(monkeypatch, verdict=verdict)
    monkeypatch.setattr(api_settings, "premise_verifier_enabled", enabled)
    resp = await forecaster.run_forecast(ForecastRequest(
        question=question,
        claim_deadline=claim_deadline,
        claim_archetype=claim_archetype,
        articles=[
            ArticleInput(
                url=f"https://source-{i}.example.test/story", title=_TITLES[i - 1],
                snippet=f"Fixture snippet long enough to be usable, variant {i}.",
                source=f"source-{i}", published_date=_FRESH, text=_BODY,
            )
            for i in (1, 2)
        ],
    ))
    return resp, calls


class TestShadowMode:
    async def test_disabled_means_no_call_at_all(self, monkeypatch):
        _resp, calls = await _run(
            monkeypatch, "[premise-off] Will the vote still happen?",
            verdict=Verdict(dead=True, reason="would flag dead"),
            enabled=False, claim_archetype="scheduled",
        )
        assert calls == []

    async def test_untriggered_request_never_calls(self, monkeypatch):
        _resp, calls = await _run(
            monkeypatch, "[premise-untriggered] Will the vote still happen?",
            verdict=Verdict(dead=True, reason="would flag dead"),
            enabled=True, claim_archetype="diffuse",
        )
        assert calls == []

    async def test_scheduled_archetype_calls_and_never_mutates_the_response(self, monkeypatch):
        resp, calls = await _run(
            monkeypatch, "[premise-scheduled] Will the vote still happen?",
            verdict=Verdict(dead=True, reason="already occurred, cites result 1"),
            enabled=True, claim_archetype="scheduled",
        )
        assert len(calls) == 1, "the trigger gate should have fired the call"
        assert resp.insufficient_data is False
        assert resp.reason is None
        assert resp.articles_used == 2, "log-only: the pool prices normally regardless of the verdict"

    async def test_past_deadline_calls_and_never_mutates_the_response(self, monkeypatch):
        resp, calls = await _run(
            monkeypatch, "[premise-deadline] Will the vote still happen?",
            verdict=Verdict(dead=True, reason="deadline passed with nothing reported"),
            enabled=True, claim_deadline=_PAST_DEADLINE,
        )
        assert len(calls) == 1
        assert resp.insufficient_data is False
        assert resp.reason is None

    async def test_a_live_verdict_also_leaves_the_response_untouched(self, monkeypatch):
        resp, calls = await _run(
            monkeypatch, "[premise-live] Will the vote still happen?",
            verdict=Verdict(dead=False, reason="only announced so far"),
            enabled=True, claim_archetype="scheduled",
        )
        assert len(calls) == 1
        assert resp.insufficient_data is False

    async def test_an_errored_verdict_is_still_just_logged(self, monkeypatch):
        resp, calls = await _run(
            monkeypatch, "[premise-errored] Will the vote still happen?",
            verdict=Verdict(dead=False, reason="verifier call failed", errored=True),
            enabled=True, claim_archetype="threshold",
        )
        assert len(calls) == 1
        assert resp.insufficient_data is False


class TestShippedDefaults:
    """Both flags default off — this slice is shadow-only by construction,
    and a silent flip to on is exactly the change that should never pass
    review unnoticed."""

    def test_verifier_ships_disabled(self):
        from forecast_api.config import ApiSettings
        assert ApiSettings.model_fields["premise_verifier_enabled"].default is False

    def test_enforce_ships_disabled_and_is_unread_this_slice(self):
        from forecast_api.config import ApiSettings
        assert ApiSettings.model_fields["premise_verifier_enforce"].default is False
