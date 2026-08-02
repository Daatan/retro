"""The settlement match gate (retro#388, retro#360) — shadow first.

Three things are under test, and the third is the one that matters.

**The prompt payload** must carry the verbatim quote, not just the claim
summary. That is the whole point: on the incident this gate exists for, the
claim summaries read as the outcome (*"The United States has agreed to grant
Ukraine a license…"*) while the quotes said the decider had *announced* it
(*"Trump said the US will grant Ukraine a licence"*). A gate that only sees
summaries sees nothing wrong.

**Parsing fails open.** An unreachable model, a timeout or an unparseable reply
must never suppress a pin — an LLM outage silently changing published numbers
is a worse failure than the one being fixed.

**Shadow means shadow.** With ``settlement_verifier_enforce`` off, a NO verdict
changes nothing at all. That is what makes it safe to enable in prod and
measure against the pins we already have ground truth for, before it is allowed
to act.
"""
from __future__ import annotations

from datetime import date, timedelta

import pytest

from forecast_api import forecaster
from forecast_api.config import settings as api_settings
from forecast_api.models import ArticleInput, ForecastRequest, PoolAggregateRequest, PoolSourceInput
from forecast_api.settlement_verifier import (
    SettlementVote,
    Verdict,
    build_prompt,
    parse_verdict,
)
from tm.models import ExtractionOutput, GatekeeperOutput, PredictionExtraction


_FRESH = (date.today() - timedelta(days=1)).isoformat()
_FRESH_EVENT = (date.today() - timedelta(days=2)).isoformat()
_BODY = (
    "Fixture article body for the match-gate suite; the gatekeeper and extractor "
    "are stubbed, so no model reads it. "
) * 3
_TITLES = [
    "Harbour quartz meridian dispatch",
    "Lantern cobalt thicket bulletin",
]


class TestPrompt:
    def test_the_quote_is_in_the_payload_not_just_the_claim(self):
        prompt = build_prompt(
            "Will the United States grant Ukraine a licence to produce Patriot systems?",
            [SettlementVote(
                outlet="pravda.com.ua",
                claim="The United States has agreed to grant Ukraine a license to produce Patriot systems.",
                quote="Trump said the US will grant Ukraine a licence to produce Patriot missiles.",
                event_date="2026-07-08",
            )],
        )
        assert "will grant" in prompt, (
            "the tense that makes this an announcement lives in the quote; a payload "
            "carrying only the claim summary cannot answer the question being asked"
        )
        assert "pravda.com.ua" in prompt
        assert "2026-07-08" in prompt

    def test_a_missing_quote_is_simply_absent(self):
        prompt = build_prompt("Will it happen?", [
            SettlementVote(outlet=None, claim="It happened.", quote=None, event_date=None),
        ])
        assert "It happened." in prompt
        assert "reported as" not in prompt


class TestParsing:
    def test_clean_json(self):
        v = parse_verdict('{"settles": false, "reason": "announcement, not the act"}')
        assert (v.settles, v.errored) == (False, False)
        assert "announcement" in v.reason

    def test_json_wrapped_in_prose_or_fences(self):
        v = parse_verdict('Here is my answer:\n```json\n{"settles": true, "reason": "reported as done"}\n```')
        assert (v.settles, v.errored) == (True, False)

    @pytest.mark.parametrize("text", ["", "no idea", '{"settles": "false"}', "{not json}"])
    def test_anything_unparseable_fails_open(self, text):
        v = parse_verdict(text)
        assert v.settles is True, "an unreadable verdict must never read as a veto"
        assert v.errored is True


def _patch(monkeypatch, claims, *, verdict: Verdict | None = None):
    async def fake_gate(**kwargs):
        return (GatekeeperOutput(
            is_prediction=True, reason="fixture gate",
            prediction_count_estimate=len(claims), relevance_score=1.0,
        ), {"total_tokens": 0})

    async def fake_extract(**kwargs):
        return (ExtractionOutput(predictions=list(claims)), {"total_tokens": 0})

    monkeypatch.setattr(forecaster, "check_is_prediction", fake_gate)
    monkeypatch.setattr(forecaster, "extract_predictions", fake_extract)
    monkeypatch.setattr(forecaster, "get_credibility_weight", lambda sid: 1.0)
    calls: list[tuple] = []
    if verdict is not None:
        async def fake_verify(question, votes, **kwargs):
            calls.append((question, list(votes)))
            return verdict
        monkeypatch.setattr(forecaster, "verify_settlement", fake_verify)
    return calls


def _settling_claim(i: int) -> PredictionExtraction:
    return PredictionExtraction(
        quote=f"Officials said the step will be taken, variant {i}.",
        claim=f"The step has been taken, variant {i}.",
        stance=1.0, certainty=0.95, settled=True, event_date=_FRESH_EVENT,
        evidence_class="reported_fact",
    )


async def _run(monkeypatch, question: str, *, verdict: Verdict | None = None, enforce: bool = False):
    calls = _patch(monkeypatch, [_settling_claim(1)], verdict=verdict)
    monkeypatch.setattr(api_settings, "settlement_verifier_enabled", True)
    monkeypatch.setattr(api_settings, "settlement_verifier_enforce", enforce)
    resp = await forecaster.run_forecast(ForecastRequest(
        question=question,
        articles=[
            ArticleInput(
                url=f"https://source-{i}.example.test/story",
                title=_TITLES[i - 1],
                snippet=f"Fixture snippet long enough to be usable, variant {i}.",
                source=f"source-{i}", published_date=_FRESH, text=_BODY,
            )
            for i in (1, 2)
        ],
    ))
    return resp, calls


class TestShadowMode:
    async def test_a_no_verdict_changes_nothing(self, monkeypatch):
        resp, calls = await _run(
            monkeypatch, "[gate-shadow] Will the step be taken?",
            verdict=Verdict(settles=False, reason="the decider only announced it"),
        )
        assert resp.settled is True, "shadow mode must not act on the verdict"
        assert resp.mean == pytest.approx(api_settings.settlement_stance)
        assert len(calls) == 1, "…but it must still have asked"

    async def test_the_verifier_sees_the_settling_claims_and_their_quotes(self, monkeypatch):
        _resp, calls = await _run(
            monkeypatch, "[gate-payload] Will the step be taken?",
            verdict=Verdict(settles=True, reason="reported as done"),
        )
        _question, votes = calls[0]
        assert votes, "the settling votes must reach the verifier"
        assert all(v.quote for v in votes)


class TestEnforcement:
    async def test_a_no_verdict_suppresses_the_pin(self, monkeypatch):
        resp, _calls = await _run(
            monkeypatch, "[gate-enforce] Will the step be taken?",
            verdict=Verdict(settles=False, reason="announced, not carried out"),
            enforce=True,
        )
        assert resp.settled is False
        assert resp.mean != pytest.approx(api_settings.settlement_stance)

    async def test_the_vetoed_rows_still_vote_as_ordinary_evidence(self, monkeypatch):
        """A vetoed settlement is a demotion, not a deletion — the articles are
        still evidence about the claim, they just may not pin it."""
        resp, _calls = await _run(
            monkeypatch, "[gate-demote] Will the step be taken?",
            verdict=Verdict(settles=False, reason="adjacent event"),
            enforce=True,
        )
        assert resp.articles_used == 2
        assert resp.mean > 0, "two stance-1.0 rows still pool positive after the veto"

    async def test_an_errored_verdict_never_suppresses(self, monkeypatch):
        resp, _calls = await _run(
            monkeypatch, "[gate-errored] Will the step be taken?",
            verdict=Verdict(settles=True, reason="verifier call failed", errored=True),
            enforce=True,
        )
        assert resp.settled is True

    async def test_a_yes_verdict_leaves_the_pin_alone(self, monkeypatch):
        resp, _calls = await _run(
            monkeypatch, "[gate-yes] Will the step be taken?",
            verdict=Verdict(settles=True, reason="the outcome is reported as done"),
            enforce=True,
        )
        assert resp.settled is True
        assert resp.mean == pytest.approx(api_settings.settlement_stance)


class TestItSkipsRatherThanGuesses:
    async def test_disabled_means_no_call_at_all(self, monkeypatch):
        calls = _patch(monkeypatch, [_settling_claim(1)],
                       verdict=Verdict(settles=False, reason="would veto"))
        monkeypatch.setattr(api_settings, "settlement_verifier_enabled", False)
        resp = await forecaster.run_forecast(ForecastRequest(
            question="[gate-off] Will the step be taken?",
            articles=[
                ArticleInput(
                    url=f"https://source-{i}.example.test/story", title=_TITLES[i - 1],
                    snippet=f"Fixture snippet long enough to be usable, variant {i}.",
                    source=f"source-{i}", published_date=_FRESH, text=_BODY,
                )
                for i in (1, 2)
            ],
        ))
        assert resp.settled is True
        assert calls == []

    async def test_the_recompute_path_without_a_question_skips(self, monkeypatch):
        """`/pool/aggregate` had no reason to carry the claim text until now.
        Without it the gate cannot run — and must say so rather than guess."""
        called: list[int] = []

        async def fake_verify(*_a, **_kw):
            called.append(1)
            return Verdict(settles=False, reason="should never be reached")

        monkeypatch.setattr(forecaster, "verify_settlement", fake_verify)
        monkeypatch.setattr(api_settings, "settlement_verifier_enabled", True)
        monkeypatch.setattr(api_settings, "settlement_verifier_enforce", True)

        resp = await forecaster.run_pool_aggregate(PoolAggregateRequest(
            sources=[
                PoolSourceInput(
                    stance=0.95, certainty=0.95, credibility_weight=1.0,
                    relevance_score=1.0, evidence_weight=1.0, published_date=_FRESH,
                    settled=True, settlement_event_date=_FRESH_EVENT,
                ),
                PoolSourceInput(
                    stance=0.9, certainty=0.95, credibility_weight=1.0,
                    relevance_score=1.0, evidence_weight=1.0, published_date=_FRESH,
                    settled=True, settlement_event_date=_FRESH_EVENT,
                ),
            ],
        ))
        assert resp.settled is True
        assert called == []

    async def test_a_row_with_no_claim_detail_skips(self, monkeypatch):
        """Legacy rows carry no per-claim data (F1/retro#364 is recent and was
        not backfilled). No claims means nothing to check, which is not the same
        as a failed check."""
        called: list[int] = []

        async def fake_verify(*_a, **_kw):
            called.append(1)
            return Verdict(settles=False, reason="should never be reached")

        monkeypatch.setattr(forecaster, "verify_settlement", fake_verify)
        monkeypatch.setattr(api_settings, "settlement_verifier_enabled", True)
        monkeypatch.setattr(api_settings, "settlement_verifier_enforce", True)

        resp = await forecaster.run_pool_aggregate(PoolAggregateRequest(
            question="Will the step be taken?",
            sources=[
                PoolSourceInput(
                    stance=0.95, certainty=0.95, credibility_weight=1.0,
                    relevance_score=1.0, evidence_weight=1.0, published_date=_FRESH,
                    settled=True, settlement_event_date=_FRESH_EVENT,
                ),
                PoolSourceInput(
                    stance=0.9, certainty=0.95, credibility_weight=1.0,
                    relevance_score=1.0, evidence_weight=1.0, published_date=_FRESH,
                    settled=True, settlement_event_date=_FRESH_EVENT,
                ),
            ],
        ))
        assert resp.settled is True
        assert called == []
