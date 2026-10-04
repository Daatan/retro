"""Jev provider switch (retro#901): one name picks URL + model + key; SSM-driven, TTL-cached."""

import asyncio
import json
import logging

import httpx

from forecast_api import jev_shadow as js
from forecast_api.config import settings


def test_no_provider_keeps_legacy_settings():
    p = js.resolve_provider()
    assert p.name == "custom" and p.url == settings.jev_shadow_api_url and p.model == settings.jev_model
    assert p.calibrated


def test_env_provider_wins(monkeypatch):
    monkeypatch.setattr(settings, "jev_provider", "openrouter")
    p = js.resolve_provider()
    assert p.url == "https://openrouter.ai/api/v1/systemone" and p.model == "typesafe/jev-1.13"
    assert p.key_ssm == "/retro/prod/secrets/OPENROUTER_JEV_API_KEY" and js.current_provider() is p


def test_ssm_provider_is_ttl_cached_and_switchable(monkeypatch, caplog):
    monkeypatch.setattr(settings, "jev_provider_ssm_name", "/retro/prod/secrets/JEV_PROVIDER")
    value = {"v": "openrouter"}
    reads = []
    monkeypatch.setattr(js, "_read_ssm", lambda name: reads.append(name) or value["v"])
    caplog.set_level(logging.INFO, logger="forecast_api.jev_shadow")
    assert js.resolve_provider().name == "openrouter"
    value["v"] = "typesafe"
    assert js.resolve_provider().name == "openrouter"  # inside the TTL: no re-read
    assert reads == ["/retro/prod/secrets/JEV_PROVIDER"]
    monkeypatch.setattr(settings, "jev_provider_ttl_seconds", 0.0)
    assert js.resolve_provider().name == "typesafe"
    assert sum("event=jev_provider name=" in r.message for r in caplog.records) == 2


def test_ssm_blip_keeps_last_choice(monkeypatch):
    monkeypatch.setattr(settings, "jev_provider_ssm_name", "/x")
    monkeypatch.setattr(settings, "jev_provider_ttl_seconds", 0.0)
    value = {"v": "openrouter"}
    monkeypatch.setattr(js, "_read_ssm", lambda name: value["v"])
    assert js.resolve_provider().name == "openrouter"
    value["v"] = None
    assert js.resolve_provider().name == "openrouter"


def test_unknown_provider_falls_back_to_legacy(monkeypatch, caplog):
    monkeypatch.setattr(settings, "jev_provider", "nope")
    assert js.resolve_provider().name == "custom"
    assert any("unknown=nope" in r.message for r in caplog.records)


def test_clef_needs_account_and_is_uncalibrated(monkeypatch):
    monkeypatch.setattr(settings, "jev_provider", "clef-flash")
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
    monkeypatch.setattr(js, "_read_ssm", lambda name: None)
    assert js.resolve_provider().name == "custom"  # no account id → legacy, never a broken URL
    js._PROVIDER.clear()
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acc123")
    p = js.resolve_provider()
    assert p.url == "https://api.cloudflare.com/client/v4/accounts/acc123/ai/run/@cf/cloudflare/clef-flash"
    assert p.model == "clef-flash" and not p.calibrated


def test_key_is_per_provider(monkeypatch):
    import tm.web_search
    asked = []
    monkeypatch.setattr(tm.web_search, "_secret", lambda env, name: asked.append((env, name)) or "k-" + env)
    assert js.resolve_api_key(provider=js.PROVIDERS["openrouter"]) == "k-OPENROUTER_JEV_API_KEY"
    assert js.resolve_api_key(provider=js.PROVIDERS["typesafe"]) == "k-TYPESAFE_API_KEY"
    js.resolve_api_key(provider=js.PROVIDERS["openrouter"])
    assert [n for _, n in asked] == ["/retro/prod/secrets/OPENROUTER_JEV_API_KEY", js.KEY_SSM_NAME]


def test_cloudflare_envelope_chunks_and_noul_schema():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        assert all("criteria" not in q for q in body["questions"].values())
        return httpx.Response(200, json={"success": True, "result": {
            "model": "clef-flash",
            "answers": {k: {"type": "noul", "noul": 0.5} for k in body["questions"]},
            "usage": {"input_tokens": 10}}})

    qs = js._selection_questions(150)
    url = "https://api.cloudflare.com/client/v4/accounts/a/ai/run/@cf/cloudflare/clef-flash"

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
            return await js._ask(c, "k", {"sentences": []}, qs, url, "clef-flash")
    out = asyncio.run(go())
    assert [len(b["questions"]) for b in seen] == [64, 64, 22]
    assert len(out["answers"]) == 150 and out["usage"]["input_tokens"] == 30 and out["model"] == "clef-flash"
