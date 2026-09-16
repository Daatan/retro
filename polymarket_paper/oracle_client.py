"""Thin client for Daatan's own Oracle forecast API (oracle.daatan.com).

Uses ``ORACLE_API_KEYS``' named-key mechanism (see
``retro/api/src/forecast_api/auth.py``), never the primary daatan-app key —
docs#57 is exactly what happens when a relay shares the prod key: staging
burned unmetered LLM spend through it. Register a capped named key for this
relay before pointing it at anything but a local test.
"""

from __future__ import annotations

import httpx

DEFAULT_BASE_URL = "https://oracle.daatan.com"

# /forecast's p99 latency is ~226s, dominated by the LLM article phase (see
# docs reference_oracle_latency_profile.md) — the usual few-seconds default
# would time out a normal run.
DEFAULT_TIMEOUT = 240.0


class OracleClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        transport: httpx.BaseTransport | None = None,
    ):
        self._client = httpx.Client(
            base_url=base_url,
            headers={"x-api-key": api_key},
            timeout=timeout,
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "OracleClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def forecast(self, question: str, resolution_criteria: str | None = None) -> dict:
        body: dict = {"question": question}
        if resolution_criteria:
            # Same pattern daatan uses for this field (daatan#1375): omit
            # rather than send an empty string, so an absent value can't
            # collide with retro's forecast cache key.
            body["resolution_criteria"] = resolution_criteria
        resp = self._client.post("/forecast", json=body)
        resp.raise_for_status()
        return resp.json()
