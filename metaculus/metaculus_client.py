"""Thin client for the Metaculus bot API (https://www.metaculus.com/api).

Auth is a per-bot ``Authorization: Token`` header. Each Daatan bot identity
(``daatan-v1``, and later ``daatan-v2``) has its own token — see README.md for
where those live in Secrets Manager. Only binary questions are supported: the
Oracul's ``mean`` stance in [-1, 1] maps cleanly to a binary probability, but
it has no numeric/date/categorical output shape today.
"""

from __future__ import annotations

import httpx

BASE_URL = "https://www.metaculus.com/api"


class MetaculusClient:
    def __init__(self, token: str, timeout: float = 30.0, transport: httpx.BaseTransport | None = None):
        self._client = httpx.Client(
            base_url=BASE_URL,
            headers={"Authorization": f"Token {token}"},
            timeout=timeout,
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "MetaculusClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def open_binary_questions(self, tournament: str, limit: int = 50) -> list[dict]:
        """List open binary questions in a tournament, newest-activity first."""
        results: list[dict] = []
        offset = 0
        while len(results) < limit:
            page_size = min(50, limit - len(results))
            resp = self._client.get(
                "/posts/",
                params={
                    "tournaments": tournament,
                    "statuses": "open",
                    "forecast_type": "binary",
                    "order_by": "-hotness",
                    "limit": page_size,
                    "offset": offset,
                },
            )
            resp.raise_for_status()
            page = resp.json()
            page_results = page.get("results", [])
            results.extend(page_results)
            if not page.get("next") or not page_results:
                break
            offset += len(page_results)
        return results[:limit]

    def count_open_questions(self, tournament: str, forecast_type: str | None = None) -> int:
        """Count open questions in a tournament, optionally filtered by ``forecast_type``.

        Used for coverage-share reporting (retro#739 / §6 O6): comparing this with
        ``forecast_type="binary"`` against a call with no filter is the same measurement
        retro#730 did by hand (binary is 50.1% of an AIB season, 38.3-57.7% of MiniBench).
        Reads the API's own ``count`` when the paginated response carries one (a single
        request); falls back to walking every page and summing ``results`` when it
        doesn't, since the pagination shape isn't pinned in tests.
        """
        params: dict[str, str | int] = {"tournaments": tournament, "statuses": "open", "limit": 50}
        if forecast_type is not None:
            params["forecast_type"] = forecast_type
        offset = 0
        total = 0
        while True:
            resp = self._client.get("/posts/", params={**params, "offset": offset})
            resp.raise_for_status()
            page = resp.json()
            if isinstance(page.get("count"), int):
                return page["count"]
            results = page.get("results", [])
            total += len(results)
            if not page.get("next") or not results:
                return total
            offset += len(results)

    def list_tournaments(self) -> list[dict]:
        """Every tournament-type project visible to this bot (``/projects/tournaments/``).

        Unpaginated on Metaculus's side (~200 rows, 2026-08). Each row carries
        ``slug``, ``name``, ``start_date``, ``close_date``, ``forecasting_end_date``
        and ``is_ongoing`` — what season auto-detection needs (retro#726).
        """
        resp = self._client.get("/projects/tournaments/")
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, list) else data.get("results", [])

    def get_question(self, post_id: int) -> dict:
        resp = self._client.get(f"/posts/{post_id}/")
        resp.raise_for_status()
        return resp.json()

    def submit_binary_forecast(self, question_id: int, probability_yes: float) -> None:
        # Metaculus retains uncertainty by clamping submissions away from the
        # extremes; keep our own margin so a rounding edge never gets rejected.
        clamped = min(max(probability_yes, 0.03), 0.97)
        resp = self._client.post(
            "/questions/forecast/",
            json=[{"question": question_id, "source": "api", "probability_yes": clamped}],
        )
        resp.raise_for_status()

    def post_comment(self, post_id: int, text: str, *, private: bool = True) -> None:
        resp = self._client.post(
            "/comments/create/",
            json={
                "text": text,
                "parent": None,
                "included_forecast": True,
                "is_private": private,
                "on_post": post_id,
            },
        )
        resp.raise_for_status()
