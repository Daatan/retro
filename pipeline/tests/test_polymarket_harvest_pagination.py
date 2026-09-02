"""Regression tests for retro#764 — CLOB cursor pagination past the old 2,100-row cap.

Before the fix, harvest() and backfill_clob_tokens() paged the Gamma API via
`offset = page * PAGE_SIZE` and treated any non-200 response as end-of-pagination.
Gamma's `offset` hard-caps around 2000, so both loops silently stopped at
~page 21 (~2,100 rows) despite the corpus having millions of markets. Both now
page the cursor-based CLOB `/markets` endpoint, which has no offset cap.
"""

import json

import pytest

pytest.importorskip("httpx")

from tm import polymarket_harvest as pm


def _clob_market(condition_id, question, yes_price=0.0, no_price=1.0, tags=None, end_date="2024-06-01T00:00:00Z"):
    return {
        "condition_id": condition_id,
        "question": question,
        "description": "",
        "market_slug": question.lower().replace(" ", "-"),
        "end_date_iso": end_date,
        "tags": tags or ["Politics", "Elections"],
        "closed": True,
        "tokens": [
            {"token_id": f"{condition_id}-yes", "outcome": "Yes", "price": yes_price, "winner": yes_price > no_price},
            {"token_id": f"{condition_id}-no", "outcome": "No", "price": no_price, "winner": no_price > yes_price},
        ],
    }


class _FakeClobClient:
    """Serves N pages of 1 market each, well past the old ~21-page Gamma cap."""

    def __init__(self, n_pages=25):
        self.n_pages = n_pages
        self.requests = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get(self, url, params=None, timeout=None):
        assert url == f"{pm.CLOB_BASE}/markets"
        cursor = (params or {}).get("next_cursor", "")
        page_idx = int(cursor) if cursor else 0
        self.requests.append(cursor)

        market = _clob_market(
            f"0xcond{page_idx}",
            f"Will candidate {page_idx} win the election?",
        )
        next_cursor = str(page_idx + 1) if page_idx + 1 < self.n_pages else pm.CLOB_END_CURSOR
        body = {"data": [market], "next_cursor": next_cursor, "limit": 1, "count": 1}
        return _Resp(200, body)


class _Resp:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self._body = body

    def json(self):
        return self._body


class TestHarvestPaginationPastOldCap:
    def test_harvest_pages_past_the_old_21_page_gamma_cap(self, tmp_path, monkeypatch):
        fake_client = _FakeClobClient(n_pages=25)
        monkeypatch.setattr(pm.httpx, "Client", lambda timeout=None: fake_client)
        monkeypatch.setattr(pm.time, "sleep", lambda s: None)

        events = pm.harvest(tmp_path, start_date="2023-01-01", end_date="2025-01-01")

        # Old code broke after ~21 pages (HTTP 422 from Gamma's offset cap).
        assert len(fake_client.requests) == 25
        assert len(events) == 25

    def test_non_200_page_is_not_treated_as_end_of_corpus(self, tmp_path, monkeypatch, caplog):
        class _FlakyClient(_FakeClobClient):
            def get(self, url, params=None, timeout=None):
                cursor = (params or {}).get("next_cursor", "")
                if cursor == "2":
                    return _Resp(503, {})
                return super().get(url, params=params, timeout=timeout)

        fake_client = _FlakyClient(n_pages=5)
        monkeypatch.setattr(pm.httpx, "Client", lambda timeout=None: fake_client)
        monkeypatch.setattr(pm.time, "sleep", lambda s: None)

        with caplog.at_level("ERROR"):
            events = pm.harvest(tmp_path, start_date="2023-01-01", end_date="2025-01-01")

        # Stopped early due to the transient failure, not because the corpus ended.
        assert len(events) == 2
        assert any("transient failure, not end of corpus" in r.message for r in caplog.records)

    def test_clob_market_adapter_maps_category_and_outcome(self):
        raw = _clob_market("0xabc", "Will X win the election?", yes_price=1.0, no_price=0.0)
        market = pm._clob_market_to_legacy_shape(raw)

        assert market["id"] == "0xabc"
        assert market["category"] == "politics"
        assert json.loads(market["outcomePrices"]) == [1.0, 0.0]
        assert market["clobTokenIds"] == ["0xabc-yes"]
        assert pm._is_political(market) is True
        assert pm._extract_outcome(market) is True
