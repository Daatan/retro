"""Tests for the Gamma keyword-search relevance guard in tm.polymarket.

Regression coverage for a bug found live via oracle-mcp-test.html: a
natural-language forecast question ("Next world record in sprint will be
broken next year") matched an unrelated "before GTA VI?" joke market with
zero shared words, because _lookup_by_keywords used to trust Gamma's own
search result[0] unconditionally. Gamma's real /markets?search= response for
that exact query was captured and used as the basis for these fixtures.
"""

from tm import polymarket


class _FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


class _FakeAsyncClient:
    """Maps a search query string to the candidate list Gamma would return."""

    def __init__(self, responses: dict):
        self._responses = responses
        self.queries_seen: list[str] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url, params=None, headers=None):
        q = params.get("search")
        self.queries_seen.append(q)
        payload = self._responses.get(q, [])
        return _FakeResponse(200, payload)


def _patch_client(monkeypatch, fake_client):
    monkeypatch.setattr(polymarket.httpx, "AsyncClient", lambda **kwargs: fake_client)


# ── _significant_words / _relevance_score ────────────────────────────────

class TestSignificantWords:
    def test_strips_stopwords_and_short_words(self):
        words = polymarket._significant_words("Will the Fed cut rates in 2026?")
        assert words == {"rates", "2026"}

    def test_lowercases_and_strips_punctuation(self):
        words = polymarket._significant_words("New Rihanna Album before GTA VI?")
        assert words == {"rihanna", "album"}


class TestRelevanceScore:
    def test_zero_when_no_shared_words(self):
        query = "Next world record in sprint will be broken next year"
        question = "New Rihanna Album before GTA VI?"
        assert polymarket._relevance_score(query, question) == 0

    def test_counts_shared_significant_words(self):
        query = "Trump reelection 2028"
        question = "Will Trump run for President in 2028?"
        assert polymarket._relevance_score(query, question) == 2  # trump, 2028


# ── _lookup_by_keywords ───────────────────────────────────────────────────

# Real Gamma response for this exact query, captured live 2026-07-28.
_GTA_VI_GARBAGE = [
    {"question": "New Rihanna Album before GTA VI?"},
    {"question": "New Playboi Carti Album before GTA VI?"},
    {"question": "Will Jesus Christ return before GTA VI?"},
    {"question": "Trump out as President before GTA VI?"},
    {"question": "Will China invades Taiwan before GTA VI?"},
]


class TestLookupByKeywords:
    async def test_returns_none_rather_than_an_unrelated_top_hit(self, monkeypatch):
        query = "Next world record in sprint will be broken next year"
        fake = _FakeAsyncClient({query: _GTA_VI_GARBAGE})
        _patch_client(monkeypatch, fake)

        result = await polymarket._lookup_by_keywords([query], query)

        assert result is None

    async def test_picks_the_relevant_candidate_over_a_higher_ranked_irrelevant_one(self, monkeypatch):
        query = "sprint world record"
        candidates = _GTA_VI_GARBAGE + [
            {"question": "New 100m sprint world record set in 2026?"},
        ]
        fake = _FakeAsyncClient({query: candidates})
        _patch_client(monkeypatch, fake)

        result = await polymarket._lookup_by_keywords([query], query)

        assert result == {"question": "New 100m sprint world record set in 2026?"}

    async def test_falls_through_to_the_next_query_when_first_has_no_relevant_match(self, monkeypatch):
        fake = _FakeAsyncClient({
            "bad phrasing": _GTA_VI_GARBAGE,
            "sprint world record": [{"question": "Sprint world record broken in 2026?"}],
        })
        _patch_client(monkeypatch, fake)

        result = await polymarket._lookup_by_keywords(["bad phrasing"], "sprint world record")

        assert result == {"question": "Sprint world record broken in 2026?"}
        assert fake.queries_seen == ["bad phrasing", "sprint world record"]

    async def test_still_matches_a_well_formed_curated_event(self, monkeypatch):
        """Guards against the relevance filter regressing the harvest pipeline's
        existing curated event_name/search_keywords matches."""
        query = "Trump reelection 2028"
        fake = _FakeAsyncClient({query: [{"question": "Will Trump run for President in 2028?"}]})
        _patch_client(monkeypatch, fake)

        result = await polymarket._lookup_by_keywords([query], query)

        assert result == {"question": "Will Trump run for President in 2028?"}
