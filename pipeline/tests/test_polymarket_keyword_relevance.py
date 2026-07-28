"""Tests for the Gamma keyword-search path in tm.polymarket.

Two bugs are covered here, found in sequence:

1. #334 — _lookup_by_keywords trusted Gamma's own result[0] unconditionally, so a
   natural-language forecast question ("Next world record in sprint will be
   broken next year") matched an unrelated "before GTA VI?" joke market with
   zero shared words. Fixed with a shared-significant-word guard.

2. #339 — the guard then exposed the real problem: the endpoint being queried,
   `/markets?search=`, ignores the query entirely and returns a volume-ranked
   default listing (verified live 2026-07-28: `search=bitcoin` returned not one
   Bitcoin market, and three unrelated queries returned byte-identical results).
   The keyword fallback could therefore never match. Fixed by moving to
   `/public-search?q=`, Polymarket's real search, with the query reduction that
   endpoint needs.

Fixtures are shaped like real captured responses from each endpoint.
"""

from tm import polymarket


class _FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


def _events_payload(*questions: str) -> dict:
    """A /public-search response: markets nested under events[]."""
    return {"events": [{"markets": [{"question": q} for q in questions]}]}


class _FakeAsyncClient:
    """Maps a /public-search `q` value to the payload Gamma would return."""

    def __init__(self, responses: dict):
        self._responses = responses
        self.queries_seen: list[str] = []
        self.paths_seen: list[str] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url, params=None, headers=None):
        self.paths_seen.append(url)
        q = (params or {}).get("q")
        self.queries_seen.append(q)
        return _FakeResponse(200, self._responses.get(q, {"events": []}))


def _patch_client(monkeypatch, fake_client):
    monkeypatch.setattr(polymarket.httpx, "AsyncClient", lambda **kwargs: fake_client)


# ── _significant_words / _relevance_score (the #334 guard) ────────────────

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


class TestRelevanceRatio:
    """Once /public-search actually works, candidates are topically adjacent, so
    a raw count stops discriminating. Both rejections below were real live
    false positives under the old `>= 1 shared word` rule."""

    def test_rejects_a_match_on_a_single_shared_entity(self):
        query = "Will Estonia adopt a nationwide four-day working week by 2027?"
        question = "Will Mart Helme be the next President of Estonia?"
        assert polymarket._relevance_score(query, question) == 1  # would have passed
        assert polymarket._relevance_ratio(query, question) < polymarket._MIN_RELEVANCE_RATIO

    def test_rejects_a_topically_adjacent_but_different_question(self):
        query = "Will Bitcoin reach $200,000 in 2026?"
        question = "Will MicroStrategy buy 200+ Bitcoin in next purchase?"
        assert polymarket._relevance_ratio(query, question) < polymarket._MIN_RELEVANCE_RATIO

    def test_accepts_a_genuine_match(self):
        query = "Israel strikes Iran"
        question = "Israel strikes Iran by January 31, 2026?"
        assert polymarket._relevance_ratio(query, question) == 1.0

    def test_accepts_a_partial_but_substantial_match(self):
        query = "Trump reelection 2028"
        question = "Will Donald Trump win the 2028 US Presidential Election?"
        assert polymarket._relevance_ratio(query, question) >= polymarket._MIN_RELEVANCE_RATIO

    def test_zero_for_an_empty_query(self):
        assert polymarket._relevance_ratio("the a of", "Israel strikes Iran?") == 0.0


class TestCompactAmounts:
    def test_rewrites_thousands_to_k_form(self):
        out = polymarket._compact_amounts("Will Bitcoin reach $200,000 in 2026?")
        # "200k" is a standalone token; the split "200"/"000" fragments are gone.
        # The bare "$" left behind is stripped later by _search_query's cleanup.
        assert "200k" in out.split()
        assert "200,000" not in out

    def test_leaves_non_round_amounts_as_digits(self):
        assert "1234567" in polymarket._compact_amounts("a 1,234,567 vote margin")

    def test_leaves_plain_numbers_untouched(self):
        assert polymarket._compact_amounts("200k by 2026") == "200k by 2026"

    def test_feeds_a_searchable_token_into_the_query(self):
        # The whole point: "200 000" finds nothing, "200k" finds the real market.
        assert polymarket._search_query("Will Bitcoin reach $200,000 in 2026?") == (
            "bitcoin 200k 2026"
        )


# ── _search_query (the #339 query reduction) ──────────────────────────────

class TestSearchQuery:
    def test_drops_stopwords_and_framing_verbs(self):
        # "will", "the", "by" are stopwords; "officially" is a framing verb that
        # never appears in a market title.
        assert polymarket._search_query("Will the treaty be officially signed by 2027?") == (
            "treaty 2027"
        )

    def test_leads_with_proper_nouns(self):
        # Gamma weights leading tokens heavily, so entities must come first even
        # when they appear late in the sentence.
        q = polymarket._search_query("a direct military confrontation between Israel and Iran")
        assert q.split()[:2] == ["israel", "iran"]

    def test_normalizes_usa_to_us(self):
        # The literal token "usa" zeroes out results — titles say "US x Iran".
        assert "usa" not in polymarket._search_query("Will the USA bomb Iran?")
        assert "us" in polymarket._search_query("Will the USA bomb Iran?").split()

    def test_caps_length(self):
        q = polymarket._search_query(
            "Ukraine Russia Kerch bridge Crimea Kharkiv Odesa Donetsk Luhansk Mariupol"
        )
        assert len(q.split()) == 6

    def test_deduplicates_tokens(self):
        q = polymarket._search_query("Iran Iran Iran strike")
        assert q.split() == ["iran", "strike"]

    def test_strips_parentheticals_and_punctuation(self):
        q = polymarket._search_query("Israel (IDF) strikes Iran!")
        assert "idf" not in q
        assert q.split()[0] == "israel"

    def test_empty_for_pure_stopwords(self):
        assert polymarket._search_query("will the of and") == ""


# ── _lookup_by_keywords ───────────────────────────────────────────────────

# Real Gamma /markets?search= response, captured live 2026-07-28 — returned
# identically for "bitcoin", "bitcoin price 2026" and a full sentence.
_GTA_VI_GARBAGE = (
    "New Rihanna Album before GTA VI?",
    "New Playboi Carti Album before GTA VI?",
    "Will Jesus Christ return before GTA VI?",
    "Trump out as President before GTA VI?",
    "Will China invades Taiwan before GTA VI?",
)


class TestLookupByKeywords:
    async def test_queries_public_search_not_markets(self, monkeypatch):
        """The #339 regression itself: /markets?search= ignores the query, so
        hitting it at all means the fallback can never match on keywords."""
        fake = _FakeAsyncClient({})
        _patch_client(monkeypatch, fake)

        await polymarket._lookup_by_keywords(["Trump reelection 2028"], "Trump 2028")

        assert fake.paths_seen, "no request was made"
        for path in fake.paths_seen:
            assert path.endswith("/public-search"), path
            assert "/markets" not in path

    async def test_sends_the_reduced_query(self, monkeypatch):
        fake = _FakeAsyncClient({})
        _patch_client(monkeypatch, fake)

        await polymarket._lookup_by_keywords(
            ["Will the USA officially bomb Iran in 2026?"], "usa iran"
        )

        # Reduced, entity-first, "usa" normalized — not the raw sentence.
        # Proper nouns lead ("us", "iran"), then remaining content words.
        assert fake.queries_seen[0] == "us iran bomb 2026"

    async def test_returns_none_rather_than_an_unrelated_top_hit(self, monkeypatch):
        raw = "Next world record in sprint will be broken next year"
        fake = _FakeAsyncClient({
            polymarket._search_query(raw): _events_payload(*_GTA_VI_GARBAGE),
        })
        _patch_client(monkeypatch, fake)

        assert await polymarket._lookup_by_keywords([raw], raw) is None

    async def test_picks_the_relevant_candidate_over_a_higher_ranked_irrelevant_one(
        self, monkeypatch
    ):
        raw = "sprint world record"
        fake = _FakeAsyncClient({
            polymarket._search_query(raw): _events_payload(
                *_GTA_VI_GARBAGE, "New 100m sprint world record set in 2026?"
            ),
        })
        _patch_client(monkeypatch, fake)

        result = await polymarket._lookup_by_keywords([raw], raw)

        assert result == {"question": "New 100m sprint world record set in 2026?"}

    async def test_flattens_markets_across_multiple_events(self, monkeypatch):
        """The match may live in the second event, not the first."""
        raw = "sprint world record"
        fake = _FakeAsyncClient({
            polymarket._search_query(raw): {
                "events": [
                    {"markets": [{"question": "New Rihanna Album before GTA VI?"}]},
                    {"markets": [{"question": "Sprint world record broken in 2026?"}]},
                ]
            },
        })
        _patch_client(monkeypatch, fake)

        result = await polymarket._lookup_by_keywords([raw], raw)

        assert result == {"question": "Sprint world record broken in 2026?"}

    async def test_retries_broader_when_the_reduced_query_returns_nothing(self, monkeypatch):
        raw = "Israel Iran direct military confrontation 2026"
        reduced = polymarket._search_query(raw)
        broader = " ".join(reduced.split()[:2])
        fake = _FakeAsyncClient({
            # reduced -> empty; broader -> a hit
            broader: _events_payload("Direct Israel-Iran military confrontation in 2026?"),
        })
        _patch_client(monkeypatch, fake)

        result = await polymarket._lookup_by_keywords([raw], raw)

        assert result == {"question": "Direct Israel-Iran military confrontation in 2026?"}
        assert fake.queries_seen[:2] == [reduced, broader]

    async def test_falls_through_to_the_next_phrasing(self, monkeypatch):
        fake = _FakeAsyncClient({
            polymarket._search_query("bad phrasing"): _events_payload(*_GTA_VI_GARBAGE),
            polymarket._search_query("sprint world record"): _events_payload(
                "Sprint world record broken in 2026?"
            ),
        })
        _patch_client(monkeypatch, fake)

        result = await polymarket._lookup_by_keywords(["bad phrasing"], "sprint world record")

        assert result == {"question": "Sprint world record broken in 2026?"}

    async def test_still_matches_a_well_formed_curated_event(self, monkeypatch):
        """Guards against the search change regressing the harvest pipeline's
        existing curated event_name/search_keywords matches."""
        raw = "Trump reelection 2028"
        fake = _FakeAsyncClient({
            polymarket._search_query(raw): _events_payload("Will Trump run for President in 2028?"),
        })
        _patch_client(monkeypatch, fake)

        result = await polymarket._lookup_by_keywords([raw], raw)

        assert result == {"question": "Will Trump run for President in 2028?"}

    async def test_keeps_closed_markets(self, monkeypatch):
        """This is a historical fetcher — a settled market is usually the target,
        unlike daatan's suggest() which filters closed ones out."""
        raw = "Israel strikes Iran"
        fake = _FakeAsyncClient({
            polymarket._search_query(raw): {
                "events": [{"markets": [
                    {"question": "Israel strikes Iran by January 31, 2026?", "closed": True},
                ]}]
            },
        })
        _patch_client(monkeypatch, fake)

        result = await polymarket._lookup_by_keywords([raw], raw)

        assert result is not None
        assert result["closed"] is True

    async def test_survives_a_malformed_payload(self, monkeypatch):
        fake = _FakeAsyncClient({polymarket._search_query("anything here"): ["not", "a", "dict"]})
        _patch_client(monkeypatch, fake)

        assert await polymarket._lookup_by_keywords(["anything here"], "anything here") is None
