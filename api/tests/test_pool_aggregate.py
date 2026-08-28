"""Tests for run_pool_aggregate() / POST /pool/aggregate (retro
docs/ORACLE_VARIABLES.md, recompute-over-pool). No search, no LLM — given a
caller-supplied set of already-extracted per-source signals, it must
reproduce exactly what a fresh /forecast run's pooling math would produce
for the same evidence (see aggregate_pool() in aggregation.py, shared by
both). Tested against the async function directly, matching this suite's
convention for authed business logic (see test_evidence_class.py /
test_settlement_hardening.py) rather than through TestClient.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pytest

from forecast_api import forecaster
from forecast_api.config import settings as api_settings
from forecast_api.models import PROVENANCE_SCHEMA_VERSION, PoolAggregateRequest, PoolSourceInput


def _prob(stance: float) -> float:
    return (stance + 1.0) / 2.0


# Relative, not hard-coded: see the note in test_settlement.py.
_FRESH = (date.today() - timedelta(days=1)).isoformat()


def _source(**over) -> PoolSourceInput:
    return PoolSourceInput(**{
        "stance": 0.5,
        "certainty": 0.6,
        "credibility_weight": 1.0,
        "relevance_score": 1.0,
        "evidence_weight": 0.6,
        "published_date": _FRESH,
        "settled": False,
        **over,
    })


class TestEmptyAndInsufficient:
    async def test_no_sources_is_insufficient(self):
        resp = await forecaster.run_pool_aggregate(PoolAggregateRequest(sources=[]))
        assert resp.insufficient_data is True
        assert resp.reason == "no_sources"
        assert resp.articles_used == 0

    async def test_all_off_topic_is_insufficient(self):
        resp = await forecaster.run_pool_aggregate(PoolAggregateRequest(
            sources=[_source(relevance_score=0.05), _source(relevance_score=0.05)],
        ))
        assert resp.insufficient_data is True
        assert resp.reason == "all_articles_off_topic"


class TestZeroWeightPool:
    async def test_all_zero_weight_sources_abstain(self):
        """R3/F14 through the endpoint: every row blocked by credibility, so
        nothing carries weight. The recompute must abstain rather than answer
        from the rows it just valued at zero."""
        resp = await forecaster.run_pool_aggregate(PoolAggregateRequest(
            sources=[
                _source(stance=0.9, credibility_weight=0.0),
                _source(stance=0.7, credibility_weight=0.0),
            ],
        ))
        assert resp.insufficient_data is True
        assert resp.reason == "no_usable_weight"
        assert resp.articles_used == 2

    async def test_legacy_row_without_evidence_weight_is_capped(self):
        """R3/F10 on the recompute path: a pre-S2-cutover row with no stored
        evidence_weight falls back to certainty under the same cap the live path
        applies to an unclassified claim, so the rows we know least about cannot
        weigh most. certainty 0.9 → 0.25, so this pool lands below the
        decisiveness floor and self-reports with a widened CI."""
        resp = await forecaster.run_pool_aggregate(PoolAggregateRequest(
            sources=[_source(stance=0.6, certainty=0.9, evidence_weight=None)],
        ))
        assert resp.insufficient_data is False
        assert (resp.ci_high - resp.ci_low) > 0.5


class TestBasicPooling:
    async def test_pools_a_single_source_toward_its_own_stance(self):
        resp = await forecaster.run_pool_aggregate(PoolAggregateRequest(
            sources=[_source(stance=0.8, evidence_weight=1.0, published_date=None)],
        ))
        assert resp.insufficient_data is False
        assert resp.articles_used == 1
        assert _prob(resp.mean) > 0.85  # near-boundary evidence pools decisively

    async def test_evidence_weight_none_falls_back_to_certainty(self):
        with_weight = await forecaster.run_pool_aggregate(PoolAggregateRequest(
            sources=[_source(evidence_weight=0.9, certainty=0.9, published_date=None)],
        ))
        # Same numeric value via the certainty fallback (evidence_weight unset).
        via_fallback = await forecaster.run_pool_aggregate(PoolAggregateRequest(
            sources=[_source(evidence_weight=None, certainty=0.9, published_date=None)],
        ))
        assert with_weight.mean == pytest.approx(via_fallback.mean)

    async def test_relevance_squared_down_weights_a_tangential_source(self):
        # Two sources disagree; the low-relevance one should barely move the
        # pool off the high-relevance one's stance.
        resp = await forecaster.run_pool_aggregate(PoolAggregateRequest(
            sources=[
                _source(stance=0.9, relevance_score=1.0, evidence_weight=1.0, published_date=None),
                _source(stance=-0.9, relevance_score=0.2, evidence_weight=1.0, published_date=None),
            ],
        ))
        assert resp.mean > 0.5  # dominated by the on-topic source


class TestRecency:
    async def test_older_article_pulls_less_than_a_fresh_one(self):
        # Two disagreeing sources, one dated far in the past (relative to
        # "now", recomputed fresh by run_pool_aggregate — not a stored value).
        resp = await forecaster.run_pool_aggregate(PoolAggregateRequest(
            sources=[
                _source(stance=0.9, published_date="2026-01-01", evidence_weight=1.0),
                _source(stance=-0.9, published_date=None, evidence_weight=1.0),  # no date -> neutral weight
            ],
        ))
        # The undated (neutral-weight) dissenter outweighs the stale one.
        assert resp.mean < 0.5


class TestSettlement:
    async def test_settlement_override_pins_through_the_endpoint(self):
        # Dated anchors: the endpoint runs with settlement_revalidate on (the
        # prod default), so an undated positive vote no longer counts — see
        # test_settlement_revalidation.py for the demotion cases.
        resp = await forecaster.run_pool_aggregate(PoolAggregateRequest(
            sources=[
                _source(stance=0.95, settled=True, settlement_event_date="2026-07-09"),
                _source(stance=0.9, settled=True, settlement_event_date="2026-07-09"),
            ],
        ))
        assert resp.settled is True
        assert resp.mean == pytest.approx(api_settings.settlement_stance)

    async def test_undated_positive_votes_are_demoted_through_the_endpoint(self):
        resp = await forecaster.run_pool_aggregate(PoolAggregateRequest(
            sources=[
                _source(stance=0.95, settled=True, published_date=None),
                _source(stance=0.9, settled=True, published_date=None),
            ],
        ))
        assert resp.settled is False
        assert resp.settlement_votes_demoted == 2

    async def test_early_undated_negative_settlement_is_still_suppressed(self):
        # Same observable as the old pin-level direction guard, new mechanism:
        # each undated pre-deadline negative is demoted per-vote
        # (undated_foreclosure) instead of the pin being blocked wholesale.
        resp = await forecaster.run_pool_aggregate(PoolAggregateRequest(
            sources=[
                _source(stance=-0.95, settled=True, published_date=None),
                _source(stance=-0.9, settled=True, published_date=None),
            ],
            claim_direction="arrival",
            claim_deadline="2099-01-01",
        ))
        assert resp.settled is False


class TestDemotionAuditLog:
    """retro#554: every ``settlement_vote_demoted`` line must identify its
    forecast (question hash) and the claim-window bounds the rule actually
    compared against — without them a demotion cannot be audited for false
    positives — and must separate a genuinely undated article from one whose
    date string failed to parse (``event_date_state``)."""

    @staticmethod
    def _demotion_lines(caplog):
        return [
            r.message for r in caplog.records
            if "event=settlement_vote_demoted" in r.message
        ]

    async def test_demotion_line_carries_question_hash_and_window_bounds(self, caplog):
        caplog.set_level(logging.WARNING, logger=forecaster.logger.name)
        question = "Will the treaty be signed by end of 2026?"
        resp = await forecaster.run_pool_aggregate(PoolAggregateRequest(
            sources=[_source(stance=0.95, settled=True, settlement_event_date="2020-01-01")],
            question=question,
            claim_created_at="2026-06-01",
            claim_deadline="2026-12-31",
        ))
        assert resp.settlement_votes_demoted == 1
        (line,) = self._demotion_lines(caplog)
        assert "reason=event_before_claim_window" in line
        assert f"question={forecaster._question_hash(question)}" in line
        assert "created=2026-06-01" in line
        assert "deadline=2026-12-31" in line
        assert "event_date=2020-01-01" in line
        assert "event_date_state=parsed" in line

    async def test_absent_date_is_distinguished_from_a_parse_failure(self, caplog):
        # Both rows demote as missing_event_date (the reason string stays
        # stable for downstream grep), but the new field tells them apart:
        # no date at all vs. a date string that failed ISO parsing.
        caplog.set_level(logging.WARNING, logger=forecaster.logger.name)
        await forecaster.run_pool_aggregate(PoolAggregateRequest(
            sources=[
                _source(stance=0.95, settled=True),
                _source(stance=0.9, settled=True, settlement_event_date="mid-July 2026"),
            ],
        ))
        absent, unparseable = self._demotion_lines(caplog)
        assert "reason=missing_event_date" in absent
        assert "event_date_state=absent" in absent
        assert "reason=missing_event_date" in unparseable
        assert "event_date_state=unparseable" in unparseable
        assert "event_date=mid-July 2026" in unparseable

    async def test_undated_foreclosure_carries_the_state_field_too(self, caplog):
        caplog.set_level(logging.WARNING, logger=forecaster.logger.name)
        await forecaster.run_pool_aggregate(PoolAggregateRequest(
            sources=[_source(stance=-0.95, settled=True)],
            claim_direction="arrival",
            claim_deadline="2099-01-01",
        ))
        (line,) = self._demotion_lines(caplog)
        assert "reason=undated_foreclosure" in line
        assert "event_date_state=absent" in line
        assert "deadline=2099-01-01" in line


class TestSourceMassCap:
    """cap_source_mass() (retro#458 Phase 1) through the full recompute path
    — run_pool_aggregate() reads PoolSourceInput.source_id (previously
    persisted-and-ignored, see the whitelist comment in run_pool_aggregate)
    purely as a cap_source_mass grouping key. Ships inert at the default
    max_source_share=1.0."""

    async def test_default_ships_inert_even_with_a_dominant_source_id(self):
        sources = [
            _source(stance=0.9, source_id="dominant", evidence_weight=1.0),
            _source(stance=0.9, source_id="dominant", evidence_weight=1.0),
            _source(stance=0.9, source_id="dominant", evidence_weight=1.0),
            _source(stance=0.9, source_id="dominant", evidence_weight=1.0),
            _source(stance=-0.9, source_id="other", evidence_weight=1.0),
        ]
        with_ids = await forecaster.run_pool_aggregate(PoolAggregateRequest(sources=sources))
        without_ids = await forecaster.run_pool_aggregate(PoolAggregateRequest(
            sources=[s.model_copy(update={"source_id": None}) for s in sources],
        ))
        assert with_ids.mean == pytest.approx(without_ids.mean)

    async def test_dominant_source_id_is_capped_once_configured(self, monkeypatch):
        # Same shape as the prod finding (one aggregator dominating the
        # pool): 4 agreeing rows from one source_id vs 1 independent
        # dissenter, equal per-row weight.
        sources = [
            _source(stance=0.9, source_id="dominant", evidence_weight=1.0),
            _source(stance=0.9, source_id="dominant", evidence_weight=1.0),
            _source(stance=0.9, source_id="dominant", evidence_weight=1.0),
            _source(stance=0.9, source_id="dominant", evidence_weight=1.0),
            _source(stance=-0.9, source_id="other", evidence_weight=1.0),
        ]
        uncapped = await forecaster.run_pool_aggregate(PoolAggregateRequest(sources=sources))
        monkeypatch.setattr(api_settings, "max_source_share", 0.3)
        capped = await forecaster.run_pool_aggregate(PoolAggregateRequest(sources=sources))
        assert capped.mean < uncapped.mean

    async def test_rows_without_a_source_id_are_never_capped_together(self, monkeypatch):
        # Legacy/anonymous rows (source_id unset) must not be treated as one
        # dominant "unknown" source just because none of them carry an id.
        monkeypatch.setattr(api_settings, "max_source_share", 0.3)
        sources = [_source(stance=0.9, evidence_weight=1.0) for _ in range(5)]
        resp = await forecaster.run_pool_aggregate(PoolAggregateRequest(sources=sources))
        assert resp.insufficient_data is False
        assert _prob(resp.mean) > 0.85  # unaffected — pools as decisively as with no cap at all


class TestPoolReportingFields:
    """evidence_mass/n_eff/age_adjusted_mass on PoolAggregateResponse
    (retro#458 Phase 2) — reporting-only visibility, not new estimator
    behaviour: mean/std/ci are unaffected by any of this class."""

    async def test_evidence_mass_and_n_eff_populated_on_a_normal_pool(self):
        resp = await forecaster.run_pool_aggregate(PoolAggregateRequest(
            sources=[
                _source(stance=0.6, evidence_weight=1.0),
                _source(stance=0.5, evidence_weight=1.0),
                _source(stance=0.4, evidence_weight=1.0),
            ],
        ))
        assert resp.insufficient_data is False
        assert resp.evidence_mass > 0.0
        # Equal-weight rows: n_eff == articles_used (Kish's ESS is exact here).
        assert resp.n_eff == pytest.approx(resp.articles_used, abs=1e-6)

    async def test_n_eff_shrinks_when_one_row_dominates(self):
        dominated = await forecaster.run_pool_aggregate(PoolAggregateRequest(
            sources=[
                _source(stance=0.6, evidence_weight=1.0),
                _source(stance=0.5, evidence_weight=0.001),
                _source(stance=0.4, evidence_weight=0.001),
            ],
        ))
        assert dominated.insufficient_data is False
        assert dominated.articles_used == 3
        assert dominated.n_eff < 1.5  # near-1, one row carries almost all the mass

    async def test_evidence_mass_and_n_eff_still_reported_on_an_insufficient_pool(self):
        # F14/no_usable_weight: the pool that abstains still had a shape (see
        # aggregation.PoolAggregateResult's docstring) — evidence_mass reads
        # 0.0 (every row was blocked), but n_eff is still Kish's ESS of that
        # same zero-weight vector, not silently dropped.
        resp = await forecaster.run_pool_aggregate(PoolAggregateRequest(
            sources=[
                _source(stance=0.9, credibility_weight=0.0),
                _source(stance=0.7, credibility_weight=0.0),
            ],
        ))
        assert resp.insufficient_data is True
        assert resp.reason == "no_usable_weight"
        assert resp.evidence_mass == 0.0

    async def test_age_adjusted_mass_equals_evidence_mass_with_no_decay_in_play(self):
        # published today: recency_weight's age gap is exactly zero, so
        # switching off decay changes nothing for this pool.
        today = date.today().isoformat()
        resp = await forecaster.run_pool_aggregate(PoolAggregateRequest(
            sources=[
                _source(stance=0.6, evidence_weight=1.0, published_date=today),
                _source(stance=0.4, evidence_weight=1.0, published_date=today),
            ],
        ))
        assert resp.insufficient_data is False
        assert resp.age_adjusted_mass == pytest.approx(resp.evidence_mass, rel=1e-6)

    async def test_age_adjusted_mass_exceeds_evidence_mass_for_a_stale_pool(self):
        # The sanity invariant this phase adds: removing recency decay can
        # only ever raise a pool's mass relative to the recency-discounted
        # sum, never lower it. Exercised with a real decay gap (well past one
        # recency half-life), not a same-day pool where the two trivially tie.
        stale = (date.today() - timedelta(days=250)).isoformat()
        resp = await forecaster.run_pool_aggregate(PoolAggregateRequest(
            sources=[
                _source(stance=0.6, evidence_weight=1.0, published_date=stale),
                _source(stance=0.4, evidence_weight=1.0, published_date=stale),
            ],
        ))
        assert resp.insufficient_data is False
        assert resp.age_adjusted_mass >= resp.evidence_mass
        assert resp.age_adjusted_mass > resp.evidence_mass  # decay actually bit here


class TestSyndicationDedup:
    """dedupe_syndicated() now runs inside run_pool_aggregate() too (retro#458
    Phase 3) — previously it only ran on the /forecast path
    (_run_forecast_inner), so a caller-persisted pool with the same wire story
    re-hosted under two outlets had zero dedup coverage and could triple its
    weight. `title_of` here is the claims_detail-derived cluster text (the
    same derivation the clustering call below already uses) since
    PoolSourceInput carries no raw title field."""

    _CLAIM = "Central bank raises interest rate to record high"
    _QUOTE = "the rate hike marks the largest increase in over a decade"

    async def test_near_duplicate_titles_collapse_to_one_row(self):
        # Same story, two different outlets/URLs, near-identical claims_detail
        # text (as a re-print of the same wire copy would extract) — the
        # exact shape run_pool_aggregate previously had zero coverage for.
        low = _source(
            stance=0.9, credibility_weight=0.3, evidence_weight=1.0,
            url="https://aggregator-a.example.test/story",
            outlet="Aggregator A",
            claims_detail=[{"claim": self._CLAIM, "quote": self._QUOTE, "stance": 0.9, "certainty": 0.8}],
        )
        high = _source(
            stance=0.9, credibility_weight=1.4, evidence_weight=1.0,
            url="https://aggregator-b.example.test/story-copy",
            outlet="Aggregator B",
            claims_detail=[{"claim": self._CLAIM, "quote": self._QUOTE, "stance": 0.9, "certainty": 0.8}],
        )
        collapsed = await forecaster.run_pool_aggregate(PoolAggregateRequest(sources=[low, high]))
        solo_high = await forecaster.run_pool_aggregate(PoolAggregateRequest(sources=[high]))
        assert collapsed.articles_used == 1
        # The higher-credibility duplicate is the one that survives — same
        # result as if the low-credibility re-print had never been sent.
        assert collapsed.mean == pytest.approx(solo_high.mean)
        assert collapsed.evidence_mass == pytest.approx(solo_high.evidence_mass)

    async def test_exact_duplicate_url_collapses_even_without_claims_detail(self):
        dup_url = "https://aggregator.example.test/story"
        a = _source(stance=0.9, credibility_weight=0.3, evidence_weight=1.0, url=dup_url)
        b = _source(stance=0.9, credibility_weight=1.4, evidence_weight=1.0, url=dup_url)
        resp = await forecaster.run_pool_aggregate(PoolAggregateRequest(sources=[a, b]))
        assert resp.articles_used == 1

    async def test_distinct_stories_are_not_merged(self):
        # Two genuinely different stories (different claims_detail text, no
        # URL overlap) must both survive — dedup must not be over-eager.
        a = _source(
            stance=0.9, credibility_weight=1.0, evidence_weight=1.0,
            url="https://a.example.test/story-one",
            claims_detail=[{"claim": "Company reports record quarterly revenue growth", "quote": "revenue climbed across every segment this quarter", "stance": 0.9, "certainty": 0.8}],
        )
        b = _source(
            stance=-0.5, credibility_weight=1.0, evidence_weight=1.0,
            url="https://b.example.test/story-two",
            claims_detail=[{"claim": "Regulator opens investigation into unrelated merger deal", "quote": "the inquiry focuses on antitrust concerns raised last month", "stance": -0.5, "certainty": 0.8}],
        )
        resp = await forecaster.run_pool_aggregate(PoolAggregateRequest(sources=[a, b]))
        assert resp.articles_used == 2


class TestAntecedentPoolSplit:
    """retro#573 Option 1 ("pool-split") — filter an already-extracted pool
    by the antecedent being asked about, before the weight loop runs. No new
    search, no new prompt: every field read here has been live on
    ClaimDetail since PR#570."""

    # Deliberately unrelated clauses, not a minimal pair — two antecedents that
    # merely swap one subject noun ("coalition wins" / "opposition wins") share
    # enough bigram structure ("wins a majority") to jaccard-match at this
    # module's threshold, which would be testing the matcher's blind spot
    # rather than the pool-split behavior this test targets.
    _COALITION = "the ruling coalition secures a parliamentary majority"
    _OPPOSITION = "an independent inquiry finds evidence of electoral fraud"

    @staticmethod
    def _conditional_source(stance: float, claim_text: str, antecedent_text_en: str, **over) -> PoolSourceInput:
        return _source(
            stance=stance,
            claims_detail=[{
                "claim": claim_text, "stance": stance, "certainty": 0.7,
                "is_conditional": True, "antecedent_text_en": antecedent_text_en,
                "antecedent_polarity": True,
            }],
            **over,
        )

    @staticmethod
    def _unconditional_source(stance: float, claim_text: str, **over) -> PoolSourceInput:
        return _source(
            stance=stance,
            claims_detail=[{"claim": claim_text, "stance": stance, "certainty": 0.7, "is_conditional": False}],
            **over,
        )

    def _mixed_pool(self) -> list:
        # Claim text deliberately distinct per source (different specific facts) so the
        # syndication dedup step above this filter — clustering on claims_detail text —
        # does not collapse same-antecedent rows into one representative before this
        # test can observe them; only antecedent_text_en repeats within a group.
        return [
            self._conditional_source(0.85, "Poll shows coalition bloc ahead by five points", self._COALITION),
            self._conditional_source(0.75, "Coalition lead widens after debate performance", self._COALITION),
            self._conditional_source(0.80, "Analyst model favors coalition after redistricting", self._COALITION),
            self._conditional_source(-0.60, "Opposition surges in latest urban district poll", self._OPPOSITION),
            self._conditional_source(-0.65, "Opposition candidate gains major union endorsement", self._OPPOSITION),
            self._conditional_source(-0.55, "Post-debate tracking poll favors opposition bloc", self._OPPOSITION),
            self._unconditional_source(0.10, "Electoral commission confirms official election date"),
            self._unconditional_source(0.05, "Officials project record turnout this cycle"),
        ]

    async def test_unconditional_query_is_unaffected_by_conditional_fields(self):
        """(a) A caller who never sets antecedent_query gets the same result
        it would have gotten before this field existed — conditional fields
        present on the rows must not leak into the estimate when nothing
        asks for them."""
        pool = self._mixed_pool()
        with_conditional_fields = await forecaster.run_pool_aggregate(PoolAggregateRequest(sources=pool))
        stripped = [s.model_copy(update={"claims_detail": None}) for s in pool]
        without_conditional_fields = await forecaster.run_pool_aggregate(PoolAggregateRequest(sources=stripped))
        assert with_conditional_fields.mean == pytest.approx(without_conditional_fields.mean)
        assert with_conditional_fields.std == pytest.approx(without_conditional_fields.std)
        assert with_conditional_fields.articles_used == without_conditional_fields.articles_used

    async def test_different_antecedents_return_materially_different_estimates(self):
        """(b) The same consequent pool, conditioned on two different
        antecedents, must not collapse to the flat-pool number — retro#573's
        core complaint was antecedent sensitivity ~= 0. Uses the issue's own
        bar: sd across antecedents for a fixed consequent > 0.15."""
        pool = self._mixed_pool()
        flat = await forecaster.run_pool_aggregate(PoolAggregateRequest(sources=pool))
        coalition = await forecaster.run_pool_aggregate(PoolAggregateRequest(
            sources=pool, antecedent_query=self._COALITION,
        ))
        opposition = await forecaster.run_pool_aggregate(PoolAggregateRequest(
            sources=pool, antecedent_query=self._OPPOSITION,
        ))
        assert coalition.insufficient_data is False
        assert opposition.insufficient_data is False
        # Each conditioned pool sees only its own antecedent's claims plus the
        # unconditional ones — the OTHER antecedent's rows must not leak in.
        assert coalition.articles_used == 5  # 3 coalition + 2 unconditional
        assert opposition.articles_used == 5  # 3 opposition + 2 unconditional
        assert coalition.mean > opposition.mean
        assert (coalition.mean - opposition.mean) > 0.3
        group_means = [coalition.mean, opposition.mean]
        mu = sum(group_means) / 2
        sd = (sum((m - mu) ** 2 for m in group_means) / 2) ** 0.5
        assert sd > 0.15
        # Neither conditioned estimate should equal the flat pool's — the flat
        # number averages across a distinction the flat pool cannot see.
        assert coalition.mean != pytest.approx(flat.mean, abs=0.05)
        assert opposition.mean != pytest.approx(flat.mean, abs=0.05)

    async def test_no_matching_antecedent_falls_back_to_insufficient_data(self):
        """(c) When nothing in the pool speaks to the asked antecedent, the
        recompute must say so rather than silently answering with the
        unfiltered pool (see antecedent_query's field docstring for why:
        answering a conditional question with an unconditional number is the
        retro#573 bug, and this must not reproduce it in a new form)."""
        pool = [
            self._conditional_source(0.85, "Poll shows coalition ahead", self._COALITION),
            self._conditional_source(-0.60, "Coalition lead evaporates in new poll", self._COALITION),
        ]
        resp = await forecaster.run_pool_aggregate(PoolAggregateRequest(
            sources=pool, antecedent_query="a meteor strikes the capital",
        ))
        assert resp.insufficient_data is True
        assert resp.reason == "no_matching_antecedent"
        assert resp.articles_used == 0

    async def test_negated_polarity_query_excludes_affirmative_claims(self):
        affirmative = self._conditional_source(0.8, "Coalition consolidates ahead of vote", self._COALITION)
        negated = _source(
            stance=-0.7,
            claims_detail=[{
                "claim": "Coalition collapse reported after defections", "stance": -0.7, "certainty": 0.7,
                "is_conditional": True, "antecedent_text_en": self._COALITION, "antecedent_polarity": False,
            }],
        )
        resp = await forecaster.run_pool_aggregate(PoolAggregateRequest(
            sources=[affirmative, negated],
            antecedent_query=self._COALITION, antecedent_query_polarity=False,
        ))
        assert resp.articles_used == 1
        assert resp.mean < 0  # only `negated` (stance -0.7) survives the filter


class TestProvenance:
    """retro#593: PoolAggregateResponse.provenance is populated on every
    return path — the normal success path, and all three insufficient_data
    early returns (no_sources, insufficient_reason, no_matching_antecedent) —
    since a caller replaying any of these numbers needs the same block."""

    async def test_normal_pool_carries_pool_method_provenance(self):
        resp = await forecaster.run_pool_aggregate(PoolAggregateRequest(
            sources=[_source(stance=0.6), _source(stance=0.4)],
        ))
        assert resp.provenance is not None
        assert resp.provenance.schema_version == PROVENANCE_SCHEMA_VERSION
        assert resp.provenance.engine == "v1"
        assert resp.provenance.method == "pool"
        assert resp.provenance.chain == []  # a recompute never searches
        # No LLM call is made recomputing over an already-extracted pool.
        assert resp.provenance.models.gatekeeper is None
        assert resp.provenance.models.extractor is None

    async def test_no_sources_still_carries_provenance(self):
        resp = await forecaster.run_pool_aggregate(PoolAggregateRequest(sources=[]))
        assert resp.insufficient_data is True and resp.reason == "no_sources"
        assert resp.provenance is not None
        assert resp.provenance.method == "pool"

    async def test_insufficient_reason_path_still_carries_provenance(self):
        resp = await forecaster.run_pool_aggregate(PoolAggregateRequest(
            sources=[_source(relevance_score=0.05), _source(relevance_score=0.05)],
        ))
        assert resp.insufficient_data is True and resp.reason == "all_articles_off_topic"
        assert resp.provenance is not None
        assert resp.provenance.method == "pool"

    async def test_no_matching_antecedent_path_still_carries_provenance(self):
        pool = [_source(
            stance=0.85,
            claims_detail=[{
                "claim": "Poll shows coalition ahead", "stance": 0.85, "certainty": 0.7,
                "is_conditional": True, "antecedent_text_en": "the coalition wins",
                "antecedent_polarity": True,
            }],
        )]
        resp = await forecaster.run_pool_aggregate(PoolAggregateRequest(
            sources=pool, antecedent_query="a meteor strikes the capital",
        ))
        assert resp.reason == "no_matching_antecedent"
        assert resp.provenance is not None
        assert resp.provenance.method == "pool"
