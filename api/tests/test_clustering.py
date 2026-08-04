"""Correlated-evidence clustering (retro#355).

The load-bearing test in this file is ``test_ships_inert`` — everything else can be
retuned, but a "refactor" that quietly moved a live estimate would be the worst
possible outcome, so the no-op property is pinned directly rather than assumed.
"""

import pytest

from forecast_api.aggregation import aggregate_pool, cluster_downweight_factors
from forecast_api.clustering import (
    cluster_text_for_claims,
    cluster_texts,
    jaccard,
    shingles,
)

POOL_KWARGS = dict(
    relevance_weight_floor=0.05,
    decisiveness_floor=0.0,
    thin_evidence_ci_inflation=1.0,
    defer_on_thin_evidence=False,
    settlement_min_sources=2,
    settlement_stance=0.94,
    logit_clamp=0.02,
    pool_dispersion_floor=0.0,
)


class TestShingles:
    def test_word_order_matters(self):
        # Same unigrams, opposite story. A bag of words would score these 1.0.
        a = shingles("russia strikes ukraine today", 3)
        b = shingles("ukraine strikes russia today", 3)
        assert jaccard(a, b) < 1.0

    def test_short_text_degrades_to_one_whole_shingle(self):
        # Not empty: a two-word claim is still a claim, and an empty set would
        # silently make the row unclusterable.
        assert shingles("kremlin mobilizes", 3) == frozenset({("kremlin", "mobilizes")})

    def test_punctuation_and_case_are_noise_digits_are_not(self):
        assert shingles("Kremlin Mobilizes, 300000!", 3) == shingles("kremlin mobilizes 300000", 3)
        assert shingles("300000 troops called", 3) != shingles("500000 troops called", 3)

    def test_empty_text_yields_nothing(self):
        assert shingles("", 3) == frozenset()
        assert shingles("!!! ,,, ---", 3) == frozenset()


class TestJaccard:
    def test_empty_pair_scores_zero_not_one(self):
        # Two text-less rows are UNKNOWN, not identical. Scoring 1.0 would collapse
        # every legacy row into one giant pseudo-story and downweight the lot.
        assert jaccard(frozenset(), frozenset()) == 0.0
        assert jaccard(shingles("a b c d", 3), frozenset()) == 0.0

    def test_identical_sets_score_one(self):
        s = shingles("the kremlin is considering a new wave", 3)
        assert jaccard(s, s) == 1.0


class TestClusterTexts:
    def test_echoes_of_one_development_group(self):
        texts = [
            "sources say the kremlin is considering a new mobilization wave",
            "the kremlin is considering a new mobilization wave sources say",
            "unrelated: the central bank held rates steady this quarter",
        ]
        ids = cluster_texts(texts, threshold=0.5)
        assert ids[0] == ids[1]
        assert ids[2] != ids[0]

    def test_single_linkage_is_transitive(self):
        # A~B and B~C, but A and C alone would miss the bar. All three cover one
        # development, so all three must land together.
        a = "the kremlin is considering a new mobilization wave in autumn"
        b = "the kremlin is considering a new mobilization wave"
        c = "kremlin considering a new mobilization wave, officials said"
        ids = cluster_texts([a, b, c], threshold=0.3)
        assert ids[0] == ids[1] == ids[2]

    def test_textless_rows_are_always_singletons(self):
        # Conservative: an unclusterable row keeps its full weight, so missing text
        # can never COST a source its vote.
        ids = cluster_texts([None, None, "a totally distinct story about rates"], threshold=0.5)
        assert len(set(ids)) == 3

    def test_ids_are_first_appearance_ordered_and_deterministic(self):
        texts = ["alpha beta gamma delta", "alpha beta gamma delta", "zeta eta theta iota"]
        assert cluster_texts(texts, threshold=0.5) == (0, 0, 1)
        # The replay harness depends on this being a pure function of the input.
        for _ in range(5):
            assert cluster_texts(texts, threshold=0.5) == (0, 0, 1)

    def test_fewer_than_two_rows_is_trivially_one_cluster(self):
        assert cluster_texts(["only one"], threshold=0.5) == (0,)
        assert cluster_texts([], threshold=0.5) == ()


class TestClusterTextForClaims:
    def test_uses_claim_and_quote(self):
        out = cluster_text_for_claims([{"claim": "A happened.", "quote": "He said A."}])
        assert "A happened." in out and "He said A." in out

    def test_falls_back_to_title_for_legacy_rows(self):
        # Rows written before claims_detail existed still cluster, on thinner text.
        assert cluster_text_for_claims(None, "A Headline") == "A Headline"
        assert cluster_text_for_claims([], "A Headline") == "A Headline"

    def test_no_caller_can_supply_a_title_today(self):
        # Pins the documented coverage bound: claims_detail is the ONLY cluster text
        # either weight site can supply, so rows without it are unclusterable and
        # event=evidence_clusters under-reports real echo (read counts as a lower bound).
        # If this fails someone added a title — pass it as `fallback` at BOTH call sites
        # and update ORACLE_VARIABLES.md, which states this bound explicitly.
        from forecast_api.models import PoolSourceInput, SourceSignal

        assert "title" not in SourceSignal.model_fields
        assert "title" not in PoolSourceInput.model_fields

    def test_reads_objects_and_dicts_identically(self):
        # /forecast passes ClaimDetail objects, /pool/aggregate passes parsed models —
        # both weight sites MUST derive the same text.
        class C:
            claim, quote = "A happened.", "He said A."

        assert cluster_text_for_claims([C()]) == cluster_text_for_claims(
            [{"claim": "A happened.", "quote": "He said A."}]
        )


class TestDownweightFactors:
    def test_zero_exponent_is_the_identity(self):
        assert cluster_downweight_factors([0, 0, 1], 0.0) is None

    def test_no_cluster_ids_is_the_identity(self):
        assert cluster_downweight_factors(None, 0.5) is None
        assert cluster_downweight_factors([], 0.5) is None

    def test_sqrt_discount_halves_a_cluster_of_four(self):
        f = cluster_downweight_factors([0, 0, 0, 0], 0.5)
        assert f == pytest.approx([0.5] * 4)
        # The cluster now carries sqrt(4) = 2 rows' worth, not 4.
        assert sum(f) == pytest.approx(2.0)

    def test_singletons_are_never_discounted(self):
        f = cluster_downweight_factors([0, 1, 2], 0.5)
        assert f == pytest.approx([1.0, 1.0, 1.0])

    def test_exponent_one_collapses_a_cluster_to_one_row(self):
        f = cluster_downweight_factors([0, 0, 0, 1], 1.0)
        assert sum(f[:3]) == pytest.approx(1.0)
        assert f[3] == pytest.approx(1.0)


class TestShipsInert:
    """The property that makes this PR safe to merge into a repo that deploys on merge."""

    @pytest.mark.parametrize("stances,weights,relevances,settled", [
        ([0.5, 0.5, 0.5], [1.0, 1.0, 1.0], [0.7, 0.7, 0.7], [False, False, False]),
        ([0.9, -0.3, 0.1, 0.4], [0.8, 0.2, 0.5, 0.9], [0.8, 0.7, 0.7, 0.6], [False] * 4),
        ([-0.9, -0.8], [1.2, 0.4], [0.9, 0.7], [True, True]),
    ])
    def test_default_exponent_moves_no_number_even_when_every_row_is_one_cluster(
        self, stances, weights, relevances, settled,
    ):
        baseline = aggregate_pool(stances, weights, relevances, settled, **POOL_KWARGS)
        clustered = aggregate_pool(
            stances, weights, relevances, settled,
            cluster_ids=tuple([0] * len(stances)),  # worst case: all one echo
            **POOL_KWARGS,
        )
        assert clustered == baseline

    def test_cluster_ids_without_an_exponent_are_inert(self):
        base = aggregate_pool([0.5, 0.5], [1.0, 1.0], [0.7, 0.7], [False, False], **POOL_KWARGS)
        same = aggregate_pool(
            [0.5, 0.5], [1.0, 1.0], [0.7, 0.7], [False, False],
            cluster_ids=(0, 0), cluster_downweight_exponent=0.0, **POOL_KWARGS,
        )
        assert same == base

    def test_a_mismatched_cluster_id_length_is_ignored_rather_than_crashing(self):
        # Defensive: a caller bug must degrade to today's behaviour, not 500 a forecast.
        base = aggregate_pool([0.5, 0.5], [1.0, 1.0], [0.7, 0.7], [False, False], **POOL_KWARGS)
        odd = aggregate_pool(
            [0.5, 0.5], [1.0, 1.0], [0.7, 0.7], [False, False],
            cluster_ids=(0,), cluster_downweight_exponent=0.5, **POOL_KWARGS,
        )
        assert odd == base


class TestDiscountActuallyBites:
    """The seam is inert, but it must be *correct* when switched on — otherwise
    enabling it later is a fresh, unreviewed change."""

    def test_echo_no_longer_outweighs_an_independent_dissenter(self):
        # Four echoes of one YES story vs one independent NO of equal per-row weight.
        stances = [0.8, 0.8, 0.8, 0.8, -0.8]
        weights = [1.0] * 5
        relevances = [0.7] * 5
        settled = [False] * 5
        undiscounted = aggregate_pool(stances, weights, relevances, settled, **POOL_KWARGS)
        discounted = aggregate_pool(
            stances, weights, relevances, settled,
            cluster_ids=(0, 0, 0, 0, 1), cluster_downweight_exponent=0.5, **POOL_KWARGS,
        )
        # 4 echoes carry sqrt(4)=2 rows' worth, so the mean moves toward the dissenter.
        assert discounted.mean < undiscounted.mean

    def test_evidence_mass_shrinks_so_thin_evidence_is_judged_on_real_independence(self):
        # The reason the discount is applied FIRST: a pool of echoes must not pass the
        # decisiveness floor on inflated mass.
        kwargs = {**POOL_KWARGS, "decisiveness_floor": 3.0, "defer_on_thin_evidence": True}
        stances, weights = [0.5] * 4, [1.0] * 4
        relevances, settled = [0.7] * 4, [False] * 4
        assert aggregate_pool(stances, weights, relevances, settled, **kwargs).insufficient_reason is None
        echoed = aggregate_pool(
            stances, weights, relevances, settled,
            cluster_ids=(0, 0, 0, 0), cluster_downweight_exponent=0.5, **kwargs,
        )
        assert echoed.insufficient_reason == "no_decisive_signal"
