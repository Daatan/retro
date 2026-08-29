"""Correlated-evidence clustering (retro#355).

The load-bearing test in this file is ``test_ships_inert`` — everything else can be
retuned, but a "refactor" that quietly moved a live estimate would be the worst
possible outcome, so the no-op property is pinned directly rather than assumed.
"""

import logging

import pytest

from forecast_api.aggregation import aggregate_pool, cluster_downweight_factors
from forecast_api.clustering import (
    SIMILARITY_BANDS,
    cluster_text_for_claims,
    cluster_texts,
    cluster_texts_with_stats,
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


class TestClusterStats:
    """The near-miss record — what the pass saw BELOW the threshold.

    The load-bearing test here is ``test_reports_a_near_miss_that_produced_no_cluster``:
    without it the instrument can only ever report echo it already found, and
    ``cluster_jaccard_threshold`` cannot be tuned downward from its own logs.
    """

    def test_reports_a_near_miss_that_produced_no_cluster(self):
        # Two clearly related write-ups that do NOT clear a 0.9 bar. The grouping is
        # silent about this; the stats are not, and that difference is the point.
        texts = [
            "the kremlin is considering a new mobilization wave this autumn",
            "the kremlin is considering a new mobilization wave, officials said",
        ]
        ids, stats = cluster_texts_with_stats(texts, threshold=0.9)
        assert len(set(ids)) == 2, "no cluster formed — the old log would say nothing"
        assert 0.0 < stats.max_jaccard < 0.9
        assert stats.pairs == 1

    def test_textful_is_the_real_denominator(self):
        # Three rows, one clusterable pair: legacy text-less rows are never compared,
        # so `rows` overstates what the pass could observe (retro#408's lower bound).
        ids, stats = cluster_texts_with_stats(
            [None, "alpha beta gamma delta", "alpha beta gamma delta", ""], threshold=0.5,
        )
        assert stats.rows == 4
        assert stats.textful == 2
        assert stats.pairs == 1
        assert ids[1] == ids[2]

    def test_pairs_is_the_textful_pair_count_and_histogram_sums_to_it(self):
        texts = ["alpha beta gamma", "delta epsilon zeta", "eta theta iota", None]
        _, stats = cluster_texts_with_stats(texts, threshold=0.5)
        assert stats.textful == 3
        assert stats.pairs == 3  # C(3, 2)
        assert sum(stats.histogram) == stats.pairs

    def test_no_comparable_pair_reports_zero_not_a_false_signal(self):
        # A pool that could never echo must not look like a pool that chose not to.
        for texts in ([], ["only one"], [None, None]):
            _, stats = cluster_texts_with_stats(texts, threshold=0.5)
            assert stats.pairs == 0
            assert stats.max_jaccard == 0.0
            assert sum(stats.histogram) == 0

    def test_identical_texts_land_in_the_top_band_not_off_the_end(self):
        # score == 1.0 would index one past the last band without the clamp.
        _, stats = cluster_texts_with_stats(
            ["alpha beta gamma delta", "alpha beta gamma delta"], threshold=0.5,
        )
        assert stats.max_jaccard == 1.0
        assert stats.histogram[-1] == 1
        assert len(stats.histogram) == SIMILARITY_BANDS

    def test_bands_are_lower_inclusive(self):
        # Exactly 0.5 belongs to band 5 ([0.5, 0.6)), not band 4. Constructed rather
        # than approximated: {(a,b,g)} vs {(a,b,g), (b,g,d)} is 1 shared of 2 total.
        texts = ["alpha beta gamma", "alpha beta gamma delta"]
        _, stats = cluster_texts_with_stats(texts, threshold=0.9)
        assert stats.max_jaccard == 0.5
        assert stats.histogram[5] == 1
        assert stats.histogram[4] == 0

    def test_stats_are_deterministic_like_the_ids(self):
        # Same purity contract as the grouping — the replay harness reads both.
        texts = ["alpha beta gamma", "alpha beta gamma delta", "zeta eta theta"]
        first = cluster_texts_with_stats(texts, threshold=0.5)
        for _ in range(5):
            assert cluster_texts_with_stats(texts, threshold=0.5) == first

    def test_grouping_is_byte_identical_to_the_stats_free_wrapper(self):
        # This change must move no estimate: cluster_texts is the old entry point and
        # has to keep returning exactly what it always did.
        texts = ["alpha beta gamma delta", "alpha beta gamma delta", "zeta eta theta iota", None]
        for threshold in (0.0, 0.3, 0.5, 0.9, 1.0):
            ids, _ = cluster_texts_with_stats(texts, threshold=threshold)
            assert cluster_texts(texts, threshold=threshold) == ids


class TestClusterLogIsUnconditional:
    """The defect this fixes: three different silences used to look identical."""

    def _log_line(self, caplog, texts):
        from forecast_api.forecaster import _cluster_ids

        caplog.clear()
        with caplog.at_level(logging.INFO, logger="forecast_api.forecaster"):
            _cluster_ids(texts, [None] * len(texts), "q-hash")
        lines = [r.getMessage() for r in caplog.records if "evidence_clusters" in r.getMessage()]
        assert len(lines) == 1, f"expected exactly one line, got {lines}"
        return lines[0]

    def test_a_pool_with_no_echo_still_logs(self, caplog):
        # Previously silent. A zero that is never written cannot be counted, which is
        # how 180 aggregates produced no observation at all.
        line = self._log_line(caplog, ["alpha beta gamma", "zeta eta theta"])
        assert "echoed_rows=0" in line
        assert "pairs=1" in line

    def test_a_pool_of_textless_legacy_rows_is_distinguishable_from_no_echo(self, caplog):
        # Same zero echo, opposite meaning: nothing was observable here.
        line = self._log_line(caplog, [None, None])
        assert "textful=0" in line
        assert "pairs=0" in line

    def test_a_pool_too_small_to_compare_still_logs(self, caplog):
        line = self._log_line(caplog, ["only one row"])
        assert "rows=1" in line
        assert "pairs=0" in line

    def test_the_near_miss_reaches_the_log(self, caplog):
        line = self._log_line(
            caplog,
            [
                "the kremlin is considering a new mobilization wave this autumn",
                "the kremlin is considering a new mobilization wave, officials said",
            ],
        )
        assert "max_jaccard=0." in line
        assert "max_jaccard=0.000" not in line, "a real near miss must not read as zero"
        assert "threshold=" in line, "a line must be self-describing after a retune"

    def test_a_short_pool_still_returns_none_so_the_discount_path_is_untouched(self):
        from forecast_api.forecaster import _cluster_ids

        assert _cluster_ids(["only one row"], [None], "q") is None
        assert _cluster_ids([], [], "q") is None
        assert _cluster_ids(
            ["alpha beta gamma", "zeta eta theta"], [None, None], "q"
        ) == (0, 1)


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
