"""Event-key clustering: same (who, what, when) rather than same words (retro#682).

The Jaccard key asks whether the extractor used the same wording. Measured over
19,926,967 pairwise comparisons in prod, 99.72% of pairs score below 0.1 and
`max_jaccard` is exactly 0.0 in two thirds of pools — pool rows are LLM paraphrases of
twenty different outlets, so one development routinely shares almost no trigram with
itself. This key is paraphrase-invariant by construction.

Reporting only: `cluster_downweight_exponent` stays 0.0 and the discount still runs off
the Jaccard ids. These tests pin the key and the measurement, not a weight change.
"""
import logging
from types import SimpleNamespace

import pytest

from forecast_api import forecaster
from forecast_api.clustering import (
    EventKeyStats,
    _normalise_entity,
    cluster_by_event_key,
    event_key,
    event_key_for_row,
)


def _claim(actors, target, edate, strength=0.8):
    return {
        "claim": "x", "stance": 0.5, "certainty": strength, "claim_strength": strength,
        "event_actors": actors, "event_target": target, "event_date": edate,
    }


class TestEntityNormalisation:
    @pytest.mark.parametrize(
        "raw", ["United States", "the United States", "  united   states  ",
                "United States.", "US", "USA", "America"],
    )
    def test_synonyms_and_noise_collapse_to_one_form(self, raw):
        assert _normalise_entity(raw) == "united states"

    def test_multi_actor_order_does_not_matter(self):
        """"United States, Israel" and "Israel, the United States" are one actor set.

        Sorting is what makes that deterministic — without it the key would depend on
        the order the extractor happened to list them in, which is not a fact about
        the world.
        """
        a = _normalise_entity("United States, Israel")
        b = _normalise_entity("Israel, the United States")
        assert a == b == "israel, united states"

    def test_duplicates_within_one_string_collapse(self):
        assert _normalise_entity("US, United States, america") == "united states"

    @pytest.mark.parametrize("raw", [None, "", "   ", ",,,", "  ,  "])
    def test_empty_inputs_yield_none(self, raw):
        assert _normalise_entity(raw) is None

    def test_distinct_entities_stay_distinct(self):
        assert _normalise_entity("Iran") != _normalise_entity("Iraq")
        assert _normalise_entity("North Korea") != _normalise_entity("South Korea")


class TestTheKeyNeedsAllThreeParts:
    def test_a_complete_key_is_built(self):
        assert event_key("United States", "Iran", "2026-02-24") == (
            "united states\x1firan\x1f2026-02-24"
        )

    def test_an_iso_timestamp_is_sliced_to_the_day(self):
        assert event_key("US", "Iran", "2026-02-24T10:30:00Z") == event_key(
            "US", "Iran", "2026-02-24"
        )

    @pytest.mark.parametrize(
        "actors,target,day",
        [(None, "Iran", "2026-02-24"), ("US", None, "2026-02-24"),
         ("US", "Iran", None), ("", "Iran", "2026-02-24")],
    )
    def test_a_missing_part_yields_no_key(self, actors, target, day):
        """None means unkeyed, and an unkeyed row is a singleton at full weight.
        Missing facets must never cost a source its vote."""
        assert event_key(actors, target, day) is None

    def test_the_dyad_alone_is_a_relationship_not_an_event(self):
        """Why the day is required at all.

        On live pools the largest actor-target-only cluster is 171 rows on
        `united states -> iran` — months of coverage collapsed into one "story". With
        the discount enabled that would crush the pool to n_eff ~ 1. The day splits that
        same group into 34 sub-clusters, largest 24.
        """
        same_dyad_different_days = [
            event_key("United States", "Iran", "2026-02-24"),
            event_key("United States", "Iran", "2026-05-19"),
        ]
        assert same_dyad_different_days[0] != same_dyad_different_days[1]

    def test_a_malformed_date_is_sliced_not_parsed(self):
        """`published_date` is free text and holds non-ISO junk (retro#714 fixes it
        going forward; stored rows keep what they were written with). A slice yields a
        stable weird key; a parse would raise inside a /forecast request over a
        reporting-only measurement."""
        k = event_key("US", "Iran", "16/09/2026")
        assert k is not None and k.endswith("16/09/2026")


class TestRowKeySourceOfTruth:
    def test_the_strongest_claim_wins(self):
        claims = [
            _claim("Israel", "Iran", "2026-02-24", strength=0.3),
            _claim("United States", "Iran", "2026-02-24", strength=0.9),
        ]
        assert event_key_for_row(claims) == event_key("United States", "Iran", "2026-02-24")

    def test_ties_break_by_array_position_deterministically(self):
        claims = [
            _claim("Israel", "Iran", "2026-02-24", strength=0.5),
            _claim("United States", "Iran", "2026-02-24", strength=0.5),
        ]
        assert event_key_for_row(claims) == event_key("Israel", "Iran", "2026-02-24")

    def test_claims_without_a_dyad_are_skipped_not_fatal(self):
        claims = [
            _claim(None, None, "2026-02-24", strength=0.99),
            _claim("United States", "Iran", "2026-02-24", strength=0.1),
        ]
        assert event_key_for_row(claims) == event_key("United States", "Iran", "2026-02-24")

    def test_row_level_facets_are_the_fallback(self):
        """Measured on prod this lifts keyed rows from 5,403 to 6,886 (+27%) with the
        largest cluster unchanged at 24 — coverage for free."""
        assert event_key_for_row(
            None, event_actors="United States", event_target="Iran",
            settlement_event_date="2026-02-24",
        ) == event_key("United States", "Iran", "2026-02-24")

    def test_published_date_is_the_last_date_resort(self):
        """`event_date` is on only 6% of voting rows ("omit entirely when the article
        states no date"), so requiring it would key 793 rows instead of ~6,900."""
        assert event_key_for_row(
            [_claim("United States", "Iran", None)], published_date="2026-02-24",
        ) == event_key("United States", "Iran", "2026-02-24")

    def test_event_date_beats_published_date_when_both_exist(self):
        assert event_key_for_row(
            [_claim("United States", "Iran", "2026-02-24")], published_date="2026-08-01",
        ) == event_key("United States", "Iran", "2026-02-24")

    def test_pydantic_style_objects_work_too(self):
        """Both call sites pass models, not dicts — SourceSignal and PoolSourceInput."""
        claims = [SimpleNamespace(
            event_actors="United States", event_target="Iran",
            event_date="2026-02-24", claim_strength=0.7,
        )]
        assert event_key_for_row(claims) == event_key("United States", "Iran", "2026-02-24")

    def test_an_unkeyable_row_returns_none(self):
        assert event_key_for_row(None) is None
        assert event_key_for_row([]) is None


class TestClustering:
    def test_paraphrases_of_one_development_share_a_cluster(self):
        keys = [
            event_key("US", "Iran", "2026-02-24"),
            event_key("the United States", "iran", "2026-02-24"),
            event_key("America", "Iran", "2026-02-24"),
        ]
        ids, stats = cluster_by_event_key(keys)
        assert ids == (0, 0, 0)
        assert stats == EventKeyStats(rows=3, keyed=3, clusters=1, echoed_rows=3, largest=3)

    def test_unkeyed_rows_are_singletons_at_full_weight(self):
        ids, stats = cluster_by_event_key([None, None])
        assert ids == (0, 1)
        assert stats.keyed == 0 and stats.echoed_rows == 0 and stats.largest == 1

    def test_ids_are_first_appearance_order_across_keyed_and_unkeyed(self):
        keys = [event_key("US", "Iran", "2026-02-24"), None,
                event_key("US", "Iran", "2026-02-24")]
        ids, _ = cluster_by_event_key(keys)
        assert ids == (0, 1, 0)

    def test_empty_input(self):
        ids, stats = cluster_by_event_key([])
        assert ids == () and stats.rows == 0 and stats.largest == 0

    def test_output_depends_on_nothing_but_the_input(self):
        """The replay harness (#350/#403) needs the same pool to give the same ids
        forever — no dict iteration order, no clock, no RNG."""
        keys = [event_key("US", "Iran", "2026-02-24"), None,
                event_key("Israel", "Iran", "2026-02-25")] * 3
        assert cluster_by_event_key(keys)[0] == cluster_by_event_key(keys)[0]


class TestTheMeasurementReachesTheLog:
    def _sig(self, actors, target, day):
        return SimpleNamespace(
            claims_detail=[_claim(actors, target, day)],
            title=None, event_actors=None, event_target=None,
            settlement_event_date=None, published_date=day,
        )

    def test_event_fields_ride_on_the_existing_line(self, caplog):
        sigs = [self._sig("US", "Iran", "2026-02-24"),
                self._sig("the United States", "iran", "2026-02-24"),
                self._sig("Israel", "Iran", "2026-02-25")]
        with caplog.at_level(logging.INFO, logger=forecaster.logger.name):
            forecaster._cluster_ids(
                [forecaster._cluster_text_of(s) for s in sigs],
                [forecaster._event_key_of(s) for s in sigs],
                "q-hash",
            )
        line = next(r.getMessage() for r in caplog.records
                    if "event=evidence_clusters" in r.getMessage())
        assert "event_keyed=3" in line
        assert "event_clusters=2" in line
        assert "event_echoed_rows=2" in line
        assert "event_largest=2" in line

    def test_the_jaccard_numbers_are_still_there(self, caplog):
        """Logged beside, not instead — the two keys have to be comparable on identical
        pools before anything is switched over."""
        sigs = [self._sig("US", "Iran", "2026-02-24")] * 2
        with caplog.at_level(logging.INFO, logger=forecaster.logger.name):
            forecaster._cluster_ids(
                [forecaster._cluster_text_of(s) for s in sigs],
                [forecaster._event_key_of(s) for s in sigs],
                "q",
            )
        line = next(r.getMessage() for r in caplog.records
                    if "event=evidence_clusters" in r.getMessage())
        for field in ("rows=", "textful=", "pairs=", "max_jaccard=", "threshold=", "hist="):
            assert field in line

    def test_the_returned_ids_are_still_the_jaccard_ones(self, caplog):
        """Reporting only. The discount is unchanged and stays gated on #403."""
        sigs = [self._sig("US", "Iran", "2026-02-24"),
                self._sig("the United States", "iran", "2026-02-24")]
        with caplog.at_level(logging.INFO, logger=forecaster.logger.name):
            ids = forecaster._cluster_ids(
                # Texts share no trigram, so Jaccard keeps them apart...
                ["alpha beta gamma delta", "zeta eta theta iota"],
                # ...while the event key says they are one development.
                [forecaster._event_key_of(s) for s in sigs],
                "q",
            )
        assert ids == (0, 1)
