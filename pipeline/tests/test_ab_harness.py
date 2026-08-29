"""Pure-logic tests for the extractor A/B harness (retro#470).

No network, no Bedrock — these cover case loading, per-facet expectation
checking, the temporal-leakage flag, and the zero-regression gate. The
live-model driver (scripts/ab_extractor_prompt.py) is exercised manually,
same as eval_extractor_adjacent_events.py; see docs/AB_HARNESS.md.
"""
import json
from pathlib import Path

import pytest

from tm.ab_harness import (
    _FACET_READERS,
    _band,
    Case,
    build_case_results,
    gate_exit_code,
    load_cases,
    unmet_facets,
)
from tm.models import PredictionExtraction

CASE_DIR = Path(__file__).resolve().parents[1] / "scripts" / "ab_cases"


def _pred(**over) -> PredictionExtraction:
    return PredictionExtraction(**{"quote": "q", "claim": "c", "stance": 0.1, "certainty": 0.5, **over})


def _case(**over) -> Case:
    return Case(**{
        "id": "c1", "event_name": "E", "event_description": "rules",
        "claim_deadline": "2026-09-05", "article_date": "2026-08-20",
        "article_text": "text", "expect": {},
        **over,
    })


class TestCaseLoading:
    def test_load_cases_round_trips_through_json(self, tmp_path: Path):
        raw = [{
            "id": "x1", "event_name": "E", "event_description": "rules",
            "claim_deadline": "2026-09-05", "article_date": "2026-08-20",
            "article_text": "text", "expect": {"stance_sign": 1},
            "tags": ["a", "b"],
        }]
        p = tmp_path / "cases.json"
        p.write_text(json.dumps(raw))

        cases = load_cases(p)

        assert len(cases) == 1
        assert cases[0].id == "x1"
        assert cases[0].tags == ("a", "b")
        assert cases[0].expect == {"stance_sign": 1}

    def test_control_event_description_defaults_to_none(self):
        assert _case().control_event_description is None


class TestTemporalLeakage:
    def test_article_after_deadline_is_leakage(self):
        c = _case(claim_deadline="2026-09-05", article_date="2026-09-10")
        assert c.is_temporal_leakage is True

    def test_article_before_deadline_is_not_leakage(self):
        c = _case(claim_deadline="2026-09-05", article_date="2026-08-20")
        assert c.is_temporal_leakage is False

    def test_article_on_deadline_is_not_leakage(self):
        c = _case(claim_deadline="2026-09-05", article_date="2026-09-05")
        assert c.is_temporal_leakage is False

    def test_unparseable_dates_are_not_leakage(self):
        c = _case(claim_deadline="whenever", article_date="2026-08-20")
        assert c.is_temporal_leakage is False


class TestUnmetFacets:
    def test_facet_met_by_any_prediction_in_any_run(self):
        expect = {"stance_sign": 1}
        runs = [[_pred(stance=-0.5)], [_pred(stance=0.9)]]
        assert unmet_facets(runs, expect) == set()

    def test_facet_unmet_when_no_run_satisfies_it(self):
        expect = {"stance_sign": 1}
        runs = [[_pred(stance=-0.5)], [_pred(stance=-0.2)]]
        assert unmet_facets(runs, expect) == {"stance_sign"}

    def test_fact_signal_null_is_its_own_checkable_facet(self):
        expect = {"fact_signal_null": True}
        runs = [[_pred(fact_signal=None)]]
        assert unmet_facets(runs, expect) == set()

    def test_multiple_facets_tracked_independently(self):
        expect = {"stance_sign": 1, "is_occurrence": False}
        runs = [[_pred(stance=0.5, is_occurrence=None)]]
        # stance_sign met, is_occurrence not (None != False)
        assert unmet_facets(runs, expect) == {"is_occurrence"}

    def test_unknown_facet_raises(self):
        with pytest.raises(KeyError):
            unmet_facets([[_pred()]], {"not_a_real_facet": 1})

    def test_empty_runs_leaves_everything_unmet(self):
        assert unmet_facets([], {"stance_sign": 1}) == {"stance_sign"}

    def test_facet_field_is_checkable(self):
        """retro#541 — the literal announcement/denial/neither field, not to be
        confused with this module's generic "facet" (expectation dimension)."""
        expect = {"facet": "neither"}
        runs = [[_pred(facet="announcement")], [_pred(facet="neither")]]
        assert unmet_facets(runs, expect) == set()

    def test_facet_field_unmet_when_no_run_matches(self):
        expect = {"facet": "neither"}
        runs = [[_pred(facet="announcement")], [_pred(facet=None)]]
        assert unmet_facets(runs, expect) == {"facet"}


class TestMagnitudeBands:
    """retro#720: the facet set could read direction but not strength.

    Every worked example in the `## Multi-stage / bracket events` section is
    positive (+0.2 … +0.6), so deleting the section outright cannot move
    `stance_sign` on any bracket case — it moves how strongly a single-stage
    "favourite" framing is read. Scored on sign alone, that deletion reports as
    a clean pass, which is the false comfort the bracket corpus exists to
    prevent.
    """

    @pytest.mark.parametrize("stance,band", [
        (0.0, "none"), (0.1, "none"), (-0.14, "none"),
        (0.15, "weak"), (0.2, "weak"), (0.3, "weak"), (0.49, "weak"),
        (0.5, "moderate"), (0.6, "moderate"), (0.79, "moderate"),
        (0.8, "strong"), (1.0, "strong"),
    ])
    def test_band_boundaries(self, stance: float, band: str):
        assert unmet_facets([[_pred(stance=stance)]], {"stance_band": band}) == set()

    def test_band_is_magnitude_only(self):
        """A band must not encode direction — that is `stance_sign`'s job, and a
        case asserting both is asserting two independent things."""
        expect = {"stance_band": "moderate"}
        assert unmet_facets([[_pred(stance=-0.6)]], expect) == set()
        assert unmet_facets([[_pred(stance=0.6)]], expect) == set()

    def test_sign_and_band_are_independently_checkable(self):
        runs = [[_pred(stance=-0.6)]]
        assert unmet_facets(runs, {"stance_sign": 1, "stance_band": "moderate"}) == {"stance_sign"}

    def test_claim_strength_band_reads_its_own_field(self):
        runs = [[_pred(stance=0.9, claim_strength=0.3)]]
        expect = {"stance_band": "strong", "claim_strength_band": "weak"}
        assert unmet_facets(runs, expect) == set()

    def test_absent_value_is_not_the_none_band(self):
        """An unfilled field is `None`, not the `"none"` band a genuine 0.0
        lands in — so a case asserting a band on a field the model left blank
        fails rather than quietly matching.

        Unreachable through the two readers registered today (`stance` and
        `claim_strength` are both required on `PredictionExtraction`), which is
        why this asserts on the helper directly. The branch exists because
        `_sign` needed exactly it the moment a nullable field — `fact_signal` —
        got a reader, and the next band reader may well be nullable too."""
        assert _band(None) is None
        assert _band(0.0) == "none"


class TestShippedCaseFiles:
    """Every committed case file must load and name only real facets.

    `unmet_facets` raises on an unknown facet, but nothing calls it until
    `compare` — so a typo'd facet name survives an entire Bedrock sweep and
    blows up after the money is spent. This is the cheap version of that check.
    """

    @pytest.mark.parametrize("path", sorted(CASE_DIR.glob("*.json")), ids=lambda p: p.name)
    def test_case_file_loads_with_known_facets(self, path: Path):
        cases = load_cases(path)
        assert cases, f"{path.name} holds no cases"
        for case in cases:
            unknown = set(case.expect) - set(_FACET_READERS)
            assert not unknown, f"{case.id}: unknown facet(s) {sorted(unknown)}"

    def test_case_ids_are_unique_across_files(self):
        """`compare` keys results by case id across whichever files an arm ran,
        so a duplicate silently scores one case against the other's runs."""
        seen: dict[str, str] = {}
        for path in sorted(CASE_DIR.glob("*.json")):
            for case in load_cases(path):
                assert case.id not in seen, \
                    f"case id {case.id!r} in both {seen[case.id]} and {path.name}"
                seen[case.id] = path.name


class TestRegressionGate:
    def test_no_regression_when_patched_keeps_baseline_facets(self):
        case = _case(expect={"stance_sign": 1})
        results = build_case_results(
            [case],
            baseline={"c1": [[_pred(stance=0.5)]]},
            patched={"c1": [[_pred(stance=0.6)]]},
        )
        assert gate_exit_code(results) == 0
        assert results[0].regressions == set()

    def test_regression_when_patched_loses_a_baseline_facet(self):
        case = _case(expect={"stance_sign": 1, "is_occurrence": True})
        results = build_case_results(
            [case],
            baseline={"c1": [[_pred(stance=0.5, is_occurrence=True)]]},
            patched={"c1": [[_pred(stance=0.5, is_occurrence=False)]]},
        )
        assert results[0].regressions == {"is_occurrence"}
        assert gate_exit_code(results) == 1

    def test_improvement_is_not_a_regression(self):
        case = _case(expect={"stance_sign": -1})
        results = build_case_results(
            [case],
            baseline={"c1": [[_pred(stance=0.5)]]},   # wrong sign
            patched={"c1": [[_pred(stance=-0.5)]]},   # now correct
        )
        assert results[0].improvements == {"stance_sign"}
        assert results[0].regressions == set()
        assert gate_exit_code(results) == 0

    def test_leakage_case_regression_excluded_from_gate_by_default(self):
        case = _case(
            claim_deadline="2026-09-05", article_date="2026-09-10",
            expect={"stance_sign": 1},
        )
        results = build_case_results(
            [case],
            baseline={"c1": [[_pred(stance=0.5)]]},
            patched={"c1": [[_pred(stance=-0.5)]]},
        )
        assert results[0].regressions == {"stance_sign"}
        assert gate_exit_code(results) == 0
        assert gate_exit_code(results, allow_leakage=True) == 1

    def test_empty_expect_never_regresses(self):
        """The 'reference case, eyeball only' pattern (retro#353 control-arm case)."""
        case = _case(expect={})
        results = build_case_results(
            [case],
            baseline={"c1": [[_pred(stance=0.9)]]},
            patched={"c1": [[_pred(stance=-0.9)]]},
        )
        assert results[0].regressions == set()
        assert gate_exit_code(results) == 0

    def test_missing_case_id_raises(self):
        case = _case()
        with pytest.raises(KeyError):
            build_case_results([case], baseline={}, patched={"c1": [[_pred()]]})

    def test_control_unmet_is_populated_when_control_results_given(self):
        case = _case(expect={"stance_sign": 1})
        results = build_case_results(
            [case],
            baseline={"c1": [[_pred(stance=-0.5)]]},
            patched={"c1": [[_pred(stance=-0.5)]]},
            control={"c1": [[_pred(stance=0.9)]]},
        )
        assert results[0].control_unmet == set()
