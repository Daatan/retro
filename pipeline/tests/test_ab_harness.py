"""Pure-logic tests for the extractor A/B harness (retro#470).

No network, no Bedrock — these cover case loading, per-facet expectation
checking, the temporal-leakage flag, and the zero-regression gate. The
live-model driver (scripts/ab_extractor_prompt.py) is exercised manually,
same as eval_extractor_adjacent_events.py; see docs/AB_HARNESS.md.
"""
import json
from pathlib import Path

import pytest

from tm.ab_harness import Case, build_case_results, gate_exit_code, load_cases, unmet_facets
from tm.models import PredictionExtraction


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
