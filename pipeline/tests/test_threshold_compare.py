"""The code-side numeric comparison (retro#683 item 1.6).

This is the half of `quantity` that has to be RIGHT rather than merely harvested.
The field exists because PR#671 measured Nova Lite returning stance +0.00 on every
between-bounds case and inverting both tone traps — so if the arithmetic that
replaces that judgment is itself shaky, the issue has moved the bug rather than
fixed it. Hence the boundary cases below get the same attention as the happy path:
`<=` at exactly the bar, `<` at exactly the bar, and a bounded report that touches
the threshold at one open endpoint.
"""
import json
from pathlib import Path
from dataclasses import replace

import pytest

from tm.ab_harness import load_cases, quantity_diagnostics
from tm.models import PredictionExtraction, Quantity
from tm.threshold_compare import QuestionThreshold, compare, normalise_unit

CORPUS = Path(__file__).resolve().parents[1] / "scripts" / "ab_cases" / "numeric_threshold_blindness.json"


def _q(value, comparator="=", unit="percent", value_hi=None):
    return Quantity(value=value, unit=unit, comparator=comparator, value_hi=value_hi)


def _t(comparator, value, unit="percent", value_hi=None):
    return QuestionThreshold(comparator=comparator, value=value, unit=unit, value_hi=value_hi)


class TestStatedLevels:
    """A reported level against every shape of threshold — the common case."""

    @pytest.mark.parametrize("value,expected", [
        (8.75, 1), (9.0, 1), (9.01, -1), (9.5, -1),
    ])
    def test_at_or_below(self, value, expected):
        assert compare(_q(value), _t("<=", 9)).sign == expected

    @pytest.mark.parametrize("value,expected", [(8.99, 1), (9.0, -1), (9.5, -1)])
    def test_strictly_below(self, value, expected):
        """9.0 against `< 9` contradicts. The inclusive/exclusive boundary is
        where a hand-rolled comparison chain gets it wrong, so it is pinned."""
        assert compare(_q(value), _t("<", 9)).sign == expected

    @pytest.mark.parametrize("value,expected", [(36, 1), (34, 1), (33, -1), (31, -1)])
    def test_strictly_above(self, value, expected):
        assert compare(_q(value, unit="seats"), _t(">", 33, unit="seats")).sign == expected

    @pytest.mark.parametrize("value,expected", [(35, 1), (30, 1), (29.9, -1)])
    def test_at_least(self, value, expected):
        assert compare(_q(value), _t(">=", 30)).sign == expected

    @pytest.mark.parametrize("value,expected", [(2.0, 1), (2.4, 1), (3.0, 1), (4.1, -1), (1.9, -1)])
    def test_between_bounds_is_inclusive(self, value, expected):
        assert compare(_q(value), _t("between", 2, value_hi=3)).sign == expected


class TestBoundedReports:
    """The article need not state a value. "stayed below 5%" is a bound, and a
    bound is answerable whenever every value it allows lands on one side."""

    def test_a_bound_entirely_inside_the_satisfying_set_satisfies(self):
        assert compare(_q(5, "<"), _t("<=", 9)).sign == 1

    def test_a_bound_entirely_outside_contradicts(self):
        assert compare(_q(10, ">"), _t("<=", 9)).sign == -1

    def test_a_bound_spanning_both_answers_abstains(self):
        """`> 5` contains 6 (satisfies `<= 9`) and 20 (does not). A sign invented
        from an interval that contains both answers is worse than no sign."""
        result = compare(_q(5, ">"), _t("<=", 9))
        assert result.sign is None
        assert result.reason == "straddles"

    def test_touching_at_an_open_endpoint_is_disjoint(self):
        """`> 9` and `<= 9` share only the point 9, which `> 9` excludes."""
        assert compare(_q(9, ">"), _t("<=", 9)).sign == -1

    def test_touching_at_two_closed_endpoints_overlaps(self):
        """`>= 9` and `<= 9` both contain 9, so they are not disjoint — the report
        allows a satisfying value and larger ones, which is a straddle, not a
        contradiction."""
        result = compare(_q(9, ">="), _t("<=", 9))
        assert result.sign is None
        assert result.reason == "straddles"

    def test_a_reported_range_inside_the_bar_satisfies(self):
        assert compare(_q(1.8, "between", value_hi=2.2), _t("<=", 3)).sign == 1

    def test_a_reported_range_across_the_bar_abstains(self):
        assert compare(_q(1.8, "between", value_hi=2.2), _t("<=", 2)).sign is None


class TestUnits:
    """A comparison that was never made must not read as one that failed."""

    def test_a_mismatch_abstains_rather_than_contradicting(self):
        result = compare(_q(40, unit="launchers"), _t(">=", 60, unit="kilometres"))
        assert result.sign is None
        assert result.reason == "unit_mismatch"

    @pytest.mark.parametrize("written", ["percent", "Percent", "  percent ", "%", "pct", "per cent", "percentage"])
    def test_the_common_spellings_of_percent_are_one_unit(self, written):
        assert normalise_unit(written) == "percent"

    def test_plurals_normalise_to_the_singular(self):
        assert normalise_unit("seats") == normalise_unit("seat")
        assert normalise_unit("daily departures") == normalise_unit("daily departure")

    def test_percentage_points_are_not_percent(self):
        """They share a prefix and are different units. A substring match here
        would turn a rate level into a rate CHANGE and answer confidently."""
        assert normalise_unit("percentage points") != normalise_unit("percent")
        assert compare(_q(0.5, unit="percentage points"), _t("<=", 9)).sign is None


class TestValidation:
    def test_between_requires_an_upper_bound(self):
        with pytest.raises(ValueError):
            QuestionThreshold(comparator="between", value=2, unit="percent")

    def test_an_upper_bound_without_between_is_rejected(self):
        with pytest.raises(ValueError):
            QuestionThreshold(comparator="<=", value=2, unit="percent", value_hi=3)

    def test_the_upper_bound_must_be_above_the_lower(self):
        with pytest.raises(ValueError):
            QuestionThreshold(comparator="between", value=3, unit="percent", value_hi=2)

    def test_the_same_rules_hold_on_the_extracted_side(self):
        """`Quantity` and `QuestionThreshold` are the same object seen from two
        sides; a rule enforced on only one of them is a rule that can be dodged."""
        with pytest.raises(ValueError):
            Quantity(value=2, unit="percent", comparator="between")
        with pytest.raises(ValueError):
            Quantity(value=2, unit="percent", comparator="<=", value_hi=3)


class TestTheRetro664Corpus:
    """The claim retro#683 rests on, checked against the corpus that produced it.

    PR#671 measured Nova Lite on these ten cases: +0.00 on every between-bounds
    case and both tone traps inverted. The code-side comparison is asserted to get
    all ten right from the known values — that gap is the whole argument for
    moving the comparison out of the model's head, so it is pinned rather than
    quoted from an issue.
    """

    @pytest.fixture(scope="class")
    def cases(self):
        return load_cases(CORPUS)

    def test_every_case_declares_a_threshold_and_a_known_value(self, cases):
        missing = [c.id for c in cases if c.question_threshold is None or c.expect_quantity is None]
        assert not missing, f"cases without threshold/known value: {missing}"

    def test_the_code_side_comparison_is_right_on_all_ten(self, cases):
        wrong = []
        for case in cases:
            got = compare(Quantity(**case.expect_quantity), case.threshold).sign
            if got != case.expect["stance_sign"]:
                wrong.append((case.id, got, case.expect["stance_sign"]))
        assert not wrong, f"code-side sign disagrees with the labelled stance sign: {wrong}"

    def test_the_archetype_detector_covers_this_whole_corpus(self, cases):
        """Every event here is decided by a number, so the detector must say so.

        It did not, until retro#748. Half this corpus failed it for two reasons that
        have nothing to do with the threshold and everything to do with how the
        number is written: a unit NOUN is not a unit symbol ("more than 33 seats"),
        and `between` was missing from the cue set. #683 pinned the miss-set rather
        than fixing it, because #688's routing ships OFF so the gap cost nothing then;
        #748 fixed it, because Phase 2 takes the comparison to live traffic and a
        question the detector cannot see is one the comparison never runs on.

        Kept as a coverage assertion rather than deleted. This corpus is the
        definition of the shape, so a future edit to `archetype.py` that loses one of
        these is a regression, and nothing else in the tree would catch it — the
        detector's own tests are hand-written examples, not this corpus.
        """
        from tm.archetype import is_threshold_shaped

        missed = {c.id for c in cases if not is_threshold_shaped(c.event_name)}
        assert missed == set(), (
            "the archetype detector stopped recognising part of the numeric corpus; "
            "these are the events it is FOR"
        )

    def test_the_corpus_file_still_round_trips(self, cases):
        """`load_cases` calls `Case(**c)`, so an unknown key in the JSON is a
        TypeError at load rather than a quiet drop. Loading it is the check."""
        assert len(cases) == len(json.loads(CORPUS.read_text()))


class TestTheDiagnostic:
    """`quantity_diagnostics` — the per-rater row the issue asks to be logged
    beside `stance`. Fed synthetic extractions so the counters are checkable."""

    @pytest.fixture(scope="class")
    def case(self):
        return next(c for c in load_cases(CORPUS) if c.id == "threshold-at-or-below-satisfied")

    @staticmethod
    def _pred(stance, quantity=None):
        return PredictionExtraction(
            quote="q", claim="c", stance=stance, claim_strength=0.7, quantity=quantity,
        )

    def test_a_correct_extraction_with_a_correct_stance_agrees(self, case):
        preds = [[self._pred(0.8, {"value": 8.75, "unit": "percent", "comparator": "="})]]
        d = quantity_diagnostics([case], {case.id: preds})[0]
        assert (d.filled, d.exact, d.agree, d.disagree) == (1, 1, 1, 0)
        assert (d.code_correct, d.stance_correct) == (1, 1)

    def test_the_gap_is_visible_when_the_number_is_right_and_the_stance_is_not(self, case):
        """The retro#664 finding in one row: the model read the number correctly
        and still scored the stance the wrong way. `code_correct` 1 against
        `stance_correct` 0 is exactly what item 1.6 exists to surface."""
        preds = [[self._pred(-0.6, {"value": 8.75, "unit": "percent", "comparator": "="})]]
        d = quantity_diagnostics([case], {case.id: preds})[0]
        assert (d.agree, d.disagree) == (0, 1)
        assert (d.code_correct, d.stance_correct) == (1, 0)

    def test_a_wrong_number_fails_the_validator_but_still_counts_as_filled(self, case):
        preds = [[self._pred(0.8, {"value": 9.75, "unit": "percent", "comparator": "="})]]
        d = quantity_diagnostics([case], {case.id: preds})[0]
        assert (d.filled, d.labelled, d.exact) == (1, 1, 0)

    def test_an_unfilled_quantity_is_not_an_abstention(self, case):
        """Nothing extracted and something extracted that code could not decide
        are different failures with different fixes; the counters keep them apart."""
        d = quantity_diagnostics([case], {case.id: [[self._pred(0.8)]]})[0]
        assert (d.predictions, d.filled, d.undecidable) == (1, 0, 0)

    def test_a_straddling_report_is_counted_as_undecidable(self, case):
        preds = [[self._pred(0.8, {"value": 5, "unit": "percent", "comparator": ">"})]]
        d = quantity_diagnostics([case], {case.id: preds})[0]
        assert (d.undecidable, d.agree, d.disagree) == (1, 0, 0)

    def test_cases_without_a_threshold_are_not_reported(self):
        """The rest of the corpus has no number for this to be about; a row of
        zeros for every bracket case would bury the ten that matter."""
        cases = load_cases(CORPUS.parent / "multi_stage_brackets.json")
        assert quantity_diagnostics(cases, {c.id: [] for c in cases}) == []

class TestTheTargetCounters:
    """`target_hit` / `target_miscomparated` — per RUN, about the one number the
    case is a test of.

    These exist because `exact` measured the wrong thing across raters. Run on the
    numeric corpus, Haiku 4.5 got every case's number, unit and comparator right in
    every run, and still scored `exact` 70% against Nova Lite's 78% — because Haiku
    quotes more generously, and its extra predictions correctly report OTHER figures
    from the same article ("the other party took 24 seats"). Those are right, not
    wrong, and enough of them invert the ranking between two raters. The validator
    asks about the case's own number, so these counters do too.
    """

    @pytest.fixture(scope="class")
    def case(self):
        return next(c for c in load_cases(CORPUS) if c.id == "threshold-at-or-below-satisfied")

    @staticmethod
    def _pred(stance, quantity=None):
        return PredictionExtraction(
            quote="q", claim="c", stance=stance, claim_strength=0.7, quantity=quantity,
        )

    def test_a_run_that_carries_the_number_hits_once_however_many_predictions(self, case):
        run = [
            self._pred(0.8, {"value": 8.75, "unit": "percent", "comparator": "="}),
            self._pred(0.2, {"value": 2.1, "unit": "percent", "comparator": "="}),
            self._pred(0.1, None),
        ]
        d = quantity_diagnostics([case], {case.id: [run]})[0]
        assert (d.runs, d.target_hit, d.target_miscomparated) == (1, 1, 0)
        assert (d.filled, d.exact, d.labelled) == (2, 1, 2), "per-prediction counts unchanged"

    def test_a_second_figure_from_the_article_is_not_a_miss(self, case):
        """The exact-vs-target divergence, pinned. Two predictions, one of them the
        target: `exact` reads 1/2, `target_hit` reads 1/1."""
        run = [
            self._pred(0.8, {"value": 8.75, "unit": "percent", "comparator": "="}),
            self._pred(0.3, {"value": 24, "unit": "seats", "comparator": "="}),
        ]
        d = quantity_diagnostics([case], {case.id: [run]})[0]
        assert (d.exact, d.labelled) == (1, 2)
        assert (d.target_hit, d.target_miscomparated) == (1, 0)

    def test_the_right_number_under_the_wrong_comparator_is_its_own_bucket(self, case):
        """Nova Lite's whole failure mode on this corpus: it finds 8.75 percent and
        writes `<` because the sentence moved downward. Not a miss — a miscomparison,
        and the distinction is the finding."""
        run = [self._pred(0.8, {"value": 8.75, "unit": "percent", "comparator": "<"})]
        d = quantity_diagnostics([case], {case.id: [run]})[0]
        assert (d.target_hit, d.target_miscomparated) == (0, 1)

    def test_finding_nothing_is_neither_a_hit_nor_a_miscomparison(self, case):
        d = quantity_diagnostics([case], {case.id: [[self._pred(0.8)]]})[0]
        assert (d.runs, d.target_hit, d.target_miscomparated) == (1, 0, 0)

    def test_a_different_unit_does_not_rescue_the_same_bare_number(self, case):
        run = [self._pred(0.8, {"value": 8.75, "unit": "seats", "comparator": "="})]
        d = quantity_diagnostics([case], {case.id: [run]})[0]
        assert (d.target_hit, d.target_miscomparated) == (0, 0)

    def test_runs_are_only_counted_where_there_is_a_label_to_count_against(self):
        """A case with a `question_threshold` and no `expect_quantity` still gets a
        row — the code-vs-stance columns are meaningful — but `runs` stays 0 so it
        cannot dilute the validator percentage."""
        case = load_cases(CORPUS)[0]
        unlabelled = replace(case, expect_quantity=None)
        d = quantity_diagnostics([unlabelled], {unlabelled.id: [[self._pred(0.8)]]})[0]
        assert d.runs == 0
