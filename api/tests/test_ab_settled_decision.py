"""Tests for the retro#697 settled-decision A/B harness.

The measurement's whole value is that the baseline arm IS the live prompt and the
pin arithmetic IS the arithmetic the retro#691 gates were scored against. Both are
easy to break silently — a heading rename empties an arm, an outlet re-derivation
shifts every pin count — so both are pinned here.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
_SPEC = importlib.util.spec_from_file_location(
    "ab_settled_decision", Path(__file__).resolve().parents[1] / "scripts" / "ab_settled_decision.py"
)
ab = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ab)


def arm_b() -> str:
    return ab.build_arms()[ab.ARM_B]


class TestArms:
    def test_baseline_is_sliced_from_the_live_prompt(self):
        from tm.extractor import PROMPT_PREFIX

        settled_rule = ab.slice_section("## SETTLED — the event already happened", level=2)
        assert settled_rule in PROMPT_PREFIX
        assert settled_rule in ab.build_arms()[ab.ARM_A]

    def test_a_renamed_heading_fails_loudly(self):
        # Silently returning an empty section would leave the control arm running
        # with no rules at all and still print a tidy comparison table.
        with pytest.raises(SystemExit, match="no longer in PROMPT_PREFIX"):
            ab.slice_section("## A HEADING THAT DOES NOT EXIST", level=2)

    def test_slice_stops_at_the_next_heading_of_that_level(self):
        section = ab.slice_section("## MATCH THE EVENT — do not credit a near-miss", level=2)
        assert section.startswith("## MATCH THE EVENT")
        assert "###" in section          # its own subsections belong to it
        assert "\n## " not in section    # the next top-level section does not

    def test_both_arms_carry_the_match_the_event_rules(self):
        for name, text in ab.build_arms().items():
            assert "decompose the RELATED EVENT into WHO" in text, name

    def test_only_the_buried_facts_paragraph_differs_in_prose(self):
        arms = ab.build_arms()
        # The live wording licenses settling on an incidental clause outright.
        assert "mark settled true regardless of how minor" in arms[ab.ARM_A]
        assert "mark settled true regardless of how minor" not in arms[ab.ARM_B]
        assert "This does NOT relax MATCH THE EVENT" in arms[ab.ARM_B]

    def test_the_buried_facts_paragraph_appears_exactly_once_per_arm(self):
        """The first run of this script appended the replacement instead of
        substituting it, so arm B carried the permissive wording AND the strict
        one, permissive first. Both arms must state the rule once."""
        for name, text in ab.build_arms().items():
            assert text.count("### Buried facts") == 1, name

    def test_arm_b_requires_the_decomposition_as_output(self):
        assert "REQUIRED OUTPUT" in arm_b()
        assert "matches_all_three" in arm_b()
        assert "REQUIRED OUTPUT" not in ab.build_arms()[ab.ARM_A]


def _row(pid, outlet, claim, url="http://x/1"):
    return {"pid": pid, "outlet": outlet, "claim": claim, "quote": "q", "url": url}


class TestPoolClassification:
    def test_one_outlet_does_not_pin(self):
        rows = [_row("p", "toi.com", "a"), _row("p", "toi.com", "b")]
        labels = {ab.candidate_key(r): "ADJACENT" for r in rows}
        assert ab.classify_pools(rows, labels)["p"]["pins_today"] is False

    def test_all_adjacent_is_zero_support(self):
        rows = [_row("p", "toi.com", "a"), _row("p", "jpost.com", "b")]
        labels = {ab.candidate_key(r): "ADJACENT" for r in rows}
        pool = ab.classify_pools(rows, labels)["p"]
        assert pool["pins_today"] and pool["zero_support"]

    def test_an_unclear_claim_disqualifies_the_strict_reading(self):
        """A pool we could not label is not a pool we know is bad — the strict
        count is what retro#691's "9 zero-support pins" refers to."""
        rows = [_row("p", "toi.com", "a"), _row("p", "jpost.com", "b")]
        labels = {ab.candidate_key(rows[0]): "ADJACENT", ab.candidate_key(rows[1]): "UNCLEAR"}
        pool = ab.classify_pools(rows, labels)["p"]
        assert pool["zero_support"] is False
        assert pool["zero_support_loose"] is True

    def test_outlet_comes_from_the_stored_column_not_the_url(self):
        # Same domain, two stored outlets: the stored identity is what the gate
        # scoring counted, so it must win.
        rows = [_row("p", "toi.com", "a", "http://feeds.example/1"),
                _row("p", "jpost.com", "b", "http://feeds.example/2")]
        labels = {ab.candidate_key(r): "ADJACENT" for r in rows}
        assert ab.classify_pools(rows, labels)["p"]["pins_today"] is True


class TestPinTable:
    LABELS = {"a": "ADJACENT", "b": "ADJACENT", "c": "SETTLES", "d": "SETTLES"}

    def _pools(self):
        rows = [_row("bad", "toi.com", "a"), _row("bad", "jpost.com", "b"),
                _row("good", "toi.com", "c"), _row("good", "jpost.com", "d")]
        labels = {ab.candidate_key(r): self.LABELS[r["claim"]] for r in rows}
        return rows, ab.classify_pools(rows, labels)

    def test_settling_everything_stops_nothing(self):
        rows, pools = self._pools()
        t = ab.pin_table(pools, {ab.candidate_key(r): True for r in rows})
        assert (t["bad_stopped"], t["bad_total"]) == (0, 1)
        assert (t["good_kept"], t["good_total"]) == (1, 1)

    def test_dropping_one_outlet_is_enough_to_stop_a_pin(self):
        rows, pools = self._pools()
        got = {ab.candidate_key(r): r["claim"] != "a" for r in rows}
        t = ab.pin_table(pools, got)
        assert t["bad_stopped"] == 1 and t["good_kept"] == 1

    def test_the_cost_side_is_counted_too(self):
        """A rewrite that settles nothing scores a perfect 1/1 on bad pins and
        must show the 0/1 good pin it costs, or the table is an advert."""
        rows, pools = self._pools()
        t = ab.pin_table(pools, {ab.candidate_key(r): False for r in rows})
        assert t["bad_stopped"] == 1 and t["good_kept"] == 0

    def test_errors_are_surfaced_as_undecided(self):
        rows, pools = self._pools()
        t = ab.pin_table(pools, {ab.candidate_key(r): None for r in rows})
        # An errored claim cannot hold a pin up, so the arm looks perfect —
        # which is exactly why the count has to be visible.
        assert t["bad_stopped"] == 1 and t["undecided"] == 4
