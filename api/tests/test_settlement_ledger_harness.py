"""Tests for scripts/score_settlement_ledger.py (retro#691).

This harness scores the settlement gates against the one label that is not a
model output — whether the outcome later contradicted the pin. That makes its
failure mode specific: a harness that quietly reconstructs the *wrong* vote-set
still prints a confident table, and the table would then be an argument for
enforcing a gate on evidence that never existed. So what is pinned here is
mostly what the harness must REFUSE to do.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "score_settlement_ledger",
    Path(__file__).resolve().parent.parent / "scripts" / "score_settlement_ledger.py",
)
mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mod)


PRED = {
    "kind": "pred", "id": "p1",
    "claim": "Israel will engage in a significant military conflict with Iran by October 27, 2026.",
    "deadline": "2026-10-27", "created": "2026-05-17",
    "archetype": "DIFFUSE", "direction": "ARRIVAL",
}


def _row(**kw) -> dict:
    base = {
        "kind": "row", "pid": "p1", "source": "example.com",
        "published": "2026-08-06", "added_at": "2026-08-06 10:00:00",
        "stance": 1.0, "certainty": 0.95, "sed": "2026-08-05",
        "occ": True, "facet": "occurrence", "actors": "Israel", "target": "Iran",
        "cls": "REPORTED",
        "claims": [{"claim": "Israel struck Iranian targets", "st": "1.0", "ct": "0.95",
                    "occ": "true", "ac": "Israel", "tg": "Iran", "ed": "2026-08-05",
                    "fc": "occurrence", "cls": "REPORTED"}],
    }
    base.update(kw)
    return base


class TestAsOfReconstruction:
    def test_rows_added_after_resolution_cannot_have_voted(self):
        """The pool keeps growing after a question resolves. Counting a row that
        arrived later as a settlement vote would credit (or blame) the pin for
        evidence it never saw."""
        late = _row(added_at="2026-09-01 10:00:00")
        assert mod.candidates_for([late], "p1", "2026-08-19 21:04:56", PRED) == []

    def test_rows_present_at_resolution_are_kept(self):
        assert len(mod.candidates_for([_row()], "p1", "2026-08-19 21:04:56", PRED)) == 1

    def test_the_temporal_guard_still_applies(self):
        """Measured on the real ledger: every reconstructable claim behind the
        contradicted Israel/Iran pin describes a 2024 or February-2026 event, so
        `settlement_vote_validity` rejects all five as
        `event_before_claim_window`. The pin's actual votes are older rows with
        no claims_detail — and a harness that ignored the guard would happily
        score these five as if they were the pin."""
        stale = _row(sed="2024-04-13", published="2024-04-13",
                     claims=[{"claim": "Israel struck Iran in 2024", "st": "1.0",
                              "ct": "0.95", "ed": "2024-04-13"}])
        assert mod.candidates_for([stale], "p1", "2026-08-19 21:04:56", PRED) == []

    def test_other_predictions_rows_are_not_borrowed(self):
        assert mod.candidates_for([_row(pid="other")], "p1", "2026-08-19 21:04:56", PRED) == []


def _run(tmp_path: Path, ledger: list[dict], data: list[dict], capsys, gates=None) -> tuple[int, str]:
    lp = tmp_path / "ledger.jsonl"
    lp.write_text("\n".join(json.dumps(e) for e in ledger))
    dp = tmp_path / "data.json"
    dp.write_text(json.dumps(data))
    argv = ["--ledger", str(lp), "--data", str(dp)]
    if gates:
        argv += ["--gates", gates]
    import sys
    old, sys.argv = sys.argv, ["prog", *argv]
    try:
        rc = mod.main()
    finally:
        sys.argv = old
    return rc, capsys.readouterr().out


def _entry(pid="p1", contradicted=False):
    return {"prediction_id": pid, "contradicted": contradicted,
            "resolved_at": "2026-08-19T21:04:56.000Z", "outcome": not contradicted}


class TestRefusals:
    def test_nothing_scoreable_exits_nonzero(self, tmp_path, capsys):
        """retro#395 shipped a settlement replay that measured nothing and read
        as a pass. An empty run must be loud and must not exit 0."""
        rc, out = _run(tmp_path, [_entry(contradicted=True)], [PRED], capsys)
        assert rc == 1
        assert "NOTHING SCOREABLE" in out

    def test_missing_claims_detail_is_named_as_such(self, tmp_path, capsys):
        _, out = _run(tmp_path, [_entry(contradicted=True)], [PRED, _row(pid="other")], capsys)
        assert "no claims_detail" in out

    def test_rows_present_but_invalid_is_a_different_message(self, tmp_path, capsys):
        """Collapsing these two into one line hides which road is blocked: a
        backfill gap is fixable, a vote-set that lives elsewhere is not."""
        stale = _row(sed="2024-04-13", published="2024-04-13",
                     claims=[{"claim": "old", "st": "1.0", "ct": "0.95", "ed": "2024-04-13"}])
        _, out = _run(tmp_path, [_entry(contradicted=True)], [PRED, stale], capsys)
        assert "rows carry claims_detail, but none is a valid vote" in out
        assert "no claims_detail" not in out

    def test_zero_contradicted_pins_scored_is_called_out(self, tmp_path, capsys):
        """The live case: four upheld pins scored, zero contradicted ones. The
        table then says nothing about whether the gates CATCH a bad pin, and the
        output has to say so or it will be read as if it did."""
        _, out = _run(tmp_path, [_entry("p1"), _entry("p2", contradicted=True)],
                      [PRED, {**PRED, "id": "p2"}, _row()], capsys)
        assert "nothing below speaks to" in out

    def test_single_digit_classes_are_labelled_not_a_measurement(self, tmp_path, capsys):
        _, out = _run(tmp_path, [_entry("p1")], [PRED, _row()], capsys)
        assert "NOT A MEASUREMENT" in out

    def test_unrecognised_gate_names_are_rejected(self, tmp_path, capsys):
        rc, _ = _run(tmp_path, [_entry()], [PRED, _row()], capsys, gates="not_a_gate")
        assert rc == 1


class TestScoring:
    def test_a_pin_that_keeps_two_outlets_is_not_blocked(self, tmp_path, capsys):
        rows = [_row(source="a.com"), _row(source="b.com")]
        _, out = _run(tmp_path, [_entry()], [PRED, *rows], capsys)
        assert "allow" in out

    def test_a_pin_demoted_below_min_sources_is_blocked(self, tmp_path, capsys):
        """facet_missing fires on both rows, leaving zero outlets."""
        rows = [_row(source="a.com", facet=None,
                     claims=[{"claim": "x", "st": "1.0", "ct": "0.95", "ed": "2026-08-05"}]),
                _row(source="b.com", facet=None,
                     claims=[{"claim": "y", "st": "1.0", "ct": "0.95", "ed": "2026-08-05"}])]
        _, out = _run(tmp_path, [_entry()], [PRED, *rows], capsys)
        assert "BLOCK" in out
        assert "settled_without_facet" in out
