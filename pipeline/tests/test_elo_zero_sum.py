"""Regression: _update_elo must conserve aggregate ELO across an uneven
correct/wrong split (it previously drifted, giving each source ±K/n)."""

from tm.scorer import Scorer


def _run(preds, outcome, base=1200.0):
    stats = {sid: {"elo": base} for sid, _ in preds}
    Scorer._update_elo(None, stats, preds, outcome)
    return stats


def test_zero_sum_with_uneven_split():
    # 3 correct (stance>0, outcome True), 1 wrong.
    preds = [("a", 0.8), ("b", 0.6), ("c", 0.7), ("d", -0.5)]
    stats = _run(preds, True)
    total = sum(s["elo"] for s in stats.values())
    assert round(total, 6) == round(1200.0 * 4, 6)  # aggregate conserved
    assert stats["a"]["elo"] > 1200.0 and stats["d"]["elo"] < 1200.0


def test_no_change_when_all_agree():
    # Everyone correct -> no counterparty -> no ELO moves.
    stats = _run([("a", 0.8), ("b", 0.6)], True)
    assert all(s["elo"] == 1200.0 for s in stats.values())


def test_empty_predictions_is_noop():
    stats = {}
    Scorer._update_elo(None, stats, [], True)
    assert stats == {}
