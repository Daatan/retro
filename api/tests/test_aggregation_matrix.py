"""R8 — the pinned aggregation trace matrix (retro#370).

The lane-soundness audit's design rule R8 says no aggregator or architecture
change ships on "backtest agreement" alone: agreement with the live system
measures *consistency*, not correctness, and at today's resolution volume
(N < 20) a Brier holdout is pure noise. What is left is fixtures — a pinned,
committed trace of what the pooling mechanics actually do today, case by case.

Every case in ``fixtures/aggregation_matrix/`` runs the **real** pipeline —
``forecaster.run_forecast`` with only the two LLM calls (gatekeeper, extractor)
and the leaderboard lookup stubbed out. Nothing about the composition is
re-implemented here: claim→article fusion, settlement grading and its
enforcers, evidence-class weighting, recency, relevance², logit pooling,
thin-evidence widening, the abstention gates and the settlement override are
all executed by production code. That is the point — a fixture that pinned a
test-local copy of the math would certify the copy, not the estimator.

Two consumers (see retro#370):

1. **Phase-1 acceptance.** Every PR in the Phase-1 set (F16, R3, F9, F20, F1)
   asserts zero *unexplained* fixture movement and states in its description
   which case IDs it intends to move, and why.
2. **F22's extraction-drift canary** reuses the same case bodies on a schedule.

Determinism: "now" is frozen at ``FIXTURE_NOW`` (both ``forecaster`` and
``aggregation`` read the clock), credibility comes from the fixture rather than
the leaderboard, and each case gets a unique question so the forecast cache
cannot serve one case's answer to another.

Adding or updating a case::

    # from api/
    AGG_MATRIX_UPDATE=1 uv run pytest tests/test_aggregation_matrix.py

That rewrites every ``expect`` block from current behaviour. Always read the
resulting git diff: it *is* the movement report a PR has to explain.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from forecast_api import aggregation, forecaster
from forecast_api.models import ArticleInput, ForecastRequest
from tm.models import ExtractionOutput, GatekeeperOutput, PredictionExtraction


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "aggregation_matrix"

# Frozen clock. Fixture article dates are chosen relative to this date, so
# recency weights (and the settlement revalidation window checks) are stable.
FIXTURE_NOW = datetime(2026, 8, 1, 12, 0, 0)

# Set AGG_MATRIX_UPDATE=1 to rewrite the `expect` blocks from live behaviour.
_UPDATE = os.environ.get("AGG_MATRIX_UPDATE") == "1"

# Snapshots are rounded to this many decimals before comparison, so an
# irrelevant float-ULP wobble across platforms can't fail the suite while a
# real mechanical change (which moves the 3rd-4th decimal at least) still does.
_ROUND = 6

# Titles must not collide: dedupe_syndicated() collapses articles whose
# title-token Jaccard is >= 0.8 into one representative. Distinct vocabulary
# per article index keeps every fixture article its own source.
_TITLE_WORDS = [
    "harbour", "quartz", "meridian", "lantern", "cobalt", "thicket",
    "vellum", "sonata", "pumice", "orchard", "gantry", "monsoon",
]

_BODY = (
    "Fixture article body. The gatekeeper and extractor are stubbed for this "
    "suite, so the text is never read by a model; it exists only to clear the "
    "pipeline's non-empty-body checks. "
) * 3


# ── fixture loading ──────────────────────────────────────────────────────────

def _load_files() -> list[tuple[Path, dict]]:
    return [
        (path, json.loads(path.read_text(encoding="utf-8")))
        for path in sorted(FIXTURE_DIR.glob("group_*.json"))
    ]


_FILES = _load_files()
_CASES = [
    pytest.param(path, case, id=case["id"])
    for path, data in _FILES
    for case in data["cases"]
]


@pytest.fixture(scope="session", autouse=True)
def _rewrite_fixtures_on_update():
    """In update mode, write every fixture file back once the session ends.

    Cases are mutated in place during the run (they are the same dict objects
    loaded above), so this just re-serialises whatever the run produced.
    """
    yield
    if not _UPDATE:
        return
    for path, data in _FILES:
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
        )


# ── driving the real pipeline ────────────────────────────────────────────────

class _FrozenDatetime(datetime):
    """``datetime`` with a pinned ``now()``; everything else is inherited (the
    aggregation module also calls ``fromisoformat`` through this name)."""

    @classmethod
    def now(cls, tz=None):  # noqa: D102 - see class docstring
        return FIXTURE_NOW if tz is None else FIXTURE_NOW.replace(tzinfo=tz)


def _url(case_id: str, source: str) -> str:
    return f"https://{source}.example.test/{case_id.lower()}"


def _title(case_id: str, index: int) -> str:
    a = _TITLE_WORDS[index % len(_TITLE_WORDS)]
    b = _TITLE_WORDS[(index + 5) % len(_TITLE_WORDS)]
    return f"{case_id} {a} {b} dispatch {index}"


def _article_input(case_id: str, index: int, spec: dict) -> ArticleInput:
    source = spec["source"]
    return ArticleInput(
        url=_url(case_id, source),
        title=_title(case_id, index),
        snippet=f"Fixture snippet for {case_id} article {index}, long enough to be usable.",
        source=source,
        published_date=spec.get("published_date", "") or "",
        text=_BODY,
    )


def _prediction(spec: dict, index: int, source: str) -> PredictionExtraction:
    """Build a claim from the fixture spec.

    Keys pass straight through to ``PredictionExtraction``, so a case can set
    any extractor field (``settled``, ``event_date``, ``evidence_class``,
    ``quantitative_estimate``, ``specificity``, ``fact_signal``, …) without
    this runner needing to know about it.

    The default claim/quote text is distinct **per article**, mirroring the
    per-article title vocabulary above: ``index`` alone is the claim's position
    *within* its article, so without ``source`` every article's first claim
    carried byte-identical text and the story clusterer (retro#355) read whole
    pools as one echo cluster — invisible while only the inert downweight
    consumed the assignment, pin-killing once the settlement count did
    (retro#372). Fixture articles are independent sources unless a case says
    otherwise by setting identical ``claim``/``quote`` explicitly.
    """
    fields: dict[str, Any] = {
        "quote": f"Fixture quote {index} per {source}.",
        "claim": f"Fixture claim {index} from {source}.",
    }
    fields.update(spec)
    return PredictionExtraction(**fields)


def _patch_pipeline(monkeypatch, case: dict) -> None:
    by_source = {a["source"]: a for a in case["articles"]}
    credibility = {
        f"{a['source']}.example.test": float(a.get("credibility", 1.0))
        for a in case["articles"]
    }

    async def fake_gate(**kwargs):
        art = by_source[kwargs["source_name"]]
        return (
            GatekeeperOutput(
                is_prediction=art.get("is_prediction", True),
                reason="fixture gate",
                prediction_count_estimate=len(art["claims"]),
                relevance_score=float(art.get("relevance", 1.0)),
            ),
            {"total_tokens": 0},
        )

    async def fake_extract(**kwargs):
        art = by_source[kwargs["source_name"]]
        return (
            ExtractionOutput(
                predictions=[
                    _prediction(c, i, art["source"]) for i, c in enumerate(art["claims"])
                ],
                author_lean=art.get("author_lean"),
                author_lean_certainty=art.get("author_lean_certainty"),
            ),
            {"total_tokens": 0},
        )

    monkeypatch.setattr(forecaster, "check_is_prediction", fake_gate)
    monkeypatch.setattr(forecaster, "extract_predictions", fake_extract)
    monkeypatch.setattr(
        forecaster, "get_credibility_weight", lambda sid: credibility.get(sid, 1.0),
    )
    monkeypatch.setattr(forecaster, "datetime", _FrozenDatetime)
    monkeypatch.setattr(aggregation, "datetime", _FrozenDatetime)


def _request(case: dict) -> ForecastRequest:
    articles = [
        _article_input(case["id"], i, spec)
        for i, spec in enumerate(case["articles"])
    ]
    return ForecastRequest(
        question=f"[{case['id']}] {case.get('question', 'Will the event occur by the deadline?')}",
        articles=articles,
        **case.get("request", {}),
    )


# ── snapshotting ─────────────────────────────────────────────────────────────

def _r(value):
    return round(value, _ROUND) if isinstance(value, float) else value


def _source_snapshot(s) -> dict:
    weight = None
    if None not in (s.evidence_weight, s.recency_weight, s.relevance_score):
        weight = s.credibility_weight * s.evidence_weight * s.recency_weight * s.relevance_score ** 2
    return {
        "source": s.source_name,
        "stance": _r(s.stance),
        "certainty": _r(s.certainty),
        "credibility_weight": _r(s.credibility_weight),
        "evidence_class": s.evidence_class,
        "evidence_weight": _r(s.evidence_weight),
        "recency_weight": _r(s.recency_weight),
        "relevance_score": _r(s.relevance_score),
        "weight": _r(weight),
        "settled": s.settled,
        "settlement_event_date": s.settlement_event_date,
        # Shadow lanes. Nothing in aggregation reads these yet, which is exactly
        # why they need pinning: F9 (clamp |fact_signal| when is_occurrence is
        # false) and the author lane both move numbers no estimate assertion
        # would catch.
        "fact_signal": _r(s.fact_signal),
        "is_occurrence": s.is_occurrence,
        "verified": s.verified,
        "author_lean": _r(s.author_lean),
    }


def _snapshot(resp) -> dict:
    return {
        "insufficient_data": resp.insufficient_data,
        "reason": resp.reason,
        "articles_used": resp.articles_used,
        "settled": resp.settled,
        "mean": _r(resp.mean),
        "std": _r(resp.std),
        "ci_low": _r(resp.ci_low),
        "ci_high": _r(resp.ci_high),
        "sources": [_source_snapshot(s) for s in resp.sources],
    }


def _diff(expected: Any, actual: Any, path: str = "") -> list[str]:
    if isinstance(expected, dict) and isinstance(actual, dict):
        out: list[str] = []
        for key in sorted(set(expected) | set(actual)):
            out += _diff(expected.get(key, "<missing>"), actual.get(key, "<missing>"), f"{path}.{key}" if path else key)
        return out
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return [f"{path}: length {len(expected)} → {len(actual)}"]
        out = []
        for i, (e, a) in enumerate(zip(expected, actual)):
            out += _diff(e, a, f"{path}[{i}]")
        return out
    if expected != actual:
        return [f"{path}: {expected!r} → {actual!r}"]
    return []


def _check_invariants(case: dict, snapshot: dict) -> list[str]:
    """Evaluate the case's human-authored directional invariants.

    These are the half of a fixture a regenerated snapshot cannot provide: the
    snapshot records what the estimator *does*, the invariant records what the
    case is *about* ("this dissenter must not flip the pool"). A PR that
    regenerates a snapshot still has to keep these true.
    """
    namespace = {
        "mean": snapshot["mean"], "std": snapshot["std"],
        "ci_low": snapshot["ci_low"], "ci_high": snapshot["ci_high"],
        "settled": snapshot["settled"], "n": snapshot["articles_used"],
        "insufficient_data": snapshot["insufficient_data"],
        "reason": snapshot["reason"], "sources": snapshot["sources"],
        "prob": lambda stance: (stance + 1.0) / 2.0,
        "abs": abs, "len": len, "any": any, "all": all,
        "sum": sum, "min": min, "max": max,
    }
    failures = []
    for expr in case.get("invariants", []):
        if not eval(expr, {"__builtins__": {}}, namespace):  # noqa: S307 - fixture-authored, test-only
            failures.append(expr)
    return failures


# ── the suite ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("path,case", _CASES)
async def test_aggregation_matrix(monkeypatch, path: Path, case: dict):
    _patch_pipeline(monkeypatch, case)
    resp = await forecaster.run_forecast(_request(case))
    actual = _snapshot(resp)

    if _UPDATE:
        case["expect"] = actual
        return

    expected = case.get("expect")
    assert expected is not None, (
        f"{case['id']} has no pinned `expect` block — run "
        f"AGG_MATRIX_UPDATE=1 pytest tests/test_aggregation_matrix.py and commit the diff."
    )

    deltas = _diff(expected, actual)
    if deltas:
        known_bad = case.get("known_bad")
        tag = (
            f"\nThis case is tagged known-bad ({known_bad['finding']}): "
            f"{known_bad['expected_behavior']}\nMovement here may be the intended fix — "
            f"say so explicitly in the PR."
            if known_bad else
            "\nThis case is NOT tagged known-bad: movement is a regression unless the PR "
            "explains it."
        )
        pytest.fail(
            f"{case['id']} ({path.name}) moved — {case['name']}\n  "
            + "\n  ".join(deltas)
            + tag
        )

    broken = _check_invariants(case, actual)
    assert not broken, (
        f"{case['id']} still matches its pinned snapshot but violates its own "
        f"invariants: {broken}"
    )


def test_every_case_id_is_unique():
    ids = [case["id"] for _, data in _FILES for case in data["cases"]]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    assert not duplicates, f"duplicate fixture case ids: {duplicates}"


def test_every_case_declares_what_it_traces():
    """A case with no ``traces`` entry is a snapshot nobody can interpret when
    it moves — which is exactly when interpretation is needed."""
    undocumented = [
        case["id"]
        for _, data in _FILES
        for case in data["cases"]
        if not case.get("traces") or not case.get("name")
    ]
    assert not undocumented, f"cases missing name/traces: {undocumented}"


def test_no_unsettled_case_publishes_a_zero_width_interval():
    """F16 (retro#365). An interval derived from between-source disagreement
    alone collapses to a point whenever a pool happens to agree — 28 of these
    cases did exactly that before the dispersion floor landed. Settled cases
    are exempt: ``_apply_pin`` replaces the interval with its own policy band.
    """
    offenders = [
        case["id"]
        for _, data in _FILES
        for case in data["cases"]
        if not case["expect"].get("insufficient_data")
        and not case["expect"].get("settled")
        and case["expect"].get("ci_high") is not None
        and case["expect"]["ci_high"] - case["expect"]["ci_low"] < 1e-9
    ]
    assert not offenders, f"zero-width published interval on: {offenders}"
