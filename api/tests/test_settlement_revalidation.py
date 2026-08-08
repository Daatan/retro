"""Aggregation-time settlement revalidation — fixtures from the 2026-07-16 prod audit.

That audit found 11 of 19 settlement-pinned forecasts wrong, all exactly 97/3.
The recompute endpoint trusted stored ``settled`` bits forever: rows extracted
before the extraction-time guards existed (or re-poisoned by re-pushes, which
re-flip flags within hours — observed live during the Phase 0 cleanup) re-pinned
wrong estimates on every recompute. ``settlement_vote_validity`` makes the
anchor requirement an invariant re-checked on every aggregation, and the pin
rule becomes unanimity: valid settled votes in BOTH directions suppress the pin
(``settlement_conflict``) instead of the larger side winning.

Each test class below encodes one real failure from the audit; the last ones
prove the LEGITIMATE prod pins survive (the over-tightening regression risk)
and that the ``SETTLEMENT_REVALIDATE=false`` kill switch restores the legacy
behavior byte-for-byte.
"""

from __future__ import annotations

import pytest

from forecast_api.aggregation import aggregate_pool, settlement_vote_validity
from forecast_api.config import settings as api_settings

TODAY = "2026-07-17"


def _kwargs(**overrides):
    kw = dict(
        relevance_weight_floor=api_settings.relevance_weight_floor,
        decisiveness_floor=api_settings.decisiveness_floor,
        thin_evidence_ci_inflation=api_settings.thin_evidence_ci_inflation,
        defer_on_thin_evidence=False,
        settlement_min_sources=2,
        settlement_stance=api_settings.settlement_stance,
        logit_clamp=api_settings.logit_clamp,
        pool_dispersion_floor=api_settings.pool_dispersion_floor,
        settlement_revalidate=True,
    )
    kw.update(overrides)
    return kw


def _pool(stances, settled, dates, published=None, weights=None, **overrides):
    n = len(stances)
    return aggregate_pool(
        stances, weights if weights is not None else [1.0] * n, [1.0] * n, settled,
        settlement_event_dates=dates,
        published_dates=published if published is not None else [None] * n,
        **_kwargs(**overrides),
    )


class TestVoteValidity:
    """Unit matrix for the pure helper (deterministic via ``today``)."""

    def test_undated_occurrence_vote_is_demoted(self):
        # The Netanyahu class: accomplished-fact language with no date.
        assert settlement_vote_validity(
            1.0, None, "2026-07-11", "arrival", "2026-12-31", None, None, today=TODAY,
        ) == "missing_event_date"

    def test_dated_occurrence_vote_stands(self):
        assert settlement_vote_validity(
            1.0, "2026-07-09", "2026-07-11", "arrival", "2026-12-31", None, None, today=TODAY,
        ) is None

    def test_occurrence_vote_after_the_deadline_is_demoted(self):
        assert settlement_vote_validity(
            1.0, "2027-01-05", "2027-01-06", "arrival", "2026-12-31", None, None, today="2027-01-07",
        ) == "event_after_deadline"

    def test_scheduled_claim_rejects_an_event_from_before_its_creation(self):
        # The 2021/2022-article class: a real, correctly-dated event from an
        # EARLIER instance of the recurring question.
        assert settlement_vote_validity(
            1.0, "2021-06-11", "2021-06-11", "arrival", "2026-12-31",
            "2026-05-19", "scheduled", today=TODAY,
        ) == "event_before_claim_window"

    def test_threshold_claim_accepts_an_event_predating_its_creation(self):
        # Bitcoin-$100k: the threshold crossed before the claim existed still
        # settles a "by DATE" claim.
        assert settlement_vote_validity(
            1.0, "2024-12-05", "2026-07-04", "arrival", "2026-12-31",
            "2026-02-13", "threshold", today=TODAY,
        ) is None

    def test_scheduled_claim_also_rejects_a_pre_creation_foreclosure(self):
        # 2021 "Bennett formed the government" extracted as a NEGATIVE vote on
        # a 2026 claim — the lower bound applies to both directions.
        assert settlement_vote_validity(
            -1.0, "2021-06-02", "2021-06-02", "arrival", "2026-12-31",
            "2026-06-03", "scheduled", today=TODAY,
        ) == "event_before_claim_window"

    def test_undated_foreclosure_before_the_deadline_is_demoted(self):
        # The fail-open class from the audit: pre-deadline negative pins with
        # no anchor at all. The old pin-level guard let these through whenever
        # the deadline had passed OR metadata was missing; per-vote this is
        # now fail-closed.
        assert settlement_vote_validity(
            -1.0, None, "2026-07-11", "arrival", "2026-12-31", None, None, today=TODAY,
        ) == "undated_foreclosure"

    def test_dated_in_window_foreclosure_stands_before_the_deadline(self):
        # France eliminated by Spain on the 14th, deadline the 19th — a
        # legitimate early NO the old direction guard wrongly suppressed.
        assert settlement_vote_validity(
            -0.98, "2026-07-14", "2026-07-14", "arrival", "2026-07-19", None, None, today="2026-07-16",
        ) is None

    def test_undated_negative_counts_once_the_window_closes(self):
        assert settlement_vote_validity(
            -1.0, None, "2026-07-16", "arrival", "2026-07-15", None, None, today=TODAY,
        ) is None

    def test_survival_positive_after_deadline_needs_no_date(self):
        # "X did NOT happen by D", reported after D — inherently undatable and
        # legitimate. A raw-sign rule would have broken this.
        assert settlement_vote_validity(
            1.0, None, "2026-07-16", "survival", "2026-07-15", None, None, today=TODAY,
        ) is None

    def test_survival_negative_is_an_occurrence_vote_and_needs_a_date(self):
        # Survival:− asserts the underlying event happened — Netanyahu-PM-Dec31
        # was pinned 3% by undated 2022-era rows exactly like this.
        assert settlement_vote_validity(
            -1.0, None, "2026-07-11", "survival", "2026-12-31", None, None, today=TODAY,
        ) == "missing_event_date"

    def test_event_dated_after_its_own_article_is_demoted(self):
        # Re-asserts the extraction-time guard on stored rows.
        assert settlement_vote_validity(
            1.0, "2026-07-12", "2026-07-10", "arrival", "2026-12-31", None, None, today=TODAY,
        ) == "event_after_article"

    def test_unclassified_claim_with_a_dated_foreclosure_counts(self):
        # Callers that don't classify claims (MCP, test console) can still get
        # a dated foreclosure through — only the undated case fails closed.
        assert settlement_vote_validity(
            -1.0, "2026-07-14", "2026-07-14", None, None, None, None, today=TODAY,
        ) is None


class TestFranceCase:
    """3 stale positives + the elimination row (prod pool cmr6a9oij…)."""

    def test_stale_undated_positives_cannot_outvote_a_dated_foreclosure(self):
        agg = _pool(
            [0.9, 1.0, 0.95, -0.98],
            [True, True, True, True],
            [None, None, None, "2026-07-14"],
            claim_direction="arrival", claim_deadline="2099-07-19",
            claim_created_at="2026-07-04", claim_archetype="scheduled",
        )
        assert agg.settled is False  # 3 demoted, 1 valid < min_sources
        assert len(agg.settlement_demotions) == 3
        assert {r for _, r in agg.settlement_demotions} == {"missing_event_date"}

    def test_two_dated_foreclosures_pin_no_before_the_deadline(self):
        # The end state France should converge to: a correct early negative pin.
        agg = _pool(
            [-0.98, -1.0],
            [True, True],
            ["2026-07-14", "2026-07-14"],
            claim_direction="arrival", claim_deadline="2099-07-19",
            claim_created_at="2026-07-04", claim_archetype="scheduled",
        )
        assert agg.settled is True
        assert agg.mean == pytest.approx(-api_settings.settlement_stance)


class TestEnglandInversionCase:
    """4 stance-inverted positives vs the correct negative (pool cmrmgga9x…).

    The inverted rows carry VALID in-window dates — (a)/(b) checks cannot catch
    them. The conflict rule can: one correct dated negative turns a wrong 4-1
    majority pin into a suppression, and re-extraction (the PR-A prompt fix)
    then heals the rows themselves.
    """

    def test_inverted_majority_with_one_correct_dissenter_is_a_conflict(self):
        agg = _pool(
            [1.0, 1.0, 1.0, 1.0, -1.0],
            [True] * 5,
            ["2026-07-15"] * 5,
            published=["2026-07-15"] * 5,
            claim_direction="arrival", claim_deadline="2099-07-15",
            claim_created_at="2026-07-15", claim_archetype="scheduled",
        )
        assert agg.settled is False
        assert agg.settlement_suppressed is True
        assert agg.suppression_reason == "settlement_conflict"

    def test_all_fresh_correct_rows_pin_no(self):
        agg = _pool(
            [-1.0, -1.0],
            [True, True],
            ["2026-07-15", "2026-07-15"],
            claim_direction="arrival", claim_deadline="2099-07-15",
            claim_created_at="2026-07-15", claim_archetype="scheduled",
        )
        assert agg.settled is True
        assert agg.mean < 0


class TestWindowCase:
    """USA-bomb-Iran-2025 / 2021-Netanyahu: dated events outside the window."""

    def test_scheduled_claim_demotes_out_of_window_votes_both_signs(self):
        agg = _pool(
            [1.0, -1.0],
            [True, True],
            ["2021-06-11", "2021-06-02"],
            published=["2021-06-11", "2021-06-02"],
            claim_direction="arrival", claim_deadline="2099-12-31",
            claim_created_at="2026-05-19", claim_archetype="scheduled",
        )
        assert agg.settled is False
        assert len(agg.settlement_demotions) == 2

    def test_threshold_claim_counts_the_same_early_events(self):
        agg = _pool(
            [1.0, 1.0],
            [True, True],
            ["2024-12-05", "2024-12-06"],
            published=["2026-07-04", "2026-07-04"],
            claim_direction="arrival", claim_deadline="2099-12-31",
            claim_created_at="2026-02-13", claim_archetype="threshold",
        )
        assert agg.settled is True


class TestKnessetFlipResidue:
    """A settled negative whose date is the OCCURRENCE, after the deadline
    (produced by enforce_deadline_arithmetic flipping a late positive)."""

    def test_not_counted_while_the_deadline_is_still_open(self):
        agg = _pool(
            [-1.0, -1.0], [True, True], ["2099-07-17", "2099-07-17"],
            claim_direction="arrival", claim_deadline="2099-07-15",
        )
        assert agg.settled is False
        assert {r for _, r in agg.settlement_demotions} == {"event_after_deadline"}

    def test_counted_once_the_window_has_closed(self):
        agg = _pool(
            [-1.0, -1.0], [True, True], ["2020-07-17", "2020-07-17"],
            claim_direction="arrival", claim_deadline="2020-07-15",
        )
        assert agg.settled is True
        assert agg.mean < 0


class TestPostWindowOccurrence:
    """USA-bomb-Iran-2025 (2026-07-19 pool audit): three prod rows settled NO at
    0.925–0.938 certainty by articles reporting the US actively bombing Iran in
    July 2026 — anchors dated ~7 months past the closed 2025 window, on a
    REPEATABLE event (and ground truth for 2025 was YES). An out-of-window
    occurrence must not settle a closed window; only the small late-arrival
    grace (the Knesset class) is honored."""

    def test_dated_anchor_far_past_the_closed_window_is_demoted(self):
        assert settlement_vote_validity(
            -1.0, "2026-07-18", "2026-07-18", "arrival", "2025-12-31", None, None,
            today="2026-07-19",
        ) == "post_window_occurrence"

    def test_dated_anchor_within_the_grace_still_counts(self):
        # The Knesset flip residue: dissolution July 17 vs a July 15 deadline.
        assert settlement_vote_validity(
            -1.0, "2020-07-17", "2020-07-17", "arrival", "2020-07-15", None, None,
            today="2026-07-19",
        ) is None

    def test_grace_boundary_is_inclusive(self):
        assert settlement_vote_validity(
            -1.0, "2025-01-14", "2025-01-14", "arrival", "2024-12-31", None, None,
            today="2026-07-19",
        ) is None  # exactly deadline + 14 days
        assert settlement_vote_validity(
            -1.0, "2025-01-15", "2025-01-15", "arrival", "2024-12-31", None, None,
            today="2026-07-19",
        ) == "post_window_occurrence"

    def test_iran_pool_shape_no_longer_pins(self):
        # The three audited rows, as a recompute would replay them.
        agg = _pool(
            [-1.0, -1.0, -1.0],
            [True] * 3,
            ["2026-07-08", "2026-07-18", "2026-07-18"],
            published=["2026-07-08", "2026-07-18", "2026-07-18"],
            claim_direction="arrival", claim_deadline="2025-12-31",
        )
        assert agg.settled is False
        assert {r for _, r in agg.settlement_demotions} == {"post_window_occurrence"}
        assert len(agg.settlement_demotions) == 3

    def test_undated_expiry_votes_still_pin_the_same_closed_window(self):
        # The honest way to settle NO on a closed window is unaffected.
        agg = _pool(
            [-1.0, -1.0], [True, True], [None, None],
            claim_direction="arrival", claim_deadline="2025-12-31",
        )
        assert agg.settled is True
        assert agg.mean < 0


class TestStaleUndatedForeclosure:
    """retro#295, the #293 residue: 12 of the same "US bombs Iran 2025" pool's
    rows had no settlement_event_date at all (so post_window_occurrence above
    doesn't see them) but were extracted from articles PUBLISHED ~7 months
    after the 2025-12-31 deadline, reporting the SAME event class recurring in
    2026 — the same non-sequitur as a dated post-window occurrence, just
    anchored on ``published`` instead of ``event``. The window genuinely being
    silent is unaffected; only undated votes from articles published long
    after the deadline are demoted."""

    def test_undated_vote_from_article_published_long_after_deadline_is_demoted(self):
        assert settlement_vote_validity(
            -1.0, None, "2026-07-14", "arrival", "2025-12-31", None, None,
            today="2026-07-19",
        ) == "stale_undated_foreclosure"

    def test_undated_vote_from_article_published_within_grace_still_pins(self):
        # 10 days after the deadline — inside post_deadline_grace_days (14d) —
        # the ordinary, honest "window closed quietly" case.
        assert settlement_vote_validity(
            -1.0, None, "2026-01-10", "arrival", "2025-12-31", None, None,
            today="2026-07-19",
        ) is None

    def test_undated_vote_with_no_published_date_is_unaffected(self):
        # Fail-open on missing metadata, matching
        # TestLegitimatePinsStillPin.test_post_deadline_undated_negatives_pin's
        # shape (no published date given at all).
        assert settlement_vote_validity(
            -1.0, None, None, "arrival", "2025-12-31", None, None,
            today="2026-07-19",
        ) is None

    def test_iran_pool_shape_undated_rows_no_longer_pin(self):
        # The 12 remaining undated rows from the audit, as a recompute would replay them.
        agg = _pool(
            [-1.0, -1.0, -1.0],
            [True] * 3,
            [None, None, None],
            published=["2026-07-12", "2026-07-14", "2026-07-16"],
            claim_direction="arrival", claim_deadline="2025-12-31",
        )
        assert agg.settled is False
        assert {r for _, r in agg.settlement_demotions} == {"stale_undated_foreclosure"}
        assert len(agg.settlement_demotions) == 3


class TestQualityFloor:
    """retro#279: settlement_min_sources counts HOW MANY valid votes agree,
    not how much evidence they carry — a pool of uniformly weak sources that
    each barely clear settlement grade could out-count its way to a pin.
    settlement_quality_floor requires the winning direction's combined weight
    (credibility * evidence_weight * recency * relevance^2) to also clear a
    bar, independent of the settlement_conflict/stale_undated_foreclosure
    checks above. Disabled by default (0.0) — every other test in this file
    doesn't set it and is unaffected."""

    def test_uniformly_weak_sources_clear_count_but_not_weight(self):
        # 2 sources (clears settlement_min_sources=2) at weight 0.05 each — a
        # never-scored source (credibility floored at 0.1) with thin relevance.
        agg = _pool(
            [1.0, 1.0], [True, True], ["2026-07-01", "2026-07-01"],
            published=["2026-07-01", "2026-07-01"],
            weights=[0.05, 0.05],
            settlement_quality_floor=0.5,
        )
        assert agg.settled is False
        assert agg.settlement_suppressed is True
        assert agg.suppression_reason == "settlement_quality_floor"
        assert agg.mean > 0  # the pooled mean stands, just unpinned

    def test_strong_sources_clear_both_count_and_weight(self):
        agg = _pool(
            [1.0, 1.0], [True, True], ["2026-07-01", "2026-07-01"],
            published=["2026-07-01", "2026-07-01"],
            weights=[0.4, 0.4],
            settlement_quality_floor=0.5,
        )
        assert agg.settled is True
        assert agg.settlement_suppressed is False

    def test_floor_disabled_by_default_weak_sources_still_pin(self):
        # Same weak-weight shape as the first test, but no floor configured —
        # byte-for-byte the pre-#279 behavior.
        agg = _pool(
            [1.0, 1.0], [True, True], ["2026-07-01", "2026-07-01"],
            published=["2026-07-01", "2026-07-01"],
            weights=[0.05, 0.05],
        )
        assert agg.settled is True

    def test_floor_ignores_demoted_votes_weight(self):
        # A high-weight negative vote that's INVALID (undated, window still
        # open — claim_deadline is 2099) never enters neg/neg_weight at all;
        # only the two valid, weak positives are tallied against the floor.
        agg = _pool(
            [1.0, 1.0, -1.0], [True, True, True],
            ["2026-07-01", "2026-07-01", None],
            published=["2026-07-01", "2026-07-01", "2026-07-01"],
            weights=[0.3, 0.3, 0.9],
            settlement_quality_floor=0.5,
            claim_direction="arrival", claim_deadline="2099-12-31",
        )
        assert {r for _, r in agg.settlement_demotions} == {"undated_foreclosure"}
        assert agg.settled is True  # the two 0.3-weight positives sum to 0.6 >= floor


class TestLegitimatePinsStillPin:
    """The 8 sound pins from the audit must survive (over-tightening risk)."""

    def test_unanimous_dated_positives_pin(self):
        # RSF/Lebanon-talks/rockets shape: many dated in-window positives.
        n = 12
        agg = _pool(
            [1.0] * n, [True] * n, ["2026-07-01"] * n,
            published=["2026-07-02"] * n,
            claim_direction="arrival", claim_deadline="2099-12-31",
            claim_created_at="2026-02-19", claim_archetype="scheduled",
        )
        assert agg.settled is True
        assert agg.settled_sources == n

    def test_post_deadline_undated_negatives_pin(self):
        # Knesset-dissolved shape: the window closed without the event.
        agg = _pool(
            [-1.0, -0.95, -1.0, -1.0], [True] * 4, [None] * 4,
            claim_direction="arrival", claim_deadline="2020-07-15",
        )
        assert agg.settled is True
        assert agg.mean < 0

    def test_unsettled_rows_never_trigger_revalidation(self):
        agg = _pool([0.6, -0.2], [False, False], [None, None])
        assert agg.settled is False
        assert agg.settlement_demotions == ()


class TestKillSwitch:
    """settlement_revalidate=False must reproduce the legacy behavior."""

    def test_legacy_majority_vote_returns_with_the_flag_off(self):
        # The France poison: 3 stale undated positives outvote the correct
        # dated negative — exactly the failure the default-on path fixes.
        agg = _pool(
            [0.9, 1.0, 0.95, -0.98],
            [True, True, True, True],
            [None, None, None, "2026-07-14"],
            settlement_revalidate=False,
        )
        assert agg.settled is True
        assert agg.mean == pytest.approx(api_settings.settlement_stance)
        assert agg.settlement_demotions == ()

    def test_legacy_direction_guard_still_reports_its_reason(self):
        agg = _pool(
            [-1.0, -1.0], [True, True], [None, None],
            claim_direction="arrival", claim_deadline="2099-01-01",
            settlement_revalidate=False,
        )
        assert agg.settled is False
        assert agg.settlement_suppressed is True
        assert agg.suppression_reason == "settlement_direction"


class TestPinOutranksAbstention:
    """retro#396: system-model §6.2 orders publish-time precedence

        settlement pin > impossibility pin > abstention > glide > pooled estimate

    but ``aggregate_pool`` computed the three abstention reasons first and
    returned early, so a pool tripping ANY of them could never publish a pin.
    The safe direction of the inversion (abstaining is quieter than pinning),
    which is why no incident traces to it — but not the contract, and the
    daatan-side twin of the same §6.2 link (daatan#1265) was the unsafe one.

    The R8 matrix pins the ``all_articles_off_topic`` half (cases C16/C17). The
    other two reasons are unreachable from a fixture — ``no_usable_weight``
    needs every weight at exactly zero and ``no_decisive_signal`` needs
    ``defer_on_thin_evidence``, which ships off — so they are pinned here,
    against ``aggregate_pool`` directly.
    """

    def test_off_topic_pool_still_publishes_its_pin(self):
        # relevance_mass 3 * 0.1^2 = 0.03 < relevance_weight_floor (0.05).
        agg = aggregate_pool(
            [1.0, 1.0, -0.4], [0.4, 0.4, 0.4], [0.1, 0.1, 0.1], [True, True, False],
            settlement_event_dates=["2026-07-01", "2026-07-01", None],
            published_dates=["2026-07-01", "2026-07-01", "2026-07-01"],
            **_kwargs(settlement_quality_floor=0.5),
        )
        assert agg.insufficient_reason is None
        assert agg.settled is True
        assert agg.mean == pytest.approx(api_settings.settlement_stance)
        # The pool's own thinness stays visible rather than being laundered by
        # the pin: a caller logging evidence_mass sees what was outranked.
        assert agg.evidence_mass == pytest.approx(1.2)
        assert agg.settlement_vote_indices == (0, 1)

    def test_thin_evidence_deferral_still_publishes_its_pin(self):
        # defer_on_thin_evidence + evidence_mass (0.6) under decisiveness_floor
        # (10.0) is "no_decisive_signal" — a deferral, not an emptiness.
        agg = aggregate_pool(
            [1.0, 1.0], [0.3, 0.3], [1.0, 1.0], [True, True],
            settlement_event_dates=["2026-07-01", "2026-07-01"],
            published_dates=["2026-07-01", "2026-07-01"],
            **_kwargs(
                defer_on_thin_evidence=True, decisiveness_floor=10.0,
                settlement_quality_floor=0.5,
            ),
        )
        assert agg.insufficient_reason is None
        assert agg.settled is True
        assert agg.thin_evidence is True

    def test_zero_weight_pool_cannot_pin_because_the_quality_floor_catches_it(self):
        # F14's "no_usable_weight" and the quality floor nest: every weight is
        # zero, so the winning direction's weight is zero too and the pin is
        # suppressed on its own merits. The abstention stands — reordering the
        # two checks did not make a worthless pin publishable.
        agg = aggregate_pool(
            [1.0, 1.0], [0.0, 0.0], [1.0, 1.0], [True, True],
            settlement_event_dates=["2026-07-01", "2026-07-01"],
            published_dates=["2026-07-01", "2026-07-01"],
            **_kwargs(settlement_quality_floor=0.5),
        )
        assert agg.insufficient_reason == "no_usable_weight"
        assert agg.settled is False
        assert agg.suppression_reason == "settlement_quality_floor"

    def test_abstention_stands_when_the_pin_is_suppressed(self):
        # Valid votes in both directions: unanimity fails, so there is no pin
        # to outrank the abstention, and the reason the caller sees is still
        # the abstention's.
        agg = aggregate_pool(
            [1.0, -1.0, 1.0, -1.0], [0.4] * 4, [0.1] * 4, [True, True, True, True],
            settlement_event_dates=["2026-07-01"] * 4,
            published_dates=["2026-07-01"] * 4,
            **_kwargs(),
        )
        assert agg.insufficient_reason == "all_articles_off_topic"
        assert agg.settled is False
        # Newly carried on the abstaining result (retro#396): without it a
        # reader cannot tell a suppressed pin from no settlement votes at all.
        assert agg.settlement_suppressed is True
        assert agg.suppression_reason == "settlement_conflict"

    def test_abstention_with_no_settlement_votes_reports_no_suppression(self):
        agg = aggregate_pool(
            [0.5, 0.3], [0.4, 0.4], [0.1, 0.1], [False, False],
            settlement_event_dates=[None, None],
            published_dates=["2026-07-01", "2026-07-01"],
            **_kwargs(),
        )
        assert agg.insufficient_reason == "all_articles_off_topic"
        assert agg.settlement_suppressed is False
        assert agg.suppression_reason is None


class TestClusterCountedVotes:
    """retro#372's second half: the trigger counts independent clusters, not rows.

    ``settlement_min_sources`` demands corroboration, and two syndicated copies
    of one report are one observation. The assignment comes from retro#355's
    clusterer; rows without claim text are singletons there, so a missing claim
    layer can never cost a vote — it just counts as itself.
    """

    def _pin(self, cluster_ids, weights=None):
        return aggregate_pool(
            [1.0, 1.0], weights if weights is not None else [1.0, 1.0],
            [1.0, 1.0], [True, True],
            settlement_event_dates=["2026-07-01", "2026-07-01"],
            published_dates=["2026-07-02", "2026-07-02"],
            cluster_ids=cluster_ids,
            **_kwargs(),
        )

    def test_two_votes_in_one_cluster_do_not_pin(self):
        agg = self._pin(cluster_ids=(0, 0))
        assert agg.settled is False
        assert agg.settled_sources == 0

    def test_two_votes_in_distinct_clusters_pin(self):
        agg = self._pin(cluster_ids=(0, 1))
        assert agg.settled is True
        assert agg.settled_sources == 2

    def test_no_assignment_falls_back_to_the_row_count(self):
        agg = self._pin(cluster_ids=None)
        assert agg.settled is True
        assert agg.settled_sources == 2

    def test_a_mismatched_assignment_is_ignored_not_misapplied(self):
        # A wrong-length assignment would mis-map votes to clusters; ignoring
        # it degrades to the row count, the pre-#372 behavior.
        agg = self._pin(cluster_ids=(0,))
        assert agg.settled is True
        assert agg.settled_sources == 2

    def test_settled_sources_reports_clusters_not_rows(self):
        # Three rows, two of them echoing one report: the pin stands on two
        # independent observations and says so.
        agg = aggregate_pool(
            [1.0, 1.0, 1.0], [1.0] * 3, [1.0] * 3, [True] * 3,
            settlement_event_dates=["2026-07-01"] * 3,
            published_dates=["2026-07-02"] * 3,
            cluster_ids=(0, 0, 1),
            **_kwargs(),
        )
        assert agg.settled is True
        assert agg.settled_sources == 2

    def test_quality_floor_still_sums_rows_not_clusters(self):
        # Mass is mass, however correlated — independence is the count's axis,
        # not the floor's. Two echoed rows at 0.15 carry 0.30 >= the 0.20
        # floor; they are still ONE observation, so no pin, and the floor is
        # not what suppressed it.
        agg = self._pin(cluster_ids=(0, 0), weights=[0.15, 0.15])
        assert agg.settled is False
        assert agg.suppression_reason is None
