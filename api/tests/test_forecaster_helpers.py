"""Unit tests for pure helpers in :mod:`forecast_api.forecaster`.

We only exercise the functions that do not require LLM/HTTP mocking:
  * ``_truncate_article``
  * ``_question_hash``
  * ``_avg``
  * ``build_claim_meta``

The full ``run_forecast`` integration path requires Bedrock and live network
and is covered by manual staging smoke tests.
"""

from forecast_api.cache import ForecastCache
from forecast_api.forecaster import (
    _avg,
    _question_hash,
    _truncate_article,
    build_claim_meta,
)
from forecast_api.models import ForecastRequest


class TestTruncateArticle:
    def test_returns_text_unchanged_when_under_cap(self):
        assert _truncate_article("short", 3000) == "short"

    def test_truncates_to_exact_cap(self):
        text = "x" * 5000
        out = _truncate_article(text, 3000)
        assert len(out) == 3000
        assert out == "x" * 3000

    def test_cap_of_zero_disables_truncation(self):
        text = "x" * 10_000
        assert _truncate_article(text, 0) == text

    def test_negative_cap_disables_truncation(self):
        text = "x" * 10_000
        assert _truncate_article(text, -1) == text


class TestQuestionHash:
    def test_is_stable_across_casing_and_whitespace(self):
        assert _question_hash("  Will X happen? ") == _question_hash("will x happen?")

    def test_short_hex_output(self):
        h = _question_hash("anything")
        assert len(h) == 12
        int(h, 16)  # raises if not hex


class TestAvg:
    def test_returns_none_when_no_values_for_key(self):
        assert _avg([{"other": 5}], "missing") is None

    def test_averages_across_entries_with_key(self):
        timings = [
            {"fetch_ms": 100},
            {"fetch_ms": 200},
            {"gate_ms": 50},  # no fetch_ms — skipped
        ]
        assert _avg(timings, "fetch_ms") == 150.0

    def test_ignores_none_values(self):
        timings = [{"fetch_ms": 100}, {"fetch_ms": None}]
        assert _avg(timings, "fetch_ms") == 100.0


def _req(**kw) -> ForecastRequest:
    return ForecastRequest(question="Will X happen by year end?", **kw)


class TestBuildClaimMeta:
    """retro#510 — ``resolution_criteria`` must discriminate the forecast cache key.

    Without it, a call that omits the criteria and one that sends them hash to the
    same key, so whichever lands first serves the other: the field would apply as a
    function of cache arrival order rather than of the request.
    """

    def test_no_metadata_at_all_stays_none(self):
        assert build_claim_meta(_req()) is None

    def test_criteria_less_requests_hash_exactly_as_before(self):
        # Regression guard on the live cache: the pre-#510 format was
        # f"{claim_direction or ''}|{claim_deadline or ''}". Any drift here
        # invalidates every cached forecast on deploy.
        assert build_claim_meta(_req(claim_direction="arrival")) == "arrival|"
        assert build_claim_meta(_req(claim_deadline="2026-12-31")) == "|2026-12-31"
        assert (
            build_claim_meta(_req(claim_direction="survival", claim_deadline="2026-12-31"))
            == "survival|2026-12-31"
        )

    def test_criteria_alone_is_enough_to_produce_a_discriminator(self):
        # Previously this returned None — a criteria-only request was indistinguishable
        # from a bare one.
        assert build_claim_meta(_req(resolution_criteria="Official gov't announcement only.")) is not None

    def test_different_criteria_produce_different_meta(self):
        a = build_claim_meta(_req(resolution_criteria="Official announcement only."))
        b = build_claim_meta(_req(resolution_criteria="Any credible press report counts."))
        assert a != b

    def test_presence_of_criteria_changes_the_cache_key(self):
        # The actual defect, asserted end-to-end at the key level: same question,
        # same article set, same temporal metadata — differing only in criteria.
        bare = _req(claim_direction="arrival", claim_deadline="2026-12-31")
        with_rules = _req(
            claim_direction="arrival",
            claim_deadline="2026-12-31",
            resolution_criteria="Only an official government announcement counts.",
        )
        k_bare = ForecastCache.make_key(bare.question, 5, "abc123", build_claim_meta(bare))
        k_rules = ForecastCache.make_key(with_rules.question, 5, "abc123", build_claim_meta(with_rules))
        assert k_bare != k_rules

    def test_identical_requests_still_share_a_key(self):
        req = _req(claim_direction="arrival", resolution_criteria="Same rules.")
        again = _req(claim_direction="arrival", resolution_criteria="Same rules.")
        assert ForecastCache.make_key(req.question, 5, None, build_claim_meta(req)) == \
               ForecastCache.make_key(again.question, 5, None, build_claim_meta(again))
