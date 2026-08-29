"""The provider's date is validated before it wins over the URL leg (retro#714).

`pipeline/tests/test_published_date_normalisation.py` pins the format rules. What this
pins is the *call site*: that `_resolve_article_date` normalises rather than trusts,
that the URL fallback behind it is reachable again, and that a rejected string is
logged instead of vanishing.
"""
import logging

from forecast_api import forecaster
from tm.web_search import SearchResult

_DATED_URL = "https://www.theguardian.com/world/2026/02/24/coalition-talks"
_UNDATED_URL = "https://www.dw.com/en/coalition-talks/a-71234567"


def _result(published_date, url=_UNDATED_URL):
    return SearchResult(
        title="Coalition talks continue", url=url,
        snippet="Negotiators met again on Tuesday.",
        source="dw.com", published_date=published_date,
    )


class TestNormalisation:
    def test_iso_passes_through(self):
        assert forecaster._resolve_article_date(_result("2026-02-24")) == "2026-02-24"

    def test_long_form_is_recovered_not_truncated(self):
        """Previously `[:10]` stored this as "Feb 24, 20" — a value nothing can parse.

        Recovery matters more than rejection here: the row used to keep its vote at the
        recency floor, so simply refusing the date would have traded a 50x under-weight
        for dropping the article outright.
        """
        assert forecaster._resolve_article_date(_result("Feb 24, 2026")) == "2026-02-24"

    def test_bidi_wrapped_date_is_recovered(self):
        assert forecaster._resolve_article_date(_result("16‏/09‏/2026")) == "2026-09-16"

    def test_relative_provider_date_is_absolutized(self):
        """Brave's `age` field arrives raw and is frequently relative."""
        out = forecaster._resolve_article_date(_result("3 days ago"))
        assert out is not None and out.startswith("20")


class TestTheFallbackLegIsReachableAgain:
    def test_unparseable_provider_date_falls_through_to_the_url(self):
        """This is the actual defect: any non-empty string used to win outright, so the
        URL leg sitting right behind it could never run."""
        assert forecaster._resolve_article_date(
            _result("not a date", url=_DATED_URL)
        ) == "2026-02-24"

    def test_undatable_url_and_bad_provider_date_yields_none(self):
        """None means the article is dropped (retro#705) — correct when neither the
        provider nor the URL knows when this was published."""
        assert forecaster._resolve_article_date(_result("not a date")) is None

    def test_ambiguous_numeric_falls_through_rather_than_guessing(self):
        assert forecaster._resolve_article_date(
            _result("05/09/2026", url=_DATED_URL)
        ) == "2026-02-24"


class TestRejectionIsVisible:
    def test_rejected_provider_date_is_logged_with_its_raw_value(self, caplog):
        with caplog.at_level(logging.INFO, logger=forecaster.logger.name):
            forecaster._resolve_article_date(_result("not a date"))
        lines = [r.getMessage() for r in caplog.records
                 if "event=provider_date_rejected" in r.getMessage()]
        assert len(lines) == 1
        assert "not a date" in lines[0]

    def test_a_genuinely_absent_date_is_not_logged_as_rejected(self, caplog):
        """An empty provider date is not a parse failure — conflating the two is what
        `event_date_state` was added to stop doing for settlement dates (retro#554)."""
        with caplog.at_level(logging.INFO, logger=forecaster.logger.name):
            forecaster._resolve_article_date(_result(""))
        assert not [r for r in caplog.records
                    if "event=provider_date_rejected" in r.getMessage()]

    def test_an_accepted_date_is_not_logged_as_rejected(self, caplog):
        with caplog.at_level(logging.INFO, logger=forecaster.logger.name):
            forecaster._resolve_article_date(_result("Feb 24, 2026"))
        assert not [r for r in caplog.records
                    if "event=provider_date_rejected" in r.getMessage()]
