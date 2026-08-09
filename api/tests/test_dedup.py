"""Unit tests for :mod:`forecast_api.dedup` — syndication collapsing.

Pure functions, no network. The key contracts: true re-prints of one wire story
collapse to a single highest-credibility source, while genuinely different stories
that merely share a topic word are left alone.
"""
from __future__ import annotations

from types import SimpleNamespace

from forecast_api.dedup import dedupe_syndicated


def _a(url, title, cred):
    return SimpleNamespace(url=url, title=title, cred=cred)


def _dedupe(items, threshold=0.8):
    return dedupe_syndicated(
        items,
        title_of=lambda r: r.title,
        url_of=lambda r: r.url,
        priority_of=lambda r: r.cred,
        threshold=threshold,
    )


def test_collapses_reprinted_story_keeping_highest_credibility():
    items = [
        _a("http://msn.com/x", "Anthropic refuses to release Claude Mythos over security concerns", 0.3),
        _a("http://reuters.com/x", "Anthropic Refuses To Release Claude Mythos Over Security Concerns", 0.9),
        _a("http://yahoo.com/x", "Anthropic refuses to release Claude Mythos over security concerns!", 0.5),
    ]
    out = _dedupe(items)
    assert len(out) == 1
    assert out[0].url == "http://reuters.com/x"  # highest credibility kept


def test_keeps_distinct_stories_sharing_a_topic_word():
    # The real Mythos forecast pull: three different stories, all about Anthropic /
    # Mythos, must NOT be merged.
    items = [
        _a("http://msn.com/1", "When will Anthropic release Mythos? Here's what the prediction market is saying", 0.3),
        _a("http://msn.com/2", "Why did Anthropic refuse to release its most dangerous AI model, Claude Mythos?", 0.3),
        _a("http://msn.com/3", "Claude Fable 5 released as Anthropic's first Mythos-class AI", 0.3),
    ]
    out = _dedupe(items)
    assert len(out) == 3


def test_exact_duplicate_urls_collapse():
    items = [
        _a("http://msn.com/x", "Totally different headline number one here", 0.3),
        _a("http://msn.com/x", "Another completely unrelated headline entirely", 0.9),
    ]
    out = _dedupe(items)
    assert len(out) == 1
    assert out[0].cred == 0.9


def test_short_titles_are_not_clustered():
    # Degenerate/terse titles carry too little signal to cluster on.
    items = [
        _a("http://a.com/1", "t", 0.5),
        _a("http://b.com/2", "t", 0.5),
    ]
    assert len(_dedupe(items)) == 2


def test_threshold_zero_disables_title_clustering():
    items = [
        _a("http://a.com/1", "Anthropic refuses to release Claude Mythos over security concerns", 0.3),
        _a("http://b.com/2", "Anthropic refuses to release Claude Mythos over security concerns", 0.9),
    ]
    assert len(_dedupe(items, threshold=0.0)) == 2  # URLs differ, titles ignored


def test_preserves_order_and_passthrough_when_unique():
    items = [
        _a("http://a.com/1", "First entirely separate story about elections somewhere", 0.5),
        _a("http://b.com/2", "Second entirely separate story about the economy today", 0.5),
    ]
    out = _dedupe(items)
    assert [r.url for r in out] == ["http://a.com/1", "http://b.com/2"]


def test_collapses_reprinted_hebrew_story():
    # Same wire story (Hebrew), one word differs (verb form) between the two
    # re-hosts — exactly the shape a syndicated re-print takes. The old
    # `[a-z0-9]+` tokenizer matched none of this text (zero tokens throughout),
    # so it could never clear `_MIN_CLUSTER_TOKENS` and these would never have
    # clustered before the script-agnostic regex (retro#458 Phase 3).
    items = [
        _a(
            "http://msn.com/he1",
            "ישראל חותמת הסכם הפסקת אש עם חמאס לאחר משא ומתן ממושך",
            0.3,
        ),
        _a(
            "http://reuters.com/he1",
            "ישראל חתמה הסכם הפסקת אש עם חמאס לאחר משא ומתן ממושך",
            0.9,
        ),
    ]
    out = _dedupe(items)
    assert len(out) == 1
    assert out[0].url == "http://reuters.com/he1"  # highest credibility kept


def test_collapses_reprinted_russian_story():
    # Same wire story (Russian), one word differs (synonym) between re-hosts.
    items = [
        _a(
            "http://yahoo.com/ru1",
            "Центральный банк повысил процентную ставку до рекордного уровня в этом году",
            0.4,
        ),
        _a(
            "http://ap.com/ru1",
            "Центральный банк поднял процентную ставку до рекордного уровня в этом году",
            0.9,
        ),
    ]
    out = _dedupe(items)
    assert len(out) == 1
    assert out[0].url == "http://ap.com/ru1"  # highest credibility kept


def test_keeps_distinct_non_latin_stories():
    # Two genuinely different Russian stories sharing only a topic word must
    # not be merged — the non-Latin analog of test_keeps_distinct_stories_sharing_a_topic_word.
    items = [
        _a("http://a.com/ru2", "Правительство объявило новый пакет экономических реформ на следующий год", 0.3),
        _a("http://b.com/ru3", "Оппозиция раскритиковала бюджетный план правительства перед голосованием в парламенте", 0.3),
    ]
    out = _dedupe(items)
    assert len(out) == 2
