from universe import DEFAULT_EVENT_SLUGS, event_slugs_from_env


def test_default_when_unset():
    assert event_slugs_from_env(None) == DEFAULT_EVENT_SLUGS
    assert event_slugs_from_env("  ") == DEFAULT_EVENT_SLUGS


def test_override():
    assert event_slugs_from_env("a, b ,,c") == ("a", "b", "c")


def test_default_universe_is_the_israeli_election_cluster():
    assert all("israel" in s or "knesset" in s for s in DEFAULT_EVENT_SLUGS)
    assert len(DEFAULT_EVENT_SLUGS) == 10
