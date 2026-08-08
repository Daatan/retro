"""Unit tests for parse_node_observations() — the comma-separated `NODE=prob`
parser shared by the /bayes/nodes route and the bayes_nodes MCP tool (retro#439:
previously copy-pasted independently in both, with zero direct test coverage
of the parsing logic itself in either)."""
from forecast_api.bayesoracle import parse_node_observations


def test_empty_string_returns_empty_dict():
    assert parse_node_observations("") == {}


def test_parses_single_pair():
    assert parse_node_observations("ELECTIONS=0.95") == {"ELECTIONS": 0.95}


def test_parses_multiple_pairs():
    assert parse_node_observations("ELECTIONS=0.95,TRUMP=0.70") == {
        "ELECTIONS": 0.95,
        "TRUMP": 0.70,
    }


def test_strips_whitespace_and_uppercases_node_id():
    assert parse_node_observations(" elections = 0.5 , trump=0.2") == {
        "ELECTIONS": 0.5,
        "TRUMP": 0.2,
    }


def test_skips_parts_missing_equals():
    assert parse_node_observations("ELECTIONS=0.95,GARBAGE,TRUMP=0.70") == {
        "ELECTIONS": 0.95,
        "TRUMP": 0.70,
    }


def test_skips_non_numeric_values():
    assert parse_node_observations("ELECTIONS=not-a-number,TRUMP=0.70") == {
        "TRUMP": 0.70,
    }


def test_trailing_comma_is_ignored():
    assert parse_node_observations("ELECTIONS=0.95,") == {"ELECTIONS": 0.95}
