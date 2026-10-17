"""retro#947 stage 1 — grouped settings views resolve to exactly the values the flat model does."""
import json
from pathlib import Path

import pytest

from forecast_api.config import ApiSettings
from forecast_api.settings_groups import GROUPS, group_fields, group_of, grouped

SNAPSHOT = Path(__file__).parent / "fixtures" / "api_settings_fields.json"


def test_every_field_is_in_exactly_one_group():
    seen: list[str] = []
    for fields in group_fields().values():
        seen.extend(fields)
    assert sorted(seen) == sorted(ApiSettings.model_fields)
    assert len(seen) == len(set(seen))


def test_field_names_match_the_snapshot():
    """Env var names are the field names upper-cased; renaming or dropping one is a deploy-time
    break. Adding a field is fine — add it to tests/fixtures/api_settings_fields.json."""
    assert sorted(ApiSettings.model_fields) == json.loads(SNAPSHOT.read_text())


def test_groups_match_flat_settings_from_the_same_environment(monkeypatch):
    env = {
        "ORACLE_API_KEY": "k", "JEV_GATE_ENABLED": "true", "JEV_GATE_THRESHOLD": "0.21",
        "SETTLEMENT_MIN_SOURCES": "7", "POOL_SHADOW_A_ENABLED": "true",
        "EXTRACTION_MEMO_TTL_HOURS": "3.5", "DATA_DIR": "/x/data", "PORT": "9001",
    }
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    s = ApiSettings(_env_file=None)
    g = grouped(s)
    for group, fields in group_fields().items():
        view = getattr(g, group)
        for f in fields:
            assert getattr(view, f) == getattr(s, f), f
    assert g.jev.jev_gate_enabled is True and g.jev.jev_gate_threshold == 0.21
    assert g.settlement.settlement_min_sources == 7
    assert str(g.caches.data_dir) == "/x/data"


def test_view_is_read_only_and_rejects_unknown_fields(monkeypatch):
    monkeypatch.setenv("ORACLE_API_KEY", "k")
    g = grouped(ApiSettings(_env_file=None))
    with pytest.raises(AttributeError):
        g.jev.jev_gate_enabled = True
    with pytest.raises(AttributeError):
        g.jev.not_a_field
    assert set(GROUPS) == set(group_fields())
    assert group_of("jev_gate_enabled") == "jev" and group_of("something_new") == "core"
