"""retro#947 item 2: report ignored env vars at startup, never change values."""
from forecast_api import settings_deprecations as sd
from forecast_api.config import ApiSettings

FIELDS = list(ApiSettings.model_fields)


def test_every_real_field_env_name_is_recognised():
    env = {f.upper(): "x" for f in FIELDS}
    assert sd.unrecognized_env(env, FIELDS) == []


def test_typo_in_own_prefix_is_reported():
    env = {"JEV_GATE_ENFORCED": "true", "JEV_GATE_ENFORCE": "true"}
    assert sd.unrecognized_env(env, FIELDS) == ["JEV_GATE_ENFORCED"]


def test_foreign_variables_are_never_flagged():
    env = {"PATH": "/bin", "AWS_REGION": "eu-central-1", "OPENROUTER_API_KEY": "k", "HOME": "/root"}
    assert sd.unrecognized_env(env, FIELDS) == []


def test_matching_is_case_insensitive():
    assert sd.unrecognized_env({"jev_gate_enforced": "1"}, FIELDS) == ["jev_gate_enforced"]
    assert sd.unrecognized_env({"jev_gate_enforce": "1"}, FIELDS) == []


def test_registered_deprecation_is_reported_once_not_as_unrecognized(monkeypatch):
    monkeypatch.setitem(sd.DEPRECATED_ENV, "JEV_OLD_FLAG", "use JEV_NEW_FLAG")
    env = {"jev_old_flag": "1"}
    assert sd.deprecated_env(env) == [("JEV_OLD_FLAG", "use JEV_NEW_FLAG")]
    assert sd.unrecognized_env(env, FIELDS) == []


def test_reporting_does_not_mutate_the_environment():
    env = {"JEV_TYPO": "1"}
    sd.unrecognized_env(env, FIELDS)
    sd.deprecated_env(env)
    assert env == {"JEV_TYPO": "1"}
