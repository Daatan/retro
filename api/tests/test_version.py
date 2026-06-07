"""Tests for version & build provenance (/version, /health, _build.build_info)."""
import json

from fastapi.testclient import TestClient

from forecast_api import _build
from forecast_api.main import app

client = TestClient(app)


def test_version_endpoint():
    r = client.get("/version")
    assert r.status_code == 200
    body = r.json()
    for key in ("version", "base_version", "build", "git_sha", "git_branch", "built_at", "source"):
        assert key in body
    assert body["version"]  # non-empty
    assert body["source"] in {"deploy", "git", "env", "unknown"}
    # when a build number is known, version composes base + build (autoincrement)
    if body["build"] is not None:
        assert body["version"] == f"{body['base_version']}+build.{body['build']}"
    else:
        assert body["version"] == body["base_version"]


def test_health_includes_provenance():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    # additive fields — existing consumers unaffected
    assert "git_sha" in body and "git_branch" in body and "built_at" in body
    assert body["version"]


def test_openapi_version_matches_build_info():
    assert app.version == _build.build_info()["version"]


def test_build_info_prefers_file_and_composes_build(tmp_path, monkeypatch):
    _build.build_info.cache_clear()
    f = tmp_path / "_build_info.json"
    f.write_text(json.dumps({"git_sha": "abc1234", "git_branch": "main", "build": 348,
                             "built_at": "2026-06-07T00:00:00Z", "source": "deploy"}))
    monkeypatch.setattr(_build, "_BUILD_FILE", f)
    info = _build.build_info()
    assert info["git_sha"] == "abc1234"
    assert info["source"] == "deploy"
    assert info["build"] == 348
    assert info["version"] == f"{info['base_version']}+build.348"
    _build.build_info.cache_clear()


def test_build_number_autoincrements_with_commit_count(monkeypatch):
    """The git fallback derives the build number from the commit count."""
    _build.build_info.cache_clear()
    monkeypatch.setattr(_build, "_BUILD_FILE", _build._BUILD_FILE.with_name("nope.json"))
    info = _build.build_info()
    # local checkout has git -> build is a positive int and version embeds it
    assert info["source"] == "git"
    assert isinstance(info["build"], int) and info["build"] > 0
    assert info["version"].endswith(f"+build.{info['build']}")
    _build.build_info.cache_clear()


def test_build_info_falls_back_gracefully(tmp_path, monkeypatch):
    """No file, no git, no env -> 'unknown', never raises; version still present."""
    _build.build_info.cache_clear()
    monkeypatch.setattr(_build, "_BUILD_FILE", tmp_path / "missing.json")
    monkeypatch.setattr(_build, "_from_git", lambda: None)
    monkeypatch.delenv("GIT_SHA", raising=False)
    info = _build.build_info()
    assert info["git_sha"] == "unknown"
    assert info["source"] == "unknown"
    assert info["version"]  # semver still single-sourced from metadata
    _build.build_info.cache_clear()


def test_build_info_env_fallback(tmp_path, monkeypatch):
    _build.build_info.cache_clear()
    monkeypatch.setattr(_build, "_BUILD_FILE", tmp_path / "missing.json")
    monkeypatch.setattr(_build, "_from_git", lambda: None)
    monkeypatch.setenv("GIT_SHA", "envsha9")
    monkeypatch.setenv("GIT_BRANCH", "feature/x")
    info = _build.build_info()
    assert info["git_sha"] == "envsha9"
    assert info["git_branch"] == "feature/x"
    assert info["source"] == "env"
    _build.build_info.cache_clear()
