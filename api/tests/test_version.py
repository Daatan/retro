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
    for key in ("version", "git_sha", "git_branch", "built_at", "source"):
        assert key in body
    assert body["version"]  # non-empty semver
    assert body["source"] in {"deploy", "git", "env", "unknown"}


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


def test_build_info_prefers_file(tmp_path, monkeypatch):
    _build.build_info.cache_clear()
    f = tmp_path / "_build_info.json"
    f.write_text(json.dumps({"git_sha": "abc1234", "git_branch": "main",
                             "built_at": "2026-06-07T00:00:00Z", "source": "deploy"}))
    monkeypatch.setattr(_build, "_BUILD_FILE", f)
    info = _build.build_info()
    assert info["git_sha"] == "abc1234"
    assert info["source"] == "deploy"
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
