"""
Version & build provenance for the Oracle API.

Resolves *what code is running* once at import (no per-request shelling), in this
order — first hit wins for the commit fields:

  1. ``_build_info.json`` next to this module — written by infra/deploy_oracle.sh
     at deploy time (authoritative in production).
  2. runtime ``git`` in the repo checkout (covers local/dev where the file is absent).
  3. env vars ``GIT_SHA`` / ``GIT_BRANCH`` (CI / systemd injection).
  4. ``"unknown"`` fallback.

The semver always comes from the installed package metadata (single source =
pyproject ``[project].version``); ``source`` records which tier supplied the
commit info, for debugging "why does it say unknown".
"""
from __future__ import annotations

import json
import os
import subprocess
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from pathlib import Path

_BUILD_FILE = Path(__file__).with_name("_build_info.json")
_REPO_ROOT = Path(__file__).resolve().parents[3]  # forecast_api -> src -> api -> repo


def _pkg_semver() -> str:
    try:
        return _pkg_version("forecast-api")
    except PackageNotFoundError:
        return "0.0.0+unknown"


def _from_file() -> dict | None:
    try:
        data = json.loads(_BUILD_FILE.read_text())
    except (OSError, ValueError):
        return None
    if not data.get("git_sha"):
        return None
    data.setdefault("source", "deploy")
    return data


def _from_git() -> dict | None:
    def _git(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args], cwd=_REPO_ROOT, capture_output=True,
                text=True, timeout=2, check=True,
            ).stdout.strip()
            return out or None
        except (OSError, subprocess.SubprocessError):
            return None

    sha = _git("rev-parse", "HEAD")
    if not sha:
        return None
    return {"git_sha": sha, "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "built_at": None, "source": "git"}


def _from_env() -> dict | None:
    sha = os.environ.get("GIT_SHA")
    if not sha:
        return None
    return {"git_sha": sha, "git_branch": os.environ.get("GIT_BRANCH"),
            "built_at": os.environ.get("BUILT_AT"), "source": "env"}


@lru_cache(maxsize=1)
def build_info() -> dict:
    """Return {version, git_sha, git_branch, built_at, source}. Never raises."""
    info = _from_file() or _from_git() or _from_env() or {
        "git_sha": "unknown", "git_branch": None, "built_at": None, "source": "unknown",
    }
    # version is always single-sourced from package metadata, regardless of tier.
    info["version"] = _pkg_semver()
    return {
        "version": info["version"],
        "git_sha": info.get("git_sha", "unknown"),
        "git_branch": info.get("git_branch"),
        "built_at": info.get("built_at"),
        "source": info.get("source", "unknown"),
    }
