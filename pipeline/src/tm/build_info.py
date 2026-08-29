"""Which Oracle build the batch lane is running (retro#744).

Vault extraction JSONs record ``prompt_version`` and the two model ids, but
until now not the *code* that produced them. The API tree stamps this on every
response (``forecast_api._build``); the batch tree is a separate checkout that
self-commits (docs/ORACLE_DEPLOY.md), so it resolves its own identity here,
from its own repo root, once per process.

Composed the same way as ``/version``: ``"{base}+build.{count}"`` where base is
the ``forecast-api`` package version (single-sourced from ``api/pyproject.toml``)
and count is ``git rev-list --count HEAD``. Never raises — ``"unknown"`` / ``None``
on any failure, because a stamp must not be able to stop an extraction.

NOT a skip key: ``_negative_marker_is_current`` and the runner's cache key
compare ``prompt_version`` + models only. A redeploy must not re-extract.
"""
from __future__ import annotations

import re
import subprocess
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parents[3]  # tm -> src -> pipeline -> repo
_API_PYPROJECT = _REPO_ROOT / "api" / "pyproject.toml"


def _git(*args: str) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", *args], cwd=_REPO_ROOT, capture_output=True, text=True, timeout=2, check=True,
        ).stdout.strip()
        return out or None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def _base_version() -> str:
    try:
        return _pkg_version("forecast-api")
    except PackageNotFoundError:
        pass
    try:
        m = re.search(r'^version\s*=\s*"([^"]+)"', _API_PYPROJECT.read_text(), re.M)
        if m:
            return m.group(1)
    except OSError:
        pass
    return "unknown"


@lru_cache(maxsize=1)
def git_sha() -> Optional[str]:
    """Full HEAD sha of the tree this process runs from, or None."""
    return _git("rev-parse", "HEAD")


@lru_cache(maxsize=1)
def oracle_version() -> str:
    """``"{base}+build.{n}"`` like ``/version``; ``"{base}"`` alone when the
    commit count is unavailable; ``"unknown"`` when nothing resolves."""
    base = _base_version()
    count = _git("rev-list", "--count", "HEAD")
    if count and count.isdigit():
        return f"{base}+build.{count}"
    return base
