import tempfile

import pytest


@pytest.fixture(autouse=True, scope="session")
def _isolate_tempdir(tmp_path_factory):
    """Point tempfile at a per-run directory for the whole session.

    web_search resolves `_QUOTA_STATE_PATH` from `tempfile.gettempdir()` at import time,
    and several tests trip a quota flag, which persists it. Without this, the suite writes
    a real `/tmp/quota_exhausted.json` on the developer's (or CI's) machine — and because
    `_fresh_ws()` re-imports the module, a file left behind by one run silently changes the
    starting state of the next one. Redirecting tempdir keeps that state per-run.
    """
    original = tempfile.tempdir
    tempfile.tempdir = str(tmp_path_factory.mktemp("tm-tmp"))
    yield
    tempfile.tempdir = original
