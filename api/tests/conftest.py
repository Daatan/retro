"""Test bootstrap.

``ApiSettings`` requires ``ORACLE_API_KEY`` and validates it at import time
(``config.py`` builds ``settings`` on import). Set a default so the suite is
runnable locally and in CI without a real secret. ``setdefault`` leaves any
real value (e.g. set by the developer) untouched.
"""

import os
import tempfile

os.environ.setdefault("ORACLE_API_KEY", "test-key")

# The forecast cache is disk-backed and shared across processes (retro#405) —
# exactly the property that makes the REAL directory poisonous under test: a
# previous pytest run's entries would satisfy this run's pipeline calls. Every
# run gets a fresh directory. Must happen before forecast_api imports, because
# cache.py builds its singleton at import time.
os.environ.setdefault("CACHE_DIR", tempfile.mkdtemp(prefix="oracle-forecast-cache-test-"))

# The settlement match gate (retro#388) ships enabled so it can shadow in prod.
# In the suite it must be OFF by default: it fires on every pinning fixture, and
# a test run should not depend on Bedrock being reachable (it fails open, so the
# assertions still hold — but each attempt costs a real timeout). Tests that
# exercise the gate turn it on explicitly.
os.environ.setdefault("SETTLEMENT_VERIFIER_ENABLED", "false")

# The verdict store (retro#532) also ships enabled, and is also OFF under test
# by default, for two reasons of its own: suites assert on verify-call counts,
# which a cross-test cache hit would silently satisfy; and the default path
# lives under data_dir, which does not exist on a dev box. Cache tests turn it
# on explicitly against a tmp_path store. Single-shot sampling for the same
# call-count reason; majority tests raise it explicitly.
os.environ.setdefault("SETTLEMENT_VERDICT_CACHE_ENABLED", "false")
os.environ.setdefault("SETTLEMENT_VERIFIER_VOTES", "1")

# The event decomposition store (retro#758) — same reasons as the verdict store
# above: its default path lives under data_dir, which does not exist on a dev
# box, and a cross-test cache hit would silently satisfy a call-count assertion.
# test_event_decomposition_store.py exercises the store directly against a
# tmp_path, unaffected by this.
os.environ.setdefault("EVENT_DECOMPOSITION_CACHE_ENABLED", "false")
