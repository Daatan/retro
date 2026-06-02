"""Test bootstrap.

``ApiSettings`` requires ``ORACLE_API_KEY`` and validates it at import time
(``config.py`` builds ``settings`` on import). Set a default so the suite is
runnable locally and in CI without a real secret. ``setdefault`` leaves any
real value (e.g. set by the developer) untouched.
"""

import os

os.environ.setdefault("ORACLE_API_KEY", "test-key")
