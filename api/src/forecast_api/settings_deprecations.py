"""Startup notices for env vars that ``ApiSettings`` silently ignores (retro#947, item 2).

``ApiSettings`` uses ``extra="ignore"``, so a misspelled or already-removed flag
(``JEV_GATE_ENFORCED`` for ``JEV_GATE_ENFORCE``) is dropped without a trace and the gate stays
at its default. This module only *reports*: it never changes a value, never raises, and the
caller wraps it so a bug here cannot stop the service from starting.

Two reports:
  * ``DEPRECATED_ENV`` — env vars we removed or renamed on purpose, with a one-line hint.
    Add an entry in the same PR that drops or renames a field.
  * ``unrecognized_env`` — env vars that start with one of this service's own prefixes but match
    no field (a typo, or a flag removed without a registry entry). Only the prefixes below are
    checked, so unrelated variables (``PATH``, ``AWS_*``) are never flagged.
"""
from __future__ import annotations

from typing import Iterable, Mapping

# env var (upper case) -> what to do instead. Empty on purpose: #933 removed
# ``openrouter_api_key`` from the api, but the shared env still feeds ``tm.config``, so warning
# on it would be noise. Add an entry only when the variable is dead everywhere.
DEPRECATED_ENV: dict[str, str] = {}

# Prefixes that belong to this service's own settings. Deliberately narrow: a false positive is
# only a log line, but a noisy one trains people to ignore it.
OWN_PREFIXES: tuple[str, ...] = (
    "JEV_", "TYPESAFE_", "SETTLEMENT_", "RESOLUTION_", "SUBJECT_", "EVIDENCE_", "EXTRACTION_",
    "EXTRACTOR_", "POLYMARKET_", "MANIFOLD_", "COGNITO_", "MCP_", "ORACLE_",
)


def deprecated_env(environ: Mapping[str, str]) -> list[tuple[str, str]]:
    return sorted((k, hint) for k, hint in DEPRECATED_ENV.items() if k in {e.upper() for e in environ})


def unrecognized_env(environ: Mapping[str, str], fields: Iterable[str]) -> list[str]:
    known = {f.upper() for f in fields}
    return sorted(
        name for name in environ
        if name.upper().startswith(OWN_PREFIXES)
        and name.upper() not in known
        and name.upper() not in DEPRECATED_ENV
    )
