"""Grouped, read-only views over ``ApiSettings`` (retro#947, stage 1).

``config.ApiSettings`` is one flat ~145-field model. This module names the groups its fields
fall into (``jev``, ``settlement``, ``resolution``, …) without touching the model: env var
names, defaults and every ``settings.<field>`` call site stay as they are. A view is a frozen
snapshot built from a settings instance, so ``grouped(settings).jev.jev_gate_enabled`` and
``settings.jev_gate_enabled`` are the same value by construction.

Membership is by field-name prefix, first matching rule wins, anything unmatched is ``core``.
``tests/test_settings_groups.py`` pins the partition (every field in exactly one group), the
field-name snapshot (an env var cannot be renamed or dropped by accident) and old-vs-new
equality under a real environment. Stage 2 — moving call sites onto the groups and splitting the
model itself — is deliberately not done here.
"""
from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping

from .config import ApiSettings

# (group, prefixes). Order matters: the first group with a matching prefix owns the field.
GROUP_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("auth", ("oracle_api_key", "cognito", "mcp", "host", "port", "per_")),
    ("jev", ("jev", "typesafe")),
    ("settlement", ("settlement", "settled")),
    ("resolution", ("resolution",)),
    ("subject", ("subject", "premise", "precursor")),
    ("evidence", ("evidence", "relevance", "recency", "cluster", "pool", "source", "max_source",
                  "harmonic", "min_stratum", "thin", "defer", "decisiveness", "logit",
                  "syndication", "hazard", "event", "conditional", "confusion")),
    ("extraction", ("extraction", "extractor", "judge", "reuse", "inject", "anchor", "query")),
    ("caches", ("cache", "data_dir", "leaderboard", "forecast_cache")),
    ("bots", ("polymarket", "manifold")),
    ("search", ("search", "retry", "degraded")),
)
GROUPS: tuple[str, ...] = tuple(g for g, _ in GROUP_RULES) + ("core",)


def group_of(field: str) -> str:
    for group, prefixes in GROUP_RULES:
        if field.startswith(prefixes):
            return group
    return "core"


def group_fields() -> dict[str, tuple[str, ...]]:
    out: dict[str, list[str]] = {g: [] for g in GROUPS}
    for f in ApiSettings.model_fields:
        out[group_of(f)].append(f)
    return {g: tuple(fs) for g, fs in out.items()}


class SettingsGroup:
    """Frozen attribute view of one group's fields, keyed by the unchanged field names."""

    __slots__ = ("_name", "_values")

    def __init__(self, name: str, values: Mapping[str, Any]) -> None:
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_values", MappingProxyType(dict(values)))

    def __getattr__(self, item: str) -> Any:
        try:
            return self._values[item]
        except KeyError:
            raise AttributeError(f"{self._name} settings group has no field {item!r}") from None

    def __setattr__(self, key: str, value: Any) -> None:
        raise AttributeError("settings groups are read-only snapshots")

    def as_dict(self) -> dict[str, Any]:
        return dict(self._values)

    def __repr__(self) -> str:
        return f"SettingsGroup({self._name}, {len(self._values)} fields)"


class GroupedSettings:
    __slots__ = tuple(GROUPS)

    def __init__(self, source: ApiSettings) -> None:
        for group, fields in group_fields().items():
            object.__setattr__(self, group, SettingsGroup(group, {f: getattr(source, f) for f in fields}))


def grouped(source: ApiSettings) -> GroupedSettings:
    return GroupedSettings(source)
