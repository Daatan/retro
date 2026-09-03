#!/usr/bin/env python3
"""
Direct-invocation driver for bayesoracle/. See SKILL.md "Run: bayesoracle".

bayesoracle/ is NOT a standalone service (no FastAPI/uvicorn anywhere in its
pyproject.toml — only numpy/scipy). Its one live HTTP surface, GET /bayes/nodes,
is served by api/'s FastAPI app (api/src/forecast_api/bayesoracle.py), which is
exercised by driver-api.sh's smoke command. This script drives the actual
inference engine (core.py) directly, the "library" pattern: load the checked-in
graph JSON, propagate, print real node probabilities. Pure numpy/scipy
computation over local JSON files — no network, no LLM, no credentials needed.

Run: cd bayesoracle && uv run python ../.claude/skills/run-retro/driver-bayesoracle.py
"""
import sys
from pathlib import Path

# bayesoracle/ is not an installed package (no pyproject entry for it as a lib) —
# this driver must be run with cwd=bayesoracle/ so `import core` resolves. Check
# and fail with a clear message *before* the import, rather than letting a
# wrong-cwd run die on a bare ModuleNotFoundError.
if not (Path.cwd() / "core.py").exists():
    print(
        "Run this from bayesoracle/: "
        "cd bayesoracle && uv run python ../.claude/skills/run-retro/driver-bayesoracle.py",
        file=sys.stderr,
    )
    sys.exit(1)

sys.path.insert(0, str(Path.cwd()))

import core  # noqa: E402


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> None:
    here = Path.cwd()

    section("1. Load + validate graph_political.json, propagate with no overrides")
    g = core.load_graph(here / "graph_political.json")
    nodes = g.compute_nodes()
    print(f"{len(nodes)} nodes loaded. First 3:")
    for n in nodes[:3]:
        print(f"  {n['id']:20s} prior={n['prior']:.3f} p={n['p']:.3f} layer={n['layer']}")

    section("2. Same graph with an observation override locking a node")
    target_id = nodes[0]["id"]
    locked_nodes = g.compute_nodes({target_id: 0.99})
    changed = [n for n in locked_nodes if n["id"] != target_id and n.get("delta")]
    print(f"Locked {target_id}=0.99; {len(changed)} downstream node(s) shifted (delta != 0).")
    for n in changed[:5]:
        print(f"  {n['id']:20s} p={n['p']:.3f} delta={n['delta']:+.3f}")

    section("3. Load + validate graph_pm.json (Polymarket-backed DAG)")
    g_pm = core.load_graph(here / "graph_pm.json")
    pm_nodes = g_pm.compute_nodes()
    print(f"{len(pm_nodes)} nodes loaded from graph_pm.json.")

    print("\nAll driver stages completed. No real network or LLM calls were made.")
    print(
        "NOT run here (need real infra/credentials): calibrate_edges.py (Bedrock + "
        "news search, ~85min/28 edges), fetch_node_history.py (Polymarket API), "
        "series/log_nodes.py (calls the live oracle.daatan.com /forecast with a "
        "real ORACLE_API_KEY)."
    )


if __name__ == "__main__":
    main()
