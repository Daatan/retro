"""
BayesOracle API adapter.

The graph definition and inference engine now live in ``bayesoracle/`` at the
repo root (``core.py`` + ``graph_political.json``) — a single source of truth
shared by the offline viewers, the backtest harness and this endpoint.  This
module is a thin adapter: it loads that graph once and exposes ``compute_nodes``
with the same contract ``main.py`` already depends on.

Previously this file carried a hand-synced *copy* of the node/edge tables and a
third propagation implementation; both are gone.
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional

# bayesoracle/ lives at the repo root: api/src/forecast_api/ -> parents[3] == repo
_BAYES_DIR = Path(__file__).resolve().parents[3] / "bayesoracle"
if str(_BAYES_DIR) not in sys.path:
    sys.path.insert(0, str(_BAYES_DIR))

import core  # noqa: E402  (resolved via _BAYES_DIR)

_GRAPH_PATH = _BAYES_DIR / "graph_political.json"


@lru_cache(maxsize=1)
def _graph() -> "core.Graph":
    return core.load_graph(_GRAPH_PATH)


def compute_nodes(observations: Optional[dict[str, float]] = None) -> list[dict]:
    """
    Compute BayesOracle node probabilities for the Israeli-politics DAG.

    observations: optional {node_id: probability} overrides (locked nodes).
    Returns a list of {id, label, layer, prior, p, delta, locked} sorted by
    layer then id — unchanged from the previous contract.
    """
    return _graph().compute_nodes(observations)
