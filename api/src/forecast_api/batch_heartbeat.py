"""Batch-loop heartbeat, surfaced on ``/health`` as ``batch_loop`` (retro#996).

``infra/ec2_run.sh`` rewrites ``<data_dir>/heartbeat.json`` (untracked) at the end of
every cycle. daatan's evidence-health cron reads it from here to tell a dead or
wedged loop from an idle one. It used to infer liveness from atlas commits on main,
but since retro#959 an idle loop with a drained queue makes no commits at all, so
"no commit for 12h" stopped meaning "dead".

``/health`` is unauthenticated, so only whitelisted fields go out: timestamps, the
sha the cycle ran on, and cell counts. Never raises — a liveness probe must not
500 because of a side file; anything unreadable is reported as ``None``.
"""
import json
from pathlib import Path
from typing import Any, Optional

HEARTBEAT_FILENAME = "heartbeat.json"

_STR_FIELDS = ("last_cycle_at", "last_cycle_ok_at", "head", "oldest_unpushed_at")
_CELL_FIELDS = ("done", "no_predictions", "failed", "pending")


def read_batch_heartbeat(data_dir: Path) -> Optional[dict[str, Any]]:
    try:
        raw = json.loads((data_dir / HEARTBEAT_FILENAME).read_text())
    except Exception:
        return None
    if not isinstance(raw, dict):
        return None

    out: dict[str, Any] = {}
    for key in _STR_FIELDS:
        val = raw.get(key)
        out[key] = val if isinstance(val, str) and val else None
    ok = raw.get("last_cycle_ok")
    out["last_cycle_ok"] = ok if isinstance(ok, bool) else None
    cells = raw.get("cells")
    if isinstance(cells, dict):
        out["cells"] = {
            k: cells[k] for k in _CELL_FIELDS if isinstance(cells.get(k), int) and not isinstance(cells.get(k), bool)
        }
    else:
        out["cells"] = None
    return out
