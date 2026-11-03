"""JSONL ledger writes shared by resolution_feedback and settlement_pin_ledger.

The ledgers are append-only except for one case: an admin reverting a mistaken
resolution (daatan#2077) removes its line. Removal rewrites the file and swaps it in,
so an append from another gunicorn worker that opened the old file would be lost.
Every write therefore takes an exclusive lock on a sidecar file, which survives the
swap, unlike a lock on the ledger itself.
"""
import fcntl
import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)


@contextmanager
def locked(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_name(f".{path.name}.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def append_line(path: Path, line: str) -> None:
    with locked(path), path.open("a") as f:
        f.write(line + "\n")


def _matches(line: str, prediction_id: str) -> bool:
    try:
        return json.loads(line).get("prediction_id") == prediction_id
    except (json.JSONDecodeError, AttributeError):
        logger.warning("Keeping malformed ledger line while removing %s", prediction_id)
        return False


def remove_prediction_lines(path: Path, prediction_id: str) -> int:
    """Drop this prediction's lines. Returns how many; other lines stay byte-identical."""
    with locked(path):
        if not path.exists():
            return 0
        lines = [line for line in path.read_text().splitlines() if line.strip()]
        keep = [line for line in lines if not _matches(line, prediction_id)]
        removed = len(lines) - len(keep)
        if removed:
            tmp = path.with_name(f".{path.name}.tmp")
            tmp.write_text("".join(line + "\n" for line in keep))
            tmp.replace(path)
        return removed
