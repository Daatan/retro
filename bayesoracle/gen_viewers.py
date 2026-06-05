#!/usr/bin/env python3
"""
Emit *.data.js for the HTML viewers so the JSON graph specs stay the single
source of truth.  The viewers load these via <script src> (which works over
file://, unlike fetch of a local .json).

Run after editing graph_political.json or regenerating graph_pm.json:
    python bayesoracle/gen_viewers.py
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
TARGETS = {
    "graph_political.json": ("graph_political.data.js", "GRAPH_POLITICAL"),
    "graph_pm.json": ("graph_pm.data.js", "GRAPH_PM"),
}


def main() -> None:
    for src, (out, var) in TARGETS.items():
        spec = json.loads((HERE / src).read_text())
        js = (f"// AUTO-GENERATED from {src} by gen_viewers.py — do not edit.\n"
              f"window.{var} = {json.dumps(spec, indent=2)};\n")
        (HERE / out).write_text(js)
        print(f"wrote {out}  (window.{var}, {len(spec['nodes'])} nodes)")


if __name__ == "__main__":
    main()
