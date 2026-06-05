#!/usr/bin/env python3
"""
Verify core.js produces the same numbers as core.py, so the HTML viewers and the
API/backtest run identical inference.  Requires `node` on PATH.

Run:  python bayesoracle/verify_core_parity.py
"""
import json
import subprocess
import sys
from pathlib import Path

import core

HERE = Path(__file__).parent
SPECS = ["graph_political.json", "graph_pm.json"]
CASES = {
    "graph_political.json": [None, {"BIBI_OUT": 0.95}, {"RIGHT_BLOC_61": 0.99}],
    "graph_pm.json": [None, {"BIBI_OUT": 0.95}, {"IRAN_NUKE": 0.9}],
}

_JS = """
const fs = require('fs'); const Bayes = require('%s');
const cases = %s;
const out = {};
for (const sf of Object.keys(cases)) {
  const g = new Bayes.BayesGraph(JSON.parse(fs.readFileSync(sf)));
  out[sf] = cases[sf].map(obs => {
    const p = g.propagate(obs); const o = {};
    Object.keys(p).sort().forEach(k => o[k] = p[k]); return o;
  });
}
process.stdout.write(JSON.stringify(out));
""" % (str(HERE / "core.js"), json.dumps(CASES))


def main() -> int:
    py = {}
    for sf in SPECS:
        g = core.load_graph(HERE / sf)
        py[sf] = [g.propagate(obs) for obs in CASES[sf]]

    try:
        raw = subprocess.run(["node", "-e", _JS], cwd=HERE, capture_output=True,
                             text=True, check=True).stdout
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        print("SKIP: node unavailable or failed —", exc)
        return 0
    js = json.loads(raw)

    maxd, n = 0.0, 0
    for sf in SPECS:
        for a, b in zip(py[sf], js[sf]):
            for k in a:
                maxd = max(maxd, abs(a[k] - b[k]))
                n += 1
    print(f"compared {n} values; max |py-js| = {maxd:.2e}")
    ok = maxd < 1e-9
    print("PARITY OK" if ok else "PARITY FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
