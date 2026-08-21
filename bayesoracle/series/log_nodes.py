#!/usr/bin/env python3
"""Daily Oracle series for the bayesoracle graph node questions (retro#577).

Asks the (v1) Oracle ``POST /forecast`` every node question in ``questions.json``
and appends one JSON line per node per UTC day to a JSONL file, so that v2 can
later be scored *paired* against v1 (design note §8) and first-difference
co-movement between nodes can be measured.

    python series/log_nodes.py --out /var/lib/oracle-series/nodes.jsonl

Idempotent per day: node ids already present for today's date are skipped, so
re-running after a partial failure only fills the gaps. Calls are sequential
with a small sleep (rate, not burst). The API key is read from the environment
(``ORACLE_API_KEY``) or from a ``.env`` file — never hardcoded.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable

HERE = Path(__file__).resolve().parent
QUESTIONS_FILE = HERE / "questions.json"
DEFAULT_ENV_FILES = (
    HERE.parent.parent / ".env",                 # repo root (Oracle box batch tree)
    HERE.parent.parent / "pipeline" / ".env",
)
DEFAULT_API = "http://127.0.0.1:8001"
GRAPHS = ("political", "pm", "caseA")


# ─── env ──────────────────────────────────────────────────────────────────────
def load_env(paths=DEFAULT_ENV_FILES) -> None:
    for path in paths:
        if not path.exists():
            continue
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def api_key() -> str:
    key = os.environ.get("ORACLE_API_KEY") or os.environ.get("ORACLE_API_KEYS", "").split(",")[0]
    if not key:
        sys.exit("ORACLE_API_KEY not set (env or .env)")
    return key.strip()


# ─── questions ────────────────────────────────────────────────────────────────
def load_questions(path: Path = QUESTIONS_FILE) -> list[tuple[str, str]]:
    """Return ``[(node_id, question)]`` in file order; node_id = ``<graph>.<id>``.

    Validates that every question is 5–500 chars (the API's bounds) and that
    ids are unique.
    """
    data = json.loads(path.read_text())
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for graph in GRAPHS:
        for nid, q in data.get(graph, {}).items():
            node_id = f"{graph}.{nid}"
            if node_id in seen:
                raise ValueError(f"duplicate node id {node_id}")
            if not isinstance(q, str) or not 5 <= len(q) <= 500:
                raise ValueError(f"{node_id}: question must be 5–500 chars")
            seen.add(node_id)
            out.append((node_id, q))
    return out


def check_graph_coverage(questions: list[tuple[str, str]], graph_dir: Path = HERE.parent) -> list[str]:
    """Node ids present in graph_political/graph_pm but missing a question."""
    have = {n for n, _ in questions}
    missing = []
    for graph, fname in (("political", "graph_political.json"), ("pm", "graph_pm.json")):
        f = graph_dir / fname
        if not f.exists():
            continue
        for node in json.loads(f.read_text())["nodes"]:
            if f"{graph}.{node['id']}" not in have:
                missing.append(f"{graph}.{node['id']}")
    return missing


# ─── JSONL ────────────────────────────────────────────────────────────────────
def logged_today(out: Path, date: str) -> set[str]:
    if not out.exists():
        return set()
    done = set()
    for line in out.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("date") == date:
            done.add(rec["node_id"])
    return done


def make_record(date: str, node_id: str, question: str, resp: dict) -> dict:
    ci = resp.get("ci")
    if isinstance(ci, dict):
        ci = [ci.get("lower", ci.get("low")), ci.get("upper", ci.get("high"))]
    return {
        "date": date,
        "node_id": node_id,
        "question": question,
        "probability": resp.get("probability"),
        "ci": ci,
        "articles_used": resp.get("articles_used"),
        "confidence": resp.get("confidence"),
        "insufficient_data": bool(resp.get("insufficient_data", False)),
        "sources": [s.get("url") for s in resp.get("sources") or [] if isinstance(s, dict) and s.get("url")],
    }


def run(
    questions: list[tuple[str, str]],
    out: Path,
    forecast: Callable[[str], dict],
    date: str | None = None,
    sleep_s: float = 2.0,
    log=print,
) -> int:
    """Append a record for every node not yet logged for ``date``. Returns count written."""
    date = date or dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = logged_today(out, date)
    written = 0
    for i, (node_id, question) in enumerate(questions):
        if node_id in done:
            continue
        try:
            resp = forecast(question)
        except Exception as e:  # keep going; re-run fills the gap
            log(f"[{node_id}] ERROR {type(e).__name__}: {e}")
            continue
        rec = make_record(date, node_id, question, resp)
        with out.open("a") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        written += 1
        log(f"[{node_id}] p={rec['probability']} n={rec['articles_used']} conf={rec['confidence']}")
        if sleep_s and i < len(questions) - 1:
            time.sleep(sleep_s)
    return written


# ─── HTTP ─────────────────────────────────────────────────────────────────────
def http_forecaster(base: str, key: str, max_articles: int = 8, timeout: float = 120.0) -> Callable[[str], dict]:
    import httpx

    client = httpx.Client(base_url=base, timeout=timeout, headers={"x-api-key": key})

    def forecast(question: str) -> dict:
        r = client.post("/forecast", json={"question": question, "max_articles": max_articles})
        r.raise_for_status()
        return r.json()

    return forecast


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path(os.environ.get("ORACLE_SERIES_OUT", "nodes.jsonl")))
    ap.add_argument("--api", default=os.environ.get("ORACLE_API_URL", DEFAULT_API))
    ap.add_argument("--sleep", type=float, default=2.0)
    ap.add_argument("--max-articles", type=int, default=8)
    ap.add_argument("--date", help="override UTC date (YYYY-MM-DD)")
    ap.add_argument("--only", help="comma-separated node ids (e.g. pm.BIBI_PM)")
    args = ap.parse_args(argv)

    load_env()
    questions = load_questions()
    missing = check_graph_coverage(questions)
    if missing:
        print(f"WARNING: graph nodes without a question: {', '.join(missing)}", file=sys.stderr)
    if args.only:
        want = set(args.only.split(","))
        questions = [(n, q) for n, q in questions if n in want]
    forecast = http_forecaster(args.api, api_key(), args.max_articles)
    n = run(questions, args.out, forecast, date=args.date, sleep_s=args.sleep)
    print(f"wrote {n} records to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
