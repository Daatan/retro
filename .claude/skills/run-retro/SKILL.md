---
name: run-retro
description: Build, run, and drive retro's three systems — the Oracle API (api/), the TruthMachine pipeline (pipeline/), and BayesOracle (bayesoracle/). Use when asked to start, run, build, test, curl, or smoke test the Oracle, the oracle API, the pipeline, TruthMachine, or BayesOracle, or to run any of retro's pytest suites.
---

`retro` is one repo with three independent Python (`uv`) projects, each with its own
`pyproject.toml`/`uv.lock`: `api/` (a real FastAPI service — drive it with
`driver-api.sh`, background-launch + curl), `pipeline/` (a batch library with no
CLI/server — drive it with `driver-pipeline.py`, direct function invocation), and
`bayesoracle/` (a numpy/scipy inference library, also no server — drive it with
`driver-bayesoracle.py`, direct function invocation). All three driver scripts live
next to this file. All paths below are relative to the repo root unless stated.

**Safety boundary — read before running anything.** The production Oracle talks to
real AWS Bedrock (costs money) and, in principle, could touch real data if pointed at
a live host. This repo has **no local/test Postgres dependency at all** — storage is
flat JSON files (`leaderboard.json`, `data/` tree) — so there is nothing to "not
connect to" there. The only real hazard is the LLM boundary. Every driver in this
skill either avoids the LLM boundary entirely (pure functions, or a genuinely
zero-LLM code path) or patches it with `unittest.mock.AsyncMock`, the same pattern
`pipeline/tests/` already uses — never a real Bedrock/OpenRouter call. As a defense in
depth, every driver command below also sets `AWS_ACCESS_KEY_ID=invalid
AWS_SECRET_ACCESS_KEY=invalid`, which (verified below) makes any accidental AWS call
fail fast (`UnrecognizedClientException`) instead of silently using ambient
credentials that may exist on the host. Never put a real key in `.env` or in this
skill — placeholders only.

## Prerequisites

`uv` (already on PATH in this environment; `uv --version` -> `uv 0.9.6` when this was
written). Nothing else — no system packages were needed for any of the three projects.

## Setup

`pipeline/` is a local editable dependency of `api/` (`[tool.uv.sources]
truthmachine-pipeline = { path = "../pipeline", editable = true }`), so sync it first.
Each project has a committed `uv.lock`; CI (`.github/workflows/tests.yml`) uses
`--frozen`, so this skill does too:

```bash
cd pipeline && uv sync --frozen --extra dev
cd ../api && uv sync --frozen --extra dev
cd ../bayesoracle && uv sync --frozen --extra dev
```

No `.env` file is required for anything in this skill. `api/tests/conftest.py`
auto-injects a dummy `ORACLE_API_KEY=test-key` for pytest; the `api/` driver below
sets its own dummy key for the live server. `pipeline/.env.example` and
`api/.env.example` document the *real* env vars (`OPENROUTER_API_KEY`,
`ORACLE_API_KEY`, search-provider keys) needed only for the production-credential
paths this skill deliberately does not run.

## Run: api

The Oracle FastAPI service (`forecast_api` package). Real production port is 8001
(`gunicorn` in prod; bare `uvicorn` for local dev — same port, per
`infra/oracle-api.service`).

```bash
.claude/skills/run-retro/driver-api.sh start   # background-launch, poll /health, print readiness
.claude/skills/run-retro/driver-api.sh smoke   # curl the endpoints that need no live Bedrock/network
.claude/skills/run-retro/driver-api.sh stop    # kill by captured PID, then by port as a fallback
```

`start` sets `ORACLE_API_KEY=dev-local-key` (override with `ORACLE_API_KEY=...
driver-api.sh start`), `DATA_DIR=/tmp/retro-driver-api-data` (empty — the app fails
open on a missing `leaderboard.json`, logging a warning and serving 0 sources rather
than erroring), and the fake-AWS-creds guardrail above. Logs go to
`/tmp/retro-driver-api.log`.

`smoke` hits only endpoints that need no live Bedrock and no real search-provider
keys:

| endpoint | auth | what it proves |
|---|---|---|
| `GET /health` | none | liveness, build info, leaderboard/cache counters |
| `GET /version` | none | build provenance |
| `GET /openapi.json` | none | full API surface is importable and serializes (18 paths at time of writing) |
| `GET /bayes/nodes` | `x-api-key` | real `bayesoracle/core.py` DAG propagation over `graph_political.json`, served through the FastAPI app — this **is** how BayesOracle's one live HTTP surface is actually exercised (see "Run: bayesoracle" below for why there's no separate bayesoracle server) |

Verified real output (this environment; `version`/`build`/path count are a snapshot at
time of writing and will drift on every merge — that's expected, not a regression):

```
{"status":"ok","version":"1.4.0+build.38947", ..., "leaderboard_sources":0, ...}
{"version":"1.4.0+build.38947","base_version":"1.4.0", ...}
18 paths
21 nodes; first: {'id': 'BEYACHAD', 'label': 'Beyachad (opp.) stays unified', 'layer': 0, 'prior': 0.68, 'p': 0.68, 'delta': 0.0, 'locked': False}
```

**Never called here** (need real production credentials/network — do not attempt):
`POST /forecast`, `POST /search`, `POST /llm`, `POST /pool/aggregate`,
`POST /relevance`, `GET /pm/markets` (Bedrock, OpenRouter, search-provider APIs, or
live Polymarket). `POST /fetch-url` is public and SSRF-guarded but was not exercised
here since it fetches an arbitrary live URL.

## Run: pipeline

`pipeline/` (package `tm`) has no CLI and no server — it's driven by direct function
calls, same as its own test suite. The real entrypoints
(`python -m tm.orchestrator {local_file|api}`, `smoke_test.py`) call Bedrock or
OpenRouter for real and must not run in this sandbox.

```bash
cd pipeline && uv run python ../.claude/skills/run-retro/driver-pipeline.py
```

The driver exercises four real code paths, from safest to most representative:

1. **Pure functions, zero LLM/network**: `tm.dedup.simhash` (near-dup fingerprint),
   `tm.extractor.has_conditional_language` (lexical pre-filter).
2. **`tm.gatekeeper.check_is_prediction()` on content-free input** — a genuinely
   unmocked, real call: `carries_proposition()` short-circuits before any LLM call
   for input with no assertable proposition, returning a canned
   `GatekeeperOutput(is_prediction=False, ...)` with `usage` all zeros. No mocking
   needed because the code path never reaches the LLM boundary.
3. **`tm.gatekeeper.check_is_prediction()` on a real article, LLM boundary mocked** —
   patches `tm.gatekeeper.complete_structured` with `unittest.mock.AsyncMock`,
   exactly the pattern in `pipeline/tests/test_gatekeeper_content_free.py`.
4. **`tm.extractor.extract_predictions()`, LLM boundary mocked** the same way,
   returning a typed `ExtractionOutput`/`PredictionExtraction`.

Verified real output (this environment):

```
simhash(...) = 0xc08581c0360a80c9
has_conditional_language('If the budget fails...') = True
has_conditional_language('The Knesset passed...') = False
is_prediction=False reason='Content-free input: the text carries no assertable proposition ...'
usage={'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
is_prediction=True reason='Article contains an explicit forward-looking claim about the budget vote.'
claim='Israeli government predicted to collapse within weeks if budget fails.' stance=-0.8 claim_strength=0.7
All driver stages completed. No real network or LLM calls were made.
```

**Never run here**: `smoke_test.py` (needs real `OPENROUTER_API_KEY`, runs 3 hardcoded
articles through the full LLM pipeline), `python -m tm.orchestrator local_file|api`
(same). `tm.render_atlas` also should not be run casually outside these — it
overwrites the tracked, multi-MB `factum_atlas.html`.

## Run: bayesoracle

**`bayesoracle/` is not a service.** Its `pyproject.toml` depends only on
numpy/scipy — no FastAPI, no `uvicorn`/`gunicorn` anywhere. Its own `README.md` is
explicit: the live `GET /bayes/nodes` endpoint is served by `api/`'s FastAPI app
(`api/src/forecast_api/bayesoracle.py`), which loads `bayesoracle/core.py` and
`graph_political.json` directly — see "Run: api" above, whose `smoke` command already
exercises this. `bayesoracle/` itself is a library: an inference engine
(`core.py`) plus calibration/backtest scripts and two static offline HTML viewers.

```bash
cd bayesoracle && uv run python ../.claude/skills/run-retro/driver-bayesoracle.py
cd bayesoracle && uv run python backtest.py   # real Brier-score backtest, local JSON only, no network
```

The driver loads and validates both checked-in graphs (`graph_political.json`,
`graph_pm.json`), propagates with no overrides, then re-propagates with one node
locked to show downstream deltas — pure numpy/scipy computation, no network, no LLM,
no credentials.

Verified real output (this environment):

```
21 nodes loaded. First 3:
  BEYACHAD             prior=0.680 p=0.680 layer=0
  BIBI_OUT             prior=0.150 p=0.150 layer=0
  HAREDI_CRISIS        prior=0.550 p=0.550 layer=0
Locked BEYACHAD=0.99; 14 downstream node(s) shifted (delta != 0).
24 nodes loaded from graph_pm.json.
```

`backtest.py` (also run in this environment, real output) reads only local
`node_history/*.json` files — no network — and reports the DAG beating a persistence
baseline on 12/16 child nodes (aggregate Brier 0.0165 vs 0.0178, +7.3% skill).

**Never run here** (need real production credentials/network):
`calibrate_edges.py` (Bedrock + live news search, ~85 min for 28 edges),
`fetch_node_history.py` (Polymarket API), `series/log_nodes.py` (calls the live
`oracle.daatan.com` `/forecast` with a real `ORACLE_API_KEY`).

## Test

```bash
cd pipeline && uv run pytest -q
cd api && uv run pytest -q
cd bayesoracle && uv run pytest -q
```

All three ran clean in this environment with no manual env setup and the same
`AWS_ACCESS_KEY_ID=invalid` guardrail — no pre-existing failures observed, all pass
(1199/1290/61 at time of writing — expect these counts to grow; a test-count string
pinned in prose goes stale the moment the suite grows, exactly like the number in
`api/README.md` that drifted from "154 tests" to well past it). LLM/network is mocked
throughout (repo `CLAUDE.md`).

## Gotchas

- **Ambient AWS credentials may exist on the host running this skill** (this box has
  a live `~/.aws/credentials`, used for legitimate SSM/prod work elsewhere in this
  workspace). Without the `AWS_ACCESS_KEY_ID=invalid AWS_SECRET_ACCESS_KEY=invalid`
  guardrail in every driver command, an accidentally-unmocked code path would use
  those real credentials instead of failing loudly. Verified: with the guardrail set,
  `api/`'s startup log shows every search-provider secret load failing with
  `UnrecognizedClientException` ("security token included in the request is
  invalid") and each provider silently disabling — the app fails open, not closed,
  so don't rely on a crash to notice a leak; rely on the fake credentials.
- **`uv run uvicorn ... --reload` spawns a reloader parent process** — `$!` captures
  the parent, and killing it can leave the actual worker running. `driver-api.sh`
  deliberately omits `--reload` and kills by port as a fallback (`lsof -ti:$PORT`).
- **`api/`'s startup is not instant, and is variable** — the FastAPI lifespan does a
  leaderboard `refresh_cache` pass, and (with the AWS guardrail active) sequentially
  fails 14 search-provider SSM lookups with retries before the app is ready. Observed
  12s on one run and >30s on another in this environment. `driver-api.sh start` polls
  for up to 60s to absorb this.
