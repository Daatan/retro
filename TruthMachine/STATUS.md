# TruthMachine / Factum Atlas — Status & Runbook

_Last updated: 2026-05-06. The "Current State" snapshot below ages fast — cross-check with the live atlas before trusting specific numbers._

---

## What Is This

TruthMachine is a retroactive media prediction pipeline. It:
1. **Ingests** news articles from 13 Israeli/international sources via GNews API
2. **Extracts** forward-looking predictions from each article using AWS Bedrock (LLM)
3. **Aggregates** predictions into a cell signal per (event × source) pair
4. **Renders** the Factum Atlas — an HTML matrix of prediction signals
5. **Publishes** the atlas to GitHub (`factum_atlas.html` on `main`)

The matrix has up to **910 cells** = 70 events × 13 sources. Not all combinations have articles (the gatekeeper rejects off-topic coverage, some sources don't cover some events), so the steady-state `done` count is lower than 910.

---

## Current State

For live numbers, look at:

- **Factum Atlas**: [daatan.github.io/retro/factum_atlas.html](https://daatan.github.io/retro/factum_atlas.html) — rendered HTML, updated on every atlas commit.
- **`data/progress.json` on `main`** — cell-by-cell status, updated by the pipeline's per-cycle commit.
- Recent atlas/progress commits: `git log --oneline --grep='atlas:\|progress:'`.

Snapshot as of last update:

| Metric | Value |
|--------|-------|
| EC2 instance | `i-00ac444b94c5ff9b2` (eu-central-1) |
| Instance type | t4g.small (ARM64, Ubuntu 24.04) |
| Batch pipeline service | `truthmachine.service` — **running** |
| Oracle API service | `oracle-api.service` — **running** (auto-deploys on merge to main via `.github/workflows/deploy-oracle.yml`) |

The pipeline loop (`infra/ec2_run.sh`) runs continuously: it sleeps 300s between cycles **only when no cells are `pending` or `failed`** — otherwise it retries immediately. So if `progress.json` shows pending cells, expect rapid re-attempts; a 5-minute gap between commits implies a steady state.

### Oracle API (oracle.daatan.com)

| Item | Status |
|------|--------|
| API skeleton (`api/`) | ✅ Merged |
| Test console | ✅ Live at [daatan.github.io/retro/oracle-test.html](https://daatan.github.io/retro/oracle-test.html) |
| Pipeline wired (web_search → gatekeeper → extractor → aggregate) | ✅ Complete |
| Leaderboard credibility weighting (TrueSkill) | ✅ Complete |
| EC2 deployment (`oracle-api.service`) | ✅ Running |
| DNS + TLS (`oracle.daatan.com`) | ✅ Live |
| nginx vhost (`infra/nginx/oracle.conf`) | ✅ Deployed |
| `/health` endpoint with version | ✅ Live (`0.1.0`) |
| daatan secrets (`ORACLE_URL` + `ORACLE_API_KEY`) | ✅ In AWS Secrets Manager (`daatan-env-{prod,staging}`) |
| daatan integration (`oracle.ts` wired into context + express guess routes) | ✅ Live in daatan v1.9.0 |
| Search provider expansion (BrightData + Nimbleway + ScrapingBee) | ✅ Live (2026-04-28) — 6 paid providers + DDG |
| Secrets Manager read permission (`openclaw-secrets-read` on `truthmachine-ec2-role`) | ✅ Applied |

### Duel: TruthMachine vs Polymarket

| Item | Status |
|------|--------|
| Polymarket harvest pipeline (`polymarket_harvest.py`) | ✅ Complete |
| PoC event generation (`poc_event_gen.py`) | ✅ Complete |
| Duel report generator (`poc_report.py`) | ✅ Complete |
| `duel.html` generated and deployed to GitHub Pages | ✅ Live |
| TM vs PM Brier comparison section | ✅ Live — 13 events scored, TM wins 8/13; aggregate Brier: TM ~0.123 vs PM ~0.377; Oracle n=1–20 per event (PRs #82/#84); [daatan.github.io/retro/duel.html](https://daatan.github.io/retro/duel.html) |

### Why is it sleeping?

`ec2_run.sh` always sleeps 300s after each cycle — even when pending cells remain. It only skips sleep when there are `failed` cells. This is a known inefficiency (see Future Fixes below).

---

## Architecture

```
ec2_run.sh (systemd loop)
  ├── 1. git pull origin main          (pick up code/config updates)
  ├── 2. gnews_ingest (batch of 10 events)
  │       └── GNews API → DDG fallback → Wayback fallback
  │           → raw_ingest/{source}/{event}/article_*.json
  ├── 3. orchestrator local_file (batches of 5 events)
  │       └── for each article:
  │           ├── Gatekeeper (Nova Micro) → is_prediction?
  │           ├── Extractor (Nova Lite) → PredictionExtraction × N
  │           ├── Article aggregator (Nova Lite) → 1 signal if spread > 0.4
  │           └── → vault2/extractions/{hash}_{event}_v1.json
  │               → atlas/{event}/{source}/entry_{hash[:8]}.json
  ├── 4. render_atlas → factum_atlas.html
  └── 5. git commit + push factum_atlas.html
```

**Models:**
- Gatekeeper: `bedrock/amazon.nova-micro-v1:0` (us-east-1)
- Extractor: `bedrock/amazon.nova-lite-v1:0` (us-east-1)
- Article aggregator: `bedrock/amazon.nova-lite-v1:0`

---

## Pipeline Stages

### Stage 1: Gatekeeper
- Uses first ~2500 chars of article
- Returns `is_prediction: bool`
- Skips article if False → cell marked `no_predictions`

### Stage 2: Extractor
- Extracts up to 10 `PredictionExtraction` objects per article
- Fields: quote, claim, stance (−1 to +1), sentiment, certainty, specificity, hedge_ratio, conditionality, magnitude, time_horizon, prediction_type, source_authority

### Stage 2b: Article Aggregator (new, 2026-03-28)
- Triggered when a single article produces predictions with stance spread > 0.4
- Collapses N predictions into 1 using editorial judgment (not averaging)
- Prompt: `TruthMachine/prompts/02b_article_aggregator.md`
- Script for post-processing: `pipeline/src/tm/reaggregate.py`

### Stage 3: Cell Signal
- Aggregates all predictions across all articles for one (event, source) cell
- Weighted mean (weight = certainty × specificity) for continuous fields
- Majority vote for categorical fields
- Written to `atlas/{event}/{source}/cell_signal.json`

---

## Failed Attempts & Issues Encountered

### 1. OpenRouter monthly quota exceeded
- **Symptom:** 403 errors on all LLM calls
- **Fix:** Switched to AWS Bedrock Nova models

### 2. `instructor.Mode.TOOLS` not supported by Bedrock
- **Symptom:** All LLM calls failed with tool-use errors
- **Fix:** Switched to `instructor.Mode.MD_JSON`

### 3. Nova Lite JSON object mode returns `<thinking>` tags
- **Symptom:** JSON parsing failed due to reasoning preamble
- **Fix:** MD_JSON mode strips markdown code blocks correctly

### 4. `KeyError: '"predictions"'` in extractor prompt
- **Symptom:** `PROMPT.format()` interpreted `{"predictions": [...]}` as a format placeholder
- **Fix:** Escaped to `{{"predictions": [...]}}`

### 5. `boto3` missing from uv venv on EC2
- **Symptom:** All LLM calls failed with `No module named 'boto3'`; cells silently marked `no_predictions` instead of `failed`
- **Root cause:** `uv sync` during bootstrap ran before venv was created (no `.venv` dir); `uv pip install` requires `--system` or an active venv
- **Fix:** `uv pip install boto3 botocore -p .venv/bin/python3` directly into the existing venv
- **Silent failure bug:** `runner.py` marks cell `failed` on exception, but `orchestrator.py` overwrites to `no_predictions` if no predictions found — hides the real error

### 6. EC2 bootstrap fails silently on secrets fetch
- **Symptom:** Bootstrap exits after `[bootstrap] Fetching secrets...` with no error
- **Root cause 1:** IAM role not propagated yet when bootstrap ran (takes ~60s)
- **Root cause 2:** `get_secret()` uses `2>/dev/null`; `set -e` exits on empty return without logging
- **Fix:** Added `|| true` to all `get_secret` calls; also `chmod 644` on `.env` (was 600, systemd runs as root)

### 7. `.env` permission denied
- **Symptom:** Service crashes immediately with `permission denied` on `.env`
- **Root cause:** Setup script ran as root via SSM, `.env` owned by root; `chmod 600` prevents ubuntu user from reading
- **Fix:** `chmod 644 /home/ubuntu/truthmachine/.env`

### 8. Git "dubious ownership" on EC2
- **Symptom:** `git fetch` fails; pipeline can't pull code updates
- **Root cause:** Repo cloned as ubuntu but SSM runs commands as root → git sees ownership mismatch
- **Fix:** `git config --global --add safe.directory /home/ubuntu/truthmachine` for ubuntu user; `chown -R ubuntu:ubuntu ~/truthmachine`

### 9. EC2 overwrites `factum_atlas.html` with empty atlas
- **Symptom:** Atlas page went blank after EC2 started pushing
- **Root cause:** EC2 started fresh with no atlas data; `commit_and_push` always renders and commits, even with 0 done cells
- **Partial fix:** Atlas data from local machine uploaded to EC2 via S3 tarball
- **Proper fix needed:** Only commit `factum_atlas.html` when `done > 0` (see Future Fixes)

### 10. DDG / Brave ingest failures
- **Symptom:** ~60–70% of article URL resolution fails
- **Root cause:** Brave API quota exhausted (402); DDG rate-limits heavily (connection errors)
- **Status:** Articles still get ingested — the ones that resolve are enough to fill the atlas gradually
- **Potential fix:** Add SerpAPI or Serper.dev as fallback; refresh Brave quota

---

## Done (historical — do not re-do)

- **Skip sleep when pending > 0** — `infra/ec2_run.sh:165-182` guards sleep on `failed + pending == 0`.
- **Don't commit empty atlas** — `infra/ec2_run.sh:52-68` guards commit on `done > 0`.
- **Boto3 bootstrap verify-and-install** — `infra/ec2_bootstrap.sh:95-100` reinstalls if `import boto3` fails post-`uv sync`.
- **Unmask silent orchestrator failures** — `pipeline/src/tm/orchestrator.run_event` catches `Exception` (not just `asyncio.TimeoutError`) around `process_article`, so any infra-level failure routes a cell to `failed` instead of being hidden as `no_predictions` or killing the whole cell loop.
- **Atlas persistence across EC2 restarts** — `data/atlas/` + `data/vault2/` are snapshotted to `s3://truthmachine-atlas-snapshots-<account>/` at the tail of each pipeline cycle (`infra/snapshot_atlas.sh`) and restored on fresh-instance bootstrap (`infra/restore_atlas.sh`). See [`docs/ATLAS_SNAPSHOTS.md`](../docs/ATLAS_SNAPSHOTS.md).

## Future Fixes (Priority Order)

### Medium Priority

1. ~~**Refresh Brave API quota or add SerpAPI fallback**~~ — resolved 2026-04-28.
   BrightData, Nimbleway, and ScrapingBee added to `web_search.py` as additional fallbacks.
   Keys in Secrets Manager. Oracle survives simultaneous exhaustion of all three original providers.
   Next: implement oracle `/search` endpoint so daatan uses the same chain (Phase 2 in TODO.md).

2. **Reduce sleep interval or make it adaptive**
   - `SLEEP_INTERVAL=60` would be more responsive
   - Or: skip sleep entirely when ingest found new articles

### Low Priority

1. **Article-level aggregation on full re-run**
   - The `reaggregate.py` script fixed 81 local entries
   - EC2 will re-aggregate inline during extraction (already wired in orchestrator)
   - No action needed unless we re-use local data

2. **Ground truth scoring**
   - `pipeline/src/tm/backtest.py` — not yet run on EC2
   - Requires resolved events with known outcomes

3. ~~**Wire TM into `duel.html`**~~ — ✅ Done (2026-05-06). PM price history harvested for 12 Atlas events; Oracle-driven TM probabilities computed; duel report live. TM wins 7/12. CLOB API ceiling reached at 12 events — pre-2023 markets (E01, E02, C01, C02, G01) not available in the CLOB system. Expanding beyond 12 requires newer events with sufficient PM CLOB history.

---

## Monitoring

```bash
# Quick status via SSM
bash /home/mark/projects/retro/infra/monitor.sh

# Live log tail
aws ssm send-command --region eu-central-1 \
  --instance-ids i-00ac444b94c5ff9b2 \
  --document-name AWS-RunShellScript \
  --parameters 'commands=["tail -30 ~/truthmachine/pipeline_log.txt"]'

# Git log (EC2 commits progress here)
cd /home/mark/projects/retro && git fetch && git log --oneline -10
```

---

## Instance Details

| Field | Value |
|-------|-------|
| Instance ID | `i-00ac444b94c5ff9b2` |
| Region | `eu-central-1` |
| AMI | Ubuntu 24.04 LTS arm64 |
| Type | t4g.small |
| IAM Role | `truthmachine-ec2-role` |
| Permissions | SSM + Bedrock (Nova) + Secrets Manager (`openclaw/*`) + S3 (`truthmachine-atlas-snapshots-*`) |
| Workdir | `/home/ubuntu/truthmachine/` |
| Service | `systemctl status truthmachine` |
| Log | `/home/ubuntu/truthmachine/pipeline_log.txt` |
| Access | AWS SSM only (port 22 blocked) |
