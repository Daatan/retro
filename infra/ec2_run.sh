#!/bin/bash
# TruthMachine EC2 Run Loop
# Continuously ingests articles, extracts predictions, renders atlas, and pushes to GitHub.
#
# Usage:
#   nohup bash ~/truthmachine/infra/ec2_run.sh >> ~/truthmachine/pipeline_log.txt 2>&1 &
#   echo $! > ~/truthmachine/run.pid
#
# Monitor:  tail -f ~/truthmachine/pipeline_log.txt
# Stop:     kill $(cat ~/truthmachine/run.pid)
set -euo pipefail

# Ensure uv is on PATH (not set in non-interactive SSM shells)
export PATH="$HOME/.local/bin:$PATH"

WORKDIR="$HOME/truthmachine"
DATA_DIR="$WORKDIR/data"
PIPELINE_DIR="$WORKDIR/pipeline"
SLEEP_INTERVAL=300   # seconds between cycles (5 min — short pause to avoid hammering APIs)

# Load secrets
set -a; source "$WORKDIR/.env"; set +a
export DATA_DIR

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

cell_stats() {
  python3 - <<'PY'
import json, os
p = os.path.join(os.environ["DATA_DIR"], "progress.json")
try:
    s = json.load(open(p))
    cells = s.get("cells", {})
    done  = sum(1 for v in cells.values() if v.get("status") == "done")
    nopred= sum(1 for v in cells.values() if v.get("status") == "no_predictions")
    fail  = sum(1 for v in cells.values() if v.get("status") == "failed")
    pend  = sum(1 for v in cells.values() if v.get("status") == "pending")
    total = len(cells)
    print(f"{done}/{total} done | {nopred} no_pred | {fail} failed | {pend} pending")
except Exception:
    print("?/? (progress.json unreadable)")
PY
}

commit_and_push() {
  local msg="$1"
  cd "$WORKDIR"
  # Render latest atlas before committing
  uv run --project "$PIPELINE_DIR" python -m tm.render_atlas 2>&1 | tail -5
  STATS=$(cell_stats)
  git add factum_atlas.html data/progress.json 2>/dev/null || true
  # Don't push an empty atlas — only commit when we have real done cells
  DONE_COUNT=$(python3 -c "
import json, os
from pathlib import Path
p = Path(os.environ['DATA_DIR']) / 'progress.json'
try:
    cells = json.loads(p.read_text()).get('cells', {})
    print(sum(1 for v in cells.values() if v.get('status') == 'done'))
except: print(0)
" 2>/dev/null)
  if [[ "${DONE_COUNT:-0}" -gt 0 ]] && ! git diff --cached --quiet; then
    git commit -m "atlas: ${msg} — ${STATS}"
    git fetch origin main
    git rebase origin/main || { git rebase --abort; log "WARNING: rebase failed, skipping push"; return 0; }
    git push origin main
    log "Pushed: ${msg} — ${STATS}"
  fi
}

# A git process that dies mid-operation leaves .git/index.lock behind, and every
# later git op in this tree then fails on it — including the `reset --hard` that is
# supposed to be the sync's own fallback. The tree keeps running whatever code it
# last had, indefinitely, while the loop reports only a generic "cycle failed".
#
# Seen twice: 2026-07-02 → 08-16 (six weeks on stale code, no atlas pushes) and
# 2026-08-17 → 08-19, the second time stranding retro#549's settlement guard on the
# API checkout while this tree — the larger producer — kept emitting the rows the
# guard exists to neutralise (retro#553).
#
# Reap it only when it is provably nobody's: no live git process in this repo, and
# the lock older than a full cycle (a lock younger than that may belong to a job
# still starting up).
reap_stale_git_lock() {
  local lock="$WORKDIR/.git/index.lock"
  [[ -e "$lock" ]] || return 0

  if pgrep -a git 2>/dev/null | grep -q "$WORKDIR"; then
    log "WARNING: .git/index.lock present but a git process is live here — leaving it"
    return 0
  fi

  local age=$(( $(date +%s) - $(stat -c %Y "$lock") ))
  if (( age < SLEEP_INTERVAL )); then
    log "WARNING: .git/index.lock is only ${age}s old — leaving it for now"
    return 0
  fi

  log "ERROR: reaping stale .git/index.lock (age ${age}s, no live git process)"
  rm -f "$lock"
}

# Bring the tree to origin/main and PROVE it landed there. Without the assertion a
# failed sync is indistinguishable from a successful one — which is exactly how the
# tree ran two-day-old code while every cycle logged a generic failure.
sync_to_main() {
  reap_stale_git_lock

  git fetch origin main
  git merge --ff-only origin/main || {
    log "WARNING: fast-forward failed — force-syncing to origin/main"
    # Tolerated deliberately. `run_pipeline` is called as `run_pipeline || log ...`,
    # and bash SUSPENDS errexit inside a function invoked that way — so a failing
    # reset here never aborted anything; the loop simply carried on and ran the
    # pipeline on stale code. That is the whole bug. Swallow it explicitly and let
    # the assertion below be the thing that decides, in both calling contexts.
    git reset --hard origin/main || log "WARNING: force-sync also failed"
  }

  local head origin behind
  head=$(git rev-parse HEAD)
  origin=$(git rev-parse origin/main)
  if [[ "$head" != "$origin" ]]; then
    behind=$(git rev-list --count "HEAD..origin/main" 2>/dev/null || echo "?")
    log "ERROR: batch tree did NOT sync — HEAD ${head:0:9} is ${behind} commit(s) behind origin/main ${origin:0:9}; REFUSING to run on stale code"
    return 1
  fi
  log "Synced to origin/main ${head:0:9}"
}

run_pipeline() {
  cd "$WORKDIR"

  # ── 1. Pull latest event/source definitions ───────────────────────────────
  # `|| return 1` is load-bearing: errexit is suspended inside this function (see
  # sync_to_main), so a bare call would log the refusal and then run anyway.
  log "Pulling latest from main..."
  sync_to_main || return 1

  # ── 2. Ingest: fetch articles in batches of 10 events per cycle ─────────
  log "Ingest starting — $(cell_stats)"
  ALL_EVENTS=$(python3 -c "
import pathlib, os, json
events_dir = pathlib.Path(os.environ['DATA_DIR']) / 'events'
print(' '.join(sorted(p.stem for p in events_dir.glob('*.json'))))
" 2>/dev/null)
  read -ra ALL_ARR <<< "$ALL_EVENTS"
  TOTAL_ALL=${#ALL_ARR[@]}
  BATCH_SIZE=10
  OFFSET_FILE="$DATA_DIR/ingest_offset"
  OFFSET=$(cat "$OFFSET_FILE" 2>/dev/null || echo 0)
  OFFSET=$(( OFFSET % TOTAL_ALL ))
  INGEST_EVENTS=("${ALL_ARR[@]:OFFSET:BATCH_SIZE}")
  # wrap around if batch crosses end of list
  REMAINING=$(( BATCH_SIZE - ${#INGEST_EVENTS[@]} ))
  if [[ $REMAINING -gt 0 ]]; then
    INGEST_EVENTS+=("${ALL_ARR[@]:0:REMAINING}")
  fi
  NEW_OFFSET=$(( (OFFSET + BATCH_SIZE) % TOTAL_ALL ))
  echo "$NEW_OFFSET" > "$OFFSET_FILE"
  log "Ingesting events ${INGEST_EVENTS[*]} (offset $OFFSET/$TOTAL_ALL)"
  uv run --project "$PIPELINE_DIR" python -m tm.gnews_ingest --events "${INGEST_EVENTS[@]}" 2>&1
  log "Ingest complete — $(cell_stats)"

  # Push progress.json after ingest so local clone sees live status
  cd "$WORKDIR"
  git add data/progress.json 2>/dev/null || true
  if ! git diff --cached --quiet; then
    git commit -m "progress: ingest batch ${INGEST_EVENTS[*]} — $(cell_stats)"
    git fetch origin main
    git rebase origin/main || { git rebase --abort; log "WARNING: progress push rebase failed"; }
    git push origin main && log "Pushed progress.json after ingest"
  fi

  # ── 3. Extract: run orchestrator in batches of 5 events, commit after each ─
  log "Extraction starting — $(cell_stats)"

  # Get all event IDs that have articles in raw_ingest
  EVENTS=$(python3 -c "
import os, pathlib
raw = pathlib.Path(os.environ['DATA_DIR']) / 'raw_ingest'
events = set()
if raw.exists():
    for src_dir in raw.iterdir():
        for evt_dir in src_dir.iterdir():
            if any(evt_dir.glob('*.json')):
                events.add(evt_dir.name)
print(' '.join(sorted(events)))
" 2>/dev/null)

  if [[ -z "$EVENTS" ]]; then
    log "No raw_ingest articles found — skipping orchestrator."
  else
    read -ra EVENT_ARR <<< "$EVENTS"
    TOTAL_EVENTS=${#EVENT_ARR[@]}
    log "Orchestrating ${TOTAL_EVENTS} events in batches of 5..."

    BATCH=0
    for (( i=0; i<TOTAL_EVENTS; i+=5 )); do
      BATCH=$((BATCH + 1))
      BATCH_EVENTS=("${EVENT_ARR[@]:i:5}")
      log "Batch ${BATCH}: events ${BATCH_EVENTS[*]}"

      timeout 600 uv run --project "$PIPELINE_DIR" python -m tm.orchestrator local_file \
        --retry-empty \
        --events "${BATCH_EVENTS[@]}" 2>&1 \
        || log "WARNING: orchestrator batch ${BATCH} timed out or failed — continuing"

      log "Batch ${BATCH} done — $(cell_stats)"
      commit_and_push "batch ${BATCH} (${BATCH_EVENTS[*]})"
    done
  fi

  log "Extraction complete — $(cell_stats)"

  # ── 4. Oracle API fill-in: search for articles for events that are still empty ─
  # Runs once every 4 cycles (offset-based) to avoid hammering Oracle on every cycle.
  ORACLE_TURN_FILE="$DATA_DIR/oracle_fill_turn"
  ORACLE_TURN=$(cat "$ORACLE_TURN_FILE" 2>/dev/null || echo 0)
  ORACLE_TURN=$(( (ORACLE_TURN + 1) % 4 ))
  echo "$ORACLE_TURN" > "$ORACLE_TURN_FILE"
  if [[ "$ORACLE_TURN" -eq 0 ]] && [[ -n "${ORACLE_API_KEY:-}" ]]; then
    EMPTY_EVENTS=$(python3 -c "
import json, os
from pathlib import Path
data = Path(os.environ['DATA_DIR'])
raw = data / 'raw_ingest'
cells = json.loads((data / 'progress.json').read_text()).get('cells', {})
# Events where every known source is no_predictions AND has no raw articles
from collections import defaultdict
event_has_articles = defaultdict(bool)
if raw.exists():
    for src_dir in raw.iterdir():
        for evt_dir in src_dir.iterdir():
            if any(evt_dir.glob('*.json')):
                event_has_articles[evt_dir.name] = True
# Events with no articles at all
events_dir = data / 'events'
all_events = {p.stem for p in events_dir.glob('*.json')}
empty = sorted(e for e in all_events if not event_has_articles[e])
print(' '.join(empty))
" 2>/dev/null)
    if [[ -n "$EMPTY_EVENTS" ]]; then
      read -ra EMPTY_ARR <<< "$EMPTY_EVENTS"
      log "Oracle API fill: ${#EMPTY_ARR[@]} events with no articles — searching via Oracle"
      for (( i=0; i<${#EMPTY_ARR[@]}; i+=5 )); do
        FILL_BATCH=("${EMPTY_ARR[@]:i:5}")
        timeout 600 uv run --project "$PIPELINE_DIR" python -m tm.orchestrator api \
          --retry-empty \
          --events "${FILL_BATCH[@]}" 2>&1 \
          || log "WARNING: Oracle fill batch timed out or failed — continuing"
        commit_and_push "oracle-fill (${FILL_BATCH[*]})"
      done
      log "Oracle API fill complete — $(cell_stats)"
    fi
  fi

  # Snapshot atlas+vault2 to S3 so a replaced EC2 instance can resume with
  # prior state. Non-fatal — failures are logged but don't break the loop.
  # Runs unconditionally after the extraction phase because vault2/ grows on
  # every article processed, not only when a commit happens.
  if [[ -x "$WORKDIR/infra/snapshot_atlas.sh" ]]; then
    bash "$WORKDIR/infra/snapshot_atlas.sh" || log "snapshot_atlas exited non-zero (non-fatal)"
  fi
}

# ── Main loop ─────────────────────────────────────────────────────────────────
log "=== TruthMachine pipeline starting ==="
log "DATA_DIR=$DATA_DIR"
log "Cycle interval: ${SLEEP_INTERVAL}s (5 min)"

while true; do
  log "─── Cycle start ───────────────────────────────────"

  run_pipeline || log "ERROR: cycle failed, will retry next interval"

  # Skip sleep if there are still failed or pending cells
  OUTSTANDING=$(python3 - <<'PY'
import json, os
from pathlib import Path
try:
    p = Path(os.environ['DATA_DIR']) / 'progress.json'
    cells = json.loads(p.read_text()).get('cells', {})
    print(sum(1 for v in cells.values() if v.get('status') in ('failed', 'pending')))
except Exception:
    print(0)
PY
)
  if [[ "${OUTSTANDING:-0}" -gt 0 ]]; then
    log "─── Cycle done. ${OUTSTANDING} cells still pending/failed — retrying immediately ────"
  else
    log "─── Cycle done. Sleeping ${SLEEP_INTERVAL}s... ────"
    sleep "$SLEEP_INTERVAL"
  fi
done
