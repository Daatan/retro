#!/bin/bash
# Zero-downtime deploy for the Oracle API.
#
# Runs ON the EC2 box (invoked by the GH Actions workflow via SSM, or by hand).
# Updates /home/ubuntu/oracle-api — the API-only checkout — to a given git ref,
# re-syncs deps, and SIGHUPs gunicorn so workers swap in the new code without
# dropping the listening socket.
#
# Explicitly does NOT touch /home/ubuntu/truthmachine (the pipeline's checkout).
# That tree carries unpushed atlas/progress commits from the ingest loop and
# has its own rebase-and-push lifecycle in infra/ec2_run.sh. Mixing the two
# responsibilities is what made earlier deploys risky; keeping them separate
# makes this script trivial.
#
# Usage:
#   bash /home/ubuntu/oracle-api/infra/deploy_oracle.sh              # -> origin/main
#   bash /home/ubuntu/oracle-api/infra/deploy_oracle.sh <commit-sha> # pin to a SHA
set -euo pipefail

API_DIR="/home/ubuntu/oracle-api"
REF="${1:-origin/main}"

# git and uv must run as ubuntu (the repo owner) regardless of whether this
# script is invoked as root via SSM. Only systemctl needs root.
AS_UBUNTU="sudo -u ubuntu HOME=/home/ubuntu"
UV="/home/ubuntu/.local/bin/uv"

log() { echo "[deploy_oracle $(date '+%H:%M:%S')] $*"; }

# ── 1. Fetch + reset ─────────────────────────────────────────────────────────
cd "$API_DIR"
log "fetching origin/main..."
$AS_UBUNTU git fetch origin main --quiet

BEFORE=$($AS_UBUNTU git rev-parse HEAD)
log "resetting to ${REF} (was ${BEFORE:0:8})"
$AS_UBUNTU git reset --hard "$REF" --quiet
AFTER=$($AS_UBUNTU git rev-parse HEAD)
log "now at ${AFTER:0:8}: $(git log -1 --pretty=%s)"

if [[ "$BEFORE" == "$AFTER" ]]; then
  log "no change — skipping sync and reload"
  exit 0
fi

# ── 2. Sync Python deps ──────────────────────────────────────────────────────
log "uv sync --frozen..."
cd "$API_DIR/api"
$AS_UBUNTU $UV sync --frozen --quiet

# ── 3. Health probe helper ───────────────────────────────────────────────────
# Require N *consecutive* 200s rather than the first 200. A SIGHUP reload runs
# new workers alongside gracefully-draining old ones; a single early 200 can be
# served by an old (still-good) worker even when the new workers are broken
# (e.g. a native-dep upgrade where SIGHUP swaps the httptools C-extension under
# an already-imported uvicorn — the bug that took the API down on 2026-06-02).
# Demanding consecutive 200s spanning the drain window catches that.
GRACEFUL_DRAIN=30   # must exceed gunicorn --graceful-timeout so old workers are gone
healthy() {
  local need=5 streak=0 http
  for _ in $(seq 1 "$((GRACEFUL_DRAIN + 12))"); do
    http=$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 http://127.0.0.1:8001/health || echo 000)
    if [[ "$http" == "200" ]]; then
      streak=$((streak + 1))
      [[ "$streak" -ge "$need" ]] && return 0
    else
      streak=0
    fi
    sleep 1
  done
  return 1
}

# ── 4. Reload, then verify — escalate to full restart if unhealthy ────────────
# SIGHUP first (zero-downtime: keeps the listening socket, no 502 window). If
# the reloaded workers aren't durably healthy, a full restart gives a fresh
# master that imports the whole upgraded stack consistently. Only if *that*
# still fails do we go red.
log "reloading oracle-api (SIGHUP)..."
sudo /bin/systemctl reload-or-restart oracle-api

if healthy; then
  log "ok — deployed ${AFTER:0:8} (healthy after SIGHUP reload)"
  exit 0
fi

log "reload unhealthy — full restart (SIGHUP can't swap native worker deps)..."
sudo /bin/systemctl restart oracle-api

if healthy; then
  log "ok — deployed ${AFTER:0:8} (healthy after full restart)"
  exit 0
fi

log "FAIL: /health not durably 200 after reload+restart — service is unhealthy"
exit 1
