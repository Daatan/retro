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

# ── 3. Zero-downtime reload ──────────────────────────────────────────────────
# SIGHUP to gunicorn master spawns fresh workers with reimported modules and
# gracefully drains the old ones. No socket close = no 502 window. If the
# dep graph changed in a way that needs a new master (rare), a separate
# `systemctl restart oracle-api` is the escape hatch.
log "reloading oracle-api (SIGHUP, or restart if stopped)..."
sudo /bin/systemctl reload-or-restart oracle-api

# ── 4. Health check ──────────────────────────────────────────────────────────
# Gunicorn's reload is async: the new workers take ~2-3s to import and start
# serving. Poll /health until we get a 200 or give up.
log "health probe..."
for i in 1 2 3 4 5 6 7 8 9 10; do
  HTTP=$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 http://127.0.0.1:8001/health || echo 000)
  if [[ "$HTTP" == "200" ]]; then
    log "ok — deployed ${AFTER:0:8} (healthy after ${i} probe$([ "$i" = 1 ] || echo s))"
    exit 0
  fi
  sleep 1
done

log "FAIL: /health never returned 200 after 10s"
exit 1
