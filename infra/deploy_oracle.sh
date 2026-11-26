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
#
# A failed deploy is safe to re-run (retro#1110). "Nothing to do" means the target
# equals the last deploy that came up healthy (.deployed_sha), not the current
# HEAD; a venv that does not sync cleanly rolls the checkout and the venv back
# before exiting, without reloading the service.
set -euo pipefail

# DEPLOY_* overrides exist for infra/tests/test_deploy_oracle.sh only.
API_DIR="${DEPLOY_API_DIR:-/home/ubuntu/oracle-api}"
REF="${1:-origin/main}"

# git and uv must run as ubuntu (the repo owner) whether this script is invoked as
# ubuntu (deploy-oracle.yml) or as root (a by-hand SSM run). Root-only steps go
# through sudo, which is passwordless for ubuntu on the box.
OWNER="${DEPLOY_OWNER:-ubuntu}"
SUDO="${DEPLOY_SUDO-sudo}"
AS_UBUNTU="${DEPLOY_AS_UBUNTU-sudo -u ubuntu HOME=/home/ubuntu}"
UV="${DEPLOY_UV:-/home/ubuntu/.local/bin/uv}"
SYSTEMCTL="${DEPLOY_SYSTEMCTL:-sudo /bin/systemctl}"
HEALTH_URL="${DEPLOY_HEALTH_URL:-http://127.0.0.1:8001/health}"
HEALTH_SLEEP="${DEPLOY_HEALTH_SLEEP:-1}"
# The last deploy that came up healthy. Untracked; written only at the end of a good deploy.
DEPLOYED_FILE="$API_DIR/.deployed_sha"

log() { echo "[deploy_oracle $(date '+%H:%M:%S')] $*"; }

# ── 0. Ownership ─────────────────────────────────────────────────────────────
# One root-owned file in a venv makes `uv sync` (run as ubuntu) fail halfway, after it
# has already removed packages: on 2026-11-26 two __pycache__ entries left by a root
# diagnostic on 11-18 cost litellm + fastapi, and the next worker recycle took the API
# down (retro#1110). Everything under the checkout belongs to ubuntu, so hand back
# whatever isn't, before git or uv touch the tree. `|| true`: find exits non-zero
# on a directory it may not enter, and that directory is itself counted anyway.
foreign=$( { find "$API_DIR" -xdev ! -user "$OWNER" 2>/dev/null || true; } | wc -l)
if [[ "$foreign" -gt 0 ]]; then
  log "ownership: $foreign entries under $API_DIR not owned by $OWNER — chown -R"
  $SUDO chown -R "$OWNER:$OWNER" "$API_DIR"
fi

# ── 1. Fetch + decide ────────────────────────────────────────────────────────
cd "$API_DIR"
log "fetching origin/main..."
$AS_UBUNTU git fetch origin main --quiet

BEFORE=$($AS_UBUNTU git rev-parse HEAD)
TARGET=$($AS_UBUNTU git rev-parse --verify "${REF}^{commit}")
DEPLOYED=$(cat "$DEPLOYED_FILE" 2>/dev/null || true)

# Compare with the last *good* deploy, not HEAD: a run that died in `uv sync` used to
# leave HEAD at the target, and the retry then skipped the sync it needed (retro#1110).
if [[ "$TARGET" == "$DEPLOYED" && "$BEFORE" == "$TARGET" ]]; then
  log "no change — ${TARGET:0:8} is already the deployed commit"
  exit 0
fi

# Where to go back to if the new commit's deps can't be installed.
ROLLBACK="${DEPLOYED:-$BEFORE}"

log "resetting to ${REF} (was ${BEFORE:0:8}, last good deploy ${DEPLOYED:-none})"
$AS_UBUNTU git reset --hard "$TARGET" --quiet
AFTER=$($AS_UBUNTU git rev-parse HEAD)
log "now at ${AFTER:0:8}: $(git log -1 --pretty=%s)"

# ── 2. Sync Python deps ──────────────────────────────────────────────────────
# Each step returns explicitly: `set -e` is suspended inside a function called under `if`.
sync_venv() {
  # An uninstall that dies halfway can leave a package's dist-info dir with its METADATA
  # already deleted. uv still counts that version as installed, so neither a re-sync nor
  # `--reinstall` restores the package's missing files; with the stub removed, one sync
  # restores the venv exactly (uv 0.11.2, reproduced for retro#1110). METADATA is
  # mandatory, so a dist-info dir without it is never a valid install.
  local stale
  stale=$( { find "$API_DIR/api/.venv/lib" -mindepth 3 -maxdepth 3 -type d -name '*.dist-info' \
             ! -exec test -e '{}/METADATA' ';' -print 2>/dev/null || true; } )
  if [[ -n "$stale" ]]; then
    log "removing dist-info dirs with no METADATA (an interrupted uninstall): $(xargs -n1 basename <<<"$stale" | tr '\n' ' ')"
    xargs -d '\n' $AS_UBUNTU rm -rf -- <<<"$stale" || return 1
  fi
  ( cd "$API_DIR/api" && $AS_UBUNTU $UV sync --frozen --quiet ) || return 1
  # The sync's exit code is not enough: after a failed sync the in-place rollback exited 0
  # with litellm's files gone. `uv pip check` failed on that venv and on the 2026-11-26
  # one (fastapi removed); importing the top-level packages passed the first. It does not
  # import forecast_api.main, which validates settings at import and needs the unit's
  # environment (oracle_api_key is required).
  ( cd "$API_DIR/api" && $AS_UBUNTU $UV pip check --python "$API_DIR/api/.venv/bin/python" ) || return 1
}

log "uv sync --frozen..."
if ! sync_venv; then
  log "FAIL: the venv did not sync cleanly at ${AFTER:0:8} — rolling back to ${ROLLBACK:0:8}"
  $AS_UBUNTU git reset --hard "$ROLLBACK" --quiet
  if sync_venv; then
    log "rolled back: checkout and venv are at ${ROLLBACK:0:8} and pass uv pip check; the service was not reloaded"
  else
    # HEAD now equals the marker, so a re-run of the last good ref would say "no change"
    # over a broken venv. Drop the marker so the next run syncs.
    $AS_UBUNTU rm -f "$DEPLOYED_FILE"
    log "ROLLBACK FAILED: the venv at ${ROLLBACK:0:8} does not pass uv pip check — the next worker recycle may fail to boot; fix the venv by hand"
  fi
  exit 1
fi
cd "$API_DIR/api"   # the steps below ran from api/ before retro#1110; keep it that way

# ── 2a. Re-install the Metaculus sync timer (retro#754) ─────────────────────
# The fetch+reset above already updated the unit files in the checkout, but a
# systemd unit file only takes effect once it's copied to /etc/systemd/system
# and daemon-reload runs — install_metaculus_timer.sh does both and is
# idempotent (see its own docstring), so it's safe and cheap to re-run on
# every real deploy rather than only when these specific files changed.
if [[ -f "$API_DIR/infra/install_metaculus_timer.sh" ]]; then
  log "re-installing metaculus-sync timer..."
  sudo bash "$API_DIR/infra/install_metaculus_timer.sh"
fi

# ── 2a'. Re-install the Polymarket paper bot timer (retro#620) ──────────────
# Same idempotent pattern; the installer also runs the bot project's uv sync.
if [[ -f "$API_DIR/infra/install_polymarket_paper_timer.sh" ]]; then
  log "re-installing polymarket-paper timer..."
  sudo bash "$API_DIR/infra/install_polymarket_paper_timer.sh"
fi

# ── 2a''. Re-install the Manifold paper bot timer (retro#918) ───────────────
if [[ -f "$API_DIR/infra/install_manifold_paper_timer.sh" ]]; then
  log "re-installing manifold-paper timer..."
  sudo bash "$API_DIR/infra/install_manifold_paper_timer.sh"
fi

# ── 2b. Write build provenance (read by /version and /health) ─────────────────
# Generated, environment-specific — not committed (see .gitignore). The runtime
# git fallback in _build.py covers dev where this file is absent. The semver is
# always read from package metadata by _build.py, so it's not written here.
BUILD_FILE="$API_DIR/api/src/forecast_api/_build_info.json"
BRANCH=$($AS_UBUNTU git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "main")
# Auto-incrementing build number = commit count (monotonic across merges to main).
BUILD=$($AS_UBUNTU git rev-list --count HEAD 2>/dev/null || echo 0)
$AS_UBUNTU bash -c "printf '{\"git_sha\":\"%s\",\"git_branch\":\"%s\",\"built_at\":\"%s\",\"build\":%s,\"source\":\"deploy\"}\n' \
  '$AFTER' '$BRANCH' '$(date -u +%FT%TZ)' '$BUILD' > '$BUILD_FILE'"
log "wrote build info: build $BUILD @ ${AFTER:0:8} ($BRANCH)"

# ── 3. Health probe helper ───────────────────────────────────────────────────
# Require N *consecutive* 200s rather than the first 200. A SIGHUP reload runs
# new workers alongside gracefully-draining old ones; a single early 200 can be
# served by an old (still-good) worker even when the new workers are broken
# (e.g. a native-dep upgrade where SIGHUP swaps the httptools C-extension under
# an already-imported uvicorn — the bug that took the API down on 2026-06-02).
# Demanding consecutive 200s spanning the drain window catches that.
GRACEFUL_DRAIN="${DEPLOY_GRACEFUL_DRAIN:-30}"   # must exceed gunicorn --graceful-timeout so old workers are gone
healthy() {
  local need=5 streak=0 http
  for _ in $(seq 1 "$((GRACEFUL_DRAIN + 12))"); do
    http=$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 "$HEALTH_URL" || echo 000)
    if [[ "$http" == "200" ]]; then
      streak=$((streak + 1))
      [[ "$streak" -ge "$need" ]] && return 0
    else
      streak=0
    fi
    sleep "$HEALTH_SLEEP"
  done
  return 1
}

# ── 4. Reload, then verify — escalate to full restart if unhealthy ────────────
# SIGHUP first (zero-downtime: keeps the listening socket, no 502 window). If
# the reloaded workers aren't durably healthy, a full restart gives a fresh
# master that imports the whole upgraded stack consistently. Only if *that*
# still fails do we go red.
log "reloading oracle-api (SIGHUP)..."
$SYSTEMCTL reload-or-restart oracle-api

if healthy; then
  $AS_UBUNTU bash -c "echo '$AFTER' > '$DEPLOYED_FILE'"
  log "ok — deployed ${AFTER:0:8} (healthy after SIGHUP reload)"
  exit 0
fi

log "reload unhealthy — full restart (SIGHUP can't swap native worker deps)..."
$SYSTEMCTL restart oracle-api

if healthy; then
  $AS_UBUNTU bash -c "echo '$AFTER' > '$DEPLOYED_FILE'"
  log "ok — deployed ${AFTER:0:8} (healthy after full restart)"
  exit 0
fi

log "FAIL: /health not durably 200 after reload+restart — service is unhealthy"
exit 1
