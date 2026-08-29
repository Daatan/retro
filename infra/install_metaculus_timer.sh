#!/bin/bash
# Install (or refresh) the Metaculus sync systemd timer on the oracle box.
#
# Runs ON the EC2 box, by hand or via SSM. Idempotent — safe to re-run after a
# deploy that changed the unit files.
#
#   sudo bash /home/ubuntu/oracle-api/infra/install_metaculus_timer.sh
#
# This installs the SCHEDULE. It does not install the CREDENTIALS: the service
# is gated on /home/ubuntu/truthmachine/.env.metaculus existing (retro#725), so
# until that file is placed the timer fires and the service is skipped cleanly
# rather than failing every 20 minutes. That is deliberate — it means the
# schedule can be installed before the key exists.
set -euo pipefail

API_DIR="/home/ubuntu/oracle-api"
ENV_FILE="/home/ubuntu/truthmachine/.env.metaculus"

log() { echo "[install_metaculus_timer $(date '+%H:%M:%S')] $*"; }

[[ $EUID -eq 0 ]] || { echo "must run as root (systemctl + /etc/systemd)"; exit 1; }

for unit in metaculus-sync.service metaculus-sync.timer; do
  src="$API_DIR/infra/$unit"
  [[ -f "$src" ]] || { echo "missing $src — is the checkout up to date?"; exit 1; }
  log "installing $unit"
  install -m 0644 "$src" "/etc/systemd/system/$unit"
done

log "daemon-reload"
systemctl daemon-reload

log "enabling timer"
systemctl enable --now metaculus-sync.timer

log "--- timer status ---"
systemctl --no-pager status metaculus-sync.timer | head -12 || true

log "--- next scheduled runs ---"
systemctl list-timers --no-pager metaculus-sync.timer || true

if [[ -f "$ENV_FILE" ]]; then
  log "credentials present at $ENV_FILE — the service will actually run"
  # Warn on loose permissions rather than fixing them silently; the file holds
  # the bot token and the Oracle relay key.
  perms=$(stat -c '%a %U' "$ENV_FILE")
  log "env file mode/owner: $perms (expected '600 root' or '600 ubuntu')"
else
  log "NOTE: $ENV_FILE does not exist yet (retro#725)."
  log "      The timer is live but every activation will be SKIPPED by"
  log "      ConditionPathExists until the credentials are placed. This is"
  log "      the intended pre-provisioning state, not a failure."
fi

log "done. Follow runs with:  journalctl -u metaculus-sync.service -f"
