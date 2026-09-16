#!/bin/bash
# Install (or refresh) the Polymarket paper bot systemd timer on the oracle box.
#
# Runs ON the EC2 box, by hand or via SSM. Idempotent; deploy_oracle.sh calls
# it on every real deploy, same as install_metaculus_timer.sh.
#
#   sudo bash /home/ubuntu/oracle-api/infra/install_polymarket_paper_timer.sh
#
# Installs the SCHEDULE only. The service is gated on the Metaculus credentials
# file existing (it reuses that relay key); until then the timer fires and the
# service is skipped cleanly.
set -euo pipefail

API_DIR="/home/ubuntu/oracle-api"
ENV_FILE="/home/ubuntu/truthmachine/.env.metaculus"
LEDGER_DIR="/home/ubuntu/truthmachine/data/polymarket_paper"

log() { echo "[install_polymarket_paper_timer $(date '+%H:%M:%S')] $*"; }

[[ $EUID -eq 0 ]] || { echo "must run as root (systemctl + /etc/systemd)"; exit 1; }

for unit in polymarket-paper.service polymarket-paper.timer; do
  src="$API_DIR/infra/$unit"
  [[ -f "$src" ]] || { echo "missing $src — is the checkout up to date?"; exit 1; }
  log "installing $unit"
  install -m 0644 "$src" "/etc/systemd/system/$unit"
done

# The bot's uv project needs its lockfile resolved once on the box (--frozen at
# run time refuses to resolve). Same pattern deploy_oracle.sh uses for api/.
if [[ -f "$API_DIR/polymarket_paper/uv.lock" ]]; then
  log "uv sync --frozen (polymarket_paper)"
  (cd "$API_DIR/polymarket_paper" && sudo -u ubuntu /home/ubuntu/.local/bin/uv sync --frozen --quiet) || log "WARN: uv sync failed; the service will fail until it succeeds"
fi

# Ledger directory owned by the user the service runs as; the API reads it.
install -d -m 0755 -o ubuntu -g ubuntu "$LEDGER_DIR"

log "daemon-reload"
systemctl daemon-reload

log "enabling timer"
systemctl enable --now polymarket-paper.timer

log "--- next scheduled runs ---"
systemctl list-timers --no-pager polymarket-paper.timer || true

if [[ -f "$ENV_FILE" ]]; then
  log "credentials present at $ENV_FILE — the service will actually run"
else
  log "NOTE: $ENV_FILE does not exist; every activation is SKIPPED by ConditionPathExists until it does."
fi

log "done. Follow runs with:  journalctl -u polymarket-paper.service -n 100"
