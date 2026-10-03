#!/bin/bash
# Install (or refresh) the bayesoracle node-series cron line (retro#577, #896)
# in the `ubuntu` user's crontab on the oracle box.
#
# Runs ON the EC2 box, as root, BY HAND (or via SSM). deploy_oracle.sh does NOT
# call it, so merging a change to the schedule never edits the box by itself.
#
#   sudo bash /home/ubuntu/truthmachine/infra/install_oracle_series_cron.sh
#
# Idempotent: every existing line that runs series/log_nodes.py is replaced by
# the single line in oracle-series.crontab; other crontab lines are kept. The
# previous crontab is backed up to /home/ubuntu/oracle-series/ first; restore
# with:  crontab -u ubuntu /home/ubuntu/oracle-series/crontab.bak.<ts>
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LINE_FILE="$HERE/oracle-series.crontab"
SCRIPT="${ORACLE_SERIES_SCRIPT:-/home/ubuntu/truthmachine/bayesoracle/series/log_nodes.py}"
BACKUP_DIR="${ORACLE_SERIES_DIR:-/home/ubuntu/oracle-series}"
CRON_USER="${ORACLE_SERIES_USER:-ubuntu}"

log() { echo "[install_oracle_series_cron $(date '+%H:%M:%S')] $*"; }

[[ $EUID -eq 0 ]] || { echo "must run as root (crontab -u $CRON_USER)"; exit 1; }
[[ -f "$LINE_FILE" ]] || { echo "missing $LINE_FILE"; exit 1; }

new_line=$(grep -v '^[[:space:]]*#' "$LINE_FILE" | grep -v '^[[:space:]]*$')
[[ $(printf '%s\n' "$new_line" | wc -l) -eq 1 ]] || { echo "$LINE_FILE must hold exactly one cron line"; exit 1; }

# The crontab passes --min-interval-days; a log_nodes.py that predates it would
# die on argparse every morning. Refuse until the batch tree has synced.
if [[ "$new_line" == *--min-interval-days* ]] && ! grep -q -- '--min-interval-days' "$SCRIPT"; then
  echo "refusing: $SCRIPT has no --min-interval-days yet (batch tree not synced to main?)"
  exit 1
fi

mkdir -p "$BACKUP_DIR"
backup="$BACKUP_DIR/crontab.bak.$(date -u '+%Y%m%dT%H%M%SZ')"
current=$(crontab -l -u "$CRON_USER" 2>/dev/null || true)
printf '%s\n' "$current" > "$backup"
chown "$CRON_USER": "$backup" 2>/dev/null || true
log "backed up current crontab to $backup"

{
  printf '%s\n' "$current" | grep -v 'series/log_nodes\.py' | grep -v '^$' || true
  printf '%s\n' "$new_line"
} | crontab -u "$CRON_USER" -

log "--- diff (old -> new) ---"
diff <(cat "$backup") <(crontab -l -u "$CRON_USER") || true
log "done. Rollback: crontab -u $CRON_USER $backup"
