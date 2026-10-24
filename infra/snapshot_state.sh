#!/bin/bash
# snapshot_state.sh — back up the small, irreplaceable Oracul-box state to S3.
#
# snapshot_atlas.sh covers atlas/ + vault2/ (expensive LLM output). This covers
# what is *not* rebuildable at any price — point-in-time ledgers and series
# that exist only on this disk (platform#42 item B):
#
#   data/polymarket_paper/            paper-bot ledger = the retro#620 scoreboard
#   data/manifold_paper/              paper-bot ledger (retro#918)
#   data/jev_values/                  diskcache, Jev pass-1 max_noul (retro#955)
#   data/resolution_feedback.jsonl    pushed by daatan /leaderboard/ingest
#   data/settlement_pin_ledger.jsonl  "
#   data/.resolution_feedback_dedup_cache/, data/.settlement_pin_ledger_dedup_cache/
#   data/*leaderboard*.json, data/resolution_tags.json
#   $HOME/oracle-series/nodes.jsonl   daily BayesOracle node series
#
# Total is a few MB. NO secrets go in this bucket — never add .env* here.
#
# Layout: s3://<bucket>/snapshots/state-<iso-ts>.tgz only. There is deliberately
# no latest pointer: the bucket's 30-day lifecycle covers snapshots/, while a
# separate overwritten key would pile up noncurrent versions (the 7-day
# noncurrent rule matches only the latest.tgz prefix). restore_state.sh picks
# the newest snapshots/state-*.tgz.
#
# diskcache dirs are SQLite in WAL mode; a plain tar of a live cache.db can
# miss the WAL tail, so each *.db is copied with the sqlite3 backup API and
# the -wal/-shm files are left out.
#
# Called from ec2_run.sh after snapshot_atlas.sh. Non-fatal on any failure.
#
# Env overrides: see snapshot_atlas.sh; SERIES_DIR (default $HOME/oracle-series)
set -eu

BUCKET="${ATLAS_SNAPSHOT_BUCKET:-truthmachine-atlas-snapshots-272007598366}"
REGION="${AWS_REGION:-eu-central-1}"
WORKDIR="${WORKDIR:-$HOME/truthmachine}"
DATA_DIR="${DATA_DIR:-$WORKDIR/data}"
SERIES_DIR="${SERIES_DIR:-$HOME/oracle-series}"

log() { echo "[snapshot_state $(date -u '+%H:%M:%S')] $*" >&2; }

# Skip on a data-wiped box: without the scoreboard ledger there is nothing
# worth saving, and an emptier tarball would become the newest restore point.
if [ ! -s "$DATA_DIR/polymarket_paper/trades.jsonl" ] && [ ! -s "$DATA_DIR/manifold_paper/trades.jsonl" ]; then
  log "no paper-bot ledgers, skipping snapshot"
  exit 0
fi

STAGE=$(mktemp -d)
TMPFILE=$(mktemp --suffix=.tgz)
trap 'rm -rf "$STAGE" "$TMPFILE"' EXIT
mkdir -p "$STAGE/data" "$STAGE/oracle-series"

cd "$DATA_DIR"
for p in polymarket_paper manifold_paper jev_values \
         resolution_feedback.jsonl settlement_pin_ledger.jsonl \
         .resolution_feedback_dedup_cache .settlement_pin_ledger_dedup_cache \
         resolution_tags.json *leaderboard*.json; do
  [ -e "$p" ] || continue
  cp -a "$p" "$STAGE/data/" || log "WARNING: copy of $p failed"
done
[ -f "$SERIES_DIR/nodes.jsonl" ] && cp -a "$SERIES_DIR/nodes.jsonl" "$STAGE/oracle-series/"

# Consistent SQLite copies: overwrite each staged *.db with a backup-API copy
# of the live file, then drop the now-meaningless -wal/-shm.
while IFS= read -r db; do
  rel="${db#"$STAGE"/data/}"
  if ! python3 - "$DATA_DIR/$rel" "$db" <<'PY'
import sqlite3, sys
src = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
dst = sqlite3.connect(sys.argv[2] + ".tmp")
src.backup(dst)
dst.close(); src.close()
PY
  then
    log "WARNING: sqlite backup of $rel failed, keeping raw copy"
    rm -f "$db.tmp"
    continue
  fi
  mv "$db.tmp" "$db"
  rm -f "$db-wal" "$db-shm"
done < <(find "$STAGE/data" -name '*.db' -type f)

if ! tar -czf "$TMPFILE" -C "$STAGE" data oracle-series; then
  log "WARNING: tar failed, skipping upload"
  exit 0
fi

SIZE=$(stat -c%s "$TMPFILE" 2>/dev/null || echo "?")
TS=$(date -u +%Y-%m-%dT%H-%M-%SZ)

if ! aws s3 cp --region "$REGION" --only-show-errors "$TMPFILE" "s3://${BUCKET}/snapshots/state-${TS}.tgz"; then
  log "WARNING: snapshots/state-${TS}.tgz upload failed"
  exit 0
fi

log "ok — ${SIZE}B to s3://${BUCKET}/snapshots/state-${TS}.tgz"
