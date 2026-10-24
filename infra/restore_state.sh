#!/bin/bash
# restore_state.sh — pull the newest snapshots/state-*.tgz (see
# snapshot_state.sh) back onto a fresh box.
#
# Called from ec2_bootstrap.sh next to restore_atlas.sh. Per-item and
# non-clobbering: each top-level entry in the tarball is restored only if it
# does not already exist locally, so running this on a healthy box is a no-op
# and a half-populated box only gets the missing pieces.
#
# Also a no-op if no state snapshot exists yet.
#
# Env overrides: see snapshot_state.sh
set -eu

BUCKET="${ATLAS_SNAPSHOT_BUCKET:-truthmachine-atlas-snapshots-272007598366}"
REGION="${AWS_REGION:-eu-central-1}"
WORKDIR="${WORKDIR:-$HOME/truthmachine}"
DATA_DIR="${DATA_DIR:-$WORKDIR/data}"
SERIES_DIR="${SERIES_DIR:-$HOME/oracle-series}"

log() { echo "[restore_state $(date -u '+%H:%M:%S')] $*" >&2; }

# ISO timestamps sort lexically, so the last key is the newest.
KEY=$(aws s3 ls --region "$REGION" "s3://${BUCKET}/snapshots/state-" 2>/dev/null \
      | awk '{print $4}' | grep '^state-.*\.tgz$' | sort | tail -n1 || true)
if [ -z "$KEY" ]; then
  log "no s3://${BUCKET}/snapshots/state-*.tgz found, nothing to restore"
  exit 0
fi

STAGE=$(mktemp -d)
TMPFILE=$(mktemp --suffix=.tgz)
trap 'rm -rf "$STAGE" "$TMPFILE"' EXIT

log "downloading snapshots/${KEY}..."
if ! aws s3 cp --region "$REGION" --only-show-errors "s3://${BUCKET}/snapshots/${KEY}" "$TMPFILE"; then
  log "WARNING: download failed, nothing restored"
  exit 0
fi
if ! tar -xzf "$TMPFILE" -C "$STAGE"; then
  log "WARNING: tar extract failed, nothing restored"
  exit 0
fi

restored=0
skipped=0
restore_into() {  # <staged dir> <target dir>
  local src="$1" dst="$2" entry name
  [ -d "$src" ] || return 0
  mkdir -p "$dst"
  for entry in "$src"/* "$src"/.[!.]*; do
    [ -e "$entry" ] || continue
    name=$(basename "$entry")
    if [ -e "$dst/$name" ]; then
      skipped=$((skipped + 1))
      continue
    fi
    cp -a "$entry" "$dst/$name"
    restored=$((restored + 1))
  done
}
restore_into "$STAGE/data" "$DATA_DIR"
restore_into "$STAGE/oracle-series" "$SERIES_DIR"

log "ok — from ${KEY}: restored ${restored}, kept ${skipped} already present"
