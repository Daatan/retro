#!/bin/bash
# Round-trip test for snapshot_state.sh / restore_state.sh (platform#42 item B).
#
# A fake `aws` on PATH maps s3://<bucket>/<key> to a local dir, so the real
# scripts run unmodified. Checks: the skip guard, a WAL-only row surviving the
# sqlite backup copy, no secrets in the tarball, restore onto an empty box, and
# restore never clobbering a file that already exists.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFRA="$SCRIPT_DIR/.."
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
FAIL=0
check() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; FAIL=1; fi; }

mkdir -p "$T/bin" "$T/s3"
cat > "$T/bin/aws" <<EOF
#!/bin/bash
# fake aws: only 's3 cp' and 's3 ls' with --region/--only-show-errors
args=(); for a in "\$@"; do case "\$a" in --region) skip=1;; --only-show-errors) ;; *) if [ -n "\${skip:-}" ]; then skip=; else args+=("\$a"); fi;; esac; done
p() { echo "$T/s3/\${1#s3://}"; }
case "\${args[1]}" in
  cp) src="\${args[2]}"; dst="\${args[3]}"
      [[ \$src == s3://* ]] && src=\$(p "\$src"); [[ \$dst == s3://* ]] && { dst=\$(p "\$dst"); mkdir -p "\$(dirname "\$dst")"; }
      cp "\$src" "\$dst";;
  ls) pre=\$(p "\${args[2]}"); d=\$(dirname "\$pre"); b=\$(basename "\$pre")
      [ -d "\$d" ] || exit 1; for f in "\$d"/"\$b"*; do [ -e "\$f" ] && echo "2026-10-24 00:00:00 1 \$(basename "\$f")"; done;;
esac
EOF
chmod +x "$T/bin/aws"
export PATH="$T/bin:$PATH" ATLAS_SNAPSHOT_BUCKET=b

# ── source box ──────────────────────────────────────────────────────────────
SRC="$T/src"; mkdir -p "$SRC/data/polymarket_paper" "$SRC/data/jev_values" "$SRC/data/.settlement_pin_ledger_dedup_cache" "$SRC/series"

# guard: no ledgers → no upload
DATA_DIR="$SRC/data" SERIES_DIR="$SRC/series" bash "$INFRA/snapshot_state.sh" 2>/dev/null
check "skip guard uploads nothing" '[ -z "$(ls "$T/s3/b/snapshots" 2>/dev/null)" ]'

echo '{"t":1}' > "$SRC/data/polymarket_paper/trades.jsonl"
echo '{"f":1}' > "$SRC/data/resolution_feedback.jsonl"
echo '{}' > "$SRC/data/resolution_leaderboard.json"
echo 'x' > "$SRC/data/.settlement_pin_ledger_dedup_cache/k"
echo 'SECRET=1' > "$SRC/data/.env"
echo '{"n":1}' > "$SRC/series/nodes.jsonl"
# keep a writer open so the row lives only in the WAL while we snapshot
python3 - "$SRC/data/jev_values/cache.db" <<'PY' &
import sqlite3, sys, time
c = sqlite3.connect(sys.argv[1]); c.execute("pragma journal_mode=wal"); c.execute("pragma wal_autocheckpoint=0")
c.execute("create table v(k)"); c.execute("insert into v values ('wal-row')"); c.commit()
open(sys.argv[1] + ".ready", "w").close(); time.sleep(5)
PY
WRITER=$!
for _ in $(seq 50); do [ -e "$SRC/data/jev_values/cache.db.ready" ] && break; sleep 0.1; done
rm -f "$SRC/data/jev_values/cache.db.ready"

DATA_DIR="$SRC/data" SERIES_DIR="$SRC/series" bash "$INFRA/snapshot_state.sh" 2>/dev/null
kill "$WRITER" 2>/dev/null || true; wait "$WRITER" 2>/dev/null || true
TGZ=$(ls "$T/s3/b/snapshots"/state-*.tgz)
# shellcheck disable=SC2034  # read inside check's eval
LIST=$(tar -tzf "$TGZ")
check "tarball uploaded" '[ -s "$TGZ" ]'
check "ledger + series + dotdir in tarball" 'grep -q "data/polymarket_paper/trades.jsonl" <<<"$LIST" && grep -q "oracle-series/nodes.jsonl" <<<"$LIST" && grep -q "data/.settlement_pin_ledger_dedup_cache/k" <<<"$LIST"'
check "no .env in tarball" '! grep -q "\.env" <<<"$LIST"'
check "no -wal/-shm in tarball" '! grep -qE "db-(wal|shm)$" <<<"$LIST"'

# ── restore onto an empty box ───────────────────────────────────────────────
DST="$T/dst"
DATA_DIR="$DST/data" SERIES_DIR="$DST/series" bash "$INFRA/restore_state.sh" 2>/dev/null
check "ledger restored" 'cmp -s "$SRC/data/polymarket_paper/trades.jsonl" "$DST/data/polymarket_paper/trades.jsonl"'
check "series restored" '[ -s "$DST/series/nodes.jsonl" ]'
check "dotdir restored" '[ -s "$DST/data/.settlement_pin_ledger_dedup_cache/k" ]'
check "WAL-only row survived" '[ "$(python3 -c "import sqlite3,sys;print(sqlite3.connect(sys.argv[1]).execute(\"select k from v\").fetchone()[0])" "$DST/data/jev_values/cache.db")" = wal-row ]'

# ── restore never clobbers ──────────────────────────────────────────────────
echo 'newer' > "$DST/data/resolution_feedback.jsonl"
rm -rf "$DST/data/polymarket_paper"
DATA_DIR="$DST/data" SERIES_DIR="$DST/series" bash "$INFRA/restore_state.sh" 2>/dev/null
check "existing file kept" '[ "$(cat "$DST/data/resolution_feedback.jsonl")" = newer ]'
check "missing item re-restored" '[ -s "$DST/data/polymarket_paper/trades.jsonl" ]'

# ── no snapshot at all → no-op ──────────────────────────────────────────────
ATLAS_SNAPSHOT_BUCKET=empty DATA_DIR="$T/none" SERIES_DIR="$T/none-s" bash "$INFRA/restore_state.sh" 2>/dev/null
check "no snapshot is a no-op" '[ ! -e "$T/none" ]'

exit $FAIL
