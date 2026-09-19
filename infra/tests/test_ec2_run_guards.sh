#!/bin/bash
# Regression tests for ec2_run.sh's sync + re-exec guards (retro#553, retro#557).
#
# These guards only execute on the unhappy path (a dead git process, a stale
# lock, a mid-flight self-update) — exactly the code that rots unnoticed, and
# the reason the batch tree ran two-day-old code twice (2026-07-02 -> 08-16 and
# 2026-08-17 -> 08-19) before anyone caught it.
#
# The functions are extracted verbatim out of ec2_run.sh (not reimplemented
# here) so a change to the real file is what these tests exercise, not a copy
# that can drift. `reap_stale_git_lock`/`sync_to_main`/`reexec_if_self_changed`
# are sourced out of the script by name; the file's own top-level `while true`
# main loop is never reached, since we never source the whole file.
#
# Two subtleties preserved on purpose (see ec2_run.sh's own comments):
#   1. `sync_to_main` is invoked the way run_pipeline calls it — errexit ON,
#      called under `||` — because bash suspends errexit for every command
#      INSIDE a function called that way, not just the function's own exit
#      status. Testing it bare (no `||`) exercises different semantics.
#   2. Every assertion checks the actual return code, not the log line — the
#      #553 bug was precisely a failure that logged a warning and kept going.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EC2_RUN="$SCRIPT_DIR/../ec2_run.sh"
FAIL=0

# ── extract the guard functions out of the real file ──────────────────────
# Each of the three targets is a top-level `name() {` ... standalone `}` block;
# `log()` is a one-liner grabbed separately since the multi-line extractor
# below assumes a closing brace on its own line.
extract_function() {
  local name="$1"
  awk -v fn="${name}() {" '
    $0 == fn { p = 1 }
    p { print }
    p && /^}$/ { exit }
  ' "$EC2_RUN"
}

LOG_LINE="$(grep -m1 '^log() {' "$EC2_RUN")"
[[ -n "$LOG_LINE" ]] || { echo "FAIL: could not find log() in $EC2_RUN"; exit 1; }
eval "$LOG_LINE"

for fn in reap_stale_git_lock reap_stale_rebase_state sync_to_main reexec_if_self_changed; do
  body="$(extract_function "$fn")"
  [[ -n "$body" ]] || { echo "FAIL: could not extract ${fn}() from $EC2_RUN"; exit 1; }
  eval "$body"
done

SLEEP_INTERVAL=300 # must match ec2_run.sh's own value — the lock-age branch depends on it

# ── test scaffolding ───────────────────────────────────────────────────────
TMPROOT="$(mktemp -d)"
trap 'rm -rf "$TMPROOT"' EXIT

# Fresh bare "origin" plus two working clones: WORKDIR (the batch tree these
# functions run inside) and PRODUCER (stands in for another contributor
# pushing to origin/main — how WORKDIR ends up behind).
ORIGIN="$TMPROOT/origin.git"
git init --bare -q -b main "$ORIGIN"

setup_clone() {
  local dir="$1"
  # An empty origin (before PRODUCER's first push) makes git warn on clone —
  # harmless, but noisy in CI logs.
  git clone -q "$ORIGIN" "$dir" 2>/dev/null
  git -C "$dir" config user.email "test@example.com"
  git -C "$dir" config user.name "Test"
}

PRODUCER="$TMPROOT/producer"
setup_clone "$PRODUCER"
echo "seed" > "$PRODUCER/file.txt"
git -C "$PRODUCER" add file.txt
git -C "$PRODUCER" commit -q -m "seed"
git -C "$PRODUCER" push -q origin main

pass() { echo "PASS: $1"; }
fail() { echo "FAIL: $1"; FAIL=1; }

# Push one more commit to origin from PRODUCER, so a freshly-cloned WORKDIR
# can be made "behind" by resetting it back to the seed commit.
advance_origin() {
  echo "advance-$RANDOM" >> "$PRODUCER/file.txt"
  git -C "$PRODUCER" commit -q -am "advance"
  git -C "$PRODUCER" push -q origin main
}

# Runs sync_to_main the way run_pipeline does: errexit on, called as the
# operand of `||`, so bash suspends errexit for every command inside it —
# not just its own final exit status. Returns sync_to_main's REAL exit code,
# not run_pipeline's collapsed `return 1`.
run_sync_to_main() {
  local rc=0
  ( set -euo pipefail; cd "$WORKDIR"; sync_to_main || exit $? )
  rc=$?
  return $rc
}

# ── case: no lock, already in sync — clean no-op ───────────────────────────
{
  WORKDIR="$TMPROOT/wd-noop"
  setup_clone "$WORKDIR"
  before="$(git -C "$WORKDIR" rev-parse HEAD)"

  rc=0
  OUT="$(run_sync_to_main 2>&1)" || rc=$?
  after="$(git -C "$WORKDIR" rev-parse HEAD)"

  if [[ $rc -eq 0 && "$before" == "$after" ]]; then
    pass "already-in-sync: clean no-op, rc=0"
  else
    fail "already-in-sync: rc=$rc before=$before after=$after — $OUT"
  fi
}

# ── case: stale lock, no live git process — reaped, fast-forwards, rc=0 ───
{
  WORKDIR="$TMPROOT/wd-stale-lock"
  setup_clone "$WORKDIR"
  advance_origin # WORKDIR was cloned before this, so it's now behind origin

  lock="$WORKDIR/.git/index.lock"
  : > "$lock"
  touch -d "-10 minutes" "$lock" # older than SLEEP_INTERVAL=300s

  rc=0
  OUT="$(run_sync_to_main 2>&1)" || rc=$?
  origin_head="$(git -C "$ORIGIN" rev-parse main)"
  wd_head="$(git -C "$WORKDIR" rev-parse HEAD)"

  if [[ $rc -eq 0 && ! -e "$lock" && "$wd_head" == "$origin_head" ]]; then
    pass "stale-lock-no-process: reaped, fast-forwarded, rc=0"
  else
    fail "stale-lock-no-process: rc=$rc lock-exists=$([[ -e "$lock" ]] && echo yes || echo no) wd=$wd_head origin=$origin_head — $OUT"
  fi
}

# ── case: fresh lock, tree behind — rc=1, refuses, lock preserved ─────────
{
  WORKDIR="$TMPROOT/wd-fresh-lock"
  setup_clone "$WORKDIR"
  advance_origin

  lock="$WORKDIR/.git/index.lock"
  : > "$lock"
  touch -d "-10 seconds" "$lock" # well under SLEEP_INTERVAL=300s — reap_stale_git_lock leaves it

  rc=0
  OUT="$(run_sync_to_main 2>&1)" || rc=$?

  if [[ $rc -eq 1 && -e "$lock" && "$OUT" == *"REFUSING to run on stale code"* ]]; then
    pass "fresh-lock-behind: rc=1, REFUSING logged, lock preserved"
  else
    fail "fresh-lock-behind: rc=$rc lock-exists=$([[ -e "$lock" ]] && echo yes || echo no) — $OUT"
  fi
}

# ── case: stale lock but a live git process in the repo — lock left alone ─
{
  WORKDIR="$TMPROOT/wd-live-process"
  setup_clone "$WORKDIR"

  lock="$WORKDIR/.git/index.lock"
  : > "$lock"
  touch -d "-10 minutes" "$lock" # stale by age alone

  # A background process whose executable basename is literally "git" (so it
  # shows up in `pgrep -a git`) and whose command line names $WORKDIR (so the
  # real `grep -q "$WORKDIR"` filter matches it) — a real stand-in, not a
  # string the function has to special-case.
  FAKEBIN="$TMPROOT/fakebin"
  mkdir -p "$FAKEBIN"
  cat > "$FAKEBIN/git" <<'EOF'
#!/bin/bash
sleep 30
EOF
  chmod +x "$FAKEBIN/git"
  "$FAKEBIN/git" "$WORKDIR" &
  LIVE_PID=$!
  # Give pgrep a moment to see the new process.
  for _ in $(seq 1 20); do
    pgrep -a git 2>/dev/null | grep -q "$WORKDIR" && break
    sleep 0.1
  done

  REAP_OUT_FILE="$TMPROOT/reap-out"
  ( set -euo pipefail; reap_stale_git_lock ) >"$REAP_OUT_FILE" 2>&1 || true
  REAP_OUT="$(cat "$REAP_OUT_FILE")"

  kill "$LIVE_PID" 2>/dev/null || true
  wait "$LIVE_PID" 2>/dev/null || true

  if [[ -e "$lock" ]]; then
    pass "stale-lock-live-process: lock left alone"
  else
    fail "stale-lock-live-process: lock was removed despite a live git process — $REAP_OUT"
  fi
}

# ── retro#838: gutted rebase-merge (autostash only) — reaped, rebase works again ─
# The exact shape found on the box: a directory `git rebase --abort` cannot read,
# which made every later `git rebase` refuse for 27 days.
{
  WORKDIR="$TMPROOT/wd-stale-rebase"
  setup_clone "$WORKDIR"

  rdir="$WORKDIR/.git/rebase-merge"
  mkdir "$rdir"
  git -C "$WORKDIR" rev-parse HEAD > "$rdir/autostash"
  touch -d "-10 minutes" "$rdir"

  # Precondition: this really is the failure — otherwise the test proves nothing.
  pre_rc=0
  git -C "$WORKDIR" rebase origin/main >/dev/null 2>&1 || pre_rc=$?

  rc=0
  OUT="$(run_sync_to_main 2>&1)" || rc=$?
  post_rc=0
  git -C "$WORKDIR" rebase origin/main >/dev/null 2>&1 || post_rc=$?
  rescued="$(git -C "$WORKDIR" for-each-ref refs/rescued | wc -l)"

  if [[ $pre_rc -ne 0 && $rc -eq 0 && ! -d "$rdir" && $post_rc -eq 0 && $rescued -eq 1 ]]; then
    pass "stale-rebase-merge: reaped, autostash ref rescued, rebase works again"
  else
    fail "stale-rebase-merge: pre_rc=$pre_rc rc=$rc dir-exists=$([[ -d "$rdir" ]] && echo yes || echo no) post_rc=$post_rc rescued=$rescued — $OUT"
  fi
}

# ── retro#838: fresh rebase-merge — may belong to a rebase still starting up ─
{
  WORKDIR="$TMPROOT/wd-fresh-rebase"
  setup_clone "$WORKDIR"
  rdir="$WORKDIR/.git/rebase-merge"
  mkdir "$rdir"

  ( set -euo pipefail; reap_stale_rebase_state ) >/dev/null 2>&1 || true

  if [[ -d "$rdir" ]]; then
    pass "fresh-rebase-merge: left alone"
  else
    fail "fresh-rebase-merge: removed although younger than a cycle"
  fi
}

# ── retro#838: ahead-only is not stale — rc=0 and the commits get published ─
# `merge --ff-only` is a successful no-op when HEAD already contains origin/main,
# so the reset fallback never ran and the equality check refused every cycle.
{
  WORKDIR="$TMPROOT/wd-ahead"
  setup_clone "$WORKDIR"
  echo "atlas" > "$WORKDIR/atlas.html"
  git -C "$WORKDIR" add atlas.html
  git -C "$WORKDIR" commit -q -m "atlas: local, unpushed"

  rc=0
  OUT="$(run_sync_to_main 2>&1)" || rc=$?
  origin_head="$(git -C "$ORIGIN" rev-parse main)"
  wd_head="$(git -C "$WORKDIR" rev-parse HEAD)"

  if [[ $rc -eq 0 && "$wd_head" == "$origin_head" && "$OUT" != *"REFUSING"* ]]; then
    pass "ahead-only: rc=0, unpushed commit published, no refusal"
  else
    fail "ahead-only: rc=$rc wd=$wd_head origin=$origin_head — $OUT"
  fi
  git -C "$PRODUCER" pull -q origin main # keep PRODUCER able to push in later cases
}

# ── retro#838: ahead-only with an unreachable remote — still not stale, rc=0 ─
{
  WORKDIR="$TMPROOT/wd-ahead-nopush"
  setup_clone "$WORKDIR"
  echo "atlas" > "$WORKDIR/atlas2.html"
  git -C "$WORKDIR" add atlas2.html
  git -C "$WORKDIR" commit -q -m "atlas: local, unpushable"
  git -C "$WORKDIR" remote set-url --push origin "$TMPROOT/no-such-remote.git"

  rc=0
  OUT="$(run_sync_to_main 2>&1)" || rc=$?

  if [[ $rc -eq 0 && "$OUT" == *"push of unpushed commits failed"* ]]; then
    pass "ahead-only-push-fails: rc=0, failure logged, cycle may run"
  else
    fail "ahead-only-push-fails: rc=$rc — $OUT"
  fi
}

# ── retro#838: diverged (ahead AND behind) — force-syncs to origin/main ────
{
  WORKDIR="$TMPROOT/wd-diverged"
  setup_clone "$WORKDIR"
  echo "atlas" > "$WORKDIR/atlas3.html"
  git -C "$WORKDIR" add atlas3.html
  git -C "$WORKDIR" commit -q -m "atlas: local, will be dropped"
  advance_origin

  rc=0
  OUT="$(run_sync_to_main 2>&1)" || rc=$?
  origin_head="$(git -C "$ORIGIN" rev-parse main)"
  wd_head="$(git -C "$WORKDIR" rev-parse HEAD)"

  if [[ $rc -eq 0 && "$wd_head" == "$origin_head" ]]; then
    pass "diverged: force-synced to origin/main, rc=0"
  else
    fail "diverged: rc=$rc wd=$wd_head origin=$origin_head — $OUT"
  fi
}

# ── retro#838: a failed cycle sleeps, even with failed/pending cells ───────
# The main loop skipped `sleep` whenever any cell was failed/pending; with six
# permanently-failed cells a refused sync became ~2 cycles/second. The loop is
# extracted from the real file; `sleep` is stubbed to record its argument and end it.
{
  LOOP="$(awk '/^while true; do$/ { p = 1 } p { print } p && /^done$/ { exit }' "$EC2_RUN")"
  if [[ -z "$LOOP" ]]; then
    fail "failed-cycle-sleeps: could not extract the main loop from $EC2_RUN"
  else
    LOOP_DATA="$TMPROOT/loop-data"
    mkdir -p "$LOOP_DATA"
    echo '{"cells": {"A01|x": {"status": "failed"}}}' > "$LOOP_DATA/progress.json"

    OUT="$( (
      export DATA_DIR="$LOOP_DATA"
      # Bounded: on a regression the loop never sleeps, and an unbounded stub
      # would hang this harness instead of failing it.
      cycles=0
      run_pipeline() { cycles=$((cycles + 1)); [[ $cycles -ge 3 ]] && exit 0; return 1; }
      sleep() { echo "SLEPT $1"; exit 0; }
      eval "$LOOP"
    ) 2>&1 )" || true
    starts="$(grep -c "Cycle start" <<< "$OUT" || true)"

    if [[ "$OUT" == *"SLEPT $SLEEP_INTERVAL"* && "$starts" -eq 1 ]]; then
      pass "failed-cycle-sleeps: one cycle, then sleep $SLEEP_INTERVAL"
    else
      fail "failed-cycle-sleeps: starts=$starts — $OUT"
    fi
  fi
}

# ── propagation: a refusing sync aborts the cycle, not just logs ──────────
# Extracted verbatim from run_pipeline's own call site, so this tracks the
# real call convention rather than a hand-written stand-in for it.
{
  CALL_SITE="$(awk '
    /sync_to_main \|\| return 1/ { print; found_sync = 1; next }
    found_sync && /reexec_if_self_changed/ { print; exit }
  ' "$EC2_RUN")"

  if [[ -z "$CALL_SITE" ]]; then
    fail "propagation: could not find the sync_to_main call site in run_pipeline"
  else
    propagation_probe() {
      eval "$CALL_SITE"
      echo "REACHED_PAST_SYNC"
    }

    WORKDIR="$TMPROOT/wd-propagation"
    setup_clone "$WORKDIR"
    advance_origin
    lock="$WORKDIR/.git/index.lock"
    : > "$lock"
    touch -d "-10 seconds" "$lock" # forces the same refusal as the fresh-lock case above

    rc=0
    OUT="$( ( set -euo pipefail; cd "$WORKDIR"; propagation_probe ) 2>&1 )" || rc=$?

    if [[ $rc -ne 0 && "$OUT" != *"REACHED_PAST_SYNC"* ]]; then
      pass "propagation: refusing sync aborts before reexec/ingest, not just logs"
    else
      fail "propagation: rc=$rc, expected abort before REACHED_PAST_SYNC — $OUT"
    fi
  fi
}

# ── re-exec: changed file re-execs into the new generation ────────────────
{
  SELF="$TMPROOT/self-changed.sh"
  MARKER="$TMPROOT/reexec-marker"
  cat > "$SELF" <<EOF
#!/bin/bash
echo REEXECD > "$MARKER"
EOF
  chmod +x "$SELF"
  SELF_HASH_AT_START="not-the-real-hash-so-it-always-looks-changed"

  rm -f "$MARKER"
  ( reexec_if_self_changed ) # exec replaces this subshell only, not the harness
  # Give the exec'd process a moment to write its marker.
  for _ in $(seq 1 20); do
    [[ -f "$MARKER" ]] && break
    sleep 0.1
  done

  if [[ -f "$MARKER" ]] && [[ "$(cat "$MARKER")" == "REEXECD" ]]; then
    pass "reexec-if-self-changed: changed file triggers exec into the new generation"
  else
    fail "reexec-if-self-changed: expected marker file was not written"
  fi
}

# ── re-exec: unchanged file is a no-op ─────────────────────────────────────
{
  SELF="$TMPROOT/self-unchanged.sh"
  echo "#!/bin/bash" > "$SELF"
  echo "echo should-not-run" >> "$SELF"
  chmod +x "$SELF"
  SELF_HASH_AT_START="$(sha256sum "$SELF" | cut -d' ' -f1)"

  rc=0
  OUT="$( ( reexec_if_self_changed; echo NO_REEXEC ) 2>&1 )" || rc=$?

  if [[ $rc -eq 0 && "$OUT" == *"NO_REEXEC"* ]]; then
    pass "reexec-if-self-changed: unchanged file is a no-op"
  else
    fail "reexec-if-self-changed: rc=$rc — $OUT"
  fi
}

if [[ $FAIL -ne 0 ]]; then
  echo "=== ec2_run.sh guard tests: FAILED ==="
  exit 1
fi
echo "=== ec2_run.sh guard tests: all passed ==="
