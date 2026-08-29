#!/bin/bash
# Regression tests for the Metaculus sync systemd units (retro#728).
#
# These units only ever execute on the oracle box, so nothing else in CI would
# notice if one of the decisions baked into them were quietly undone. Each test
# below pins a specific decision that cost measurement or an incident to reach —
# not the file's formatting.
set -uo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE="$DIR/metaculus-sync.service"
TIMER="$DIR/metaculus-sync.timer"

pass=0; fail=0
ok()   { echo "  ok: $1"; pass=$((pass+1)); }
bad()  { echo "  FAIL: $1"; fail=$((fail+1)); }
check(){ if eval "$2"; then ok "$1"; else bad "$1"; fi; }

echo "== metaculus-sync units =="

check "service file exists"                "[[ -f '$SERVICE' ]]"
check "timer file exists"                  "[[ -f '$TIMER' ]]"
[[ -f "$SERVICE" && -f "$TIMER" ]] || { echo "missing unit files"; exit 1; }

# --- Cadence -----------------------------------------------------------------
# The whole point of retro#728. A 6h poll misses most of a 1.5h question window;
# an earlier workflow comment proposed exactly that, on a latency assumption
# retro#617 later disproved (p99 25.0s over n=20,000, not multi-minute).
period=$(grep -oP '^OnCalendar=\*:0/\K[0-9]+' "$TIMER" || true)
check "timer uses a minute-granularity OnCalendar" "[[ -n '$period' ]]"
check "poll interval is <= 20 min (not hours)"     "[[ -n '$period' && '$period' -le 20 ]]"

# Persistent catches up after downtime instead of silently skipping a window.
check "timer is Persistent"                 "grep -q '^Persistent=true' '$TIMER'"

# --- Not-yet-provisioned state is skipped, not failed ------------------------
# Without this the timer turns "credentials not placed yet" (retro#725) into a
# failure every 20 minutes.
check "service gated on the credentials file" \
  "grep -q '^ConditionPathExists=/home/ubuntu/truthmachine/.env.metaculus' '$SERVICE'"

# --- Correct checkout --------------------------------------------------------
# /home/ubuntu/truthmachine is the pipeline's tree and carries the ingest loop's
# unpushed commits; deploy_oracle.sh deliberately never touches it. Running out
# of it would couple this job to that tree's lifecycle.
check "runs from the oracle-api checkout" \
  "grep -q '^WorkingDirectory=/home/ubuntu/oracle-api/metaculus' '$SERVICE'"
check "does not run from the pipeline checkout" \
  "! grep -q '^WorkingDirectory=/home/ubuntu/truthmachine' '$SERVICE'"

# --- Dependency pinning ------------------------------------------------------
# A drifted lockfile must fail loudly rather than resolve a different dependency
# set on a box we only observe through logs.
check "uv run is --frozen"                  "grep -q 'uv run --frozen' '$SERVICE'"

# --- A hung run cannot wedge every later run ---------------------------------
timeout=$(grep -oP '^TimeoutStartSec=\K[0-9]+' "$SERVICE" || true)
check "service has a start timeout"         "[[ -n '$timeout' ]]"
check "timeout is inside the poll period"   "[[ -n '$timeout' && -n '$period' && '$timeout' -lt \$(( period * 60 )) ]]"

# --- Credentials are not committed -------------------------------------------
check "no inline token in the unit" \
  "! grep -qiE '^Environment=.*(API_KEY|TOKEN)=.+' '$SERVICE'"
check "credentials come from an EnvironmentFile" \
  "grep -q '^EnvironmentFile=' '$SERVICE'"

echo
echo "passed=$pass failed=$fail"
[[ $fail -eq 0 ]]
