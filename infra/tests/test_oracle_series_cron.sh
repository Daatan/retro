#!/bin/bash
# Regression tests for the bayesoracle node-series cron line (retro#896).
# The line only ever runs on the oracle box; this pins the decisions in it.
set -uo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CRON="$DIR/oracle-series.crontab"
INSTALLER="$DIR/install_oracle_series_cron.sh"
DEPLOY="$DIR/deploy_oracle.sh"

pass=0; fail=0
ok()   { echo "  ok: $1"; pass=$((pass+1)); }
bad()  { echo "  FAIL: $1"; fail=$((fail+1)); }
check(){ if eval "$2"; then ok "$1"; else bad "$1"; fi; }

echo "== oracle-series cron =="
line=$(grep -v '^[[:space:]]*#' "$CRON" | grep -v '^[[:space:]]*$')

check "exactly one cron line"            "[[ \$(printf '%s\n' \"\$line\" | wc -l) -eq 1 ]]"
# Cadence lives in the script guard, not in cron: `*/3` in day-of-month resets
# at month boundaries. Cron must fire daily and pass the interval.
check "fires daily at 06:30 UTC"         "[[ \"\$line\" == '30 6 * * * '* ]]"
check "no */N day-of-month stride"       "! printf '%s' \"\$line\" | awk '{print \$3}' | grep -q '/'"
check "passes --min-interval-days 3"     "[[ \"\$line\" == *'--min-interval-days 3'* ]]"
check "runs from the batch tree"         "[[ \"\$line\" == *'cd /home/ubuntu/truthmachine/bayesoracle'* ]]"
check "appends to the persistent series" "[[ \"\$line\" == *'--out /home/ubuntu/oracle-series/nodes.jsonl'* ]]"
# Merging must never change the box's schedule by itself (retro#896).
check "deploy_oracle.sh does not run the installer" "! grep -q install_oracle_series_cron '$DEPLOY'"
check "installer refuses an unsynced script" "grep -q 'refusing:' '$INSTALLER'"
check "installer backs up before writing"    "grep -q 'crontab.bak' '$INSTALLER'"

echo "== $pass passed, $fail failed =="
[[ $fail -eq 0 ]]
