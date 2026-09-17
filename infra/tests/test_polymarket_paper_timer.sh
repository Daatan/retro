#!/bin/bash
# Regression tests for the Polymarket paper bot systemd units (retro#620).
# Same shape as test_metaculus_timer.sh: each check pins a decision, not formatting.
set -uo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE="$DIR/polymarket-paper.service"
TIMER="$DIR/polymarket-paper.timer"
INSTALLER="$DIR/install_polymarket_paper_timer.sh"

pass=0; fail=0
ok()   { echo "  ok: $1"; pass=$((pass+1)); }
bad()  { echo "  FAIL: $1"; fail=$((fail+1)); }
check(){ if eval "$2"; then ok "$1"; else bad "$1"; fi; }

echo "== polymarket-paper units =="
check "service file exists" "[[ -f '$SERVICE' ]]"
check "timer file exists"   "[[ -f '$TIMER' ]]"
[[ -f "$SERVICE" && -f "$TIMER" ]] || { echo "missing unit files"; exit 1; }

# Cadence: 4 runs/day. Once a day would make MAX_MARKETS_PER_RUN the daily
# budget and turn one oneshot into an hour-long run; every 20 min would burn
# Oracle calls for a market that moves once a day.
check "timer runs 4 times a day" "[[ \$(grep -oP '^OnCalendar=\*-\*-\* \K[0-9,]+' '$TIMER' | tr ',' '\n' | wc -l) -eq 4 ]]"
check "timer is Persistent"      "grep -q '^Persistent=true' '$TIMER'"

# Skipped, not failed, until credentials exist — and it reuses the relay key.
check "service gated on the bot's own credentials file (not .env.metaculus — it never landed)" \
  "grep -q '^ConditionPathExists=/home/ubuntu/truthmachine/.env.polymarket-paper' '$SERVICE'"
check "metaculus env file is optional" \
  "grep -q '^EnvironmentFile=-/home/ubuntu/truthmachine/.env.metaculus' '$SERVICE'"
check "bot env file is required and read last (wins)" \
  "[[ \$(grep '^EnvironmentFile=' '$SERVICE' | tail -1) == 'EnvironmentFile=/home/ubuntu/truthmachine/.env.polymarket-paper' ]]"
check "installer derives the env file from ORACLE_API_KEYS" \
  "grep -q 'ORACLE_API_KEYS' '$INSTALLER' && grep -q 'O_EXCL, 0o600' '$INSTALLER'"
check "installer kicks a first run only when the ledger is empty, without blocking the deploy" \
  "grep -q 'snapshots.jsonl' '$INSTALLER' && grep -q 'systemctl start --no-block polymarket-paper.service' '$INSTALLER'"

check "ledger dir is under the API's data_dir" \
  "grep -q '^Environment=PAPER_LEDGER_DIR=/home/ubuntu/truthmachine/data/polymarket_paper' '$SERVICE'"

check "runs from the oracle-api checkout" \
  "grep -q '^WorkingDirectory=/home/ubuntu/oracle-api/polymarket_paper' '$SERVICE'"
check "does not run from the pipeline checkout" \
  "! grep -q '^WorkingDirectory=/home/ubuntu/truthmachine' '$SERVICE'"
check "uv run is --frozen" "grep -q 'uv run --frozen' '$SERVICE'"

timeout=$(grep -oP '^TimeoutStartSec=\K[0-9]+' "$SERVICE" || true)
check "service has a start timeout"        "[[ -n '$timeout' ]]"
check "timeout is inside the 6h period"    "[[ -n '$timeout' && '$timeout' -lt 21600 ]]"

check "no inline token in the unit" "! grep -qiE '^Environment=.*(API_KEY|TOKEN)=.+' '$SERVICE'"

# Structural no-trading guarantee: no signing/web3 dependency can be present.
check "no web3/signing dependency in the bot project" \
  "! grep -vE '^[[:space:]]*#' '$DIR/../polymarket_paper/pyproject.toml' | grep -qiE 'web3|eth[-_]account|py_clob_client|clob-client|ethers'"

echo
echo "passed=$pass failed=$fail"
[[ $fail -eq 0 ]]
