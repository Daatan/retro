#!/bin/bash
# Regression tests for the Manifold paper bot systemd units (retro#918).
# Same shape as test_metaculus_timer.sh: each check pins a decision, not formatting.
set -uo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE="$DIR/manifold-paper.service"
TIMER="$DIR/manifold-paper.timer"
INSTALLER="$DIR/install_manifold_paper_timer.sh"

pass=0; fail=0
ok()   { echo "  ok: $1"; pass=$((pass+1)); }
bad()  { echo "  FAIL: $1"; fail=$((fail+1)); }
check(){ if eval "$2"; then ok "$1"; else bad "$1"; fi; }

echo "== manifold-paper units =="
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
  "grep -q '^ConditionPathExists=/home/ubuntu/truthmachine/.env.manifold-paper' '$SERVICE'"
check "metaculus env file is optional" \
  "grep -q '^EnvironmentFile=-/home/ubuntu/truthmachine/.env.metaculus' '$SERVICE'"
check "bot env file is required and read last (wins)" \
  "[[ \$(grep '^EnvironmentFile=' '$SERVICE' | tail -1) == 'EnvironmentFile=/home/ubuntu/truthmachine/.env.manifold-paper' ]]"
check "installer derives the env file from ORACLE_API_KEYS" \
  "grep -q 'ORACLE_API_KEYS' '$INSTALLER' && grep -q 'O_EXCL, 0o600' '$INSTALLER'"
check "installer kicks a first run only when the ledger is empty, without blocking the deploy" \
  "grep -q 'snapshots.jsonl' '$INSTALLER' && grep -q 'systemctl start --no-block manifold-paper.service' '$INSTALLER'"

check "ledger dir is under the API's data_dir" \
  "grep -q '^Environment=PAPER_LEDGER_DIR=/home/ubuntu/truthmachine/data/manifold_paper' '$SERVICE'"

check "runs from the oracle-api checkout" \
  "grep -q '^WorkingDirectory=/home/ubuntu/oracle-api/manifold_paper' '$SERVICE'"
check "does not run from the pipeline checkout" \
  "! grep -q '^WorkingDirectory=/home/ubuntu/truthmachine' '$SERVICE'"
check "uv run is --frozen" "grep -q 'uv run --frozen' '$SERVICE'"

timeout=$(grep -oP '^TimeoutStartSec=\K[0-9]+' "$SERVICE" || true)
check "service has a start timeout"        "[[ -n '$timeout' ]]"
check "timeout is inside the 6h period"    "[[ -n '$timeout' && '$timeout' -lt 21600 ]]"

check "no inline token in the unit" "! grep -qiE '^Environment=.*(API_KEY|TOKEN)=.+' '$SERVICE'"

# Structural no-write guarantee: Phase 1 has no Manifold key and no bet client.
# (pytest's AST test in manifold_paper/tests pins the client itself; this is the
# cheap shell-level tripwire for a key or a write endpoint sneaking in elsewhere.
# manifold_client.py's docstring names /v0/bet to say it is absent, hence the filter.)
check "no Manifold API key or Authorization header in the bot project" \
  "! grep -rEqi 'MANIFOLD_API_KEY|Authorization:|Key \\{' '$DIR/../manifold_paper' --include=*.py --include=*.toml --exclude-dir=tests --exclude-dir=.venv"
check "no write call on the Manifold client" \
  "! grep -rEq '\\.(post|put|delete|patch)\\(' '$DIR/../manifold_paper/manifold_client.py' '$DIR/../manifold_paper/paper.py'"
check "bot project has no dependency beyond httpx" \
  "[[ \$(sed -n '/^dependencies = \[/,/^\]/p' '$DIR/../manifold_paper/pyproject.toml' | grep -c '\"') -eq 1 ]]"
check "no Manifold key in the unit" "! grep -qiE '^Environment=MANIFOLD_[A-Z_]*(KEY|TOKEN)' '$SERVICE'"

# Must not collide with polymarket-paper's :30 slot on the same box.
check "timer offset differs from polymarket-paper" \
  "! diff -q <(grep -oP '^OnCalendar=.*' '$TIMER') <(grep -oP '^OnCalendar=.*' '$DIR/polymarket-paper.timer') >/dev/null"

echo
echo "passed=$pass failed=$fail"
[[ $fail -eq 0 ]]
