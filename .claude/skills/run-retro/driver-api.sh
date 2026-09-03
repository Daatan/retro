#!/usr/bin/env bash
# Driver for the Oracle FastAPI service (api/). See SKILL.md "Run: api".
#
# Subcommands:
#   start   - launch uvicorn in the background, poll /health, print PID/port
#   smoke   - curl the endpoints that need no live Bedrock/network access
#   stop    - kill the server by the port it's bound to
#
# Safety: sets a fake ORACLE_API_KEY (never a real one) and does not touch
# /forecast, /search, /llm, /pool/aggregate, /relevance, /pm/markets — those
# need real Bedrock/OpenRouter/Polymarket credentials this sandbox must not use.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
API_DIR="$REPO_ROOT/api"
PORT="${PORT:-8001}"
API_KEY="${ORACLE_API_KEY:-dev-local-key}"
LOG_FILE="${LOG_FILE:-/tmp/retro-driver-api.log}"
DATA_DIR="${DATA_DIR:-/tmp/retro-driver-api-data}"

cmd="${1:-}"

start() {
  mkdir -p "$DATA_DIR"
  cd "$API_DIR"
  echo "Starting api/ on :$PORT (log: $LOG_FILE, DATA_DIR=$DATA_DIR)"
  ORACLE_API_KEY="$API_KEY" DATA_DIR="$DATA_DIR" \
    AWS_ACCESS_KEY_ID=invalid AWS_SECRET_ACCESS_KEY=invalid AWS_DEFAULT_REGION=eu-central-1 \
    uv run uvicorn forecast_api.main:app --port "$PORT" --host 127.0.0.1 \
    >"$LOG_FILE" 2>&1 &
  echo $! > /tmp/retro-driver-api.pid
  # Startup does a leaderboard refresh_cache pass and (with the AWS guardrail
  # active) sequentially fails 14 search-provider SSM lookups with retries —
  # observed 12-45s in this environment, so poll generously.
  for i in $(seq 1 60); do
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
      echo "Ready after ${i}s."
      return 0
    fi
    sleep 1
  done
  echo "Server did not become ready in 60s; last log lines:" >&2
  tail -n 40 "$LOG_FILE" >&2
  return 1
}

smoke() {
  echo "== GET /health =="
  curl -sf "http://127.0.0.1:$PORT/health" | tee /dev/stderr | python3 -c 'import json,sys; d=json.load(sys.stdin); assert d["status"]=="ok"'
  echo
  echo "== GET /version =="
  curl -sf "http://127.0.0.1:$PORT/version"
  echo
  echo "== GET /openapi.json (schema only, count of paths) =="
  curl -sf "http://127.0.0.1:$PORT/openapi.json" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(len(d["paths"]), "paths")'
  echo "== GET /bayes/nodes (pure computation over graph_political.json, needs x-api-key) =="
  curl -sf -H "x-api-key: $API_KEY" "http://127.0.0.1:$PORT/bayes/nodes" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(len(d["nodes"]), "nodes; first:", d["nodes"][0])'
}

stop() {
  if [ -f /tmp/retro-driver-api.pid ]; then
    kill "$(cat /tmp/retro-driver-api.pid)" 2>/dev/null || true
    rm -f /tmp/retro-driver-api.pid
  fi
  # belt-and-suspenders: also free the port
  lsof -ti:"$PORT" -sTCP:LISTEN 2>/dev/null | xargs -r kill 2>/dev/null || true
  echo "Stopped."
}

case "$cmd" in
  start) start ;;
  smoke) smoke ;;
  stop) stop ;;
  *) echo "usage: $0 {start|smoke|stop}" >&2; exit 1 ;;
esac
