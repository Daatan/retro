#!/bin/bash
# TruthMachine log viewer
#
# Usage:
#   bash infra/logs.sh              — tail last 30 lines
#   bash infra/logs.sh tail [N]     — tail last N lines (default 30)
#   bash infra/logs.sh warn         — show warnings/errors only
#   bash infra/logs.sh progress     — show only progress lines (done | )
#   bash infra/logs.sh grep <pat>   — grep for pattern
#   bash infra/logs.sh settlement [YYYY-MM-DD]
#                                   — settlement shadow-gate vs verifier report
#                                     (reads the ORACLE log, not the pipeline one)

INSTANCE="i-00ac444b94c5ff9b2"
REGION="eu-central-1"
LOG="/home/ubuntu/truthmachine/pipeline_log.txt"
# The Oracle API logs somewhere else entirely, and journald has neither
# (repo CLAUDE.md). Settlement events are only ever in this one.
ORACLE_LOG="/home/ubuntu/truthmachine/oracle_log.txt"

run_remote() {
  local CMD_ID
  CMD_ID=$(aws ssm send-command \
    --region "$REGION" \
    --instance-ids "$INSTANCE" \
    --document-name "AWS-RunShellScript" \
    --parameters "commands=[\"$1\"]" \
    --query "Command.CommandId" --output text 2>/dev/null)
  # Poll rather than sleep a fixed 6s: a tail returns instantly, but a scan of
  # the 350MB oracle log does not, and a fixed wait silently returns a partial
  # answer that reads like a complete one.
  local STATUS
  for _ in $(seq 1 40); do
    STATUS=$(aws ssm get-command-invocation \
      --region "$REGION" --command-id "$CMD_ID" --instance-id "$INSTANCE" \
      --query "Status" --output text 2>/dev/null)
    case "$STATUS" in Success|Failed|Cancelled|TimedOut) break ;; esac
    sleep 3
  done
  aws ssm get-command-invocation \
    --region "$REGION" \
    --command-id "$CMD_ID" \
    --instance-id "$INSTANCE" \
    --query "StandardOutputContent" --output text 2>/dev/null
}

MODE="${1:-tail}"
ARG="${2:-30}"

case "$MODE" in
  tail)
    echo "=== Last $ARG lines ==="
    run_remote "tail -$ARG $LOG 2>/dev/null"
    ;;
  warn|warnings)
    echo "=== Warnings & errors (aggregated) ==="
    run_remote "grep -iE 'warning|error|failed|quota|429|401|403|timeout' $LOG \
      | grep -v 'Slug → HTTP' \
      | sort | uniq -c | sort -rn | head -30"
    ;;
  progress)
    echo "=== Progress lines ==="
    run_remote "grep -E 'done \|' $LOG | tail -20"
    ;;
  settlement)
    # `$2` is a date here, not a line count, so the shared ARG default of 30 is
    # wrong — fall back to a week, which is the window the enforce/don't-enforce
    # decision was scoped to (retro#691).
    SINCE="${2:-$(date -u -d '7 days ago' +%F 2>/dev/null || date -u -v-7d +%F)}"
    echo "=== Settlement shadow-gate report (since $SINCE) ==="
    # The script ships in both checkouts — the batch tree self-syncs from main,
    # the API tree is rewritten by deploy-oracle.yml. Prefer whichever is there.
    run_remote "for D in /home/ubuntu/truthmachine /home/ubuntu/oracle-api; do \
      if [ -f \$D/infra/settlement_report.py ]; then \
        python3 \$D/infra/settlement_report.py --log $ORACLE_LOG --since $SINCE; exit 0; fi; \
      done; echo 'settlement_report.py not found in either checkout'; exit 1"
    ;;
  grep)
    PAT="${ARG}"
    echo "=== grep: $PAT ==="
    run_remote "grep -E '$PAT' $LOG | tail -30"
    ;;
  *)
    echo "Usage: $0 [tail [N] | warn | progress | grep <pattern> | settlement [YYYY-MM-DD]]"
    exit 1
    ;;
esac
