#!/bin/bash
# TruthMachine EC2 Monitor — full status dashboard
# Usage: bash infra/monitor.sh

INSTANCE="i-00ac444b94c5ff9b2"
REGION="eu-central-1"
WORKDIR="/home/ubuntu/truthmachine"

ssm_simple() {
  local CMD_ID
  CMD_ID=$(aws ssm send-command \
    --region "$REGION" \
    --instance-ids "$INSTANCE" \
    --document-name "AWS-RunShellScript" \
    --parameters "commands=[\"$1\"]" \
    --query "Command.CommandId" --output text 2>/dev/null)
  sleep 7
  aws ssm get-command-invocation \
    --region "$REGION" \
    --command-id "$CMD_ID" \
    --instance-id "$INSTANCE" \
    --query "StandardOutputContent" --output text 2>/dev/null
}

clear
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  TruthMachine Monitor  $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Instance: $INSTANCE ($REGION)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Run the on-EC2 stats script (avoids SSM quoting issues entirely)
ssm_simple "bash $WORKDIR/infra/remote_stats.sh 2>/dev/null" | \
  sed 's/^=== /\n  /; s/ ===//'

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  bash infra/logs.sh [tail [N] | warn | progress | grep <pat>]"
echo "  bash infra/logs.sh settlement [YYYY-MM-DD]   # shadow gates vs verifier"
echo "  bash infra/check_keys.sh"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
