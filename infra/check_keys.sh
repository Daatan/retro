#!/bin/bash
# TruthMachine API key health check
# Tests all external API keys and AWS Bedrock access
# Usage: bash infra/check_keys.sh

INSTANCE="i-00ac444b94c5ff9b2"
REGION="eu-central-1"

SERPERDEV_KEY=$(aws secretsmanager get-secret-value \
  --secret-id daatan/serperdev-key \
  --region "$REGION" \
  --query SecretString --output text 2>/dev/null)

ok()   { echo "  ✓  $1"; }
fail() { echo "  ✗  $1"; }
warn() { echo "  ⚠  $1"; }

run_remote() {
  local CMD_ID
  CMD_ID=$(aws ssm send-command \
    --region "$REGION" \
    --instance-ids "$INSTANCE" \
    --document-name "AWS-RunShellScript" \
    --parameters "commands=[\"$1\"]" \
    --query "Command.CommandId" --output text 2>/dev/null)
  sleep 6
  aws ssm get-command-invocation \
    --region "$REGION" \
    --command-id "$CMD_ID" \
    --instance-id "$INSTANCE" \
    --query "StandardOutputContent" --output text 2>/dev/null
}

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  TruthMachine Key Check  $(date '+%Y-%m-%d %H:%M:%S')"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── Every secret web_search.py reads must actually resolve ──────────────
#
# `_secret()` falls back to Secrets Manager and returns None on a miss, so a secret
# that doesn't exist doesn't raise — the provider is just silently skipped, and the
# search chain quietly degrades instead of failing. Assert existence explicitly.
echo ""
echo "  SECRET EXISTENCE (names read by pipeline/src/tm/web_search.py)"
MISSING=0
for S in dataforseo-key serpapi-key serperdev-key brave-api-key brightdata-api-key \
         nimbleway-api-key scrapingbee-api-key tavily-api-key oracle-api-key \
         news-indexer-url news-indexer-api-key gcp-service-account-key; do
  if aws secretsmanager get-secret-value --secret-id "daatan/$S" --region "$REGION" \
       --query SecretString --output text >/dev/null 2>&1; then
    ok "daatan/$S"
  else
    fail "daatan/$S — MISSING; this provider will be silently skipped"
    MISSING=$((MISSING + 1))
  fi
done
[[ "$MISSING" -gt 0 ]] && warn "$MISSING secret(s) missing — the search chain is degraded, not broken, so nothing will alarm"

# ── Serper.dev ──────────────────────────────────────────
echo ""
echo "  SEARCH KEYS"
if [[ -n "$SERPERDEV_KEY" ]]; then
  STATUS=$(curl -s -o /dev/null -w "%{http_code}" -X POST "https://google.serper.dev/search" \
    -H "X-API-KEY: $SERPERDEV_KEY" \
    -H "Content-Type: application/json" \
    -d '{"q":"test","num":1}' 2>/dev/null)
  if [[ "$STATUS" == "200" ]]; then
    ok "Serper.dev (HTTP $STATUS)"
  else
    fail "Serper.dev (HTTP $STATUS)"
  fi
else
  warn "Serper.dev — key not found in Secrets Manager (daatan/serperdev-key)"
fi

# ── Brave ───────────────────────────────────────────────
BRAVE_KEY=$(aws secretsmanager get-secret-value \
  --secret-id daatan/brave-api-key \
  --region "$REGION" \
  --query SecretString --output text 2>/dev/null)
if [[ -n "$BRAVE_KEY" ]]; then
  STATUS=$(curl -s -o /dev/null -w "%{http_code}" \
    "https://api.search.brave.com/res/v1/web/search?q=test&count=1" \
    -H "Accept: application/json" \
    -H "X-Subscription-Token: $BRAVE_KEY" 2>/dev/null)
  if [[ "$STATUS" == "200" ]]; then
    ok "Brave Search (HTTP $STATUS)"
  elif [[ "$STATUS" == "402" ]]; then
    fail "Brave Search — quota exhausted (402)"
  else
    warn "Brave Search (HTTP $STATUS)"
  fi
else
  warn "Brave Search — key not found in Secrets Manager"
fi

# ── AWS Bedrock (from EC2 via IAM role) ─────────────────
echo ""
echo "  AI / BEDROCK (tested from EC2)"
BEDROCK_OUT=$(run_remote "cd /home/ubuntu/truthmachine && export PATH=\$HOME/.local/bin:\$PATH && uv run --project pipeline python3 -c \"import boto3; c=boto3.client('bedrock-runtime',region_name='us-east-1'); r=c.invoke_model(modelId='amazon.nova-micro-v1:0',body='{\\\"messages\\\":[{\\\"role\\\":\\\"user\\\",\\\"content\\\":[{\\\"text\\\":\\\"hi\\\"}]}]}',contentType='application/json'); print('OK')\" 2>&1 | tail -1")
if echo "$BEDROCK_OUT" | grep -q "^OK"; then
  ok "Bedrock Nova Micro (us-east-1)"
else
  fail "Bedrock Nova Micro — $BEDROCK_OUT"
fi

# ── GitHub PAT ──────────────────────────────────────────
echo ""
echo "  GIT / GITHUB"
GH_TOKEN=$(aws secretsmanager get-secret-value \
  --secret-id daatan/github-pat \
  --region "$REGION" \
  --query SecretString --output text 2>/dev/null)
if [[ -n "$GH_TOKEN" ]]; then
  STATUS=$(curl -s -o /dev/null -w "%{http_code}" \
    -H "Authorization: token $GH_TOKEN" \
    "https://api.github.com/user" 2>/dev/null)
  if [[ "$STATUS" == "200" ]]; then
    ok "GitHub PAT (HTTP $STATUS)"
  else
    fail "GitHub PAT (HTTP $STATUS)"
  fi
else
  warn "GitHub PAT — key not found in Secrets Manager"
fi

# ── EC2 .env contents ───────────────────────────────────
echo ""
echo "  EC2 .env (keys present)"
run_remote "grep -E '^[A-Z_]+=.' /home/ubuntu/truthmachine/.env | sed 's/=.*/=<set>/' | sed 's/^/    /'"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
