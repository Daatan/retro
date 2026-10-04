#!/usr/bin/env bash
# Show or switch the live Jev-API provider (retro#901). Workers pick a change up within 60 s,
# no restart. Usage: infra/jev-provider.sh                 # show
#                    infra/jev-provider.sh openrouter      # switch (typesafe|openrouter|clef-flash|clef)
# Clef also needs SSM /retro/prod/secrets/CLOUDFLARE_AI_API_TOKEN and CLOUDFLARE_ACCOUNT_ID, and
# runs log-only (uncalibrated). See docs/ORACLE_DEPLOY.md § Switching the Jev provider.
set -euo pipefail
NAME=/retro/prod/secrets/JEV_PROVIDER
if [[ $# -eq 0 ]]; then
  aws ssm get-parameter --region eu-central-1 --name "$NAME" --query Parameter.Value --output text 2>/dev/null \
    || echo "(unset — the JEV_SHADOW_API_URL / JEV_MODEL / JEV_API_KEY_SSM_NAME drop-in decides)"
  exit 0
fi
case "$1" in typesafe|openrouter|clef-flash|clef) ;; *) echo "unknown provider: $1" >&2; exit 2;; esac
aws ssm put-parameter --region eu-central-1 --name "$NAME" --type String --overwrite --value "$1" \
  --description "Live Jev-API provider (retro#901)" --query Version --output text
echo "JEV_PROVIDER=$1 — live within 60 s; watch for 'event=jev_provider' in oracle_log.txt"
