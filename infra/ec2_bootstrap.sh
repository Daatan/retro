#!/bin/bash
# TruthMachine EC2 Bootstrap — run once on a fresh instance.
# Access via SSM: aws ssm start-session --target <instance-id> --region us-east-1
set -euo pipefail

# Auto-detect region from EC2 metadata (IMDSv2), fallback to us-east-1
TOKEN=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" \
  -H "X-aws-ec2-metadata-token-ttl-seconds: 60" 2>/dev/null || true)
REGION=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" \
  "http://169.254.169.254/latest/meta-data/placement/region" 2>/dev/null || echo "us-east-1")
REPO="https://github.com/Daatan/retro.git"
WORKDIR="$HOME/truthmachine"

log() { echo "[bootstrap] $*"; }

# ── 1. System dependencies ────────────────────────────────────────────────────
log "Installing system packages..."
sudo apt-get update -q
sudo apt-get install -y git curl python3-pip python3-venv jq unzip

# AWS CLI v2 (not in apt on Ubuntu 24.04 arm64)
if ! command -v aws &>/dev/null; then
  log "Installing AWS CLI v2..."
  curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip" -o /tmp/awscliv2.zip
  unzip -q /tmp/awscliv2.zip -d /tmp
  sudo /tmp/aws/install
  rm -rf /tmp/aws /tmp/awscliv2.zip
fi
log "AWS CLI $(aws --version 2>&1 | head -1)"

# Install uv (fast Python package manager)
if ! command -v uv &>/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
fi
log "uv $(uv --version)"

# ── 2. Pull secrets from AWS Secrets Manager ─────────────────────────────────
log "Fetching secrets..."
get_secret() {
  aws secretsmanager get-secret-value \
    --secret-id "$1" --region "$REGION" \
    --query SecretString --output text 2>/dev/null
}

OPENROUTER_KEY=$(get_secret "openclaw/openrouter-api-key") || true
BRAVE_KEY=$(get_secret "openclaw/brave-api-key") || true
SERPAPI_KEY=$(get_secret "openclaw/serpapi-key") || true
SERPERDEV_KEY=$(get_secret "openclaw/serperdev-key") || true
GH_TOKEN=$(get_secret "openclaw/github-pat") || true

if [[ -z "$OPENROUTER_KEY" ]]; then
  echo "ERROR: openclaw/openrouter-api-key not found in Secrets Manager" >&2
  exit 1
fi
if [[ -z "$GH_TOKEN" ]]; then
  echo "ERROR: openclaw/github-pat not found in Secrets Manager" >&2
  echo "Create it: aws secretsmanager create-secret --name openclaw/github-pat --secret-string ghp_..." >&2
  exit 1
fi

# ── 3. Clone repository ───────────────────────────────────────────────────────
log "Cloning repo..."
mkdir -p "$WORKDIR"

# Use token-authenticated URL so the run loop can push
AUTHED_REPO="https://x-token:${GH_TOKEN}@github.com/Daatan/retro.git"

if [[ -d "$WORKDIR/.git" ]]; then
  log "Repo already cloned, pulling latest..."
  git -C "$WORKDIR" remote set-url origin "$AUTHED_REPO"
  git -C "$WORKDIR" pull --ff-only origin main
else
  git clone "$AUTHED_REPO" "$WORKDIR"
fi

# Git identity for commits from this machine
git -C "$WORKDIR" config user.email "pipeline@truthmachine"
git -C "$WORKDIR" config user.name "TruthMachine EC2"

# ── 4. Write .env file ────────────────────────────────────────────────────────
log "Writing .env..."
cat > "$WORKDIR/.env" <<EOF
OPENROUTER_API_KEY=${OPENROUTER_KEY}
BRAVE_API_KEY=${BRAVE_KEY}
SERPAPI_KEY=${SERPAPI_KEY}
SERPERDEV_KEY=${SERPERDEV_KEY}
EOF
chmod 644 "$WORKDIR/.env"

# ── 5. Install Python dependencies ───────────────────────────────────────────
log "Installing Python dependencies..."
uv sync --project "$WORKDIR/pipeline"
# uv sync sometimes misses boto3 if venv was pre-created; install directly to be sure
if ! uv run --project "$WORKDIR/pipeline" python3 -c "import boto3" 2>/dev/null; then
  log "boto3 missing from venv — installing directly..."
  uv pip install boto3 botocore -p "$WORKDIR/pipeline/.venv/bin/python3"
fi
log "boto3: $(uv run --project "$WORKDIR/pipeline" python3 -c 'import boto3; print(boto3.__version__)' 2>/dev/null)"

# ── 6. Create data directories ────────────────────────────────────────────────
mkdir -p "$WORKDIR/data/events" \
         "$WORKDIR/data/sources" \
         "$WORKDIR/data/raw_ingest" \
         "$WORKDIR/data/vault2/articles" \
         "$WORKDIR/data/vault2/extractions" \
         "$WORKDIR/data/atlas"

# ── 6a. Restore prior atlas+vault2 from S3 if available ─────────────────────
# No-op if atlas is already populated (safety on re-bootstrap) or if no
# snapshot exists yet (fresh account). See infra/snapshot_atlas.sh for what
# lives in the tarball.
if [[ -x "$WORKDIR/infra/restore_atlas.sh" ]]; then
  log "Restoring atlas snapshot from S3..."
  WORKDIR="$WORKDIR" bash "$WORKDIR/infra/restore_atlas.sh" || log "restore_atlas exited non-zero (non-fatal, continuing)"
fi

# ── 7. Install systemd service ────────────────────────────────────────────────
log "Installing systemd service..."
sudo cp "$WORKDIR/infra/truthmachine.service" /etc/systemd/system/truthmachine.service
sudo systemctl daemon-reload
sudo systemctl enable truthmachine
sudo systemctl start truthmachine
log "Service status: $(sudo systemctl is-active truthmachine)"

log "Bootstrap complete."
log "Monitor: sudo journalctl -u truthmachine -f"
log "Or:      tail -f $WORKDIR/pipeline_log.txt"
