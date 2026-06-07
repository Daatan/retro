# Oracle API — Deploy

Zero-downtime deploy flow for the Oracle API on the retro EC2 box.

## Topology

One EC2 box, **two independent checkouts** of this repo:

| Path | Role | Git lifecycle |
|------|------|---------------|
| `/home/ubuntu/truthmachine/` | Pipeline worktree — ingest loop, orchestrator, atlas render | `ec2_run.sh` commits `data/progress.json` + `factum_atlas.html`, rebases on `origin/main`, pushes. May accumulate unpushed commits when rebase fails. |
| `/home/ubuntu/oracle-api/`   | Oracle API worktree — FastAPI service under `oracle-api.service` | Deploys `reset --hard` to `origin/main`. Never diverges. |

Both checkouts read the same `data/` directory — it lives in the pipeline's tree, and the API's `.env` points `DATA_DIR=/home/ubuntu/truthmachine/data` at it. Data is shared, **code is not**. This is what lets auto-deploy stay trivial: the deploy script never has to reason about the pipeline's work-in-progress commits because it only touches the API checkout.

## Deploy flow

`infra/deploy_oracle.sh` runs on the box and does:

1. `git fetch origin main` in `/home/ubuntu/oracle-api`
2. `git reset --hard <ref>` (defaults to `origin/main`, override with a SHA to pin)
3. `uv sync --frozen` in `api/`
4. `sudo systemctl reload-or-restart oracle-api` — gunicorn SIGHUPs its workers, new code is imported into fresh workers, old workers drain gracefully. The listening socket on `:8001` is never closed, so there's no 502 window.
5. **Health gate (hardened 2026-06-02):** poll `/health` and require **5 consecutive** 200s over a window that spans the graceful-drain. If the reload is not durably healthy, **escalate to a full `systemctl restart`** and re-verify; only if *that* also fails does the script `exit 1` (deploy goes red).

No-op fast-path: if `git reset` produces the same SHA as before, the script skips `uv sync` and the reload entirely.

> **Why the consecutive-200 gate + restart escalation (incident 2026-06-02).** A
> dependency upgrade took the live API down: every request returned `400 "Invalid
> HTTP request received."` SIGHUP does **not** replace the long-lived gunicorn
> master — it forks new workers that load an upgraded native extension (e.g.
> `httptools`) from disk while inheriting the *old* `uvicorn` already imported in
> the master → ABI mismatch → garbage HTTP parsing. The old health check exited on
> the **first** 200, which a still-draining *old* (good) worker happily served, so
> the deploy reported success while real traffic was broken. The hardened gate
> requires sustained health (so a draining good worker can't mask broken new ones)
> and, when a reload can't serve, does a full restart that brings up a fresh master
> with the whole upgraded stack imported consistently. Net: native-dependency
> upgrades self-heal, and a genuinely broken deploy fails **red** instead of
> silently breaking production.

### Invocation

There are three supported ways to deploy. **Normal operation is path 1 — do nothing; merge to `main` and let the workflow run.**

#### 1. Automatic: merge to `main`

`.github/workflows/deploy-oracle.yml` runs on every push to `main` that touches `api/**`, `pipeline/**`, `infra/deploy_oracle.sh`, `infra/oracle-api.service`, or the workflow file itself. It authenticates to AWS via OIDC, calls `aws ssm send-command` against the box, waits for completion, and prints the deploy script's stdout/stderr into the Actions log. A `no-op` fast-path in `deploy_oracle.sh` makes the workflow cheap when the resolved HEAD already matches what's on the box.

Manual trigger (for rollbacks or redeploying the same SHA): **Actions → Deploy Oracle API → Run workflow**, optionally pinning `ref` to a prior commit.

#### 2. Ad-hoc via SSM (no SSH needed)

When you want to poke the box from a laptop with AWS creds but without opening port 22:

```bash
aws ssm send-command \
  --region eu-central-1 \
  --instance-ids i-00ac444b94c5ff9b2 \
  --document-name AWS-RunShellScript \
  --comment "manual deploy" \
  --parameters 'commands=["sudo -u ubuntu bash /home/ubuntu/oracle-api/infra/deploy_oracle.sh"]' \
  --output text --query 'Command.CommandId'
# then:
aws ssm get-command-invocation --region eu-central-1 \
  --instance-id i-00ac444b94c5ff9b2 --command-id <CMD_ID>
```

This is exactly what the GH Actions workflow does internally; it's the canonical way to reach the box.

#### 3. On the box (SSH, if available)

```bash
ssh ubuntu@<oracle-public-ip>
bash /home/ubuntu/oracle-api/infra/deploy_oracle.sh
```

SSH isn't guaranteed to be open — the security group may block port 22, and key rotation history is spotty. Prefer path 1 or 2.

## Why the split?

Before this split, a single checkout at `/home/ubuntu/truthmachine/` served both roles. The pipeline's continuous `git commit` / `git rebase` / `git push` loop for atlas data meant:

- The working tree was never in sync with `origin/main` — there were always a few local WIP commits.
- Any `git pull --rebase` during deploy could fail, block, or (worse) interleave with an in-flight pipeline commit.
- A `git reset --hard` would silently nuke unpushed atlas/progress data the pipeline had just produced.

Separating the two checkouts removes the class of problem. The pipeline's messy branch state is now irrelevant to deploys.

## One-time migration (historical)

Completed on 2026-04-17 via SSM on `i-00ac444b94c5ff9b2`: `/home/ubuntu/oracle-api/` checkout created, `uv sync --frozen` completed, unit file swapped, service restarted, `/health` green. Old unit backed up to `/etc/systemd/system/oracle-api.service.pre-migration.bak`. If the box is ever rebuilt from scratch, the commands are below.

<details>
<summary>Migration runbook (re-run only on a fresh box)</summary>

```bash
# Run locally with AWS creds loaded:
aws ssm send-command --region eu-central-1 --instance-ids i-00ac444b94c5ff9b2 \
  --document-name AWS-RunShellScript --comment "oracle-api initial setup" \
  --parameters 'commands=[
    "set -eu",
    "sudo -u ubuntu git clone https://github.com/Daatan/retro.git /home/ubuntu/oracle-api",
    "cd /home/ubuntu/oracle-api/api && sudo -u ubuntu /home/ubuntu/.local/bin/uv sync --frozen",
    "sudo cp /home/ubuntu/oracle-api/infra/oracle-api.service /etc/systemd/system/oracle-api.service",
    "sudo systemctl daemon-reload",
    "sudo systemctl restart oracle-api",
    "sleep 3 && curl -s http://127.0.0.1:8001/health"
  ]'
```

</details>

## Rollback

Three options, in order of preference:

1. **Via GH Actions**: Actions → Deploy Oracle API → Run workflow → set `ref` to the last known-good SHA. Deploys it via the same path as a normal deploy.
2. **Via SSM from a laptop**: `aws ssm send-command ... "commands=[\"sudo -u ubuntu bash /home/ubuntu/oracle-api/infra/deploy_oracle.sh <prev-sha>\"]"`
3. **Hard-reset on the box**: `sudo systemctl restart oracle-api` (2-5s 502 window) — the escape hatch when the service is wedged. Note the deploy script already auto-escalates to a full restart when a reload isn't durably healthy (see the health gate above), so this is rarely needed by hand.

## GitHub Actions → AWS auth

The `deploy-oracle.yml` workflow uses OIDC, not static AWS keys. It reads two repository variables (not secrets; these aren't sensitive):

| Name | Purpose | Default |
|------|---------|---------|
| `AWS_DEPLOY_ROLE_ARN` | IAM role the workflow assumes | **required, no default** |
| `AWS_REGION` | Region for SSM calls | `eu-central-1` |
| `ORACLE_INSTANCE_ID` | EC2 instance ID to target | `i-00ac444b94c5ff9b2` |

### One-time IAM setup

The IAM role must have a trust policy that allows GitHub's OIDC provider to assume it from this repo's `main` branch, and a permissions policy that allows `ssm:SendCommand` / `ssm:GetCommandInvocation` / `ssm:ListCommandInvocations` on the target instance. A minimal setup:

```bash
# 1. Register GitHub's OIDC provider (once per AWS account; skip if already present).
aws iam create-open-id-connect-provider \
  --url https://token.actions.githubusercontent.com \
  --client-id-list sts.amazonaws.com \
  --thumbprint-list 6938fd4d98bab03faadb97b34396831e3780aea1

# 2. Create the role with a scoped trust policy (see infra/iam/gha-deploy-oracle-trust.json).
aws iam create-role --role-name gha-deploy-oracle \
  --assume-role-policy-document file://infra/iam/gha-deploy-oracle-trust.json

# 3. Attach an inline policy allowing only SSM calls against the oracle instance.
aws iam put-role-policy --role-name gha-deploy-oracle \
  --policy-name ssm-deploy --policy-document file://infra/iam/gha-deploy-oracle-policy.json

# 4. Add the ARN as a repo variable.
gh variable set AWS_DEPLOY_ROLE_ARN --body "arn:aws:iam::<account-id>:role/gha-deploy-oracle"
```

Example policy documents live under `infra/iam/`.
