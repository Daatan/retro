# IAM templates

Templates for AWS IAM policies referenced by the infra. They are not applied automatically; a human runs the setup commands (see each section) once, edits the placeholders, and wires any resulting ARN into the workflow / config as called out.

## 1. GitHub Actions → AWS (OIDC, used by `deploy-oracle.yml`)

See setup in [`docs/ORACLE_DEPLOY.md`](../../docs/ORACLE_DEPLOY.md#one-time-iam-setup). Set the resulting role ARN as the `AWS_DEPLOY_ROLE_ARN` repo variable.

| File | Purpose |
|------|---------|
| `gha-deploy-oracle-trust.json` | Who can assume the role — scoped to GitHub Actions runs from this repo on `main` (plus manual `workflow_dispatch`). |
| `gha-deploy-oracle-policy.json` | What the role can do — `ssm:SendCommand` scoped to the one oracle instance *and* only the `AWS-RunShellScript` document, plus read-only `ssm:Get*/List*Command*` for polling results. See "Why `SendCommand` is two separate statements" below. |

## 2. TruthMachine EC2 → S3 (snapshot/restore atlas state)

Inline policy for the existing `truthmachine-ec2-role` (attached to the batch-pipeline EC2 instance). Grants Get/Put/Delete on the snapshot bucket only. See setup in [`docs/ATLAS_SNAPSHOTS.md`](../../docs/ATLAS_SNAPSHOTS.md#one-time-iam-setup).

| File | Purpose |
|------|---------|
| `truthmachine-ec2-s3-snapshots-policy.json` | Read/write access to `truthmachine-atlas-snapshots-<ACCOUNT_ID>` only. Lists the bucket, and gets/puts/deletes objects (including versioned deletes so the per-cycle `snapshots/` lifecycle rule works). |

## 3. TruthMachine EC2 → Secrets Manager (search API keys)

Inline policy for the existing `truthmachine-ec2-role`. Grants read-only access to the `openclaw/*` secret namespace so `web_search.py` can fetch search API keys at startup without needing them in the `.env` file.

| File | Purpose |
|------|---------|
| `truthmachine-ec2-secrets-policy.json` | `secretsmanager:GetSecretValue` on `arn:aws:secretsmanager:eu-central-1:<ACCOUNT_ID>:secret:openclaw/*`. No write, no list, no other namespaces. |

**Apply:**
```bash
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
sed "s/<ACCOUNT_ID>/$ACCOUNT_ID/g" infra/iam/truthmachine-ec2-secrets-policy.json > /tmp/secrets-policy.json
aws iam put-role-policy \
  --role-name truthmachine-ec2-role \
  --policy-name openclaw-secrets-read \
  --policy-document file:///tmp/secrets-policy.json
```

**Before applying, also store the new keys in Secrets Manager** (values come from daatan's env bundle):
```bash
aws secretsmanager create-secret --region eu-central-1 \
  --name openclaw/brightdata-api-key --secret-string "<VALUE>"
aws secretsmanager create-secret --region eu-central-1 \
  --name openclaw/nimbleway-api-key --secret-string "<VALUE>"
aws secretsmanager create-secret --region eu-central-1 \
  --name openclaw/scrapingbee-api-key --secret-string "<VALUE>"
```

**After applying, reload oracle and restart the pipeline** to pick up the new keys:
```bash
# Via SSM:
sudo systemctl reload oracle-api
sudo systemctl restart truthmachine
```

## Placeholders to replace

Before applying:

| Placeholder | Where | What to put |
|-------------|-------|-------------|
| `<ACCOUNT_ID>` | all files | Your 12-digit AWS account ID (`aws sts get-caller-identity --query Account --output text`) |
| `<REGION>` | `gha-deploy-oracle-policy.json` | The region the oracle instance lives in (currently `eu-central-1`) |
| `<INSTANCE_ID>` | `gha-deploy-oracle-policy.json` | The oracle instance ID (currently `i-00ac444b94c5ff9b2`) |

## Scope rationale

- **Trust**: `token.actions.githubusercontent.com` pinned to this repo by its immutable `repository_id` (`1184236342`, stable across org transfer / rename) plus a `sub` restricted to `ref:refs/heads/main` and `environment:*` — so only workflow runs on `main` (and `workflow_dispatch`, which still runs from whatever branch you pick) can assume the role. A PR branch cannot. This matches our deploy trigger.
- **Permissions**: `ssm:SendCommand` is the sharp tool — anyone with it can run arbitrary shell as root on the target instance. We scope it two ways: the instance ARN is restricted to the one oracle box, and the document ARN is restricted to `AWS-RunShellScript` (blocks e.g. `AWS-RunPowerShellScript` or custom documents). `ssm:GetCommandInvocation` / `ssm:ListCommandInvocations` are scoped to the region but not per-command (the command-id is only known after `SendCommand` returns).

### Why `SendCommand` is two separate statements

`ssm:SendCommand` is a [multi-resource action](https://docs.aws.amazon.com/service-authorization/latest/reference/list_amazonsystemsmanager.html#amazonsystemsmanager-actions-as-permissions): IAM evaluates it against the **instance** ARN *and* the **document** ARN at the same time, and both must match an `Allow`. We need to express "instance X AND document Y", not "instance X OR document Y".

A single statement with both ARNs in the `Resource` array expresses OR (the semantics of IAM `Resource` lists), which fails the two-resource check. The earlier form of this template did exactly that:

```json
// DO NOT — denied with "no identity-based policy allows the ssm:SendCommand action on resource: instance/..."
"Resource": [
  "arn:aws:ec2:...:instance/<INSTANCE_ID>",
  "arn:aws:ssm:...::document/AWS-RunShellScript"
],
"Condition": { "StringEquals": { "ssm:DocumentName": "AWS-RunShellScript" } }
```

Splitting into two statements (one per resource) gives the evaluator an independent `Allow` for each check, which is the form AWS documents. The `ssm:DocumentName` condition was redundant belt-and-suspenders in the broken form and isn't needed now that the document ARN is scoped directly.

Diagnosed live during the first smoke test of the Tier 4 workflow; see PR #49 comments for context.
