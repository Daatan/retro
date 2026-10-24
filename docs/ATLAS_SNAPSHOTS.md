# Atlas Snapshots — S3 Backup & Restore

The pipeline holds ~14 MB of expensive-to-regenerate state on EC2 that isn't in git:

- `data/atlas/` — per-cell `entry_*.json` + `cell_signal.json` (the aggregated LLM output per event × source).
- `data/vault2/` — raw articles + their extracted predictions (what the LLM calls actually produced).

Losing it would cost ~$0.05–$0.15 in re-run Bedrock fees plus ~a day of wall time to re-ingest and re-extract from scratch, *if* GNews/Brave quotas cooperate. This document describes the S3-based backup that makes that loss recoverable.

What is **not** backed up:

- `data/raw_ingest/` — re-fetchable from GNews/DDG. Cheap to regenerate.
- `data/pages/` — local scrape cache, disposable.
- `data/events/`, `data/sources/`, `data/progress.json` — tracked in git; always authoritative there.

## Topology

```
EC2 (truthmachine.service)
  ├── data/atlas/       ─┐
  └── data/vault2/      ─┴→ snapshot_atlas.sh  →  s3://truthmachine-atlas-snapshots-<ACCOUNT_ID>/
                                                    ├── latest.tgz                  (versioned, 7d noncurrent retention)
                                                    └── snapshots/<iso-ts>.tgz      (30d lifecycle expiry)
                                                                     ▲
  ec2_bootstrap.sh (on fresh instance)  ←  restore_atlas.sh   ───────┘
```

- **State archive** (`snapshots/state-<iso-ts>.tgz`, platform#42 item B): the small irreplaceable files — paper-bot ledgers (`data/polymarket_paper/` = the retro#620 scoreboard, `data/manifold_paper/`), `data/jev_values/`, the daatan-pushed resolution feedback / settlement-pin ledgers + their dedup caches, `data/*leaderboard*.json`, `data/resolution_tags.json`, and `~/oracle-series/nodes.jsonl`. A few MB per cycle. Written by [`infra/snapshot_state.sh`](../infra/snapshot_state.sh) right after the atlas snapshot; SQLite diskcache files are copied with the sqlite3 backup API (WAL-safe). No latest pointer — [`infra/restore_state.sh`](../infra/restore_state.sh) takes the newest `state-*` key and restores only items missing locally. **Never put `.env*` or any secret in this bucket.** Round-trip test: `infra/tests/test_state_snapshot.sh`.
- **Per-cycle archive** (`snapshots/<iso-ts>.tgz`): written on every successful extraction cycle, expires after 30 days. Purpose: point-in-time recovery (e.g. "restore state as of two days ago because today's batch was poisoned").
- **Latest pointer** (`latest.tgz`): overwritten each cycle. Versioning keeps the previous N for 7 days, so `aws s3api list-object-versions` can still pull a recent good version if the most recent upload is corrupt. Purpose: one-object restore on boot.

## Wiring

- **Backup** ([`infra/snapshot_atlas.sh`](../infra/snapshot_atlas.sh)) — called at the tail of `run_pipeline()` in [`infra/ec2_run.sh`](../infra/ec2_run.sh) on every cycle. Runs unconditionally after extraction; exits 0 non-fatally on any failure so a broken S3 never stalls the pipeline.
- **Restore** ([`infra/restore_atlas.sh`](../infra/restore_atlas.sh)) — called between data-dir creation (step 6) and service start (step 7) in [`infra/ec2_bootstrap.sh`](../infra/ec2_bootstrap.sh). Only restores if `data/atlas/` is empty (never clobbers a populated box). Exits 0 non-fatally if no snapshot exists (fresh account starts empty).

## One-time setup

### 1. Create the bucket

```bash
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
BUCKET="truthmachine-atlas-snapshots-${ACCOUNT_ID}"
REGION=eu-central-1

aws s3api create-bucket --bucket "$BUCKET" --region "$REGION" \
  --create-bucket-configuration "LocationConstraint=$REGION"

aws s3api put-public-access-block --bucket "$BUCKET" \
  --public-access-block-configuration \
  "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"

aws s3api put-bucket-versioning --bucket "$BUCKET" \
  --versioning-configuration Status=Enabled

aws s3api put-bucket-lifecycle-configuration --bucket "$BUCKET" --lifecycle-configuration '{
  "Rules": [
    { "ID": "expire-per-cycle-snapshots", "Status": "Enabled",
      "Filter": {"Prefix": "snapshots/"}, "Expiration": {"Days": 30},
      "NoncurrentVersionExpiration": {"NoncurrentDays": 1} },
    { "ID": "prune-latest-history", "Status": "Enabled",
      "Filter": {"Prefix": "latest.tgz"},
      "NoncurrentVersionExpiration": {"NoncurrentDays": 7} }
  ]
}'
```

### 2. One-time IAM setup

Attach an inline policy to the existing `truthmachine-ec2-role` (the instance role already attached to the pipeline box):

```bash
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
# substitute <ACCOUNT_ID> in the template
POLICY=$(sed "s/<ACCOUNT_ID>/${ACCOUNT_ID}/g" infra/iam/truthmachine-ec2-s3-snapshots-policy.json)

aws iam put-role-policy \
  --role-name truthmachine-ec2-role \
  --policy-name truthmachine-s3-snapshots \
  --policy-document "$POLICY"
```

The policy scopes:

- `s3:ListBucket` to `truthmachine-atlas-snapshots-<ACCOUNT_ID>` only.
- `s3:GetObject`, `s3:PutObject`, `s3:DeleteObject` (+ their `*Version` variants so the lifecycle rule can clean up) to objects in that bucket only.

No cross-account access. No access to any other S3 bucket in the account.

### 3. Seed the first snapshot (optional)

The first backup will land on the next pipeline cycle. If you want to verify the path eagerly, run it once via SSM:

```bash
aws ssm send-command --region eu-central-1 \
  --instance-ids i-00ac444b94c5ff9b2 \
  --document-name AWS-RunShellScript \
  --parameters 'commands=["sudo -u ubuntu bash /home/ubuntu/truthmachine/infra/snapshot_atlas.sh"]'
```

Then verify:

```bash
aws s3 ls --region eu-central-1 s3://truthmachine-atlas-snapshots-<ACCOUNT_ID>/
# expect: latest.tgz  snapshots/
```

## Operations

### Manual restore (e.g. to a new box)

```bash
# Assumes truthmachine-ec2-role is attached and ATLAS_SNAPSHOT_BUCKET points at the bucket
bash /home/ubuntu/truthmachine/infra/restore_atlas.sh
```

The script is idempotent — it won't overwrite an already-populated atlas, so you can run it defensively.

### Rollback to a specific point-in-time snapshot

```bash
aws s3 ls --region eu-central-1 s3://truthmachine-atlas-snapshots-<ACCOUNT_ID>/snapshots/ | sort
# pick a timestamp, e.g. 2026-04-17T15-30-00Z
aws s3 cp --region eu-central-1 \
  s3://truthmachine-atlas-snapshots-<ACCOUNT_ID>/snapshots/2026-04-17T15-30-00Z.tgz \
  /tmp/restore.tgz

# stop the service, wipe runtime state, extract, restart
sudo systemctl stop truthmachine
rm -rf /home/ubuntu/truthmachine/data/atlas /home/ubuntu/truthmachine/data/vault2
cd /home/ubuntu/truthmachine/data && tar -xzf /tmp/restore.tgz
sudo systemctl start truthmachine
```

### Recover from a corrupt `latest.tgz`

Bucket versioning retains the previous 7 days of `latest.tgz` versions:

```bash
aws s3api list-object-versions --bucket truthmachine-atlas-snapshots-<ACCOUNT_ID> \
  --prefix latest.tgz --query 'Versions[?IsLatest==`false`].[VersionId,LastModified]' \
  --output table
aws s3api get-object --bucket truthmachine-atlas-snapshots-<ACCOUNT_ID> \
  --key latest.tgz --version-id <version-id> /tmp/restore.tgz
```

## Cost

~31 MB (atlas) + a few MB (state) compressed per cycle, ~70 cycles/day, 30-day retention → ~70 GB steady state, ~$1.7/month.

**The bucket is versioned, so `Expiration` alone never frees anything:** it turns each expired `snapshots/` object into a delete marker and keeps the data as a noncurrent version forever. Until the `NoncurrentVersionExpiration` line above was added (platform#42, 2026-10-24) the bucket had grown to ~457 GB (≈ $11/month) and was adding ~1.5 GB/day. Apply the lifecycle block above to an existing bucket the same way; the backlog is purged within ~1–2 days.

## Known limitations

- **No encryption at rest beyond S3 default SSE-S3.** The atlas data is derived from public news articles, so this is acceptable; upgrade to KMS if the pipeline ever ingests private/restricted sources.
- **Single-region.** Bucket is in `eu-central-1` to match the EC2 box. Cross-region replication would add redundancy but also cost and complexity; skipped for now.
- **Snapshot is not atomic with git push.** If the pipeline commits `factum_atlas.html` at time T and then snapshots at T+δ with new extractions that landed in between, on restore we'd have cell entries that aren't reflected in `factum_atlas.html` yet. Safe — the next cycle's `render_atlas` run will pick them up.
