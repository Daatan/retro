# retro/terraform

Infrastructure-as-code for the **Oracul / TruthMachine box** (`i-00ac444b94c5ff9b2`)
— the EC2 instance that serves `oracle.daatan.com` (Oracle FastAPI), `bayes.daatan.com`
(BayesOracle), and runs the `truthmachine.service` batch pipeline.

This box was hand-provisioned and operated out-of-band (via `infra/*.sh` over SSM).
It is load-bearing for daatan (the forecast path calls it through `ORACLE_URL`), so it
is brought under terraform here to make it auditable and rebuildable.

## State
- Backend: S3 `daatan-terraform-state`, key `retro/terraform.tfstate`, region `eu-central-1`.
- Locking: S3-native (`use_lockfile`). No DynamoDB table needed.
- Isolated from the daatan app states (`prod/`, `staging/` keys) and news-indexer.

## What's managed
- `aws_instance.oracle` — imported, not recreated. `lifecycle.prevent_destroy = true`
  guards the box against accidental replacement.
- The security group `sg-0c9c7cee5ebcf853d` ("openclaw-sg") is **referenced by id**, not
  managed here — it predates this stack.

## Usage
```bash
terraform init
terraform plan     # expect: No changes. Infrastructure matches configuration.
```
Never run `apply` for a change you have not reviewed in `plan`. Use `-target` for
surgical changes; never a blanket apply.

## Elastic IP (done 2026-08-27)
The box previously had only an **ephemeral** public IP (`3.120.185.111`), so a stop/start
would break `oracle`/`bayes` DNS. `eip.tf` now manages:
- `aws_eip.oracle` + `aws_eip_association.oracle` — address is now `3.122.48.104`
  (`eipalloc-05a6e2750d63d416e`), stable across stop/start
- `aws_route53_record.oracle` / `.bayes` — **imported**, not created, then repointed at the
  EIP. TTL lowered 300 → 60 first so the one-time address change drained in under a minute.

This unblocks any window that needs a stop, notably the retro#436 encrypted-root-volume swap.
