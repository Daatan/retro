# retro/terraform

Infrastructure-as-code for the **Oracle / TruthMachine box** (`i-00ac444b94c5ff9b2`)
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

## Deferred (follow-up PR — needs a low-traffic DNS-cutover window)
The box currently has only an **ephemeral** public IP (`3.120.185.111`), so a stop/start
would break `oracle`/`bayes` DNS. The follow-up adds:
- `aws_eip` + association (one-time IP change)
- `oracle.daatan.com` / `bayes.daatan.com` A records repointed at the EIP
