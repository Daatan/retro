# Claude / agent guidance

Loaded automatically by Claude Code (and other CLAUDE.md-aware agents) when working in this repo. Keep it terse; it's read on every turn. Cross-repo rules (PR-only, terraform, SSM) come from the Daatan workspace CLAUDE.md — this file holds only what's specific to retro.

## What this repo is

Three related systems, one repo:

- **`pipeline/`** — TruthMachine / Factum Atlas: retroactive media-analysis pipeline (ingest → gatekeeper → extractor → aggregate → Brier/ELO scoring). Python, `uv` project, package `tm`.
- **`api/`** — the **Oracle**: FastAPI microservice at `oracle.daatan.com` (`forecast_api` package). Takes a binary question, returns a calibrated probability with per-source credibility weighting. The `tm` package (search/LLM) is shared internally.
- **`bayesoracle/`** — BayesOracle at `bayes.daatan.com`.

[`readme.md`](./readme.md) is the product/vision doc. The technical map lives in [`docs/ARCHITECTURE.md`](./docs/ARCHITECTURE.md) — start there. Oracle API contract: [`docs/ORACLE_API.md`](./docs/ORACLE_API.md); deploy: [`docs/ORACLE_DEPLOY.md`](./docs/ORACLE_DEPLOY.md).

## Hard rules

- **PR-only.** Never push to `main`, even for docs. Merge via the GitHub UI so CI runs.
- **Merging to `main` deploys.** `deploy-oracle.yml` redeploys the Oracle host on any push to `main` touching `api/**` or `pipeline/**`; `deploy-atlas.yml` republishes the GitHub Pages atlas on changes to the root HTML files. There is no separate prod tag — treat every merge as a deploy.
- **`net_guard` is duplicated on purpose.** `pipeline/src/tm/net_guard.py` has a second copy in news-indexer (`src/news_indexer/net_guard.py`). Fix both together — a nightly drift-check CI in news-indexer fails if they diverge.
- **Don't "fix" leftover `openclaw` references.** The openclaw→daatan rename (819f188) was deliberately code/docs-only; live secret names and the local `infra/openclaw/` dir stay `openclaw` until a separate migration.
- **Terraform**: state key `retro/` in `daatan-terraform-state`; never blanket `apply`, use `-target` (workspace rule).

## Running & testing

- Pipeline tests: `cd pipeline && uv run pytest -q` (LLM/network mocked)
- API tests: `cd api && uv run pytest -q`
- Search provider chain (the order matters): `pipeline/src/tm/web_search.py` — news-indexer is step 0, before GDELT and the paid SERP providers.

## Infra cheat-sheet

- Oracle host: EC2 `i-00ac444b94c5ff9b2` (`oracle.daatan.com` / `bayes.daatan.com`), eu-central-1. **SSM only, no SSH.**
- Oracle logs: `/home/ubuntu/truthmachine/oracle_log.txt` — **not** journald; `journalctl` finds nothing useful.
- Services on the box: `truthmachine.service` (batch pipeline loop) + `oracle-api.service` (FastAPI).
- Latency profile: `/forecast` slow tail is dominated by the LLM article phase (p99 ≈ 226 s) and slow GDELT failures — not by the search providers themselves.
- LLM: AWS Bedrock — Nova Micro (gatekeeper) + Nova Lite (extractor/aggregator). Prompts are served from Bedrock via SSM parameter ARNs (shared prod+staging, 5-min cache) — editing a repo fallback prompt alone does **not** change prod.
- Live pages: atlas https://daatan.github.io/retro/ · test console `/oracle-test.html` · Polymarket duel `/duel.html`

## Before opening a PR

1. `git fetch origin main && git rebase origin/main`
2. `uv run pytest -q` in every package you touched (`pipeline/`, `api/`)
3. Update the matching doc in `docs/` if you changed the API surface, pipeline stages, or deploy flow
4. Verify `gh pr view --json mergeable` shows `MERGEABLE` before announcing the PR
