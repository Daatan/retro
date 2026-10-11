# Manifold paper bot (retro#918)

A calibration arena, not a trading bot. Every six hours it asks Oracul for a
probability on liquid, public-interest binary markets on
[Manifold Markets](https://manifold.markets), records Manifold's price next to
it, and books a **hypothetical** M$100 position when the two disagree by more
than 10 points. Positions are held to resolution. Nothing is ever sent to
Manifold.

Why: Metaculus (retro#619) needs a DTUA whose §3 forbids commercial use of
anything "derived from the Data"; Polymarket (retro#620) is one election.
Manifold has hundreds of short-horizon markets that resolve within days, a
public API that permits bots, and no agreement to sign. It answers one
question: *is the Oracle's probability calibrated, and does it beat the crowd?*

## Phase 1: read-only, by design

* No Manifold account, no API key, no `Authorization` header. Only the two
  unauthenticated GET endpoints `/v0/search-markets` and `/v0/market/{id}`.
* The client has no write method; `tests/test_manifold_client.py` (AST-level)
  and `infra/tests/test_manifold_paper_timer.sh` fail if a POST/PUT/DELETE, a
  `/v0/bet` path or a key appears.
* Every trade row is stamped `"paper": true`.
* Real-Mana bets (a bot-flagged account, a key in Secrets Manager, capped
  stakes) are a **separate, gated Phase 2** with its own issue and module. This
  project must not grow that capability incrementally.

## Evaluation only: Manifold's ToS

Manifold's Terms and API docs: *"You may not use data obtained from the Site or
through our API to train artificial intelligence or machine learning models for
commercial purposes without first obtaining a data license."* Academic and
non-commercial use is permitted; commercial licensing is `data@manifold.markets`.

So: this ledger and its scorecard are an **evaluation** of the Oracle. They are
not training, fine-tuning, few-shot or prompt-tuning data for the production
Oracle. If a finding here would change a prompt, a threshold or a model, take it
to Mark first (it may need a data licence). Prize drawings forbid automated
entries — this bot does not enter any.

API limits: 500 requests/minute/IP; a run makes a few hundred requests at most.

## Universe

`GET /v0/search-markets?contractType=BINARY&filter=closing-month` (3 pages),
then `universe.py`: at least `MIN_BETTORS` unique bettors and `MIN_VOLUME` Mana,
closing between `MIN_HOURS_TO_CLOSE` and `MAX_DAYS_TO_CLOSE`, not
personal/self-referential (keyword filter). Markets already in the ledger skip
admission and are re-fetched by id every run until they resolve, so a market
that falls out of the search still gets its outcome recorded. The per-run
Oracle cap goes to the soonest-closing markets.

`resolution` of YES/NO is scored; MKT (probabilistic) and CANCEL are recorded
(`resolution_raw`) but never scored.

## Scorecard

`GET /manifold/paper` (`?format=md`), built by
`api/src/forecast_api/manifold_paper.py` from the ledger alone:

* Brier and log-loss, Oracle vs market, same snapshot (the last with an Oracle number)
* Brier difference (oracle − market) with a seeded bootstrap 95% CI
* reliability table (5 bins), coverage (share of markets that got an Oracle number)
* the **pre-registered bar**: ≥150 scored markets and Oracle Brier ≤ market + 0.01.
  Otherwise stop and note it in `Daatan/docs` `decisions.md`.

## Layout

| File | Role |
|---|---|
| `paper.py` | entrypoint, universe gathering, run loop, pure decision functions |
| `manifold_client.py` | GET-only client, `Market` dataclass, resolution mapping |
| `universe.py` | admission filter |
| `oracle_client.py` | `POST /forecast` on the Oracle API (copy of the Polymarket one) |
| `ledger.py` | append-only JSONL: `snapshots.jsonl`, `trades.jsonl` |
| `../api/src/forecast_api/manifold_paper.py` | scorecard for `GET /manifold/paper` |
| `../infra/manifold-paper.{service,timer}`, `install_manifold_paper_timer.sh` | systemd oneshot, 00:45/06:45/12:45/18:45 UTC |

## Env

See the docstring of `paper.py`. On the box `ORACLE_API_KEY` comes from
`.env.manifold-paper`, which the installer derives from the `ORACLE_API_KEYS`
map (`manifold-paper` entry, else `polymarket-paper`, else `metaculus`). A
capped named entry is preferred (docs#57: never the primary daatan-app key).
The ledger lives in `/home/ubuntu/truthmachine/data/manifold_paper`, which the
API reads.

Cost: ≈ 12 Oracle calls per run × 4 runs, each market re-asked at most every
20 h — about the same ≈ $0.60/day as the Polymarket bot.

```bash
curl -s https://oracle.daatan.com/manifold/paper?format=md
curl -s https://oracle.daatan.com/manifold/paper | jq .summary
```

## Develop

```bash
cd manifold_paper
uv sync --frozen --extra dev
uv run pytest -q
DRY_RUN=1 PAPER_LEDGER_DIR=/tmp/mf uv run python paper.py   # reads Manifold, no Oracle call, writes nothing
```
