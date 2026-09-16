# Polymarket paper trading (retro#620)

A scoreboard, not a trading bot. Every six hours it asks Oracul for a
probability on each open market in a fixed Polymarket cluster, records the
market price next to it, and books a **hypothetical** $100 position whenever
the two disagree by more than 10 points. Positions are held to resolution and
never closed early. Nothing is ever sent to Polymarket.

Why: none of the "make money with the Oracle" ideas can be judged until the
Oracle has a public, externally-priced track record. This produces one, on a
fixed date, at a cost of about $30 in LLM calls.

## No trading, by design

* The only Polymarket dependency is the public, read-only **Gamma** API
  (`gamma-api.polymarket.com`). There is no CLOB client, no wallet, no
  `web3`/`py-clob-client` dependency, no private key anywhere in this tree —
  `infra/tests/test_polymarket_paper_timer.sh` fails if one is added.
* Every trade row is stamped `"paper": true`.
* The report is public (`GET /pm/paper`) because there is nothing to protect.

Flipping this into real trading is a separate decision with its own issue;
this project must not grow that capability incrementally.

## Relation to `duel.html`

The older public "Duel" page scores TruthMachine vs Polymarket on a hand-picked set of
already-resolved events (13 at the time of writing). This bot is the forward-looking
version: forecasts are recorded *before* resolution, on a fixed universe, at a fixed
cadence, so nothing can be selected after the fact.

## Universe

The Israeli Knesset election of **2026-10-27** — ten Gamma events, about
30 binary markets with live prices (`universe.py`; override with
`PAPER_EVENT_SLUGS`, comma-separated). Placeholder markets ("Party A" with
no price) and non-yes/no markets are skipped. The cluster was chosen because
it is the one topic where Daatan's news coverage is deepest, so if Oracul has
an edge anywhere, it should show up here.

Resolution timing, so the scorecard is read correctly:

| Event(s)                                         | Resolves           |
|--------------------------------------------------|--------------------|
| election winner, hung parliament, Likud loses seats, held-as-scheduled | ~2026-10-28 (official results) |
| seat buckets (Likud, UTJ, Ra'am), Shas vote share, parties over threshold | 2026-11-30 event end |
| next Prime Minister                              | on swearing-in — weeks after |

## Files

| File | Role |
|------|------|
| `paper.py` | The run: universe → snapshots → Oracle calls → paper trades. Entry point. |
| `gamma_client.py` | Gamma API client + `Market` parsing (prices, closed, resolved outcome). |
| `oracle_client.py` | Copy of `metaculus/oracle_client.py` — `POST /forecast` → `mean`. |
| `ledger.py` | Append-only JSONL: `snapshots.jsonl`, `trades.jsonl`. |
| `universe.py` | Default event slugs. |
| `../api/src/forecast_api/pm_paper.py` | Reads the ledger and renders the scorecard for `GET /pm/paper`. |
| `../infra/polymarket-paper.{service,timer}`, `install_polymarket_paper_timer.sh` | systemd oneshot, 00:30/06:30/12:30/18:30 UTC. |

## Run parameters (environment)

| Variable | Default | Meaning |
|----------|---------|---------|
| `ORACLE_API_KEY` | required | Relay key for `oracle.daatan.com`. On the box it comes from `.env.metaculus`. |
| `ORACLE_BASE_URL` | `https://oracle.daatan.com` | |
| `PAPER_LEDGER_DIR` | `./data` | On the box: `/home/ubuntu/truthmachine/data/polymarket_paper`, which is what the API reads. |
| `PAPER_EVENT_SLUGS` | the ten Israeli-election events | |
| `MAX_MARKETS_PER_RUN` | 12 | Cap on Oracle calls per run. Staleness decides who goes first. |
| `STALE_AFTER_HOURS` | 20 | Re-ask the Oracle about a market only after this long (≈ once a day). |
| `EDGE_THRESHOLD` | 0.10 | \|Oracle − market\| needed to open a paper position. |
| `TRADE_PRICE_MIN/MAX` | 0.05 / 0.95 | Don't paper-trade near-certain markets. |
| `ORACLE_PRICE_MIN/MAX` | 0.02 / 0.98 | Don't even ask about markets the price already settles. |
| `NOTIONAL_USD` | 100 | Per position. |
| `DRY_RUN` | unset | `1` = fetch and forecast, write nothing. |

**Relay-key caveat.** The bot reuses the Metaculus relay key, which the API
caps at `max_articles=5` (prod interactive default is 10). So this measures a
slightly handicapped Oracul. Every snapshot records `articles_used` and
`articles_found`, so the handicap is visible in the data. To remove it, add a
`polymarket-paper` entry to `ORACLE_API_KEYS` without the cap and put that key
in `/home/ubuntu/truthmachine/.env.polymarket-paper` (read after
`.env.metaculus`, so it overrides).

## Cost

About 30 markets, one Oracle call each per day (the 20 h staleness plus the
12-per-run cap spread them over the day), about 2¢ per call → roughly
$0.60/day, **about $30 for the 42 days to the election**. Gamma is free.

## Reading the scorecard

```
curl -s https://oracle.daatan.com/pm/paper?format=md
curl -s https://oracle.daatan.com/pm/paper | jq .summary
```

Two views, both from the same ledger:

1. **Mark-to-market** (available from day one, headline for 2026-10-28):
   every paper position valued at the latest market price; `unrealized_pnl_usd`
   and, once markets resolve, `realized_pnl_usd`, `n_won`, `n_lost`. A
   positive number means the market moved toward the Oracle after it
   disagreed; it is suggestive, not proof.
2. **Brier on resolution** (fills in as markets resolve): for each resolved
   market, the Oracle's *last* probability and the market price *at that same
   snapshot*, both scored against the outcome. `brier_oracle < brier_market`
   is the only number that says the Oracle beats the market. With ~30
   markets, one election and heavily correlated questions, treat it as a
   single data point, not a distribution.

What would count as a result on 10-28: Brier on the ~15 markets that resolve
with the official count, plus MTM on the rest. What would not: any P&L
figure alone.

## Local run

```
cd polymarket_paper
uv sync --frozen
ORACLE_API_KEY=... DRY_RUN=1 uv run python paper.py      # no writes
uv run pytest -q                                          # 34 tests, all offline
```
