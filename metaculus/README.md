# Metaculus sync

Submits Daatan Oracle forecasts into a Metaculus tournament, under the
`daatan-v1` bot account. Built for daatan#1554 / the Trello "Compete on
Metaculus arena" card, to get an external, public benchmark of Oracle's
calibration alongside FutureSearch and others.

Only binary questions are handled — Oracle's `mean` stance in `[-1, 1]` maps
directly to a probability via `(mean + 1) / 2`; there's no numeric/date/
categorical output shape today.

## How it works

`sync.py` polls open binary questions in a target tournament
(`METACULUS_TOURNAMENT`, default `bot-testing-area` — unscored, intended for
exactly this kind of testing), skips any it has already forecast recently
(`STALE_AFTER_HOURS`), asks Oracle (`oracle_client.py`, hitting
`oracle.daatan.com/forecast`) for a probability + rationale, and submits the
forecast plus a private rationale comment via `metaculus_client.py`.

It does **not** maintain its own state — "already forecast" is read straight
from Metaculus's own `question.my_forecasts.latest`, so the job is safely
idempotent across runs without needing to commit anything back to the repo
(unlike `bayesoracle/`'s daily data refresh).

## Required secrets

Two identities exist, one used today:

| Secret | Metaculus bot | Used for |
|---|---|---|
| `metaculus/oracle-v1-api-key` | `daatan-v1` | Live — the only bot wired up so far |
| `metaculus/oracle-v2-api-key` | `daatan-v2` | Stored, **not yet wired in** — v2/conditionals has no shipped forecast-generation code distinct from v1 yet, so there's nothing meaningfully different to submit |

Plus a relay key into Oracle itself, distinct from the primary daatan-app
key (docs#57: a shared key previously burned unmetered LLM spend) — register
it as a new named, capped entry in `ORACLE_API_KEYS` on the oracle-api host
before pointing this at anything but a local/dry-run test. Store the value
this script sends as `x-api-key` under `metaculus/oracle-relay-api-key`.

## Local run

```
cd /home/mark/projects/retro
uv run --project metaculus python metaculus/sync.py
```

with `METACULUS_API_KEY`, `ORACLE_API_KEY` set (and optionally `DRY_RUN=true`
to log without submitting).

## Cadence

Tournament questions are open for **1.5 hours** (temporarily 3h), launch at
random hours up to 5 at a time, and are scored on **spot peer score** — only
the last forecast before close counts.

Two consequences, both measured rather than assumed:

- **Poll every ~20 minutes.** retro#617 measured `/forecast` over n=20,000
  production calls: p50 7.3s, p99 25.0s, max 45.9s, zero timeouts against a
  90s cap. Latency is not a constraint, so the 6-hour cadence an earlier draft
  proposed would simply miss most questions.
- **`STALE_AFTER_HOURS` defaults to 0.75**, not 24. At 45 minutes we get ~2
  forecasts per question, so news breaking mid-window still reaches the
  forecast that actually gets scored. A 24h value meant one forecast at
  discovery and no update, which throws away the spot-score mechanic.

## Scheduling — a systemd timer on the oracle box, not Actions cron

Discovery runs from `infra/metaculus-sync.timer` (every 20 minutes), not from a
`schedule:` block in `.github/workflows/metaculus-sync.yml`. **Do not add a cron
there as well** — two schedulers would double-submit into the same question
window.

The reason is not preference. GitHub Actions cron is delayed and skipped under
load, and that unreliability is the stated reason Metaculus temporarily widened
the question window from 1.5h to 3h. Scheduling our discovery on the same
substrate would inherit precisely the risk the widening exists to absorb, and on
a 1.5h window a missed poll is a forfeited question (retro#728).

Install or refresh the units (idempotent, safe after any deploy that changed
them):

```
sudo bash /home/ubuntu/oracle-api/infra/install_metaculus_timer.sh
```

The service is gated on `ConditionPathExists=/home/ubuntu/truthmachine/.env.metaculus`,
so **the schedule can be installed before the credentials exist** (retro#725):
until that file is placed, each activation is skipped cleanly rather than
failing every 20 minutes. That file holds `METACULUS_API_KEY`, `ORACLE_API_KEY`
(the named capped relay key, never the primary) and `METACULUS_TOURNAMENT`; the
box has no Secrets Manager access, so it is placed by hand.

Logs go to the journal, **not** to `truthmachine/oracle_log.txt` — that file is
the Oracle API's convention, is already ~341 MB and is unrotated, and a job that
fires 72 times a day does not belong in it:

```
journalctl -u metaculus-sync.service -n 100
systemctl list-timers metaculus-sync.timer
```

`workflow_dispatch` on the Actions workflow stays, and is still the right way to
run a manual dry run without touching the box.

## Not yet done

- The `ORACLE_API_KEYS` named-key entry for this relay hasn't been added to
  the live oracle-api host — required before any real (even unscored) run
  (retro#725). Neither has `/home/ubuntu/truthmachine/.env.metaculus`, which the
  timer is gated on.
- The systemd units are committed but **not yet installed on the box** — run
  `infra/install_metaculus_timer.sh` there once the credentials are placed.
- Still points at `bot-testing-area` (unscored) — deliberately not pointed at
  a real AIB/MiniBench tournament yet.
