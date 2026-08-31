# Metaculus sync

Submits Daatan Oracul forecasts into a Metaculus tournament, under the
`daatan-v1` bot account. Built for daatan#1554 / the Trello "Compete on
Metaculus arena" card, to get an external, public benchmark of Oracul's
calibration alongside FutureSearch and others.

Only binary questions are handled — Oracul's `mean` stance in `[-1, 1]` maps
directly to a probability via `(mean + 1) / 2`; there's no numeric/date/
categorical output shape today.

## How it works

`sync.py` polls open binary questions in a target tournament
(`METACULUS_TOURNAMENT`, default `bot-testing-area` — unscored, intended for
exactly this kind of testing; `auto` resolves the current FutureEval/AIB
season by itself — see "Season auto-detect" below), skips any it has already forecast recently
(`STALE_AFTER_HOURS`), asks Oracul (`oracle_client.py`, hitting
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

Plus a relay key into Oracul itself, distinct from the primary daatan-app
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

Tournament questions are open for **3.0 hours**, launch at random hours up to 5
at a time, and are scored on **spot peer score** — only the last forecast
before close counts.

The 3.0h is measured, not quoted: every one of the 60 questions in the
2026-08-24 MiniBench round had exactly a 3.00h window (min = p50 = max), and
that has held for the five rounds since 2026-06-15. Metaculus's own resources
notebook still says 1.5h; it is stale. Do not reason from 1.5h.

Two knobs, routinely confused — one is about *catching* a question, the other
about *how many times we answer it*:

- **Poll every ~20 minutes.** retro#617 measured `/forecast` over n=20,000
  production calls: p50 7.3s, p99 25.0s, max 45.9s, zero timeouts against a
  90s cap. Latency is not a constraint, so the 6-hour cadence an earlier draft
  proposed would simply miss most questions. Unchanged by anything below.
- **`STALE_AFTER_HOURS` defaults to 4** — deliberately *above* the 3h window,
  so each question gets exactly one forecast. The FutureEval rules state that
  "bot makers should only submit one forecast per question in these bot-only
  tournaments". An earlier 0.75 default was chosen to get ~2 forecasts and let
  mid-window news reach the scored submission; that is better for the score and
  against the rules, so retro#755 inverted it. The cost is real — we forgo the
  mid-window update — and it is the price of entering compliant.

`bot-testing-area` is the exception the rule itself names ("testing areas where
resubmission is encouraged"), so a lower value is fine there.

## Scheduling — a systemd timer on the oracle box, not Actions cron

Discovery runs from `infra/metaculus-sync.timer` (every 20 minutes), not from a
`schedule:` block in `.github/workflows/metaculus-sync.yml`. **Do not add a cron
there as well** — two schedulers would double-submit into the same question
window.

The reason is not preference. GitHub Actions cron is delayed and skipped under
load, and that unreliability is the stated reason Metaculus widened the question
window to 3h. Scheduling our discovery on the same substrate would inherit
precisely the risk the widening exists to absorb, and inside a 3h window a
missed poll is a forfeited question (retro#728).

This matters more than the window length suggests. A MiniBench round is not a
leisurely fortnight: all ~60 questions of the 2026-08-24 round opened and closed
inside **46 hours** (first open 08-24 00:00:00Z, last close 08-25 22:04:35Z),
across 21 distinct UTC hours, and that burst has compressed from ~6.7 days in
May to ~1.8 days now. There is no quiet hour to miss.

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
the Oracul API's convention, is already ~341 MB and is unrotated, and a job that
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

## Season auto-detect (retro#726)

There is no per-season registration on Metaculus: a bot account simply
forecasts on whatever the current season's questions are. Seasons start every
September, January and May (`spring-aib-2026` → `summer-futureeval-2026` → …)
and the new slug is not announced anywhere machine-readable ahead of time, so
`METACULUS_TOURNAMENT=auto` asks `/api/projects/tournaments/` (authenticated —
the bot token is enough) and picks the latest-started tournament whose slug or
name says `futureeval`/`aib` and whose `forecasting_end_date` — the day
questions stop opening; `close_date` is months later, after resolution — is
still ahead. On the ~week where Summer's window overlaps Fall's start, Fall
wins. If nothing is open the run logs that and exits 0.

Set the explicit slug instead of `auto` whenever you want to pin a season
(e.g. `bot-testing-area` for a dry run — `auto` never selects it, it carries
no season marker).
