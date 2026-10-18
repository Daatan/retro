# studies

Hand-built retro case studies, one directory per event, published to GitHub Pages at
`https://daatan.github.io/retro/studies/` by `.github/workflows/deploy-atlas.yml`.

Each study is a self-contained `index.html` (Russian) and `en.html` (English), data
inlined, no build step, plus the rated article table as CSV. Add a new study as
`studies/<slug>/` and a card in both `studies/index.html` and `studies/en.html`.

| Study | Event | Built |
|---|---|---|
| [`ukraine-2022/`](ukraine-2022/) | E01, Russia's full-scale invasion of Ukraine, 2022-02-24 | 2026-09-23 |
| [`israel-2022/`](israel-2022/) | E02, 25th Knesset election: does Netanyahu's bloc reach 61? 2022-11-01 | 2026-09-24 |
| [`energy-2022/`](energy-2022/) | E03, Brent ≥ $120 (ICE settle) and TTF > EUR 200/MWh in 2022; both crossed 2022-03-07 | 2026-10-15 |

## ukraine-2022 method, in short

- **Source.** GDELT GKG via BigQuery over 122 domains (news-indexer sources plus major
  world, Russian and Ukrainian outlets and think tanks), 2021-11-20 to 2022-02-23, plus
  Wayback CDX listings and a few hand-found seeds.
- **Filter.** URL/title keywords, Claude Haiku 4.5 title triage, and full-text relevance.
  Dedupe per speaker and day. Drop texts with post-2022-02-24 hindsight. Page dates are
  cross-checked against the listed date.
- **Rating.** Haiku 4.5 rates the main voice of each article: stance −1..1, claim
  strength, expected scope and a verbatim quote. `P = 0.5 + 0.5·stance`, capped at 0.35
  for Donbas-only and 0.45 for limited strikes.
- **Table.** 200 balanced rows, at most 8 per domain and 2 per speaker, with sceptics
  raised to 35%. The aggregates on the page use the full 910-row pool.
- **Gaps.** FT, WSJ and Bloomberg are missing. Metaculus and Telegram are not covered.
  25 quotes are model paraphrases and are flagged.

## israel-2022 method, in short

- **Question.** Will the Likud + Religious Zionism + Shas + UTJ bloc win 61+ seats?
  Outcome: yes, 64.
- **Source.** 326 seed domains (Hebrew, English, Russian and Arabic Israeli press, world
  outlets, think tanks, pollsters). GDELT GKG via BigQuery for 2022-05-01 to 2022-10-31
  matched 157 of them. Wayback CDX added Israel Hayom, Haaretz English, NewsRu.co.il,
  INSS, JPPI and others.
- **Filter.** Keywords in four languages, Haiku title triage (8,780 → 2,261), full-text
  relevance, dedupe per speaker and day and per pollster and week, hindsight check.
  Articles whose page date falls in an earlier campaign (2019–2021) are dropped (37).
  A second Haiku pass on spring-2022 articles drops 33 where "61" is about the current
  Knesset (no-confidence votes, defections, an alternative government), not the election.
- **Rating.** Same `P = 0.5 + 0.5·stance`. For seat polls, stance follows a fixed rule
  on the bloc total (≤58 −0.6, 59 −0.4, 60 −0.2, 61–62 +0.3, ≥63 +0.6), and the
  article's framing may shift it by up to ±0.2.
- **Table.** 300 rows, at most 15 per domain and 3 per speaker. Polls are capped at 30%
  and sceptics raised to 35%. The aggregates use the full 1,100-row pool.
- **Gaps.** N12, Kan and Channel 13 are barely covered, and TV is not covered.
  15 quotes are model paraphrases and are flagged.

## energy-2022 method, in short

- **Questions.** Two independent legs: does ICE front-month Brent settle at or above $120 at
  some point in 2022, and does TTF front-month close above EUR 200/MWh. Outcome: yes and yes,
  first crossed 2022-03-07 (Brent max settle $127.98 on 03-08, TTF max EUR 339 on 08-26).
  Statements window 2021-09-01 to 2022-02-24. No derived score; two P tracks.
- **Source.** 146 seed domains (wires, trade and energy press, agencies, banks, think tanks,
  Russian and European outlets). GDELT GKG via BigQuery, 12,537 energy-titled URLs; Wayback
  for text. Bloomberg, FT and WSJ are barely covered.
- **Filter.** Keyword score and Haiku title triage (2,551 → 817), full-text rating of 498,
  then horizon filter (only `year_2022` / `next_months`; "prices are high today" is dropped),
  hindsight check, page-date check for every year, leg-scope guard (US-gas pieces cannot feed
  the TTF leg), dedupe per speaker and day.
- **Rating.** Same `P = 0.5 + 0.5·stance` per leg; a stated probability wins. Price forecasts
  follow a fixed clamp rule (Brent <90 −0.6 … ≥130 +0.6; TTF <100 −0.6 … ≥250 +0.6).
- **Table.** 211 rows, at most 15 per domain and 3 per speaker. Aggregates use the 264-row pool.
- **Gaps.** Most articles name no price, so P is the model's directional reading, not a
  calibrated probability. TTF coverage by banks and agencies is thin (18 and 4 statements).

Working files (fetch lists, raw texts, rating JSONL) are kept outside the repo.
