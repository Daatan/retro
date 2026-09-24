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

Working files (fetch lists, raw texts, rating JSONL) are kept outside the repo.
