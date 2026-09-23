# studies

Hand-built retro case studies, one directory per event, published to GitHub Pages at
`https://daatan.github.io/retro/studies/` by `.github/workflows/deploy-atlas.yml`.

Each study is a self-contained `index.html` (data inlined, no build step) plus the
rated article table as CSV. Add a new study as `studies/<slug>/` and a card in
`studies/index.html`.

| Study | Event | Built |
|---|---|---|
| [`ukraine-2022/`](ukraine-2022/) | E01, Russia's full-scale invasion of Ukraine, 2022-02-24 | 2026-09-23 |

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
  raised to 35%. The aggregates on the page use the full 913-row pool.
- **Gaps.** FT, WSJ and Bloomberg are missing. Metaculus and Telegram are not covered.
  25 quotes are model paraphrases and are flagged.

Working files (fetch lists, raw texts, rating JSONL) are kept outside the repo.
