# case-studies

`index.html` — a **self-contained** marketing/explainer page presenting Retro
Analysis (בדיעבד) case studies. Plain HTML styled with Tailwind via CDN and
Google Fonts; no build step and no dependency on the pipeline or atlas data
(the narrative content is inline).

## Status

- **Not currently published by CI.** `.github/workflows/deploy-atlas.yml` copies
  `factum_atlas.html`, `oracle-test.html`, `duel.html`, and `bayesoracle/` to
  GitHub Pages — it does **not** include `case-studies/`. So this page is not live
  at `daatan.github.io/retro/` today; it's shared/opened directly.
- It's the plain-HTML sibling of the React prototype in [`ui-prototype/`](../ui-prototype/)
  — same case-study idea, two unconnected implementations.

## If you want it published

Add a copy step to `deploy-atlas.yml` (e.g. `cp -r case-studies _site/case-studies`)
and link it from the atlas index. Until then, treat it as a standalone artifact —
and decide whether it's still maintained or should be retired to avoid drift with
the live Daatan site.
