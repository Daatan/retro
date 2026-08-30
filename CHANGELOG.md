# Changelog

All notable changes to the Oracle (the `api/` service and the `pipeline/` batch lane) are
recorded here. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

**Version scheme (retro#742).** The number is the *product generation*, not an API contract:
**1.4.x** is the live v1 engine (what `0.4.1` mislabels today), **1.5.x** is reserved for the
Oracle 1.5 programme and **2.x** for the graph engine — each flip is an explicit decision, not a
side effect of a merge. `/version` composes this base with an auto-incrementing `+build.N`
(commit count, written by `infra/deploy_oracle.sh`). Response-shape compatibility is gated on
`provenance.schema_version`, not on this number.

**Prompt changes** are tracked in [`docs/PROMPT_VERSIONS.md`](docs/PROMPT_VERSIONS.md) and
referenced here by version (e.g. "extractor prompt v8"), never duplicated.

Every PR that touches `api/**` or `pipeline/src/**` adds a line under **Unreleased** (CI
enforces it; label `no-changelog` opts out). Cutting a release moves that section under a
version heading — see `docs/ORACLE_DEPLOY.md` § Cutting a release.

## [Unreleased]

### Fixed
- Threshold archetype: a cue with a bare number behind it now counts as a magnitude ("more than 33 seats", "at least 8 goals"), and `between` joins the cue set (retro#748). The two gaps hid half the numeric-threshold A/B corpus and four live prod forecasts from `is_threshold_shaped`; the bare integer is bound to the cue rather than recognised anywhere, so prod firing moves 4.4% → 6.6% with no false positive among the five newly-caught claims.

## [1.4.0] — 2026-08-30

First release under the generation scheme (retro#742): `0.4.1` → `1.4.0`, same engine. `pipeline/pyproject.toml` moves `0.1.0` → `1.4.0` too — one product.

### Added
- Batch lane: every vault extraction JSON (done and gate_rejected/no_predictions markers) now carries `oracle_version` + `git_sha` of the running pipeline tree (retro#744) (#747)
- CHANGELOG, `release.yml` (tag → GitHub release) and the changelog-touched PR check (retro#743)
- Provenance block + `schema_version` on `/forecast` and `/pool/aggregate` (retro#593, #612); prompt version/hash in `ProvenanceModels` (retro#627, #628); gatekeeper prompt version/hash on `/relevance` (#639); effective article ceiling in provenance (retro#652, #653)
- Oracle 2.0 playground — `/v2/forecast` traced jobs + `oracle-v2-test.html` (#595, #597, #599, #614)
- Confidence bucket on plain `/forecast` and `/pool/aggregate` (retro#618, #630)
- Opt-in per-request extractor model override (retro#652, #654)
- Antecedent filtering of the pool, wired into the live `/forecast` path (retro#573, #582, #585)
- Premise-check step, shadow/log-only (retro#575, #587)
- Shadow-only precursor candidate-match — Daatan bank + Polymarket (retro#608, #611, #658)
- Settled-grounding shadow slice (retro#609, #613)
- Retry-relaxed-search fallback ladder rung 1, shadow-only (retro#621, #648, #661)
- Deterministic settlement semantic gates: backtest harness, 387 labelled pairs, `facet_missing` gate, shadow wiring, outcome-contradicted-pin scoring, `logs.sh settlement` (retro#691, #693, #694, #696, #698, #699, #701)
- Oracle 1.5 Phase 1 elicited fields (shadow): `certainty` → `claim_strength` (retro#680, #692), `reader_confidence` (retro#681, #695), `report_kind` + `consensus_view` (retro#686, #719), `quantity` + the code-side threshold comparison and its per-rater A/B diagnostic (retro#683); extractor prompt v9; deterministic confusion flags, log-only (retro#687, #712); `(actor, target, day)` clustering beside the Jaccard key (retro#682, #717)
- Threshold-shaped batch events routed to a second extractor, ships off (retro#688, #702)
- Metaculus: calibration backtest driver (retro#619, #624); sync service for the daatan-v1 bot (daatan#1554, #723, #733, #736, #738); IAM for `metaculus/*` secrets (retro#725, #735)
- Periodic check for the resolution-shadow-credibility gate (retro#604, #629)
- Extractor audits: log-only wrong-entity dyad mismatch (retro#545, #586, #645, #646), fabricated-quote provenance (#660), `fact_signal` sign-mismatch warning (retro#602, #610), `author_lean` sign-mismatch guard (retro#326, #656)
- A/B harness: refuse a dead arm (retro#561, #675); multi-stage/bracket + magnitude-facet corpus (retro#720, #734); stance sign-flip rate in `eval_extractor_stability` (retro#664, #665, #671)
- CI: prompt version bumps enforced (#633); response schema hashed into the prompt lock (retro#700, #710); five appended prompt blocks locked (#731, #732); `ec2_run.sh` guard regression tests (#557, #606); terraform fmt + validate gate (#598)
- BayesOracle viewers: resolved-market flags, Daatan forecast links, live GitHub Pages links (#580, #581)

### Changed
- Extractor prompt: FACET block excludes polls/projections (retro#541, #584); tone-vs-stance disambiguation (retro#545, #649); settlement bounded to the claim window at extraction time (retro#704, #707); lead claim's reading carried across article aggregation (retro#721, #722)
- Gatekeeper: guard against same-place-different-story confabulation (retro#623, #625)
- Gate-0 evidence window enforced (retro#545, #659)
- Articles that cannot be dated are dropped rather than dated to today (retro#705, #706); provider publication dates normalised (retro#714, #715); search-engine link wrappers dropped instead of stored (retro#709, #713)
- Bedrock gatekeeper/extractor/ground-truth defaults pinned to the `us.` region (retro#548, #596)
- Search-provider keys and Cognito Google OAuth moved from Secrets Manager to SSM (docs#122, #603, #605, #607, #672)
- Oracle box: Elastic IP (retro#436, #668); unit file synced with the retro#600 mitigations (#636); nginx `client_max_body_size` 16m (#716, #718); Bedrock alarms on a `us-east-1` SNS topic (#674)
- Docs: batch/live prompt-version divergence made explicit (retro#631, #634); extraction/elicitation split adopted in prose (docs#156, #667); extractor/gatekeeper/settlement-verifier model survey (#647); claim-issues-first rule (#594); "track record" wording (#662)
- Deps: aiohttp, cryptography, h2, starlette, pyasn1, pydantic-settings (#638); official uv binary in Docker, compose SIGTERM handling (#640)

### Fixed
- Skip Bedrock prompt caching for models that don't support it (retro#650, #651)
- `short-form` split from "has no article page" (#690)
- `test_web_search` made network-free (retro#708, #711)
- Pipeline: retro#600 stalled-event-loop false-positive timeout (#635); GDELT BigQuery `job.result()` timeout (#663)
- `duel_report.py`'s dead oracle-api-key reference pointed at SSM (docs#122, #605)

## [0.4.1] — 2026-08-22
- fix(v2 playground): no fake pool-split edges; same-question gate on Polymarket anchors (#599)

## [0.4.0] — 2026-08-22
- feat(api): Oracle 2.0 playground — `POST /v2/forecast` job + `GET /v2/jobs/{id}` trace, `oracle-v2-test.html` (#595, #597)

## [0.3.2] — 2026-08-05
- fix(cache): share `forecast_cache` between workers via diskcache (retro#405, #411)

## [0.3.1] — 2026-07-16
- fix(oracle-mcp): proxy authorize/token to strip `offline_access` (Cognito rejects it) (#283)

## [0.3.0] — 2026-07-15
- feat(oracle-mcp): DCR façade for the human Claude-connector login (#277)

## [0.2.1] — 2026-07-15
- fix(oracle-mcp): allow public Host on the MCP transport (DNS-rebinding guard) (#275)

## [0.2.0] — 2026-07-15
- feat(oracle): MCP server exposing forecast + Polymarket-edge tools, OAuth via Cognito (#270)

## [0.1.0] — 2026-04-14
- feat: Oracle API — FastAPI prediction service skeleton (#25)

[Unreleased]: https://github.com/Daatan/retro/compare/v1.4.0...HEAD
[1.4.0]: https://github.com/Daatan/retro/compare/924538f3e...v1.4.0
[0.4.1]: https://github.com/Daatan/retro/compare/302648267...924538f3e
[0.4.0]: https://github.com/Daatan/retro/compare/c02ff7aca...302648267
[0.3.2]: https://github.com/Daatan/retro/compare/acfcb26e3...c02ff7aca
[0.3.1]: https://github.com/Daatan/retro/compare/c92569fbd...acfcb26e3
[0.3.0]: https://github.com/Daatan/retro/compare/199ff8c3e...c92569fbd
[0.2.1]: https://github.com/Daatan/retro/compare/cb0c271ed...199ff8c3e
[0.2.0]: https://github.com/Daatan/retro/compare/4f169c111...cb0c271ed
[0.1.0]: https://github.com/Daatan/retro/commit/4f169c111
