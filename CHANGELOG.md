# Changelog

All notable changes to the Oracul (the `api/` service and the `pipeline/` batch lane) are
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
- Extractor prompt **v13** (retro#770 class 1): an incidental office title ("Prime Minister Benjamin Netanyahu") was extracted as a **verified, occurrence-typed claim at stance +1.0** on questions about whether that person holds the office at a *future* date — a systematic upward bias on every incumbent-survival forecast, and the source of four rounds of manual evidence-pool exclusions. New `PROMPT_PREFIX` section: a title used to IDENTIFY someone is a naming convention, so on a future-dated tenure question it is a precondition (`|stance| <= 0.2`, `claim_strength <= 0.3`, `is_occurrence false`, never settled), and a fragment carrying no proposition — a broadcast billing, a caption, a colon lead-in — extracts nothing. Behavioural rule only: no new field, schema and `provenance.schema_version` unchanged. Two of the five worked examples are positive controls so the rule cannot read as "ignore incumbent coverage" (the retro#720 failure mode). **The anonymised A/B could not see this bug** — a placeholder "Leader L" reproduces nothing on either rater, since it needs the model's own knowledge that Netanyahu IS the incumbent, so the cases carry the real entity, question and article text. Measured: Haiku 4.5 gate 0, zero regressions, tokens +2.2%, and on the real defect rows 3/3 runs `+1.0/settled=true` → no extraction and `+0.9` → `+0.2` with the tenure control's full negative signal preserved. Nova Lite gate 1 at 3 runs on one bracket case that is satisfied on **both** arms at 15 runs (verdict is sample-size dependent), tokens +2.3%; it never had the defect (0/15 baseline, 1/13 patched). **Gate on Haiku** — daatan's evidence pool only ever calls the live API. Class 2 (sign inversion) deliberately not attempted: it is the retro#657 negation shape that has failed three times on Nova

### Added
- Once-per-question WHO/WHAT/SCOPE decomposition, injected into the extractor's RELATED EVENT input rather than requested as output (retro#758, control arm to retro#697): appended to the prompt's RELATED EVENT block when present, byte-identical to today when absent — zero prompt or schema change. Measured on the deadline/denial regression sentinel: Nova Lite 11/15 → 15/15. **Ships off by default** (`settings.inject_event_decomposition`) — the sentinel alone doesn't establish the retro#691 387-pair adjacency measurement this issue calls for as the real acceptance test
- Extractor prompt **v12** (retro#763, Oracle 1.5 Phase 1, unparked from retro#673 §1): one elicited shadow field at the schema tail — `grounds` `{kind, basis}`, what the quote's position RESTS ON (`event_observed` / `authority_asserted` / `market_or_poll_number` / `expert_inference` / `historical_base_rate` / `writer_assertion`, plus the phrase naming it). Lets the pool count *reasons* rather than articles (three outlets repeating one statement are one ground) and recognise same-grounds-same-stance duplicates. Not an `evidence_class` extension — class is the route, grounds is what was seen. Threads to `ClaimDetail` per claim and rolls up to `SourceSignal` as the dominant claim's; `provenance.schema_version` 1.5 → 1.6. **v12d renames the GROUNDS kinds to share no lexical stem with `evidence_class`** (v12a-c leaked `official_statement` into `evidence_class` 21 times on the full-corpus A/B — a silent down-weight the gate doesn't check). A/B numbers in `docs/PROMPT_VERSIONS.md`
- Extractor prompt **v11** (retro#697, Oracle 1.5 Phase 1): the WHO / WHAT / WITHIN WHAT SCOPE decomposition `PROMPT_PREFIX` § MATCH THE EVENT has required since v1 is now REPORTED instead of discarded — `claim_actor` (`{name, type}`), `claim_predicate` and `claim_scope`, three shadow fields at the `ExtractionOutput` tail. Question-level, not per-claim: they describe the RELATED EVENT, so they ride on `SourceSignal` beside `consensus_view` and deliberately get no `ClaimDetail` copy. `provenance.schema_version` 1.4 → 1.5. Their consumer is `settlement_semantic.claim_subject_from_fields` (new, `proxy=False`), which replaces the regex proxy that makes `gate_predicate_echo`'s 0.68/0.46 a documented lower bound. Measured: Haiku 4.5 gate 0 on all five corpus files, 130/130 fill, consistency 1.00, tokens +12.7%; Nova Lite 3 improvements, 1 regression on a documented comparator case (retro#683/#688 territory, reproduced at 0/15 by an input-only change to the v10 prompt), tokens +4.1%. No prose paragraph — five measured variants showed it never carried the fill. Spin-outs retro#757 (harness voting) and retro#758 (inject the decomposition as input)
- Extractor prompt **v10** (retro#684, Oracle 1.5 Phase 1): two elicited shadow fields at the schema tail — `tone` (`approve`/`neutral`/`alarm`, the quote's own register, so an evaluation stops being read as a direction) and `voice` (`{kind, attributed_to}`, whose assertion the quote is, so a wire carried by thirty outlets counts once). Both thread to `ClaimDetail` per claim and roll up to `SourceSignal` as the dominant claim's; `provenance.schema_version` 1.3 → 1.4. Zero A/B regressions on both raters, `token_usage` +7.9%. **`tone` is Haiku-only** — Nova Lite answers `neutral` on 145/145 and trips the field's kill criterion; `voice` clears it on both

### Changed
- Rename Oracle → Oracul in this repo's code prose (Daatan/retro#766 wave 2, retro#768): 101 mentions in 48 files across `api/`, `pipeline/`, `metaculus/` and `bayesoracle/`. **Comments, docstrings, log and console messages, and duel-report copy only — not one symbol, file or import was renamed, so behaviour is byte-identical apart from log text.** The one string a third party sees is the Metaculus comment rationale, now "Forecast from Daatan Oracul.". Everything a client can depend on keeps the Oracle name: the whole of `mcp_server.py` (each `@mcp.tool()` docstring *is* the description a registered OAuth connector reads, and `oracle_probability` is a response field), the `main.py` FastAPI title `TruthMachine Oracle API`, the `models.py` served `Field(description=...)`, `api/pyproject.toml`, and every line naming `ORACLE_API_KEY`, `oracle.daatan.com`, `oracle-api.service` or an `oracle-mcp/*` OAuth scope. `infra/`, `terraform/`, `data/` and `.github/` are untouched, as is `Oracle 1.5` / `Oracle 2.0` — those name the product *generation* (retro#742), not the service
- Prose rename Oracle → Oracul across this repo's docs (Daatan/retro#766 wave 1, retro#767): 131 mentions in 30 markdown files. **Documentation only — no code, no config, no deploy artefact.** Every wire identifier keeps the Oracle name permanently per the wave 3 decision (`oracle.daatan.com`, `ORACLE_URL`/`ORACLE_API_KEY`, `oracle-api.service`, `oracle.conf`, `oracle.tf`, the served FastAPI title and the MCP server description), as do `Oracle 1.5`/`Oracle 2.0`, which name the product generation rather than the product
- `per_article_timeout_seconds` 25 → 35 (retro#697 follow-up). After the v11 deploy, 5+-article batches lost every article to the per-article budget in 8/15 forecasts in the first 20 min (vs ~6% on 08-27..29); extract p50 ~8s / p90 ~13s on Haiku, and the A/B harness ran 15–27% slower on output-heavy files. Primary + relaxed retry (2 × 35s) still fit inside the 90s `forecast_timeout_seconds`

### Fixed
- Threshold archetype: a cue with a bare number behind it now counts as a magnitude ("more than 33 seats", "at least 8 goals"), and `between` joins the cue set (retro#748). The two gaps hid half the numeric-threshold A/B corpus and four live prod forecasts from `is_threshold_shaped`; the bare integer is bound to the cue rather than recognised anywhere, so prod firing moves 4.4% → 6.6% with no false positive among the five newly-caught claims.
- `docs/ORACLE_VARIABLES.md`: `quantity` had no row after retro#683 shipped it, and `stance_vs_quantity` was still documented as inert with *both* inputs missing when the claim side had landed

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
- Oracul box: Elastic IP (retro#436, #668); unit file synced with the retro#600 mitigations (#636); nginx `client_max_body_size` 16m (#716, #718); Bedrock alarms on a `us-east-1` SNS topic (#674)
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
- feat: Oracul API — FastAPI prediction service skeleton (#25)

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
