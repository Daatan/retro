# TruthMachine / Factum Atlas — 6-Month MVP Execution Plan

**Goal A:** 100×200 Middle East matrix, 5+ years retro, fully operational.
**Goal B:** Daatan Forecast Android app live, 200k MAU.
**Budget:** ~$262,000 (see INVESTOR_COSTS.md)
**Team:** 2 founders + 1 mathematician/data engineer (to hire)
**Revenue in period:** $0 (MVP phase)

---

## Phase 0 — Pre-Start (Before Month 1)

These must be done before the clock starts on salaries.

### Legal & Corporate
- [ ] Choose company structure (Ltd. in Israel — *Chevra Bet* is standard for tech startups)
- [ ] Engage startup-focused Israeli lawyer (referral from accelerator network or F6S)
- [ ] Incorporate: ~1–2 weeks, ~₪8,000–12,000 all-in
- [ ] Open business bank account (Leumi Tech or Discount startup tracks)
- [ ] Draft founder agreement: equity split, vesting schedule (4 years, 1-year cliff)
- [ ] Assign IP to company (critical before any investor conversation)

### Hiring
- [ ] Write job description for mathematician / data engineer
- [ ] Post on: LinkedIn, Facebook ML Israel group, JobMaster, academic ML dept boards (TAU, Technion, Hebrew U, BGU)
- [ ] Target profile: probability theory background + Python/LightGBM/scikit-learn + interest in forecasting
- [ ] **Months 1–3: paid consultant** (~₪15,000/mo freelance) — feature design, model architecture, TruthMachine spec
- [ ] **Month 4: transitions to full-time employee** at ₪45,000/mo

---

## Month 1 — Pipeline Repair & Foundation

**Theme: Fix the data gap. Nothing else matters until cells are filling.**

### Pipeline (Priority 1)
The current pipeline has an 83% "no predictions" rate (218/264 cells). Root causes to diagnose and fix:

- [ ] Audit `no_predictions` cells: are articles being fetched at all, or fetched but empty?
- [ ] Fix CDX scan: increase `CDX_SCAN_LIMIT` (currently 150) and `CDX_FETCH_LIMIT` (currently 15) for historical queries
- [ ] Haaretz: test Wayback snapshot quality; if consistently paywalled, subscribe to Haaretz digital archive (~$150/mo)
- [ ] Globes / Calcalist: identify if paywall is blocking Wayback snapshots; add direct subscription if needed
- [ ] Reuters: low `done` rate despite being a major source — investigate URL pattern issues in CDX
- [ ] Add per-cell diagnostic logging: distinguish "no articles found" vs "articles found but gatekeeper rejected all"
- [ ] Fix the 5 `failed` cells (JSON parse errors from Gemini) — add retry with stricter prompt

### Matrix Definition
- [ ] Define the canonical list of 200 Middle East events (event taxonomy: war, elections, diplomatic, economic, terror incidents)
- [ ] Define the canonical list of 100 sources (Israeli Hebrew, Israeli English, Arabic, Turkish, international wire)
- [ ] Identify which ~20 sources need paid access — subscribe to those before retro fill begins
- [ ] Confirm date range: 5 years back = 2021–2026; extend to 7 years (2019–2026) if CDX coverage allows at no extra cost

### Translation Pipeline
- [ ] Integrate Google Translate API into `gnews_ingest.py` for Arabic and Turkish sources
- [ ] Add language detection (langdetect library) before translation call to avoid billing for already-English content
- [ ] Test on 3 Arabic sources + 2 Turkish sources

### Infrastructure
- [ ] Set up separate AWS environments: `dev` and `prod`
- [ ] Enable AWS Cost Explorer alerts at $200/mo and $500/mo thresholds
- [ ] Set up pipeline monitoring: failed cell alerts via email/Telegram

### Deliverable: end of Month 1
- Pipeline filling cells at >50% rate (down from 17% currently)
- 200 events + 100 sources defined and locked
- Mathematician/data engineer hired and onboarded

---

## Month 2 — Retro Fill Begins + TruthMachine Design

**Theme: Get data flowing. Mathematician designs the model architecture.**

### Retro Fill
- [ ] Begin systematic retro fill: run pipeline on full 100×200 grid, 2021–2026
- [ ] Process in batches by year: start with 2024–2025 (freshest CDX coverage), then go backward
- [ ] Target: 5,000+ cells with predictions by end of month (25% of matrix)
- [ ] Monitor translation costs weekly — pause and optimize if Google Translate spend exceeds $500 in first run

### TruthMachine — R&D Phase
The mathematician/data engineer leads this track. Existing prototype: `backtest.py` uses `lgb.LGBMClassifier`.

- [ ] Audit existing `backtest.py`: understand current feature set, label definition, train/test split
- [ ] Define the prediction target formally: what exactly is TruthMachine predicting? (probability that a claim resolves true, given source stance scores + metadata)
- [ ] Design feature matrix:
  - Stance scores (mean, variance, skew across sources)
  - Source authority weights
  - Sentiment + certainty + hedge_ratio from extractor
  - Time horizon, prediction type
  - Cross-source agreement/disagreement signals
  - Polymarket baseline (where available)
- [ ] Identify minimum viable training set size: how many labeled (claim → outcome) pairs are needed?
- [ ] Map existing Polymarket data (`polymarket.py`) to resolved events — this is the ground truth source

### Legal
- [ ] App privacy policy + Terms of Service drafted (covers data collection, prediction disclaimers)
- [ ] GDPR-light compliance review (Israeli Privacy Law 5741 + EU users if any)
- [ ] Review top 5 source ToS for scraping/extraction legality — flag any red lines

### Daatan Forecast — Spec
- [ ] Write product spec: what does the app show? (event list, source comparison, forecast probability, historical accuracy)
- [ ] Design Android app wireframes
- [ ] Define data API: what endpoints does the app need from the backend?

### Deliverable: end of Month 2
- 5,000+ cells filled
- TruthMachine feature spec finalized
- App wireframes approved
- Legal foundation complete

---

## Month 3 — Retro Fill 50% + TruthMachine Alpha Training

**Theme: First model. First working app screen.**

### Retro Fill
- [ ] Continue fill: target 10,000+ cells by end of month (50% of matrix)
- [ ] Cover years 2022–2023 in this month
- [ ] QA spot-check: human review of 200 random extracted predictions for accuracy
- [ ] Fix any systematic extraction errors found in QA

### TruthMachine — Alpha
- [ ] Assemble training dataset: join extracted predictions with Polymarket resolved outcomes
- [ ] Minimum target: 500 labeled examples (claim + features → binary outcome)
- [ ] Train first LightGBM model on available data
- [ ] Evaluate: Brier score, calibration curve, feature importance
- [ ] Iterate: add/remove features based on importance; retrain
- [ ] Document model card: what the model predicts, training data, known limitations

### Daatan Forecast — Android Development
- [ ] Set up Android project (Kotlin + Jetpack Compose)
- [ ] Google Play Developer account ($25 one-time)
- [ ] Backend API: FastAPI or Flask serving pre-computed forecast data
- [ ] Screen 1: Event list with forecast probability badges
- [ ] Screen 2: Event detail — source breakdown, TruthMachine score, historical chart
- [ ] Auth: simple email/password or Google Sign-In

### Infrastructure
- [ ] Set up production database (RDS PostgreSQL) for forecast data
- [ ] Set up S3 bucket for pre-computed results
- [ ] Automate weekly pipeline re-run for new events (cron)

### Deliverable: end of Month 3
- 10,000 cells filled (50% retro complete)
- TruthMachine alpha: Brier score baseline established
- Android app: 2 core screens working against real data

---

## Month 4 — Retro Fill 80% + Model Integration + App Beta

**Theme: Connect everything. Internal testing.**

### Retro Fill
- [ ] Push to 16,000+ cells (80%) — cover 2019–2021 (older CDX, expect lower coverage)
- [ ] Identify structural gaps: events with <5 sources covered — flag for manual archive research
- [ ] Handle Arabic/Turkish gaps: if Google Translate quality is poor for certain source styles, add post-translation cleanup

### TruthMachine — Beta Model
- [ ] Retrain on expanded dataset (now ~2,000+ labeled examples as retro fill grows)
- [ ] Add cross-validation: 5-fold CV to check for overfitting
- [ ] Tune hyperparameters: LightGBM num_leaves, learning_rate, min_data_in_leaf
- [ ] Integrate model into backend API: `/api/forecast/{event_id}` returns TruthMachine score
- [ ] Calibration: apply isotonic regression or Platt scaling to output probabilities

### Daatan Forecast — Beta
- [ ] Internal beta: founders + 10–20 trusted testers (journalists, analysts)
- [ ] Collect feedback: UI clarity, trust in forecasts, missing features
- [ ] Performance: ensure app loads in <2 seconds for any event
- [ ] Push notifications: alert users when a forecast changes by >10 points
- [ ] Android widget (optional stretch goal for virality)

### Content Access
- [ ] Review Wayback coverage for 2019–2021 — if gaps are critical, evaluate Factiva or LexisNexis trial access
- [ ] Finalize content budget: if paid archive spend is tracking above $5,000, reprioritize sources

### Deliverable: end of Month 4
- 80% retro matrix complete
- TruthMachine beta: calibrated model in production
- Android app beta: tested by 20+ users
- All legal docs live on app store listing

---

## Month 5 — Full Matrix + Soft Launch

**Theme: Ship it. Controlled growth.**

### Retro Fill — Completion
- [ ] Push to 100% of reachable cells (expect ~18,000–19,000 of 20,000 due to genuine coverage gaps)
- [ ] Document unreachable cells: which events × sources have zero historical coverage and why
- [ ] Generate retro analysis pages (using `render_atlas.py`) for all completed cells
- [ ] QA final pass: 500-sample human review of extraction quality across Arabic, Turkish, Hebrew, English

### TruthMachine — Production
- [ ] Final model trained on full retro dataset
- [ ] Backtest report: accuracy by event category, by source language, by time horizon
- [ ] Model versioning: store model artifacts in S3 with version tags
- [ ] Automated weekly retraining as new resolved events accumulate

### Daatan Forecast — Soft Launch
- [ ] Submit to Google Play (expect 2–3 day review)
- [ ] Soft launch: no press, share only in targeted communities
  - Israeli forecasting / rationalist communities (Facebook, Telegram)
  - Arabic-language political analysis groups
  - Political science Twitter/X accounts covering ME
- [ ] Target: 1,000–5,000 installs in soft launch
- [ ] Monitor: crash rate, retention (Day 1, Day 7), session length
- [ ] Fix any critical bugs from real-user data

### Infrastructure — Scale Prep
- [ ] Load test: simulate 50k concurrent users
- [ ] Set up auto-scaling group on AWS
- [ ] CloudFront CDN for static assets and pre-computed forecast pages
- [ ] Set up error alerting (Sentry or AWS CloudWatch)

### Deliverable: end of Month 5
- Full ME retro matrix live
- App on Google Play
- 1,000–5,000 real installs
- Infrastructure ready for viral load

---

## Month 6 — Growth Push & B2B Groundwork

**Theme: Hit 200k MAU. Open the B2B door.**

### Daatan Forecast — Growth
- [ ] Identify what drove installs in Month 5 — double down on those channels
- [ ] Viral mechanics:
  - Shareable forecast cards (image export of event forecast for social media)
  - "Was I right?" notifications when events resolve
  - Leaderboard: top forecasters (user predictions vs TruthMachine)
- [ ] Localization: Arabic UI (RTL layout) — significant for ME audience
- [ ] Press outreach: 3–5 Israeli tech/media journalists; 2–3 Arabic-language tech media
- [ ] Target: 200,000 MAU by end of Month 6 (**this requires a viral event or significant press coverage — the single biggest execution risk in this plan**)

### B2B Groundwork
- [ ] Identify first 10 B2B targets: think tanks, NGOs, news organizations, financial institutions tracking ME risk
- [ ] Build B2B demo: custom event matrix export, API access, white-label forecast widget
- [ ] Price anchoring: B2B annual license at $10,000–50,000/year depending on data volume
- [ ] Reach out to 10 targets with personalized demos

### TruthMachine — Continuous Improvement
- [ ] Integrate user prediction data (from app leaderboard) as additional signal
- [ ] Explore ensemble: LightGBM + logistic regression as a simple ensemble
- [ ] Publish accuracy report publicly: transparent Brier scores build trust

### Investor Prep
- [ ] Metrics deck: DAU/MAU, retention curves, retro matrix coverage stats, TruthMachine Brier score
- [ ] Pipeline due diligence package: architecture diagram, data sourcing methodology, legal status of content access
- [ ] Series A / seed pitch: position as "Bloomberg Terminal for geopolitical forecasting"

### Deliverable: end of Month 6
- 200k MAU (or documented path to it with current growth rate)
- Full ME retro public
- B2B pipeline: 10 prospects contacted, 2–3 demos delivered
- Investor pitch deck ready

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| 200k MAU target not hit organically | High | High | Lower target to 50k MAU; add budget for targeted ads ($5–15k) |
| Retro fill coverage below 80% due to CDX gaps | Medium | Medium | Budget $3–5k additional for paid archive access |
| Mathematician/data engineer hire takes >6 weeks | Medium | High | Start recruiting in Phase 0, before salaries begin |
| TruthMachine Brier score not competitive vs naive baseline | Medium | High | Pivot to displaying raw source disagreement (no ML needed for v1 value) |
| Arabic/Turkish extraction quality poor post-translation | Medium | Medium | Reduce scope to Hebrew/English only for retro v1 if quality is unacceptable |
| Legal challenge from a paywalled Israeli publisher | Low | High | Proactive ToS review in Month 2; content access via subscriptions not scraping |
| AWS costs exceed budget at 200k MAU scale | Low | Medium | Cost alerts + auto-scaling caps; pre-computed data model limits per-request cost |

---

## Monthly Budget Burn

| Month | Personnel | Infra | Other | Total |
|---|---|---|---|---|
| 1 | $31,100 (2 founders) + $3,300 (consultant) | $300 | $6,500 (incorporation + 1-mo subscriptions) | ~$41,200 |
| 2 | $31,100 + $3,300 (consultant) | $400 | $3,000 (translation retro run 1) | ~$37,800 |
| 3 | $31,100 + $3,300 (consultant) | $500 | $2,000 (translation + QA) | ~$36,900 |
| 4 | $46,700 (+ full hire) | $700 | $1,500 | ~$48,900 |
| 5 | $46,700 | $900 | $1,000 (translation top-up) | ~$48,600 |
| 6 | $46,700 | $1,200 | $500 | ~$48,400 |
| **Total** | **$245,700** | **$4,000** | **$14,500** | **~$264,200** |

> LLM API costs (~$700 total) absorbed into "Other" above. Translation front-loaded in months 2–3 during retro fill.

---

*This plan is a living document. Update after each monthly review.*

---

---

# Phase 2 — Worldwide Expansion ($1,000,000 budget, Months 7–18)

**Trigger:** Phase 1 complete — ME retro live, Daatan Forecast on Android, traction proven.
**Goal:** Global coverage (5 regions), 10M MAU, B2B revenue stream open, Series A ready.
**Budget:** $1,000,000
**Team:** Grow from 3 to ~10 people.

---

## Budget Allocation

| Category | Amount | % |
|---|---|---|
| Personnel (scale team to ~10) | $580,000 | 58% |
| Global content acquisition | $80,000 | 8% |
| Campaigns & growth marketing | $150,000 | 15% |
| Cloud infrastructure (global scale) | $60,000 | 6% |
| Translation infrastructure | $40,000 | 4% |
| Legal / IP (international) | $40,000 | 4% |
| B2B product development | $30,000 | 3% |
| Data feeds (global markets) | $20,000 | 2% |
| **Total** | **$1,000,000** | **100%** |

---

## Team Expansion (Months 7–18)

| Role | Start | Salary (NIS/mo) | Rationale |
|---|---|---|---|
| Mathematician / data engineer (already hired) | Month 4 | 20,000 | TruthMachine |
| Backend engineer #2 | Month 7 | 30,000 | API scale + B2B integrations |
| Mobile developer (Android + iOS) | Month 8 | 30,000 | iOS launch, global app stores |
| Growth / community manager | Month 8 | 20,000 | Campaign execution |
| Data researcher (global events) | Month 9 | 22,000 | Expand event matrix beyond ME |
| Arabic / Turkish content specialist | Month 9 | 20,000 | Translation QA + source expansion |
| B2B sales / partnerships | Month 10 | 25,000 + commission | Enterprise deals |
| DevOps / infrastructure | Month 10 | 30,000 | Multi-region AWS |
| **Total monthly at full team (Mo 12+)** | | **~₪197,000 (~$53,000/mo)** | |

---

## Month 7–8 — iOS + Europe Launch

**Theme: Double the platform. Enter a second geography.**

- [ ] Launch iOS app (App Store) — same feature set as Android
- [ ] Choose second geography: **Europe** (Ukraine/Russia conflict coverage) or **Latin America** (Venezuela, Brazil elections)
- [ ] Begin Europe/LATAM event matrix: 50 events × 50 sources as pilot
- [ ] Hire backend engineer #2 and mobile developer
- [ ] Growth budget: $20,000 — targeted social ads (Meta, X) in target geographies
- [ ] B2B: first paid pilot agreement with a think tank or news organization (~$10k deal)
- [ ] TruthMachine v2: retrain on global data, add region as a feature

**Deliverable:** iOS live · Second region pipeline running · First B2B revenue

---

## Month 9–10 — Three Regions + B2B Product

**Theme: Prove the model scales beyond the Middle East.**

- [ ] Add third region: **East Asia** (Taiwan, Korea, China — high Polymarket signal density)
- [ ] 100×100 matrix per region — standardize event taxonomy globally
- [ ] Launch B2B product: white-label forecast widget, API access, custom matrix exports
- [ ] B2B pricing: $15,000–50,000/year depending on region and data volume
- [ ] Target 5 B2B clients signed by Month 10
- [ ] Arabic RTL + Turkish localization fully polished
- [ ] Campaign: $30,000 — influencer partnerships in ME + Europe political analysis communities
- [ ] Milestone: **1,000,000 MAU**

**Deliverable:** 3 regions live · B2B product launched · 5 clients · 1M MAU

---

## Month 11–12 — Global Matrix + Series A Prep

**Theme: Establish global credibility. Raise the next round.**

- [ ] Add regions 4 and 5: **Sub-Saharan Africa** + **South/Southeast Asia**
- [ ] Full global matrix: ~500 events × ~150 sources across 5 regions
- [ ] TruthMachine v3: cross-region model — does consensus across geographies improve accuracy?
- [ ] Publish public accuracy report: Brier scores by region, event type, time horizon — this is the investor and press hook
- [ ] Campaign: $50,000 — PR push targeting mainstream financial and political media (Bloomberg, FT, Politico)
- [ ] B2B: 15+ clients, $200k+ ARR
- [ ] Milestone: **5,000,000 MAU**
- [ ] Series A deck: $5–10M raise, positioned as infrastructure layer for geopolitical intelligence

**Deliverable:** 5 regions · 5M MAU · $200k ARR · Series A process started

---

## Month 13–18 — Scale to 10M MAU + Series A Close

**Theme: Grow into the raise. Become the default geopolitical forecast layer.**

- [ ] Complete global coverage: add remaining high-signal regions (Central Asia, Balkans, West Africa)
- [ ] Premium subscription tiers: individual ($9/mo), professional ($49/mo), institutional ($199/mo)
- [ ] B2B enterprise tier: custom data pipelines, dedicated analyst support
- [ ] iOS + Android: widgets, Apple Watch / WearOS, news app integrations
- [ ] Prediction league: global leaderboard, public forecaster reputation scores
- [ ] Partner integrations: embed Daatan forecasts in news publishers, research platforms
- [ ] Campaign: $100,000 total across months 13–18 — performance marketing + conference presence (Davos, AIPAC, SXSW)
- [ ] Milestone: **10,000,000 MAU** by Month 18
- [ ] Series A close: $5–10M to fund global team, enterprise sales, and model R&D

**Deliverable:** 10M MAU · Series A closed · $1M+ ARR · Global brand established

---

## Monthly Burn (Phase 2)

| Period | Personnel | Infra | Marketing | Other | Monthly Total |
|---|---|---|---|---|---|
| Mo 7–8 | $25,000 | $3,000 | $10,000 | $5,000 | ~$43,000 |
| Mo 9–10 | $38,000 | $5,000 | $15,000 | $6,000 | ~$64,000 |
| Mo 11–12 | $48,000 | $7,000 | $25,000 | $7,000 | ~$87,000 |
| Mo 13–18 | $53,000 | $8,000 | $17,000 | $5,000 | ~$83,000 |
| **Total Phase 2** | **~$580,000** | **~$60,000** | **~$150,000** | **~$80,000** | **~$870,000** |

> Remaining ~$130,000 held as reserve for unexpected content licensing, legal, or hiring costs.

---

## Risk Register (Phase 2)

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| 10M MAU not reached without paid acquisition | High | High | Series A funds performance marketing; viral loop must be proven before Mo 12 |
| B2B sales cycle longer than expected | Medium | High | Start outreach Month 7; close first paid pilot before building full B2B product |
| TruthMachine accuracy degrades on non-ME regions | Medium | Medium | Train region-specific models; don't force one global model prematurely |
| Content licensing becomes a legal bottleneck at scale | Medium | High | Engage IP counsel Month 7 for international framework |
| Key hire (B2B sales) fails to close deals | Medium | High | Founders lead first 3 B2B deals before delegating |
| Competitor (Metaculus, Manifold, PredictIt) copies ME matrix format | Low | Medium | Speed + language coverage + retro depth are the moat — publish accuracy reports early |

---

## Combined Budget Summary

| Phase | Period | Budget | Primary goal |
|---|---|---|---|
| Phase 1 — MVP | Months 1–6 | ~$264,000 (pre-seed / YC batch) | ME retro + 200k MAU |
| Phase 2 — Worldwide | Months 7–18 | ~$1,000,000 (Series A) | 5 regions + 10M MAU + B2B |
| **Total** | **18 months** | **~$1,264,000** | **Global geopolitical forecast platform** |

*This plan is a living document. Update after each monthly review.*
