# TruthMachine / Factum Atlas — Investor Budget

**Two sequenced stages over 18 months. Technical and infrastructure costs only — personnel and GTM are scoped separately.** Stage 1 builds the MVP and ships Daatan Forecast; Stage 2 covers the baseline cost of operating it at 100K users with production-grade infrastructure.

| Stage | Period | Headline cost | Primary outcome |
|---|---|---|---|
| 1 — MVP build | Months 1–6 | **~$15,000** ($10.5K–$21.4K range) | 100×200 ME matrix, 5+ years retro fill, Daatan Forecast Android build |
| 2 — Scale baseline | Months 7–18 | **$103,370** ($100K recurring + $3,370 one-time) | Multi-AZ HA infra, ML training capacity, oracle pipeline at scale |
| **Total ask** | **18 months** | **~$124,370** | |

> **Relationship to MVP_PLAN_6M.md.** That document describes an *aggressive* Months 7–18 plan — "Phase 2 — Worldwide Expansion" at ~$1M (Series A) targeting 5 regions and 10M MAU. The Stage 2 figure here is the **conservative operating baseline** for the same window: what it costs to keep the MVP running at 100K users without the worldwide push. They are two different scenarios for the same period, not duplicate accounting.

---

# Stage 1 — MVP Build (Months 1–6, ~$21,000)

**Scope:** 100×200 Middle East matrix, 5+ years retro fill + Daatan Forecast Android app.
**Assumptions:** Zero revenue, AWS infrastructure.

## 1. Paid content access — archives, paywalls, licensing

~20 of 100 sources require paid access (Haaretz archive, Calcalist, Globes, key Arabic sources).
CDX/Wayback covers the remaining ~80% for historical content.

**One-time retro scrape** (subscribe, scrape, cancel):

| Item | Cost |
|---|---|
| Haaretz digital archive | $150–300 |
| Hebrew business press (Globes, Calcalist) | $100–200 |
| Arabic source archives (2–3 key outlets) | $100–300 |
| Contingency (additional sources) | $150–400 |
| **One-time total** | **$500–1,200** |

**Ongoing subscriptions** (certain sources require active credentials for live ingestion; LLM extraction quality degrades significantly on truncated/paywalled content):

| Source | Est. annual cost |
|---|---|
| Haaretz (English + Hebrew editions) | ~$200–300 |
| Jerusalem Post | ~$100–200 |
| Other paywalled sources (contingency) | ~$200–500 |
| **Ongoing total** | **~$500–1,000/yr** |

## 2. Cloud infrastructure (AWS)

Currently <$100/mo. Scales with Daatan Forecast user growth.

| Period | Monthly | Total |
|---|---|---|
| Months 1–3 (build + small user base) | ~$250–400 | ~$900 |
| Months 4–6 (post-launch) | ~$600–1,200 | ~$2,400 |
| **Total 6 months** | | **$3,000–5,500** |

> Includes EC2 (pipeline + API), RDS, S3, CloudFront, load balancer.
> Post-launch, the app is read-heavy with pre-computed data — manageable on 2–3 mid-range instances.

## 3. LLM API cost (retro fill + ongoing)

Pipeline uses Gemini 2.0 Flash Lite via OpenRouter (~$0.075/1M input tokens).

| Item | Estimate |
|---|---|
| Retro fill: 20,000 cells × ~10 articles, gatekeeper + extractor passes | ~$200–400 one-time |
| Ongoing monthly (new events, rescoring) | ~$50–100/mo → ~$300 over 6 months |
| **Total** | **~$500–700** |

> This is the cheapest line item. Gemini Flash Lite makes LLM costs almost negligible at this scale.

## 4. Translation infrastructure (Google Translate API)

Arabic, Turkish, Hebrew — contingency/fallback only. LLMs read these natively; translation reserved for edge cases or source-specific formatting issues.

| Item | Estimate |
|---|---|
| **Total** | **~$500** |

## 5. Legal / IP counsel + incorporation

Company not yet incorporated.

| Item | Cost |
|---|---|
| Israeli company incorporation (lawyer + gov fees) | $1,500–3,000 |
| App ToS, privacy policy, data processing agreements | $1,500–3,500 |
| Light ongoing counsel (scraping ToS review, licensing) | $500–1,000/mo → $2,000–4,000 |
| **Total** | **$5,000–10,500** |

## 6. Prediction market & financial data feeds

Polymarket currently scraped free. Minimal additional feeds needed for ME matrix.

| Item | Cost |
|---|---|
| Licensed / stable data feeds (contingency) | $500–2,000 |
| **Total** | **$500–2,000** |

## Stage 1 — Consolidated Budget

| Category | Low | High |
|---|---|---|
| Paid content access — one-time scrape | $500 | $1,200 |
| Paid content access — ongoing subscriptions | $500 | $1,000 |
| Cloud infrastructure (AWS) | $3,000 | $5,500 |
| Legal / incorporation | $5,000 | $10,500 |
| LLM API (Gemini Flash Lite) | $500 | $700 |
| Translation API (contingency) | $500 | $500 |
| Data feeds | $500 | $2,000 |
| **Total** | **$10,500** | **$21,400** |

**Working figure: ~$15,000 for 6 months.**

## Stage 1 — Key observations

1. **LLM costs are negligible** — Gemini Flash Lite at $0.075/1M tokens makes the pipeline economics extremely favorable.
2. **Translation is not a major cost** — LLMs read Hebrew, Arabic, and Turkish natively; Google Translate is contingency-only.
3. **Content access risk is real** — 83% of current test cells return no predictions, partly due to paywall gaps in Wayback. Ongoing subscriptions for Haaretz and JPost are cheap insurance for coverage quality.

---

# Stage 2 — Scale Baseline (Months 7–18, $103,370)

**Horizon:** 12 months · **Infrastructure ceiling:** $100K · **One-time backfill:** $3,370

> **Current state:** PostgreSQL runs inside Docker on a single t3.small with no dedicated DB server, no Multi-AZ, no load balancer, and no Redis. This is fine for hundreds of users but collapses under 100K.

### 1. Compute — Production (100K users)

Current setup has one public subnet in one AZ — any instance failure takes the site down. At scale:

| Resource | Spec | Annual |
|---|---|---|
| Application Load Balancer | ALB across 2 AZs | ~$600 |
| App servers — auto-scaling group | 4–8× c6g.large (avg 5), 2 AZs | ~$6,200 |
| NAT Gateways | 2× (one per AZ, required for private subnets) | ~$800 |
| Elastic IPs, VPC endpoints | | ~$200 |
| **Subtotal** | | **~$7,800/yr** |

> ⚠️ **Risk — traffic spike:** A viral news event (war, election) can push 10× normal traffic in minutes. Auto-scaling group max capacity needs headroom. Burst cost: up to $2,000 in a single bad week.

### 2. Database — Move from Docker to Managed RDS

PostgreSQL currently runs in Docker on the same t3.small as the app. No connection pooling, no read replicas, no automatic failover. **One corrupted volume = total data loss.**

| Resource | Spec | Annual |
|---|---|---|
| RDS PostgreSQL — primary | db.r6g.large (2 vCPU, 16 GB RAM), Multi-AZ | ~$3,650 |
| RDS — read replica | db.r6g.large (offload analytics + leaderboard queries) | ~$1,825 |
| Extended backup retention | 35 days PITR + cross-region snapshot | ~$400 |
| Performance Insights + Enhanced Monitoring | | ~$200 |
| **Subtotal** | | **~$6,075/yr** |

> ⚠️ **Risk — data growth:** 100K users × years of predictions + RetroAnalysis articles = DB could hit 500 GB. r6g.large has 128 GB max storage before needing a larger instance class (db.r6g.xlarge = double cost ~$7,300/yr Multi-AZ).

> ⚠️ **Risk — connection exhaustion:** Next.js serverless routes open a new DB connection per request. At scale this hits PostgreSQL's connection limit hard — needs PgBouncer or RDS Proxy ($0.015/hr = ~$130/mo = **$1,560/yr** extra).

### 3. Cache — Redis

No Redis today. Every leaderboard render, every credibility score lookup, every session check hits the DB directly.

| Resource | Spec | Annual |
|---|---|---|
| ElastiCache Redis — clustered | cache.r6g.large, 2 nodes (primary + replica) | ~$3,300 |
| **Subtotal** | | **~$3,300/yr** |

> ⚠️ **Risk:** Redis is also the rate-limiter for oracle API. If it goes down, rate limiting fails open → uncontrolled LLM/search spend until restarted.

### 4. ML Training — GPU Compute

Training custom stance/extraction models on domain-specific geopolitical text requires serious GPU:

| Resource | Spec | Use | Annual |
|---|---|---|---|
| EC2 p4d.24xlarge (spot) | 8× A100 80 GB, ~$10/hr spot | Full model training runs, ~300 hrs | ~$3,000 |
| EC2 g5.12xlarge (spot) | 4× A10G 24 GB, ~$1.70/hr spot | Experiments, fine-tuning, eval | ~$2,900 |
| EFS | 1 TB (shared training dataset) | Training data mount | ~$360 |
| S3 | 1 TB (checkpoints + artifacts) | Model versioning | ~$276 |
| **Subtotal** | | | **~$6,536/yr** |

> ⚠️ **Risk — experiment sprawl:** Hyperparameter search, ablation studies, failed runs. A single poorly supervised training campaign can burn 3–5× the planned GPU budget in a week. Actual range: **$6K–$20K/yr** depending on research intensity.

> ⚠️ **Risk — model size creep:** A 13B or 70B fine-tune multiplies training costs 5–10×. p4d.24xlarge spot is ~$10/hr — a 3-day training run = $720.

### 5. RetroAnalysis — 10-Year Israeli Media Archive

**Scope:** ~50 major outlets (Haaretz, Ynet, Times of Israel, Jerusalem Post, Kan, i24, Walla, Maariv, Calcalist…) × ~30 articles/day × 10 years ≈ **6–8 million articles**.

| Item | Resource | Cost |
|---|---|---|
| **One-time ingestion** | BrightData bulk scraping, ~7M pages | ~$2,800 |
| **One-time embeddings** | text-embedding-3-small, 7M articles × 500 tokens | ~$70 |
| **One-time LLM processing** | Gemini Flash, gatekeeper + extractor × 7M | ~$500 |
| **Vector database** | Pinecone Professional, 7M vectors × 1536 dims | ~$3,000/yr |
| **S3 raw archive** | ~300 GB text + metadata | ~$84/yr |
| **Ongoing delta crawl** | Daily ingest of new articles, BrightData | ~$600/yr |
| **Annual re-embedding** | Model upgrades trigger re-index | ~$500 |
| **One-time total** | | **~$3,370** |
| **Annual ongoing** | | **~$4,184/yr** |

> ⚠️ **Risk — scope expansion:** "Israeli media" likely expands to Arabic-language regional media, international coverage of Israel, or a second country entirely. 7M → 50M articles = Pinecone costs scale to $25K+/yr or require self-hosted Weaviate on a dedicated server (~$2,400/yr EC2).

> ⚠️ **Risk — scraping blocks:** Israeli news sites (especially Haaretz, paywalled) actively block scrapers. BrightData may require premium residential proxies ($3–5× more expensive) for paywalled content.

### 6. Search APIs — Scaled for 100K Users

This is the **fastest-growing cost** at scale. The oracle runs searches for every forecast request.

**Volume estimate:**
- 100K users, 7% DAU = 7,000 daily active
- 3 oracle calls/user/day = 21,000 oracle calls/day
- 5 searches/oracle call = **105,000 searches/day = 3.15M/month**

| Provider | Role | Annual |
|---|---|---|
| Serper.dev Enterprise | Primary (~3M queries/mo, negotiated rate) | ~$12,000 |
| BrightData SERP | Fallback (~500K queries/mo, $0.002/query) | ~$12,000 |
| **Subtotal** | | **~$24,000/yr** |

> ⚠️ **Risk — BrightData cost:** BrightData SERP pay-as-you-go hits hard at volume. 500K queries/mo × $0.002 = $1,000/mo. Negotiate a flat monthly contract early.

> ⚠️ **Risk — geopolitical spike:** A major Middle East event (war escalation, election) drives a sudden 5–10× search spike in a single day. At $0.002/query, 1M queries in 24 hours = $2,000 in one day — with no automatic spend cap unless WAF/rate-limiting is in place.

> ⚠️ **Risk — Serper quota exhaustion:** Happened once already (April 2026). At enterprise scale, a single bad day of scraping can exhaust the monthly quota. Need prepaid buffer or auto-downgrade to BrightData.

### 7. LLM APIs — Oracle Pipeline at Scale

At 21,000 oracle calls/day, each running gatekeeper + extractor on 5 articles:
= 210,000 LLM calls/day × 1,000 tokens avg = **210M tokens/day = 6.3B tokens/month**

| Provider | Use | Monthly | Annual |
|---|---|---|---|
| OpenAI GPT-4o-mini | Oracle gatekeeper + extractor | ~$1,100 | ~$13,200 |
| OpenAI GPT-4o | Hard questions (est. 10% of calls) | ~$500 | ~$6,000 |
| Google Gemini Flash | RetroAnalysis bulk + research | ~$100 | ~$1,200 |
| **Subtotal** | | **~$1,700/mo** | **~$20,400/yr** |

> ⚠️ **Risk — model upgrade:** If GPT-4o-mini quality is insufficient for accurate extraction and 50% of calls must use GPT-4o ($2.50/1M input, $10/1M output), the LLM bill doubles to **$40K+/yr**.

> ⚠️ **Risk — context window growth:** Adding more articles per oracle call (from 5 to 10), longer snippets, or chain-of-thought prompting → token count per call doubles, cost doubles.

### 8. Storage, CDN & Egress

Often overlooked — AWS charges for data leaving the network.

| Resource | Spec | Annual |
|---|---|---|
| S3 (all buckets, ~1 TB at scale) | Backups, uploads, archives, tf-state | ~$280 |
| EBS snapshots (database + EC2) | ~200 GB/mo retained | ~$360 |
| CloudFront CDN | ~10 TB/mo served to 100K users | ~$1,020 |
| EC2 → Internet egress | ~50 GB/day not covered by CloudFront | ~$1,620 |
| Cross-AZ traffic | Multi-AZ DB + Redis replication | ~$300 |
| **Subtotal** | | **~$3,580/yr** |

> ⚠️ **Risk — egress surprise:** AWS egress ($0.09/GB) is a silent killer. A single feature that sends large article bodies to the client instead of summaries can multiply egress 10×. CloudFront must be in front of all content delivery.

### 9. Security

| Resource | Annual |
|---|---|
| AWS WAF (web ACL + managed rule groups) | ~$600 |
| AWS Shield Advanced (DDoS — prediction markets are targets) | ~$3,000 |
| Annual penetration test (third party) | ~$4,000 |
| **Subtotal** | **~$7,600/yr** |

> ⚠️ **Risk — prediction market DDoS:** Forecasting platforms are targets for manipulation attacks (flood the oracle with requests to exhaust quotas right before an event). Shield Advanced is not optional at 100K users.

### 10. Observability & On-call

| Tool | Plan | Annual |
|---|---|---|
| Grafana Cloud Pro | 10 users, 30-day retention, Loki + Prometheus | ~$350 |
| AWS CloudWatch enhanced + log retention | App logs, RDS metrics, Lambda logs | ~$600 |
| AWS X-Ray | Distributed tracing across oracle + daatan | ~$300 |
| PagerDuty | 2 on-call users, incident routing | ~$576 |
| **Subtotal** | | **~$1,826/yr** |

### 11. AWS Support

At ~$5K/mo AWS bill, Business Support is necessary (access to Cloud Support engineers within 1 hour for production incidents):

| | Annual |
|---|---|
| AWS Business Support (~10% of monthly bill) | ~$5,000 |
| **Subtotal** | **~$5,000/yr** |

### 12. Existing Services (Terraform — both repos)

Already provisioned, cost continues unchanged:

| Service | Annual |
|---|---|
| EC2 t4g.medium (retro pipeline, us-east-1) | ~$480 |
| EC2 2× t3.small (daatan staging, eu-central-1) | ~$480 |
| SES + Lambda mail forwarder | ~$24 |
| Route53 (3 hosted zones) | ~$36 |
| ECR (container registry) | ~$120 |
| Secrets Manager (~30 secrets) | ~$150 |
| Lambda (cron functions) | ~$60 |
| SNS | ~$60 |
| GitHub Team + Actions minutes | ~$480 |
| Claude Code subscription | ~$240 |
| Domain names | ~$120 |
| **Subtotal** | **~$2,250/yr** |

## Stage 2 — Infrastructure Budget Summary

| Category | Annual | One-time |
|---|---|---|
| Compute — production (HA, multi-AZ) | $7,800 | — |
| Database — RDS Multi-AZ + read replica | $6,075 | — |
| Cache — Redis clustered | $3,300 | — |
| ML Training — GPU | $6,536 | — |
| RetroAnalysis pipeline | $4,184 | $3,370 |
| Search APIs (3M queries/mo) | $24,000 | — |
| LLM APIs (oracle at scale) | $20,400 | — |
| Storage, CDN, egress | $3,580 | — |
| Security (WAF + Shield + pentest) | $7,600 | — |
| Observability + on-call | $1,826 | — |
| AWS Support (Business) | $5,000 | — |
| Existing services & tooling | $2,250 | — |
| **Infrastructure recurring** | **$92,551** | **$3,370** |
| **Contingency (8%)** | **$7,449** | — |
| **Infrastructure total** | **$100,000** | **$3,370** |

## Stage 2 — Cost Overrun Scenarios

| Scenario | Probability | Extra cost |
|---|---|---|
| Search spike during geopolitical event (1M queries in 48hrs) | High | +$2,000 one-time |
| LLM upgrade from GPT-4o-mini → GPT-4o for quality | Medium | +$20,000/yr |
| DB scale-up needed (r6g.large → r6g.2xlarge) | Medium | +$3,600/yr |
| RetroAnalysis expands to 2nd country (50M articles) | Medium | +$18,000/yr vector DB |
| ML training experiment sprawl (3× planned GPU hrs) | High | +$13,000/yr |
| BrightData no enterprise deal (pay-as-you-go) | Low | +$10,000/yr |
| DDoS attack without Shield Advanced | Low | +$15,000 incident |
| RDS Proxy needed for connection pooling | Medium | +$1,560/yr |

## Stage 2 — Grand Total

| Category | 12-month |
|---|---|
| Infrastructure (recurring) | $100,000 |
| RetroAnalysis backfill (one-time) | $3,370 |
| **Total** | **$103,370** |

---

# Combined 18-Month Ask

| Stage | Period | Total |
|---|---|---|
| Stage 1 — MVP build | Months 1–6 | ~$15,000 |
| Stage 2 — Scale baseline | Months 7–18 | $103,370 |
| **Combined ask** | **18 months** | **~$118,370** |
