# Bediavad — Seed Pitch Deck

> **Format:** 10 slides, narrative-first | **Stage:** Seed | **Last updated:** 2026-03-19

---

## Slide 1: Title

**Company:** Bediavad (בדיעבד)

**Tagline:** *Retroactive Media Analysis*

**One-liner:** *The past is our data. The future is our product.*

> _Visual: clean, dark background. "Bediavad" large, "Retroactive Media Analysis" smaller below, one-liner at bottom. No clutter._

---

## Slide 2: The Problem

**Israeli media shapes critical decisions — but no one measures if it was right.**

Every day, journalists, analysts, and commentators publish predictions about elections, wars, economic shifts, and diplomatic moves. Governments, funds, and enterprises make decisions based on this coverage.

But there is no score. No accountability. No way to know:
- Who called the Oct 7 threat before it happened?
- Which outlet predicted the Moody's downgrade — and which dismissed it as alarmist?
- Who has been systematically wrong about Iranian escalation for five years?
- Which sources predicted the US capture of Maduro — and which called it impossible?

**Credibility is determined by reach, not by record.**

> _Visual: screenshot of the Maduro case study page — teal column (WSJ, Miami Herald, PanamPost called it) vs red column (NYT, Al Jazeera, Reuters got it wrong). Outcome badge: SUCCESS. The product speaks for itself._

---

## Slide 3: Why Now?

**Three forces converging — none of which existed five years ago.**

**1. The post-truth crisis is measurable — and people are angry.**
We live in an era of media manipulation, echo chambers, and outlets that write what users want to hear rather than what is true. But "post-truth" is a myth. Every factual claim, given enough time, is verifiable. The technology to do that verification at scale now exists. The demand for accountability has never been higher.

**2. Institutional demand for accountable information has never been higher.**
Funds, governments, and enterprises are making billion-dollar decisions based on media coverage — and getting burned. After Oct 7, after the Moody's downgrade, after every missed call, the question is the same: *who should we have been listening to?* There is no systematic answer. Until now.

**3. The LLM breakthrough made this possible.**
Retroactive analysis of thousands of articles in Hebrew and English — extracting claims, scoring accuracy, calibrating models — was computationally and economically impossible five years ago. Today, cheap, high-quality LLMs make it an engineering problem. We are the first to apply this capability systematically to media credibility.

> _Visual: three icons — broken megaphone (post-truth) / burning newspaper (institutional missed calls) / neural network (LLMs). Simple, bold._

---

## Slide 4: The Solution & The Moat

**We retroactively audit five years of Israeli and international media — and score every prediction.**

Using an LLM pipeline, we process thousands of articles, extract every forward-looking claim (explicit or implied), and score each one against what actually happened. We call this **Outcome-Supervised Narrative Calibration (OSNC)**.

The result:
- A **competitive ranking** of journalists and outlets by domain — who is actually reliable on security, economics, politics
- A **credibility layer** for the information ecosystem, built on mathematical ground truth
- The **Factum Atlas** — a proprietary labeled dataset of 125 events × 25 sources, scored predictions with verified outcomes, already being built

*We are not a fact-checker. Fact-checkers verify what happened. We measure who predicted it correctly — before it happened.*

**The moat:** Our advantage is not the technology — LLMs are available to everyone. It is the system we built on top: the event taxonomy, the scoring rubric, the source selection, and the Factum Atlas we are already accumulating. A competitor can copy the approach, but they start from zero. We don't.

> _Visual: pipeline diagram (Articles → Extract → Score → Rank → Dataset) + screenshot of actual case study page showing teal/red two-column layout with source quotes._

---

## Slide 5: The Oracle (Product 2)

**Because we know who was right in the past, we can predict the future more accurately than market consensus.**

Once the Factum Atlas is built, we train a gradient-boosted model on the full forensic vector: 11 per-prediction metrics (stance, certainty, hedge ratio, specificity, contrarianism, and more), source track record by domain, time-to-event, and market-implied probability where available. Every forecast is a calibrated probability — explainable by design, showing which sources and signals drove it.

The model learns non-obvious patterns: which sources matter in which domains, how source combinations interact, when contrarian voices outperform consensus. Not a weighted average — a learned function trained on reality.

For example:
- `"Probability Netanyahu survives no-confidence vote: 0.83"`
- `"Likelihood of IDF ground operation in Lebanon within 30 days: 0.61"`
- `"Moody's second downgrade by Q2: 0.44"`

This is **TruthMachine** — a calibrated oracle API for geopolitical and economic events, sold to institutional clients.

> _Visual: large probability gauge at 0.76, surrounded by weighted source inputs._

---

## Slide 6: Traction

**The infrastructure is built. The first results are live.**

- ✅ Pipeline live: gatekeeper → forensic extractor → scoring (Hebrew + English)
- ✅ 125 seed events defined across 9 domains (politics, war, economy, tech, energy)
- ✅ 25 sources selected (15 Israeli, 6 international, Polymarket as auxiliary)
- ✅ **3 fully scored case studies published** — live at [daatan.github.io/retro](https://daatan.github.io/retro):
  - 🇻🇪 **Operation Absolute Resolve** — US capture of Maduro; Bloomberg & Reuters called it, ynet/Israel Hayom missed it
  - 📉 **Trump Trade Wars 2025** — tariff escalation & dollar decline; Haaretz called it, Calcalist dismissed it
  - ⚡ **Energy Volatility 2026** — European gas price spike; TheMarker called it, Globes missed it

**Next milestone:** First 20 events × 25 sources fully scored and published.

> _Visual: actual screenshot of the live case study — teal column (YES/ACTION → SUCCESS) vs red column (NO/DIPLOMACY → FAILED), source name + date + quote cards, "PREDICTED OUTCOME" badge. This is a real page, not a mockup._

---

## Slide 7: Business Model

**Freemium leaderboard → Institutional API**

| Tier | Audience | Product | Price |
|---|---|---|---|
| **Free** | Anyone | Public credibility leaderboard — journalist & outlet scores by domain | $0 |
| **Pro** | Individual traders, analysts | Oracle API — probability estimates for current events; full source credibility data | $299/year |
| **Enterprise** | Quant funds, government risk desks, media orgs | Raw forensic dataset, custom domain scoring, white-label, dedicated support | $2,400/year |

The free leaderboard drives awareness and positions us as the credibility standard.
The API is the revenue engine — individual traders at $299, institutions at $2,400.

> _Visual: three-tier pricing table, clean._

---

## Slide 8: Market

**Who pays for better predictions about Israel and the Middle East?**

- **Quantitative funds & traders** — Israeli equities, FX, energy exposure; need edge over consensus
- **Prediction market participants** — Polymarket traders seeking data-driven advantage
- **Corporate & government risk desks** — geopolitical and supply chain scenario planning
- **Media & PR** — organizations wanting to benchmark their own credibility

**Market Size**

| | |
|---|---|
| **TAM** | ~$500M — geopolitical intelligence + prediction market data globally |
| **SAM** | ~$15M ARR — Middle East focus; 48K prediction market traders + 500 institutional clients |
| **SOM (Year 1)** | ~$62K ARR — 1 enterprise client, 200 Pro subscribers |

The market is forming now: Kalshi grew 8.5× in one year to 5.1M users. A single Iran strike market hit $188M in volume on Polymarket. We are building the intelligence layer this industry does not yet have.

The Israeli market is the proof of concept. The long-term ambition is to expand in every direction simultaneously:

- **More regions** — MENA, Eastern Europe, Gulf states, any high-stakes media ecosystem
- **More languages** — Arabic, Farsi, Turkish, Russian
- **More time** — historical archives going back decades, not just five years
- **More coverage** — eventually, every published article, not just event-centric sampling

With the right infrastructure, **total media coverage is not a moonshot — it is an engineering problem.**

> _Visual: TAM/SAM/SOM table + concentric circles expanding outward — Israel → MENA → Global → All Languages → All Time._

---

## Slide 9: Competition

**No one does both. We do.**

| | Retroactive prediction scoring | Calibrated probability API | Media/journalist tracking |
|---|---|---|---|
| **Bediavad** | ✅ Outcome-based | ✅ Geopolitical oracle | ✅ Israeli + international |
| TipRanks / Seeking Alpha | ✅ Finance only | ❌ | Partial (financial analysts) |
| Good Judgment / Metaculus | Partial | ✅ Crowd-based | ❌ |
| Eurasia Group / Oxford Analytica | ❌ | ❌ | ❌ |
| NewsGuard / Biasly | ❌ Criteria-based | ❌ | ✅ Process-based |

**Our position:** TipRanks for geopolitical media × Good Judgment for the Middle East.

> _Visual: comparison table with checkmarks. Bediavad row highlighted._

---

## Slide 10: Team & The Ask

**Team**

**Andrey** — Media analyst and content strategist; deep knowledge of Israeli press landscape; chess Candidate Master.

**Marik** — 20+ years software engineering and DevOps; built and scaled TableSurfing (global P2P accommodation network); now building the pipeline end-to-end.

We are hiring a third team member: a data analyst with NLP background to run the scoring pipeline.

We are building at the intersection of investigative journalism, quantitative finance, and AI infrastructure.

---

**The Ask: $100,000** *(Stage 1 — validation)*

| Use of funds | % | ~USD |
|---|---|---|
| Team (3 people, 6 months) | 66% | $66,000 |
| Data ingestion (Perigon / Event Registry) | 18% | $18,000 |
| Infrastructure, LLM & services | 7% | $7,000 |
| Legal & compliance | 5% | $5,000 |
| Reserve | 4% | $4,000 |

**Stage 1 — $100K — 6 months:**
Prove the thesis. 5-year Israeli press analysis complete. Scoring system validated. Statistically significant accuracy advantage over Kalshi/Polymarket consensus demonstrated on the same questions.

**Stage 2 — $300K — upon hitting Stage 1 milestones:**
Build the product and expand the dataset. Oracle API live. First paying institutional clients. Public credibility leaderboard launched.

Dataset expansion is a first-class goal — not a side effect:
- **More time** — extend history from 5 to 10+ years
- **More sources** — from 25 to 50+, adding Arabic and Farsi outlets
- **More events** — from 125 to 500+, enabling statistically reliable model training

The larger the dataset, the stronger the alpha. A Polymarket trader in New York cannot read Haaretz, Al-Monitor, and Al Arabiya simultaneously. We can — and we score all of them.

> _Visual: two-phase timeline. Stage 1: validate. Stage 2: build & sell._

---

*"The past is our data. The future is our product."*
