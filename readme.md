# Project Overview: Daatan

**Daatan** is a suite of tools designed to bring objective, mathematical measurement to public predictions — turning the unaccountable world of expert opinion into a scored, ranked, and queryable dataset.

---

### The Problem: Israeli Media Has Never Been Held to Account

For decades, Israeli journalists, analysts, and commentators have shaped public understanding of some of the most consequential events in the region — elections, wars, diplomatic breakthroughs, economic shocks. Yet there is no mechanism to track whether they were right.

A columnist who confidently predicted the judicial reform would collapse, a security correspondent who dismissed the Hamas threat as managed, a financial analyst who called the Moody's downgrade "alarmist" — none of them are ever scored. Their reach and reputation are determined by their platform, not their track record.

This is not only a problem of explicit predictions. Much of the signal is hidden: in the framing of a headline, the certainty of a source quoted, the sentiment underneath a neutral-sounding analysis. **Daatan extracts and scores all of it** — not just clear forecasts, but implied direction, vague sentiment, and the story being told between the lines.

---

### The Daatan Suite

Daatan addresses this problem through three interconnected products:

**Daatan News** — A news portal covering Israeli and regional media. Readers don't just consume news: they engage with it by joining forecasts, staking positions, or flagging claims as wrong. This gamification layer passively generates reputation data — every interaction is a signal about source credibility, by topic, by domain, over time.

**Daatan Forecast** — An active prediction market where users, analysts, and bots make explicit forward-looking claims and stake reputation on them. Forecasts are resolved against real-world outcomes, building verifiable track records.

**Retro Analysis (בדיעבד)** — The retroactive audit engine. Rather than waiting for sources to opt in, Retro processes the public trail of what media *already* published — and scores it. It bypasses the cold-start problem entirely: reputation scores are generated from history, not from future participation.

---

### The Engine: The Factum Atlas

Retro Analysis is built around **The Factum Atlas**, a definitive map of historical reality against which all media claims are measured. It cross-references major events against media outlets, journalists, and think tanks.

For every article in scope, a multi-stage LLM pipeline extracts a forensic prediction vector:

- **Stance** — directional outlook on the event (-1.0 bearish → +1.0 bullish)
- **Certainty** — how linguistically confident the author is
- **Sentiment** — the emotional charge beneath the words
- **Specificity** — how falsifiable and concrete the claim is
- **Contrarianism** — how far the prediction deviates from the consensus of peers at the same moment in time
- ...and more proprietary dimensions

Each prediction is matched to a seed event, scored against historical ground truth, and used to update the source's **Brier Score** (calibration accuracy) and **ELO rating** (relative rank).

The resulting dataset — thousands of scored predictions, ranked sources, and verified outcomes — becomes the training corpus for **The TruthMachine**.

### Outcome-Supervised Narrative Calibration (OSNC)

Because we mathematically understand how past language correlates with actual outcomes, The TruthMachine can read a cluster of articles published today and ask: *given who is saying what, and how accurate they have historically been, what is the probability this event occurs?*

This methodology, **Outcome-Supervised Narrative Calibration (OSNC)**, allows the TruthMachine to identify the specific "linguistic signatures" of accuracy.

Output examples:
- `"Chances Netanyahu survives no-confidence vote: 0.83"`
- `"Probability of IDF ground operation in Lebanon within 30 days: 0.61"`
- `"Likelihood of Moody's second downgrade by Q2: 0.44"`

This shifts Daatan from a media-auditing tool into a forward-looking intelligence product. Primary clients:

- **Quantitative funds & traders** — API access to high-signal alternative data on Israeli and regional events
- **Prediction market participants** — data-driven edge for Polymarket and similar platforms
- **Corporate & government risk desks** — calibrated probability estimates for geopolitical and economic planning

The TruthMachine inference layer is exposed as the **Oracle API** (`oracle.daatan.com`) — a FastAPI microservice that accepts a binary question and returns a calibrated probability distribution, with per-source credibility weighting derived from the Factum Atlas. See [`docs/ORACLE_API.md`](docs/ORACLE_API.md) and the [live test console](https://daatan.github.io/retro/oracle-test.html).

---

### The Vision

Daatan's goal is to create a definitive reliability layer for the information ecosystem — starting with Israel. By objectively measuring who was right, who was wrong, and who saw it coming when no one else did, we surface hidden analytical talent, penalize confident noise, and provide the most honest probabilities available for the region's most consequential events.

---

## Appendix A: LLM & NLP Pipeline

**Hybrid architecture:** High-volume filtering uses fast models (AWS Bedrock Nova Micro); nuanced forensic extraction uses mid-tier models (Bedrock Nova Lite). Heavy models are called only when necessary.

**Multilingual extraction:** The pipeline natively processes Hebrew and English, capturing signals in Israeli media before they surface in international coverage.

**Scope:** Every article that passes the prediction filter is processed — not just op-eds with explicit forecasts, but news analysis, commentary, sourced claims, and any text that implies a directional view on a future outcome.

---

## Appendix B: The Mathematics of Truth

**Brier Score** — measures calibration: how close the extracted probability was to the actual binary outcome. Lower is better.

**ELO Rating** — zero-sum relative ranking: sources that correctly predict events their peers got wrong absorb significant rating gains, instantly surfacing hidden analytical alpha regardless of platform size or brand recognition.

---

## Appendix C: Infrastructure

**Compute:** AWS EC2 t4g.small (Graviton, ARM), `eu-central-1` (Frankfurt). Access via AWS SSM only (port 22 blocked).

**Storage (MVP):** Filesystem JSON — one file per event/source pair (~3,200 atlas entries). Migrates to PostgreSQL + pgvector as data volume and API demand grows.

**Orchestration:** `truthmachine.service` (batch pipeline loop) + `oracle-api.service` (FastAPI forecast API at `oracle.daatan.com`).

**LLM:** AWS Bedrock — Nova Micro (gatekeeper) + Nova Lite (extractor, aggregator).

**Live pages:**
- Factum Atlas: https://daatan.github.io/retro/
- Oracle Test Console: https://daatan.github.io/retro/oracle-test.html
- TruthMachine vs Polymarket Duel: https://daatan.github.io/retro/duel.html
