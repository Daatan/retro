from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class ApiSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    oracle_api_key: str  # required — startup fails with clear error if missing
    openrouter_api_key: Optional[str] = None  # DEAD: /llm now routes to Bedrock via tm.llm. Kept to avoid breaking env wiring; safe to remove later.

    data_dir: Path = Path("/home/ubuntu/truthmachine/data")
    leaderboard_path: Path = Path("")  # empty = data_dir/leaderboard.json
    leaderboard_refresh_seconds: int = 86400  # pipeline writes this at most once/day

    max_articles: int = 10
    host: str = "127.0.0.1"
    port: int = 8001

    # Hard ceiling on a single /forecast pipeline run. The pipeline has no
    # natural deadline — slow article fetches plus Bedrock throttling/retries
    # can stack past the client's own timeout (oracle-test.html aborts at 120s),
    # leaving the caller hanging. When this fires we cancel the run and return a
    # placeholder so the caller gets a fast, clean answer. Set below the client
    # abort to guarantee we win the race.
    forecast_timeout_seconds: int = 90

    # Per-article wall-clock ceiling for the gatekeeper+extractor LLM work.
    # Articles are processed in parallel (asyncio.gather), so a single slow LLM
    # call (gatekeeper stragglers of 30s+ observed; article-phase p99 ~226s)
    # stalls the whole batch and blows past the caller's timeout. Bounding each
    # article means one straggler is dropped instead of holding up the rest.
    # Set comfortably above normal article latency (~3-4s) but well under
    # forecast_timeout_seconds.
    per_article_timeout_seconds: int = 25

    # Cap article body fed to LLMs (both the gatekeeper relevance screen and the
    # extractor). The thesis is usually in the lead, but the relevance signal can
    # sit deeper, so we keep a generous window before latency/$$ diminish returns.
    max_article_chars: int = 4000

    # ── Aggregation (logit pooling + recency) ──────────────────────────────
    # Sources are pooled in log-odds space, weighted by credibility × certainty
    # × recency. Recency uses exponential decay with this half-life (days): an
    # article this many days old counts half as much as one published today.
    # Aggressive (7d) by design — the latest reporting should dominate as an
    # event resolves, so stale pre-resolution coverage stops diluting a decided
    # outcome.
    recency_half_life_days: float = 7.0
    # Floor for the recency weight so very old articles still count a little.
    recency_floor: float = 0.02
    # Probability clamp before taking log-odds (keeps logits finite). Also caps
    # how extreme a single pooled estimate can get: [clamp, 1-clamp].
    logit_clamp: float = 0.01
    # Each source's aggregation weight is multiplied by relevance_score² (the
    # gatekeeper's graded topic relevance). If the summed relevance mass
    # Σ relevance² across surviving articles is below this floor, the whole set is
    # treated as off-topic and the forecast returns insufficient_data
    # (reason="all_articles_off_topic") rather than pooling junk. Conservative:
    # 0.05 ≈ one article at relevance ~0.22. Tune down using daatan's logged
    # relevance_score / all_articles_off_topic data.
    relevance_weight_floor: float = 0.05
    # Weight premium for a source whose extractor output carries an explicit
    # quantitative_estimate (a named model/poll/market probability cited for the
    # event itself, e.g. "Opta gives France 18.83%"). Without this, such a source
    # is just one more equal-weight vote and gets outvoted by volume — several
    # qualitative "favorite" match reports pooled a France World Cup forecast to
    # 75% against a cited Opta baseline of 18.83%. 1.0 disables the premium
    # (identical to pre-fix behaviour). See quantitative_anchor_multiplier().
    quantitative_anchor_weight: float = 4.0
    # Syndication dedupe: two search results are treated as the same (re-hosted)
    # story when their title-token Jaccard is >= this. Kept high so genuinely
    # different stories sharing a topic word are not merged; only true re-prints
    # collapse to one source. 0.0 disables title clustering (URL-dedupe still runs).
    syndication_title_similarity: float = 0.8
    # Decisiveness floor: the certainty-weighted evidence mass
    # (Σ credibility·certainty·recency·relevance² over surviving articles) at or
    # above which the pooled CI is trusted as-is. BELOW it the pool is thin/hedged,
    # so rather than abstain (which surfaced as "no AI estimate" even for on-topic
    # coverage) we *widen the CI* toward maximal uncertainty in proportion to the
    # shortfall — a thin on-topic pool then self-reports as a low-confidence estimate
    # with a wide band instead of a deceptively tight one. 0.5 ≈ one solid, on-topic,
    # confident article's worth of evidence. Abstention is now reserved for the
    # relevance floor (genuinely off-topic) and a truly empty pool. See
    # thin_evidence_ci_inflation and defer_on_thin_evidence.
    decisiveness_floor: float = 0.5
    # Maximum half-width (in probability space, [0,1]) added to the pooled 95% CI
    # when evidence_mass → 0, scaled linearly by the decisiveness shortfall
    # (deficit = (floor − mass)/floor). pool_sources' interval reflects only how much
    # the sources *disagree*, not how *much* evidence there is, so a thin pool that
    # happens to agree gets a deceptively tight CI; this term restores the missing
    # uncertainty. 0.45 ≈ a near-zero-mass on-topic pool spans almost the full [0,1].
    # 0.0 disables the widening (CI reflects dispersion only).
    thin_evidence_ci_inflation: float = 0.45
    # Escape hatch: when True, restore the old behaviour — a pool below
    # decisiveness_floor returns insufficient_data (reason="no_decisive_signal")
    # instead of emitting a wide-CI estimate. Default False (widen, don't defer).
    defer_on_thin_evidence: bool = False
    # ── Settlement override ─────────────────────────────────────────────────
    # When at least this many independent sources carry a settlement claim (the
    # extractor's `settled` flag: the outcome is reported as an accomplished
    # fact, not a prediction) agreeing in direction, the pooled estimate is
    # pinned to ±settlement_stance and the response carries settled=true.
    # Pooling can never exceed its most confident member, so a decided event
    # otherwise tops out wherever the extractor's stances land (the Knicks
    # "82% the day after the title" case). 2 = one wire story can't settle a
    # forecast alone; 0 disables the override entirely.
    settlement_min_sources: int = 2
    # Stance the pinned estimate takes on settlement (0.94 stance = 0.97
    # probability). Deliberately short of 1.0 — sources can be wrong together.
    settlement_stance: float = 0.94
    # A settlement claim counts toward the pin only when the extractor followed
    # its own accomplished-fact rules: near-boundary stance and high certainty
    # (the prompt mandates ±1.0 / ≥0.9). Enforced in code because the 2026-07-08
    # F-35 false pin was driven by "settled" claims at stance −0.8 / certainty
    # 0.52 — hedged half-settlements the prompt should never produce. Claims
    # failing the gates are demoted to ordinary (non-settled) evidence.
    settlement_min_claim_stance: float = 0.9
    settlement_min_claim_certainty: float = 0.9

    # Forecast-response cache keyed by sha256(question, max_articles).
    # cache_ttl_seconds=0 disables caching entirely.
    cache_ttl_seconds: int = 3600
    cache_max_entries: int = 512

    # Search-result cache keyed by sha256(question, limit). Longer TTL than
    # forecast cache — article lists for a given query are stable for hours.
    # search_cache_ttl_seconds=0 disables search caching.
    search_cache_ttl_seconds: int = 14400  # 4 hours
    search_cache_max_entries: int = 256

    @property
    def resolved_leaderboard_path(self) -> Path:
        if self.leaderboard_path != Path(""):
            return self.leaderboard_path
        return self.data_dir / "leaderboard.json"


settings = ApiSettings()
