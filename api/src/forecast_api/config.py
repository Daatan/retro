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

    max_articles: int = 5
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
    # Decisiveness floor: minimum total certainty-weighted evidence mass
    # (Σ credibility·certainty·recency·relevance² over surviving articles) required
    # to emit a forecast. Below this the pool is too thin/low-certainty to mean
    # anything, so we return insufficient_data (reason="no_decisive_signal") and
    # let the caller keep its base-rate estimate instead of overwriting it with a
    # ~50% coin-flip. 0.5 ≈ one solid, on-topic, confident article's worth of
    # evidence. Deferring is a safe degradation (caller falls back to its LLM base
    # rate); a genuinely balanced ~50% backed by strong coverage clears this easily.
    decisiveness_floor: float = 0.5
    # Per-source certainty floor: an article whose certainty-weighted claims average
    # below this is dropped before aggregation entirely (not just down-weighted).
    # certainty ∈ [0,1] is the extractor's linguistic confidence — 0.1–0.2 is hedged
    # speculation ("could", "implies", "potentially"), the kind of tangential claim a
    # search match on a common word produces. Dropping these (rather than letting
    # their small weight still tug the pool and pad the evidence mass) makes a pool of
    # only-speculative sources collapse to insufficient_data via the floors above,
    # instead of emitting a confident-looking estimate from claims that barely bear on
    # the question. 0.0 disables the gate. Conservative default — tune up using
    # daatan's logged per-source certainty distribution.
    certainty_floor: float = 0.2

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
