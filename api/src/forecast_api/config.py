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
    # Credibility feedback loop, step 1 (docs/ORACLE_VARIABLES.md §9 —
    # resolution-outcome feedback loop). Append-only JSONL of resolved
    # forecasts' per-source stances, pushed by daatan via
    # POST /leaderboard/ingest.
    resolution_feedback_path: Path = Path("")  # empty = data_dir/resolution_feedback.jsonl
    # Step 3: shadow-scored output of replaying resolution_feedback_path
    # through OpenSkill on every ingest (resolution_scorer.py). Separate from
    # leaderboard_path — get_credibility_weight() never reads this file.
    resolution_leaderboard_path: Path = Path("")  # empty = data_dir/resolution_leaderboard.json

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
    # When a caller supplies a gatekeeper verdict on an ArticleInput (relevance +
    # is_prediction — news-indexer's POST /relevance result, threaded through daatan),
    # reuse it instead of re-running check_is_prediction. The SAME claim-aware judge already
    # ran once upstream before the article was pushed; re-judging is a duplicate call whose
    # only effect is to let the push decision and the aggregation weight disagree by
    # nondeterministic noise. On by default; the flag remains as a kill switch —
    # setting REUSE_SUPPLIED_RELEVANCE=false restores re-judging without a code change.
    # Oracle-discovered (SERP/GDELT) articles carry no verdict and are always judged.
    # Design: news-indexer docs/MATCHING_ARCHITECTURE.md §3.
    reuse_supplied_relevance: bool = True
    # ── Evidence-class weighting (S2 cutover, retro docs/ORACLE_VARIABLES.md §5) ─
    # Per-claim weight component for the cross-article `weight` term, keyed by
    # PredictionExtraction.evidence_class. Replaces the old certainty-as-weight
    # term, the 0.9 certainty floor resolve_stance_certainty applied on a cited
    # quantitative_estimate, and the standalone ×4 quantitative_anchor_multiplier
    # with one lookup table (see evidence_class_weight() in aggregation.py).
    # cited_probability keeps the old ×4 premium verbatim — it protects the same
    # France World Cup regression (a 75% pooled estimate against a cited Opta
    # baseline of 18.83%; see TestFranceWorldCupRegression). The other four are
    # new calibration: reported_fact and cited_share sit below the anchor premium
    # since neither is a stated probability for the event itself; reporting and
    # opinion are down-weighted relative to the old ~0.5-0.7 typical certainty
    # range to make room for that separation. Expect retuning once more
    # real-traffic evidence_class_weighted log volume accumulates.
    evidence_class_weight: dict[str, float] = {
        "cited_probability": 4.0,
        "reported_fact": 1.0,
        "cited_share": 1.5,
        "reporting": 0.6,
        "opinion": 0.25,
    }
    # Fallback for an evidence_class string absent from the map above (defensive
    # only — PredictionExtraction.evidence_class is a Literal of the five keys
    # above, so pydantic already rejects anything else at extraction time).
    # Unclassified evidence (evidence_class is None — the extractor omitted it)
    # does NOT use this: it falls back to the claim's own certainty instead, so
    # partial classification coverage doesn't regress weighting quality for
    # claims the classifier skipped. See evidence_class_weight() in
    # aggregation.py.
    evidence_class_weight_default: float = 0.6
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

    # ── MCP server / OAuth 2.1 (Cognito) ───────────────────────────────────
    # The /mcp endpoint (docs/ORACLE_MCP.md) is a Model Context Protocol server
    # exposing the Oracle's tools to AI agents. Auth is MCP-native OAuth 2.1:
    # the Oracle is a Resource Server that verifies Cognito-issued JWT access
    # tokens. The whole mount is CONDITIONAL on cognito_user_pool_id being set —
    # a deploy without these vars simply omits /mcp rather than failing startup,
    # so the REST API is never held hostage to Cognito config. (Unset here rather
    # than required precisely so the two-PR rollout — Cognito infra, then this
    # code — can't crash the box if the env lags the merge.)
    cognito_user_pool_id: Optional[str] = None  # e.g. eu-central-1_ABC123 — presence enables /mcp
    cognito_region: Optional[str] = None  # falls back to the region prefix of the pool id
    # Comma-separated Cognito app-client IDs allowed to call /mcp (the Claude
    # public client + daatan's M2M client). Empty = reject all (fail closed).
    cognito_allowed_client_ids: str = ""
    # Public URL of this Resource Server — the OAuth resource indicator and the
    # `resource` value in the protected-resource metadata. Must match what
    # clients request tokens for.
    mcp_resource_url: str = "https://oracle.daatan.com/mcp"

    @property
    def cognito_region_resolved(self) -> Optional[str]:
        """Region for the Cognito issuer/JWKS. Explicit override, else the
        prefix of the pool id (Cognito pool ids are '<region>_<hash>')."""
        if self.cognito_region:
            return self.cognito_region
        if self.cognito_user_pool_id and "_" in self.cognito_user_pool_id:
            return self.cognito_user_pool_id.split("_", 1)[0]
        return None

    @property
    def cognito_issuer(self) -> Optional[str]:
        region = self.cognito_region_resolved
        if not (self.cognito_user_pool_id and region):
            return None
        return f"https://cognito-idp.{region}.amazonaws.com/{self.cognito_user_pool_id}"

    @property
    def cognito_jwks_url(self) -> Optional[str]:
        iss = self.cognito_issuer
        return f"{iss}/.well-known/jwks.json" if iss else None

    @property
    def cognito_allowed_client_id_set(self) -> set[str]:
        return {c.strip() for c in self.cognito_allowed_client_ids.split(",") if c.strip()}

    @property
    def mcp_enabled(self) -> bool:
        """True when enough Cognito config is present to mount /mcp. When False,
        main.py skips the mount entirely (REST API unaffected)."""
        return bool(self.cognito_issuer)

    @property
    def resolved_leaderboard_path(self) -> Path:
        if self.leaderboard_path != Path(""):
            return self.leaderboard_path
        return self.data_dir / "leaderboard.json"

    @property
    def resolved_resolution_feedback_path(self) -> Path:
        if self.resolution_feedback_path != Path(""):
            return self.resolution_feedback_path
        return self.data_dir / "resolution_feedback.jsonl"

    @property
    def resolved_resolution_leaderboard_path(self) -> Path:
        if self.resolution_leaderboard_path != Path(""):
            return self.resolution_leaderboard_path
        return self.data_dir / "resolution_leaderboard.json"


settings = ApiSettings()
