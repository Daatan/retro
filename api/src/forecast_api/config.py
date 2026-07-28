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
    # Author-scoring lane (author-scoring redesign, Phase 1 step 3): shadow
    # per-(author, outlet) board replayed from the same feedback file's
    # author_signals (resolution_scorer.rescore_authors_from_disk).
    resolution_author_leaderboard_path: Path = Path("")  # empty = data_dir/resolution_author_leaderboard.json
    # Step 4 — the cutover (docs/ORACLE_VARIABLES.md §9). When True,
    # get_credibility_weight() sources credibility from the resolution-informed
    # shadow board above instead of the vault-curated leaderboard.json. Full
    # REPLACEMENT, not a blend: under the flag the vault is never consulted,
    # and a source without enough resolution history falls back to neutral 1.0
    # rather than to a 2022 backtest of 5 outlets that has been frozen since
    # 2026-03-28 (nothing in prod regenerates it — see the §8 "credibility
    # still ≈1.0" note; the vault is legacy, kept only as this flag's OFF path).
    #
    # Default False — new behaviour, unlike settlement_revalidate's default-on
    # bug fix. Flipping it is a manual step after enough real resolutions
    # accumulate AND a human reviews the backtest
    # (pipeline/scripts/backtest_shadow_credibility.py): set
    # RESOLUTION_SHADOW_CREDIBILITY_ENABLED=true in the env and restart
    # oracle-api.service. Revert is the same in reverse — no deploy either way,
    # same revert story as SETTLEMENT_REVALIDATE below.
    resolution_shadow_credibility_enabled: bool = False
    # Global gate: scoreable resolutions (resolution_scorer.count_resolutions)
    # required before ANY source's shadow score is trusted; below it every
    # source gets neutral 1.0. 50 is where simulated correlation between a
    # source's true accuracy and its resulting weight reaches ~0.97 and
    # stabilises — a starting floor, not a proven optimum.
    resolution_shadow_min_global_predictions: int = 50
    # Credibility is derived from the source's Brier score, NOT from its
    # OpenSkill skill_conservative: sigma barely moves in these large
    # multi-team matches, so mu-3*sigma stays pinned near 0 and the vault's
    # 1.0 + conservative/25 transform maps every source to ~1.0 (measured
    # spread 1.03x on the real board, 1.09x simulated at 100 resolutions —
    # i.e. the cutover would have been a no-op). Brier separates properly
    # (spread ~2.5x at 100 resolutions) and needs no ranking counterparty.
    # The OpenSkill fields stay on the board for display/ranking only.
    #
    # Shrinkage toward the uninformed prior (Brier 0.25) instead of a minimum
    # per-source count: it degrades smoothly rather than at a cliff, so a
    # lucky new source with 2 near-perfect calls lands at ~1.08, not ~1.48.
    # Higher = more conservative. Expressed in pseudo-resolutions.
    resolution_shadow_brier_prior_n: float = 10.0
    # Slope mapping Brier distance-from-0.25 to weight. Brier 0.25 (uninformed
    # or consistently hedged) maps to exactly 1.0, so an unknown source is
    # neutral by construction; a perfect source approaches the upper clamp.
    resolution_shadow_brier_slope: float = 2.0
    # Hard bounds on the resulting weight, so no single source can dominate or
    # be zeroed out of a pool on scoring alone.
    resolution_shadow_weight_min: float = 0.25
    resolution_shadow_weight_max: float = 2.0

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
    # Re-validate every settlement vote at aggregation time against its stored
    # anchor date and the claim's window (settlement_vote_validity,
    # aggregation.py), instead of trusting the caller's settled bits — a stale
    # or poisoned pool row otherwise re-pins the estimate on every recompute
    # forever (the 2026-07-16 false-pin audit: 11 of 19 pins wrong). Also
    # replaces the majority vote with unanimity: valid settled votes in BOTH
    # directions suppress the pin (settlement_conflict) rather than letting
    # the larger side win. Kill switch: SETTLEMENT_REVALIDATE=false in the env
    # restores the legacy trust-the-flags behavior without a deploy (restart
    # oracle-api.service after changing it).
    settlement_revalidate: bool = True
    # A dated NON-occurrence settlement vote is honored at most this many days
    # past a closed claim window (settlement_vote_validity). Within the grace it
    # is the flipped late-arrival class (the Knesset dissolving July 17 against
    # a July 15 deadline settles NO — the occurrence itself proves the miss);
    # beyond it, it is the repeatable-event non-sequitur the 2026-07-19 pool
    # audit caught (July-2026 US-strikes articles settling "US bombs Iran in
    # 2025" NO at 0.93+ certainty — an out-of-window occurrence says nothing
    # about the window, and ground truth there was YES). Undated expiry votes
    # are unaffected.
    settlement_post_deadline_grace_days: int = 14
    # Aggregate quality floor (retro#279): settlement_min_sources only counts
    # HOW MANY valid votes agree, not how much evidence they carry — a pool of
    # uniformly weak sources (low credibility, thin relevance, recency-decayed)
    # that each barely clear settlement grade could still out-count its way to
    # a pin. When set above 0, requires the winning direction's votes to also
    # carry at least this much combined weight (credibility · evidence_weight ·
    # recency · relevance²) — same units as decisiveness_floor's evidence_mass,
    # just scoped to the settling subset instead of the whole pool. Below it
    # the pin is suppressed (suppression_reason="settlement_quality_floor") and
    # the pooled mean stands, same as any other suppressed pin.
    #
    # Default 0 (disabled): unlike decisiveness_floor/relevance_weight_floor,
    # there is no audited incident to calibrate this against yet (#279 doesn't
    # specify a number either) — 0.5 looked like a natural reuse of
    # decisiveness_floor's scale but broke multiple legitimate-pin tests once
    # wired through real per-source weights (credibility/recency/relevance
    # multiplied together lands lower than a single flat floor assumes). Tune
    # from real pool data (per-source weight distribution of past legitimate
    # vs. false settlements) before enabling in prod.
    settlement_quality_floor: float = 0.0

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
    # ── DCR façade (human Claude-connector login) ──────────────────────────
    # Cognito has no Dynamic Client Registration, which Claude's MCP connector
    # requires (it refuses a static client_id and hard-fails discovery without a
    # registration_endpoint in the AS metadata). When BOTH of these are set, the
    # Oracle origin advertises ITSELF as the authorization server (mcp_dcr.py) so
    # it can inject a registration_endpoint that hands back the one pre-provisioned
    # public client. Unset = the protected-resource metadata points straight at
    # Cognito (M2M client_credentials path only, no human login). See ORACLE_MCP.md.
    cognito_hosted_ui_domain: Optional[str] = None  # e.g. https://daatan-oracle.auth.eu-central-1.amazoncognito.com
    cognito_claude_client_id: Optional[str] = None  # public PKCE client id the /register façade returns

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
    def mcp_allowed_hosts(self) -> list[str]:
        """Host header values the MCP streamable-HTTP transport accepts.

        The mcp SDK's DNS-rebinding guard auto-restricts to localhost when the
        app binds 127.0.0.1; behind nginx the app sees the public Host header
        (e.g. oracle.daatan.com) and every authenticated call would 421. Allow
        the resource URL's host (exact + any explicit port) plus localhost for
        in-box health probes."""
        from urllib.parse import urlparse

        host = urlparse(self.mcp_resource_url).netloc
        localhost = ["127.0.0.1:*", "localhost:*"]
        return [host, f"{host}:*", *localhost] if host else localhost

    @property
    def mcp_allowed_origins(self) -> list[str]:
        """Origin header values the MCP transport accepts. Non-browser clients
        omit Origin (allowed outright); a browser client on the resource host's
        own origin is allowed. Also allows the GitHub Pages origin that hosts
        oracle-mcp-test.html — same trust boundary as the CORSMiddleware
        allow_origins list in main.py, since that page calls /mcp directly
        from the browser with a real Bearer token, not just REST endpoints."""
        from urllib.parse import urlparse

        parsed = urlparse(self.mcp_resource_url)
        if not parsed.netloc:
            return ["https://daatan.github.io"]
        origin = f"{parsed.scheme}://{parsed.netloc}"
        return [origin, f"{origin}:*", "https://daatan.github.io"]

    @property
    def mcp_as_issuer(self) -> Optional[str]:
        """The authorization-server issuer the Oracle advertises for the DCR
        façade — its own origin (scheme://host of mcp_resource_url). This is the
        value clients fetch the AS metadata from, so it must equal metadata.issuer."""
        from urllib.parse import urlparse

        parsed = urlparse(self.mcp_resource_url)
        return f"{parsed.scheme}://{parsed.netloc}" if parsed.netloc else None

    @property
    def cognito_authorize_endpoint(self) -> Optional[str]:
        """Cognito hosted-UI OAuth2 authorize endpoint (browser login)."""
        d = (self.cognito_hosted_ui_domain or "").rstrip("/")
        return f"{d}/oauth2/authorize" if d else None

    @property
    def cognito_token_endpoint(self) -> Optional[str]:
        """Cognito hosted-UI OAuth2 token endpoint (code→token exchange)."""
        d = (self.cognito_hosted_ui_domain or "").rstrip("/")
        return f"{d}/oauth2/token" if d else None

    @property
    def dcr_enabled(self) -> bool:
        """True when the DCR façade should be served (the human Claude login
        flow). Requires the RS enabled plus the Cognito hosted-UI domain and the
        static public client id. When False the M2M-only path is served unchanged."""
        return bool(
            self.mcp_enabled
            and self.cognito_hosted_ui_domain
            and self.cognito_claude_client_id
            and self.mcp_as_issuer
        )

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

    @property
    def resolved_resolution_author_leaderboard_path(self) -> Path:
        if self.resolution_author_leaderboard_path != Path(""):
            return self.resolution_author_leaderboard_path
        return self.data_dir / "resolution_author_leaderboard.json"


settings = ApiSettings()
