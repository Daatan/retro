from typing import Literal, Optional
from pydantic import BaseModel, Field


# ── Meta ──────────────────────────────────────────────────────────────────────

class VersionResponse(BaseModel):
    version: str = Field(description="Composed version: '{base}+build.{n}' (PEP 440 local) or base")
    base_version: str = Field(description="Human-set semver from pyproject [project].version")
    build: Optional[int] = Field(default=None, description="Auto build number (git commit count)")
    git_sha: str = Field(description="Deployed commit SHA, or 'unknown'")
    git_branch: Optional[str] = Field(default=None, description="Branch the deploy came from")
    built_at: Optional[str] = Field(default=None, description="UTC ISO-8601 deploy timestamp")
    source: str = Field(description="Where commit info came from: deploy | git | env | unknown")


# ── Search ────────────────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500, description="Search query string")
    limit: int = Field(default=5, ge=1, le=30, description="Max results to return")
    date_from: Optional[str] = Field(default=None, description="ISO date lower bound YYYY-MM-DD")
    date_to: Optional[str] = Field(default=None, description="ISO date upper bound YYYY-MM-DD")
    enrich_snippets: bool = Field(default=False, description="Scrape article URLs to fill empty snippets (adds latency)")
    distill: bool = Field(default=True, description="On 0 verbatim results, distill the query to keywords (LLM; also translates non-Latin questions) and retry once")


class SearchResultItem(BaseModel):
    title: str
    url: str
    snippet: str = ""
    source: str = ""
    published_date: str = ""


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResultItem]
    count: int
    provider: str = Field(default="", description="Provider that returned results (e.g. 'gdelt', 'brave', 'none')")
    provider_chain: list[str] = Field(default_factory=list, description="Full fallback chain attempted in order")
    distilled_query: Optional[str] = Field(default=None, description="Keywords actually searched when the verbatim query returned 0 and distillation kicked in; null otherwise")


class ProviderStatus(BaseModel):
    configured: bool
    exhausted: bool = Field(description="In-process quota-exhausted flag (resets on restart)")
    status: str = Field(description="'ok' | 'exhausted' | 'not_configured' | 'error'")
    credits: Optional[int] = Field(default=None, description="Remaining credits from provider API, if available")
    error: Optional[str] = Field(default=None)


class SearchHealthResponse(BaseModel):
    providers: dict[str, ProviderStatus]
    overall: str = Field(description="'ok' (≥2 usable) | 'degraded' (1 usable) | 'down' (0 usable on EC2)")
    usable_count: int = Field(description="Number of configured, non-exhausted providers (excluding DDG)")


# ── Forecast ──────────────────────────────────────────────────────────────────

class ArticleInput(BaseModel):
    url: str
    title: str = ""
    snippet: str = ""
    source: str = ""
    published_date: str = ""
    text: Optional[str] = Field(
        default=None,
        description="Pre-fetched article body. If omitted, oracle fetches via trafilatura.",
    )
    # Caller-supplied gatekeeper verdict (news-indexer's POST /relevance, threaded through
    # daatan). When BOTH are set and settings.reuse_supplied_relevance is on, the Oracle
    # reuses them instead of re-running check_is_prediction — the SAME judge already ran once
    # upstream. Additive and fail-open: omit them and today's behavior (re-judge) is unchanged.
    # The extractor still runs regardless; only the duplicate relevance judgment is skipped.
    relevance: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Caller's graded gatekeeper relevance [0,1]; reused with is_prediction when reuse_supplied_relevance is on.",
    )
    is_prediction: Optional[bool] = Field(
        default=None,
        description="Caller's gatekeeper pass/reject verdict; reused with relevance when reuse_supplied_relevance is on.",
    )


class ForecastRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=500, description="Binary question to forecast")
    max_articles: Optional[int] = Field(default=None, ge=1, le=30)
    articles: Optional[list[ArticleInput]] = Field(
        default=None,
        description=(
            "Pre-fetched articles. If provided, oracle skips its internal search and analyzes "
            "these directly. max_articles is ignored when this field is set."
        ),
    )
    debug: bool = Field(default=False, description="Include debug telemetry in response (token counts, gatekeeper scores, prompts)")
    # Optional temporal metadata (daatan's claim classifier; TEMPORAL_MODEL_PLAN.md #3.4).
    # Additive and fail-open: callers that don't classify claims omit them and get
    # today's behavior. When present they gate EARLY settlement direction — before
    # claim_deadline only "the event occurred" may pin (arrival → YES, survival → NO).
    claim_direction: Optional[Literal["arrival", "survival"]] = Field(
        default=None,
        description="Temporal direction of the claim: arrival = true only if the event happens by the deadline; survival = true if it does not.",
    )
    claim_deadline: Optional[str] = Field(
        default=None,
        description="Claim deadline (ISO date, e.g. 2026-12-31). Before it, only occurrence-direction settlements may pin the estimate.",
    )
    claim_created_at: Optional[str] = Field(
        default=None,
        description="When the caller's prediction was created (ISO date or timestamp). With claim_archetype='scheduled', a settlement event dated before it cannot settle the claim — it belongs to an earlier instance of the recurring event, not this one. Accepted now, enforced by settlement revalidation.",
    )
    claim_archetype: Optional[Literal["scheduled", "diffuse", "threshold", "none"]] = Field(
        default=None,
        description="Temporal archetype from the caller's claim classifier. 'scheduled' claims (an election, a match) bound settlement event dates to [claim_created_at, claim_deadline]; other archetypes bound only the deadline side (a threshold may legitimately have been crossed before the claim was created). Accepted now, enforced by settlement revalidation.",
    )
    prediction_id: Optional[str] = Field(
        default=None,
        description="Caller's identifier for the prediction this forecast relates to (e.g. daatan's context_snapshots key). Log correlation only — never used in scoring.",
    )


class SourceSignal(BaseModel):
    source_id: str
    source_name: str
    url: str
    stance: float = Field(description="Extracted stance [-1, 1]")
    certainty: float = Field(description="Author certainty [0, 1]")
    credibility_weight: float = Field(description="Source trust from leaderboard [0, ∞], 1.0 = neutral")
    claims: list[str] = Field(description="Extracted claim summaries")
    published_date: Optional[str] = Field(default=None, description="Article publish date (YYYY-MM-DD) if known; drives recency weighting")
    recency_weight: Optional[float] = Field(default=None, description="Time-decay weight applied to this source in aggregation [recency_floor, 1.0]")
    relevance_score: Optional[float] = Field(default=None, description="Graded topic relevance [0,1]; its square multiplies this source's aggregation weight")
    settled: Optional[bool] = Field(default=None, description="True when this source reports the event's outcome as an accomplished fact (settlement claim), not a prediction")
    quantitative_estimate: Optional[float] = Field(default=None, description="Explicit modeled/poll/market probability [0,1] this source cited for the event itself, if any; carries a weight premium via evidence_class=cited_probability (see evidence_class_weight)")
    evidence_weight: Optional[float] = Field(default=None, description="This source's evidence_class-derived weight component (S2 cutover) — the linear factor `credibility * evidence_weight * recency * relevance^2` reduces to when recomputing a pool of already-extracted sources without redoing extraction. NOT credibility/recency/relevance combined, just the evidence_class-weighting term; see evidence_class_weight() in aggregation.py.")
    evidence_class: Optional[Literal["reported_fact", "cited_probability", "cited_share", "reporting", "opinion"]] = Field(default=None, description="This article's most common evidence_class among its extracted claims (an article can carry several claims with different classes; None if every claim was unclassified). Needed by the credibility feedback loop (docs/ORACLE_VARIABLES.md §9) to exclude opinion-class articles from the resolution-outcome signal — evidence_weight alone can't distinguish an opinion-class article from a low-certainty unclassified one, since both can land at a similar numeric weight.")
    settlement_event_date: Optional[str] = Field(default=None, description="When settled: the ISO date anchoring this source's settlement — the outcome's occurrence date for a positive settlement, the foreclosing event's date for a negative one; None when the settlement is legitimately undated (e.g. post-deadline expiry). Taken from the highest-certainty settlement-grade claim whose sign matches this source's collapsed stance. Persist alongside `settled` — aggregation-time revalidation re-checks it on every pool recompute.")
    author_lean: Optional[float] = Field(default=None, description="The BYLINE author's OWN directional forecast of the event (retro #308/#309) [-1, 1]: +1 the author expects it to happen, -1 they expect it will NOT, 0 they weigh both sides. None when the author only reports facts or relays others' views. Deliberately SEPARATE from `stance`/the estimate — carried here only so daatan can score author accuracy later; nothing in aggregation reads it.")
    author_lean_certainty: Optional[float] = Field(default=None, description="How firmly the byline author commits to `author_lean` [0, 1] (0 = heavily hedged, 1 = emphatic). None when `author_lean` is None. Scoring lane only — not used in the estimate.")
    fact_signal: Optional[float] = Field(default=None, description="EXPERIMENTAL shadow (Phase 2, author-scoring redesign) — the fact-lane counterpart of `stance`: what the REPORTED FACTS alone imply about the event [-1, 1], un-fused from author assertion/framing. Claim-weighted MEAN over this article's fact-bearing claims — the SAME reduction as `stance` (over the same scored claims), so the offline fact-lane backtest compares mean-to-mean. None when no scored claim carried a fact_signal (e.g. pure opinion). Nothing in aggregation reads it yet — carried only for daatan persistence + the offline backtest.")
    event_actors: Optional[str] = Field(default=None, description="EXPERIMENTAL shadow — WHO acts in the fact behind `fact_signal`, from this article's DOMINANT (max |fact_signal|) claim; for the estimator's actor-pair (dyad) check against the claim's subject. None when `fact_signal` is None.")
    event_target: Optional[str] = Field(default=None, description="EXPERIMENTAL shadow — the TARGET of the action in the fact behind `fact_signal`, from the dominant claim; with `event_actors` this is the fact's dyad. None when `fact_signal` is None.")
    is_occurrence: Optional[bool] = Field(default=None, description="EXPERIMENTAL shadow — True when the dominant fact IS the event itself (or its definitive outcome), False when it is only a precursor/precondition/escalation. From the dominant claim; None when `fact_signal` is None.")
    verified: Optional[bool] = Field(default=None, description="EXPERIMENTAL shadow — True when the dominant fact is independently reported, False when only claimed by an interested party. From the dominant claim; None when `fact_signal` is None.")


class ArticleDebug(BaseModel):
    url: str
    outcome: str = Field(description="ok | gate_rejected | no_predictions | fetch_error | gate_error | extract_error | empty_text")
    gate_passed: Optional[bool] = None
    gate_reason: Optional[str] = None
    gate_prediction_count_estimate: Optional[int] = None
    gate_tokens: Optional[int] = None
    extract_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    fetch_ms: Optional[float] = None
    gate_ms: Optional[float] = None
    extract_ms: Optional[float] = None


class DebugInfo(BaseModel):
    search_query: str
    search_provider: str = Field(description="Provider that returned results: serper | serpapi | brightdata | ddg | search_cache | caller | none")
    search_provider_chain: list[str] = Field(description="Full fallback chain attempted before a result was returned")
    gatekeeper_model: str
    extractor_model: str
    articles_fetched: int
    articles_gate_passed: int
    articles_extracted: int
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int
    per_article: list[ArticleDebug]
    gatekeeper_prompt: str
    extractor_prompt: str


# ── LLM proxy ────────────────────────────────────────────────────────────────

class LlmMessage(BaseModel):
    role: str
    content: str


class LlmRequest(BaseModel):
    model: Optional[str] = Field(
        default=None,
        description="litellm model ID (e.g. bedrock/amazon.nova-lite-v1:0); defaults to the server's configured Bedrock model",
    )
    messages: list[LlmMessage]
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)


class LlmResponse(BaseModel):
    content: str
    model: str


# ── Article fetch ─────────────────────────────────────────────────────────────

class FetchUrlRequest(BaseModel):
    url: str = Field(..., description="Article URL to fetch and extract")


class FetchUrlResponse(BaseModel):
    text: str
    title: Optional[str] = None
    date: Optional[str] = None   # ISO date string YYYY-MM-DD
    source: Optional[str] = None


# ── Forecast ──────────────────────────────────────────────────────────────────

class ForecastResponse(BaseModel):
    question: str
    mean: float = Field(description="Credibility-weighted mean stance [-1, 1]. Convert to probability: (mean + 1) / 2")
    std: float = Field(description="Weighted standard deviation")
    ci_low: float = Field(description="95% confidence interval lower bound")
    ci_high: float = Field(description="95% confidence interval upper bound")
    articles_used: int
    sources: list[SourceSignal]
    placeholder: bool = Field(default=False, description="True if this is a stub response (pipeline not yet wired)")
    insufficient_data: bool = Field(default=False, description="True when the forecast could not be computed (no usable articles). mean/ci are NOT a real estimate — render 'couldn't answer', not 0%.")
    reason: Optional[str] = Field(default=None, description="When insufficient_data: why. One of no_search_results | all_articles_off_topic | no_usable_weight | no_decisive_signal | all_fetches_failed | extraction_errors | no_usable_predictions | timeout | no_result. (no_usable_weight: every surviving source carried zero aggregation weight — blocked by credibility and/or zeroed by relevance — so there was nothing to pool. no_decisive_signal only when defer_on_thin_evidence is set; otherwise thin evidence yields a wide-CI estimate.)")
    articles_found: int = Field(default=0, description="How many search results were considered before filtering")
    outcome_counts: dict[str, int] = Field(default_factory=dict, description="Per-article outcome histogram (gate_rejected, gate_error, empty_text, extract_error, unhandled_error, ok, …) — explains an empty forecast")
    provider: str = Field(default="", description="Search provider that served the underlying article search. May be a pseudo-provider: 'caller' (articles supplied by the request), 'search_cache', or 'none'.")
    provider_chain: list[str] = Field(default_factory=list, description="Full search fallback chain attempted, in order")
    distilled_query: Optional[str] = Field(default=None, description="Keywords the question was distilled to before searching; null when no distillation was applied")
    settled: bool = Field(default=False, description="True when enough independent sources report the event's outcome as an accomplished fact (see settlement_min_sources). mean/ci are pinned near the boundary; the forecast is a candidate for resolution, not further updates.")
    debug: Optional[DebugInfo] = Field(default=None, description="Debug telemetry — only present when request includes debug=true")


# ── Pool aggregate (recompute-over-pool) ────────────────────────────────────

class PoolSourceInput(BaseModel):
    """One already-extracted per-source signal, as a caller's evidence pool
    would persist it — the same fields a live /forecast call's SourceSignal
    already returns (see docs/DATABASE.md "Evidence pool" in the daatan repo)."""
    stance: float = Field(ge=-1.0, le=1.0, description="This source's already-computed avg_stance")
    certainty: float = Field(ge=0.0, le=1.0, description="Fallback weight component used only when evidence_weight is unset (legacy rows predating the S2 cutover, PR #248/#249)")
    credibility_weight: float = Field(ge=0.0, description="Source trust from the leaderboard; 1.0 = neutral")
    relevance_score: float = Field(ge=0.0, le=1.0, description="Graded topic relevance; squared in the weight formula")
    evidence_weight: Optional[float] = Field(default=None, ge=0.0, description="This source's evidence_class-derived weight (see SourceSignal.evidence_weight); falls back to certainty when unset")
    published_date: Optional[str] = Field(default=None, description="Article publish date (YYYY-MM-DD); recency is recomputed against now, not a stored value")
    settled: bool = Field(default=False, description="True when this source cleared the settlement grade (SourceSignal.settled)")
    settlement_event_date: Optional[str] = Field(default=None, description="The settlement anchor date this source carried when extracted (SourceSignal.settlement_event_date); consumed by aggregation-time settlement revalidation")


class PoolAggregateRequest(BaseModel):
    """Recompute a pooled estimate over an already-extracted evidence pool —
    no search, no LLM calls. Same claim_direction/claim_deadline/claim_created_at/
    claim_archetype semantics as ForecastRequest (gates settlement votes)."""
    sources: list[PoolSourceInput] = Field(default_factory=list)
    claim_direction: Optional[Literal["arrival", "survival"]] = Field(default=None)
    claim_deadline: Optional[str] = Field(default=None)
    claim_created_at: Optional[str] = Field(default=None, description="See ForecastRequest.claim_created_at")
    claim_archetype: Optional[Literal["scheduled", "diffuse", "threshold", "none"]] = Field(default=None, description="See ForecastRequest.claim_archetype")


class PoolAggregateResponse(BaseModel):
    mean: float = Field(description="Credibility-weighted mean stance [-1, 1]. Convert to probability: (mean + 1) / 2")
    std: float = Field(description="Weighted standard deviation")
    ci_low: float = Field(description="95% confidence interval lower bound")
    ci_high: float = Field(description="95% confidence interval upper bound")
    articles_used: int
    settled: bool = Field(default=False, description="Same settlement-override semantics as ForecastResponse.settled")
    insufficient_data: bool = Field(default=False, description="True when no usable estimate could be pooled. mean/ci are NOT a real estimate.")
    reason: Optional[str] = Field(default=None, description="When insufficient_data: no_sources | all_articles_off_topic | no_usable_weight | no_decisive_signal")
    settlement_suppressed: bool = Field(default=False, description="True when a would-be settlement pin was suppressed — 'settlement_conflict' (valid settled votes in both directions, revalidation path) or 'settlement_direction' (temporal guard, legacy path). Diagnostics only; the pooled mean stands.")
    settlement_suppression_reason: Optional[str] = Field(default=None, description="Why the pin was suppressed, when settlement_suppressed")
    settlement_votes_demoted: int = Field(default=0, description="How many of the request's settled votes failed aggregation-time revalidation (settlement_vote_validity) and were counted as ordinary evidence instead. Per-vote reasons are logged server-side (event=settlement_vote_demoted).")


# ── Resolution feedback ingest (credibility feedback loop, step 1) ─────────
# docs/ORACLE_VARIABLES.md "Open, in suggested order" — resolution-outcome
# feedback loop. Storage-only for now: daatan pushes one resolved forecast's
# per-source stances here; nothing reads this yet (that's step 2+).

class ResolutionSourceInput(BaseModel):
    """One source's stance on a forecast that has since resolved — the same
    fields daatan's EvidencePoolArticle already persists per article."""
    source: str = Field(description="Outlet/source id, matches leaderboard.json's source_id")
    stance: float = Field(ge=-1.0, le=1.0, description="This source's stance at the time it was extracted")
    evidence_class: Optional[Literal["reported_fact", "cited_probability", "cited_share", "reporting", "opinion"]] = Field(default=None, description="opinion-class articles are excluded from scoring downstream — recorded here regardless so the exclusion rule can be changed without re-ingesting")
    credibility_weight: Optional[float] = Field(default=None, ge=0.0)
    evidence_weight: Optional[float] = Field(default=None, ge=0.0)


class AuthorSignalInput(BaseModel):
    """One byline author's directional lean on a forecast that has since
    resolved — the author-scoring lane counterpart of ResolutionSourceInput
    (author-scoring redesign, Phase 1 step 3). Unlike the stance lane,
    opinion-class rows belong here: author_lean is the author's own lean and
    opinion is precisely where it lives."""
    author: Optional[str] = Field(default=None, description="Byline author string as extracted; None/empty is scored per-outlet as '(no byline)'")
    outlet_name: Optional[str] = Field(default=None)
    author_lean: float = Field(ge=-1.0, le=1.0, description="The author's lean toward the claim resolving YES, at extraction time")
    author_lean_certainty: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    evidence_class: Optional[Literal["reported_fact", "cited_probability", "cited_share", "reporting", "opinion"]] = Field(default=None, description="Recorded for later analysis, never filtered on in this lane")


class IngestResolutionRequest(BaseModel):
    """One resolved forecast's per-source stances, pushed by daatan once a
    Prediction resolves. Idempotent on prediction_id — re-ingesting the same
    prediction_id is a no-op, not an error, so a fire-and-forget retry on the
    daatan side can never double-count a source's history."""
    prediction_id: str
    outcome: bool = Field(description="True if the claim resolved YES/TRUE (Prediction.status == RESOLVED_CORRECT for a claim asserting YES)")
    resolved_at: Optional[str] = Field(default=None)
    sources: list[ResolutionSourceInput] = Field(default_factory=list)
    author_signals: list[AuthorSignalInput] = Field(default_factory=list)


class IngestResolutionResponse(BaseModel):
    accepted: bool = Field(default=True)
    already_ingested: bool = Field(description="True when prediction_id was already on record — sources_recorded is 0 in that case, nothing was written")
    sources_recorded: int
    author_signals_recorded: int = Field(default=0)


# ── Relevance ─────────────────────────────────────────────────────────────────

class RelevanceRequest(BaseModel):
    """One (claim, article) pair to judge. Exposes the gatekeeper on its own so a
    caller can ask 'does this article bear on this claim?' *before* committing to a
    full /forecast run — news-indexer uses it to rescue articles its embedding
    cosine ranks poorly (the cosine misranks; see docs/ORACLE_API.md)."""
    claim: str = Field(..., min_length=3, description="The claim to judge against — daatan's Prediction.claimText")
    article_text: str = Field(..., min_length=1, description="Article body text")
    source_name: str = Field(default="", description="Outlet/source name, as the gatekeeper prompt's Source line")
    article_date: str = Field(default="", description="Article date, as the gatekeeper prompt's Date line")
    short_form: bool = Field(
        default=False,
        description=(
            "The item is a social-media / messaging post (e.g. a journalist's Telegram channel), "
            "not a news article. Judge it on content rather than length: the gatekeeper's default "
            "'under ~200 words is insubstantial' rule targets paywall stubs, and misfires on terse "
            "posts that carry real evidence. Off by default; /forecast never sets it."
        ),
    )


class RelevanceResponse(BaseModel):
    """`tm.gatekeeper.GatekeeperOutput` over the wire, plus the model that judged —
    callers persist `model` so a verdict can always be traced to what produced it."""
    is_prediction: bool = Field(description="Coarse gate: could a forecaster's estimate of THIS outcome move after reading the article?")
    relevance_score: float = Field(ge=0.0, le=1.0, description="Graded relevance [0,1]; its square multiplies the source's aggregation weight in /forecast")
    reason: str
    prediction_count_estimate: int = Field(ge=0)
    model: str = Field(description="litellm model ID that produced this verdict")
