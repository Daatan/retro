from typing import Optional
from pydantic import BaseModel, Field


# ── Search ────────────────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500, description="Search query string")
    limit: int = Field(default=5, ge=1, le=30, description="Max results to return")
    date_from: Optional[str] = Field(default=None, description="ISO date lower bound YYYY-MM-DD")
    date_to: Optional[str] = Field(default=None, description="ISO date upper bound YYYY-MM-DD")
    enrich_snippets: bool = Field(default=False, description="Scrape article URLs to fill empty snippets (adds latency)")


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


class SourceSignal(BaseModel):
    source_id: str
    source_name: str
    url: str
    stance: float = Field(description="Extracted stance [-1, 1]")
    certainty: float = Field(description="Author certainty [0, 1]")
    credibility_weight: float = Field(description="Source trust from leaderboard [0, ∞], 1.0 = neutral")
    claims: list[str] = Field(description="Extracted claim summaries")


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
    model: str = Field(default="google/gemini-2.5-flash", description="OpenRouter model ID")
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
    debug: Optional[DebugInfo] = Field(default=None, description="Debug telemetry — only present when request includes debug=true")
