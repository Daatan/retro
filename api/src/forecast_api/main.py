import asyncio
import logging
import re
from contextlib import asynccontextmanager
from urllib.parse import urlparse

import httpx
from fastapi import Depends, FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from slowapi.errors import RateLimitExceeded

from ._build import build_info
from .auth import ApiKeyClient, verify_api_key
from .bayesoracle import compute_nodes, parse_node_observations
from .cache import forecast_cache
from .config import settings
from .forecaster import (
    GATEKEEPER_PROMPT_HASH,
    GATEKEEPER_SCHEMA_HASH,
    GATEKEEPER_PROMPT_VERSION,
    run_forecast,
    run_pool_aggregate,
)
from .leaderboard import (
    background_refresh_loop,
    get_leaderboard_data,
    leaderboard_size,
    leaderboard_snapshot_date,
    refresh_cache,
    refresh_shadow_cache,
)
from .resolution_feedback import ingest_resolution
from .resolution_scorer import load_shadow_leaderboard, rescore_authors_from_disk, rescore_from_disk
from .settlement_pin_ledger import load_ledger, record_settlement_pin
from tm.config import settings as _pipeline_settings
from tm.gatekeeper import check_is_prediction
from tm.llm import complete_text_once_with_usage
from .article_fetch import ArticleFetchError, fetch_and_extract
from .limiter import limiter
from .mcp_auth import SCOPE_FORECAST, SCOPE_READ
from .mcp_server import build_mcp
from .models import ForecastRequest, ForecastResponse, FetchUrlRequest, FetchUrlResponse, IngestResolutionRequest, IngestResolutionResponse, LlmRequest, LlmResponse, PoolAggregateRequest, PoolAggregateResponse, RelevanceRequest, RelevanceResponse, SearchRequest, SearchResponse, SearchHealthResponse, SettlementPinReportResponse, TokenUsage, VersionResponse
from .searcher import run_search, run_search_health
from .v2_playground import V2ForecastRequest, V2JobCreated, get_job, start_job

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)


# MCP server (None when Cognito OAuth isn't configured — then /mcp is not mounted
# and the REST API runs unaffected). Built once at import; its stateless-HTTP
# session manager is driven by the lifespan below.
_mcp = build_mcp()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    path = settings.resolved_leaderboard_path
    await refresh_cache(path)
    logger.info("Oracle API starting — leaderboard: %d sources, port: %d", leaderboard_size(), settings.port)
    # The resolution-shadow board only feeds credibility when the cutover flag
    # is on, so skip the extra disk reads entirely when it isn't.
    shadow_board = shadow_feedback = None
    if settings.resolution_shadow_credibility_enabled:
        shadow_board = settings.resolved_resolution_leaderboard_path
        shadow_feedback = settings.resolved_resolution_feedback_path
        await refresh_shadow_cache(shadow_board, shadow_feedback)
    refresh_task = asyncio.create_task(
        background_refresh_loop(path, settings.leaderboard_refresh_seconds, shadow_board, shadow_feedback)
    )
    try:
        if _mcp is not None:
            # StreamableHTTPSessionManager must be running for the mounted /mcp
            # app to serve requests.
            async with _mcp.session_manager.run():
                yield
        else:
            yield
    finally:
        # Shutdown
        refresh_task.cancel()
        try:
            await refresh_task
        except asyncio.CancelledError:
            pass
        logger.info("Oracle API shut down")


_CORS_ORIGIN = "https://daatan.github.io"
_CORS_ORIGINS = {_CORS_ORIGIN, "https://bayes.daatan.com"}


def _rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    origin = request.headers.get("origin", "")
    headers = {}
    if origin in _CORS_ORIGINS or origin.startswith("http://localhost") or origin.startswith("http://127.0.0.1"):
        headers["Access-Control-Allow-Origin"] = origin
        headers["Access-Control-Allow-Headers"] = "Content-Type, x-api-key"
        headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return JSONResponse({"detail": "Rate limit exceeded — max 10 requests/minute"}, status_code=429, headers=headers)


app = FastAPI(
    title="TruthMachine Oracle API",
    description="Calibrated probability estimates for binary questions, weighted by historical source accuracy.",
    version=build_info()["version"],  # single-sourced from pyproject [project].version
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)

# Allow oracle-test.html / oracle-mcp-test.html on GitHub Pages to call this API.
# "Authorization" is for oracle-mcp-test.html's browser-side /mcp calls (Bearer
# token from the Cognito PKCE flow) — REST endpoints still use x-api-key only.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://daatan.github.io",
        "https://bayes.daatan.com",
    ],
    # Starlette matches allow_origins exactly (wildcard ports don't work there),
    # so localhost dev origins need a regex.
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "x-api-key", "Authorization"],
)


@app.get("/", include_in_schema=False)
async def root():
    """Redirect to the interactive test console."""
    return RedirectResponse("https://daatan.github.io/retro/oracle-test.html")


@app.get("/bayes/nodes", tags=["BayesOracle"])
async def bayes_nodes(
    observations: str = Query(
        default="",
        description=(
            "Comma-separated node=probability overrides, e.g. "
            "'ELECTIONS=0.95,TRUMP=0.70'. "
            "Overrides are clamped to [0, 1]. Unlisted nodes are computed from the DAG."
        ),
    ),
    _: ApiKeyClient = Depends(verify_api_key),
):
    """
    Return BayesOracle probabilities for the Israeli-politics DAG.

    Each node has a prior; children are computed from their parents via a
    fitted-intercept logistic CPT (exact enumeration over parent states), so
    baseline reproduces the priors and mutually-exclusive outcome groups stay on
    the probability simplex.  The graph lives in ``bayesoracle/graph_political.json``.
    Supply ``observations`` to lock specific nodes and see how the rest shifts.
    """
    obs = parse_node_observations(observations)
    return {"nodes": compute_nodes(obs or None)}


@app.post("/v2/forecast", response_model=V2JobCreated, status_code=202, tags=["V2"])
@limiter.limit("5/minute")
async def v2_forecast(request: Request, body: V2ForecastRequest, _: ApiKeyClient = Depends(verify_api_key)):  # request: required by slowapi
    """Oracle 2.0 playground (retro#595): start a traced query-path run and
    return its job id. Poll ``GET /v2/jobs/{id}`` for the growing trace."""
    job = start_job(body)
    return V2JobCreated(job_id=job["id"], status=job["status"])


@app.get("/v2/jobs/{job_id}", tags=["V2"])
async def v2_job(job_id: str, _: ApiKeyClient = Depends(verify_api_key)):
    """The full trace of a playground run: nodes, edges, every prompt and
    response, the log, and the result once done. 404 for unknown ids."""
    job = get_job(job_id)
    if job is None:
        return JSONResponse(status_code=404, content={"detail": "unknown job"})
    return job


@app.get("/leaderboard", tags=["Meta"])
async def leaderboard(_: ApiKeyClient = Depends(verify_api_key)):
    """
    Return the source credibility leaderboard, sorted by skill conservative score (μ − 3σ).

    A FROZEN snapshot (see `snapshot_date`) — the legacy vault, kept only as
    the credibility fallback. Watched for manual rewrites every N seconds, but
    nothing has regenerated it since 2026-03-28. See GET /leaderboard/resolution-shadow
    for the newer, resolution-informed board (small sample size).
    """
    return {
        "sources": get_leaderboard_data(),
        "count": leaderboard_size(),
        "snapshot_date": leaderboard_snapshot_date(),
    }


@app.post("/leaderboard/ingest", response_model=IngestResolutionResponse, tags=["Meta"])
@limiter.limit("60/minute")
async def leaderboard_ingest(
    request: Request,  # required by slowapi
    body: IngestResolutionRequest,
    _: ApiKeyClient = Depends(verify_api_key),
):
    """
    Credibility feedback loop, steps 1+3 (docs/ORACLE_VARIABLES.md §9).
    Ingest one resolved forecast's per-source stances, then (if it's new,
    not a duplicate) replay the full accumulated history through OpenSkill
    to refresh the resolution-informed shadow score — see
    GET /leaderboard/resolution-shadow.

    Storage/shadow only while resolution_shadow_credibility_enabled is off
    (the default) — the rescored board is then read by nothing but
    GET /leaderboard/resolution-shadow, and leaderboard.json is untouched
    either way. Idempotent on prediction_id: re-sending the same
    prediction_id is a no-op (already_ingested=true), safe for a
    fire-and-forget retry on the caller's side.

    Also feeds the settlement-pin ledger (retro#361 Phase 1): when the
    payload carries a ``settlement_snapshot``, records the pin's declared
    direction against ``outcome`` for GET /leaderboard/settlement-pin-report.
    Independent idempotency from the resolution-feedback store above — keyed
    on prediction_id in its own ledger file — so a retry that adds a
    snapshot the first push omitted can still land it.
    """
    result = await ingest_resolution(settings.resolved_resolution_feedback_path, body)
    await record_settlement_pin(settings.resolved_settlement_pin_ledger_path, body)
    if not result.already_ingested:
        await asyncio.to_thread(
            rescore_from_disk,
            settings.resolved_resolution_feedback_path,
            settings.resolved_resolution_leaderboard_path,
        )
        await asyncio.to_thread(
            rescore_authors_from_disk,
            settings.resolved_resolution_feedback_path,
            settings.resolved_resolution_author_leaderboard_path,
        )
        # Pick the freshly-scored board up immediately rather than waiting out
        # the daily refresh timer — otherwise live weights would lag a new
        # resolution by up to leaderboard_refresh_seconds.
        if settings.resolution_shadow_credibility_enabled:
            await refresh_shadow_cache(
                settings.resolved_resolution_leaderboard_path,
                settings.resolved_resolution_feedback_path,
            )
    return result


@app.get("/leaderboard/resolution-shadow", tags=["Meta"])
async def leaderboard_resolution_shadow(_: ApiKeyClient = Depends(verify_api_key)):
    """
    Credibility feedback loop, step 3 (docs/ORACLE_VARIABLES.md §9) — the
    resolution-informed shadow score, recomputed from scratch on every
    /leaderboard/ingest call. For comparing against the live /leaderboard
    (vault-curated) during the observation step (step 4). Does not affect
    get_credibility_weight() or /forecast.
    """
    data = await asyncio.to_thread(load_shadow_leaderboard, settings.resolved_resolution_leaderboard_path)
    return {"sources": data, "count": len(data)}


@app.get("/leaderboard/author-shadow", tags=["Meta"])
async def leaderboard_author_shadow(_: ApiKeyClient = Depends(verify_api_key)):
    """
    Author-scoring lane shadow board (author-scoring redesign, Phase 1 step
    3): per-(byline author, outlet) resolution-time Brier + OpenSkill over
    the ingested author_signals, recomputed from scratch on every
    /leaderboard/ingest call. Shadow only — nothing reads this into
    /forecast weighting or get_credibility_weight().
    """
    data = await asyncio.to_thread(load_shadow_leaderboard, settings.resolved_resolution_author_leaderboard_path)
    return {"authors": data, "count": len(data)}


@app.get("/leaderboard/settlement-pin-report", response_model=SettlementPinReportResponse, tags=["Meta"])
async def leaderboard_settlement_pin_report(
    include_confirmed: bool = Query(default=False, description="Include pins that agreed with their resolution too, not just contradicted ones"),
    _: ApiKeyClient = Depends(verify_api_key),
):
    """
    Settlement-pin quality report (retro#361 Phase 1) — every settlement pin
    on record, compared against its eventual resolved outcome. Recomputed
    from the ledger on disk on every call, same as GET /leaderboard/*-shadow.

    Defaults to contradicted pins only — the direct answer to "which pins
    were wrong". Pass include_confirmed=true for the full ledger (both
    directions), e.g. to compute a denominator for pin precision.

    Deliberately does NOT classify *why* a contradicted pin was wrong
    (extraction error / question semantics / genuine miss / pin-correct) —
    that heuristic is the issue's Phase 2, scoped out here as follow-up work
    requiring its own review.
    """
    all_entries = await load_ledger(settings.resolved_settlement_pin_ledger_path)
    contradicted = [e for e in all_entries if e.contradicted]
    return SettlementPinReportResponse(
        entries=all_entries if include_confirmed else contradicted,
        contradicted_count=len(contradicted),
        total_settled_pins=len(all_entries),
    )


@app.get("/health", tags=["Meta"])
async def health():
    """Liveness probe — no auth required."""
    bi = build_info()
    return {
        "status": "ok",
        "version": bi["version"],
        "build": bi["build"],
        "git_sha": bi["git_sha"],
        "git_branch": bi["git_branch"],
        "built_at": bi["built_at"],
        "leaderboard_sources": leaderboard_size(),
        "cache": {
            "enabled": forecast_cache.enabled,
            **forecast_cache.stats().as_dict(),
        },
    }


@app.get("/version", response_model=VersionResponse, tags=["Meta"])
async def version():
    """Build provenance — version + deployed commit. No auth required."""
    return build_info()


@app.post("/forecast", response_model=ForecastResponse, tags=["Forecast"])
@limiter.limit("10/minute")
async def forecast(
    request: Request,  # required by slowapi
    body: ForecastRequest,
    client: ApiKeyClient = Depends(verify_api_key),
):
    """
    Given a binary question, return a calibrated probability distribution.

    The `mean` field is in stance space [-1, 1].
    Convert to probability [0, 1] with: `p = (mean + 1) / 2`
    """
    return await run_forecast(body, client=client)


@app.post("/pool/aggregate", response_model=PoolAggregateResponse, tags=["Forecast"])
@limiter.limit("60/minute")
async def pool_aggregate(
    request: Request,  # required by slowapi
    body: PoolAggregateRequest,
    _: ApiKeyClient = Depends(verify_api_key),
):
    """
    Recompute a pooled estimate over an already-extracted evidence pool —
    no search, no LLM calls. Reuses the exact same pooling math `/forecast`
    uses live, so a recompute can never silently drift from a fresh run of
    the same evidence (retro docs/ORACLE_VARIABLES.md, recompute-over-pool).

    The `mean` field is in stance space [-1, 1], same as `/forecast`.
    """
    return await run_pool_aggregate(body)


@app.post("/relevance", response_model=RelevanceResponse, tags=["Forecast"])
@limiter.limit("120/minute")
async def relevance(
    request: Request,  # required by slowapi
    body: RelevanceRequest,
    _: ApiKeyClient = Depends(verify_api_key),
):
    """
    Judge one (claim, article) pair with the gatekeeper — the same claim-aware
    screener `/forecast` already runs on every article it aggregates
    (`tm.gatekeeper.check_is_prediction`, Nova Micro). Same prompt, same model:
    this endpoint only exposes it standalone, so a caller can ask "does this
    article bear on this claim?" without paying for a whole `/forecast` run.

    Why it exists: news-indexer gates forecast↔article matching on embedding
    cosine, which *misranks* — a tangential article can outscore an on-topic one,
    so genuinely relevant coverage never reaches the Oracle at all. news-indexer
    calls this to give poorly-ranked candidates a second, claim-aware opinion.

    Stateless, and no search — one LLM call. It does not aggregate, does not
    touch the leaderboard, and does not affect `/forecast`'s behaviour.
    """
    try:
        out, usage = await check_is_prediction(
            article_text=body.article_text,
            source_name=body.source_name,
            article_date=body.article_date,
            event_name=body.claim,
            short_form=body.short_form,
            language=body.language,
        )
    except Exception as exc:
        logger.warning("relevance check failed claim=%.60s err=%s", body.claim, exc)
        return JSONResponse({"detail": f"Relevance check failed: {exc}"}, status_code=502)
    return RelevanceResponse(
        is_prediction=out.is_prediction,
        relevance_score=out.relevance_score,
        reason=out.reason,
        prediction_count_estimate=out.prediction_count_estimate,
        model=_pipeline_settings.gatekeeper_model,
        gatekeeper_prompt_version=GATEKEEPER_PROMPT_VERSION,
        gatekeeper_prompt_hash=GATEKEEPER_PROMPT_HASH,
        gatekeeper_schema_hash=GATEKEEPER_SCHEMA_HASH,
        token_usage=TokenUsage.from_usages([usage]),
    )


@app.post("/search", response_model=SearchResponse, tags=["Search"])
@limiter.limit("60/minute")
async def search(
    request: Request,  # required by slowapi
    body: SearchRequest,
    _: ApiKeyClient = Depends(verify_api_key),
):
    """
    Search for news articles using the full provider fallback chain.

    Tries: SerpAPI → Serper → Brave → BrightData → Nimbleway → ScrapingBee → DDG.
    DDG is skipped when the service is running on EC2.
    """
    return await run_search(body)


@app.get("/search/health", response_model=SearchHealthResponse, tags=["Search"])
async def search_health(_: ApiKeyClient = Depends(verify_api_key)):
    """
    Per-provider search health: key configured, in-process quota flag, and live credit
    count where the provider exposes a credit API (Serper, SerpAPI, ScrapingBee).
    """
    return await run_search_health()


@app.post("/llm", response_model=LlmResponse, tags=["IBI"])
@limiter.limit("30/minute")
async def llm_proxy(
    request: Request,
    body: LlmRequest,
    client: ApiKeyClient = Depends(verify_api_key),
):
    """
    Proxy LLM calls to Bedrock (via tm.llm → litellm) using the server's AWS creds.
    Accepts model + messages, returns the assistant's text content. ``model`` is a
    litellm ID and defaults to the server's configured Bedrock model; callers may
    override it (e.g. bedrock/amazon.nova-micro-v1:0).

    Uses the non-retrying complete_text_once_with_usage: this is an interactive
    endpoint and daatan's caller times out at 60s, so we don't want the
    [30,60,120] backoff.

    retro#432: a ``max_articles``-capped key exists specifically to stop
    unmetered LLM spend (the staging-key-cutover incident, see auth.py). That
    cap has no natural meaning here — this route has no article concept to
    scale it against — so a capped client is denied the endpoint outright
    rather than partially metered; the whole point of the cap is that this
    key must not drive uncontrolled spend.
    """
    if client.max_articles is not None:
        logger.warning("event=llm_proxy_denied_capped_client client=%s", client.name)
        return JSONResponse(
            {"detail": "This API key is spend-capped and cannot use the raw /llm proxy"},
            status_code=403,
        )
    model = body.model or _pipeline_settings.extractor_model
    messages = [{"role": m.role, "content": m.content} for m in body.messages]
    try:
        content, usage = await complete_text_once_with_usage(
            model,
            messages=messages,
            max_tokens=1024,
            temperature=body.temperature,
            timeout=55,
        )
    except Exception as exc:
        logger.warning("llm proxy failed model=%s err=%s", model, exc)
        return JSONResponse({"detail": f"LLM call failed: {exc}"}, status_code=502)
    return LlmResponse(content=content, model=model, token_usage=TokenUsage.from_usages([usage]))


@app.post("/fetch-url", response_model=FetchUrlResponse, tags=["IBI"])
@limiter.limit("30/minute")
async def fetch_url(request: Request, body: FetchUrlRequest):
    """
    Fetch an article URL and extract its text, title, and publication date.
    Uses trafilatura. Intentionally public (no x-api-key) so the IBI tool works
    without a key, but rate-limited and SSRF-guarded: rejects non-http(s) schemes
    and hosts that resolve to non-public addresses.

    Extraction logic lives in article_fetch.fetch_and_extract, shared with the
    MCP `fetch_article` tool so the SSRF-sensitive path has one implementation.
    """
    try:
        return await fetch_and_extract(body.url)
    except ArticleFetchError as exc:
        return JSONResponse({"detail": exc.detail}, status_code=exc.status)


_GAMMA_BASE = "https://gamma-api.polymarket.com"
_GAMMA_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; TruthMachine/1.0)"}
_ID_RE = re.compile(r"^\d+(,\d+)*$")


@app.get("/pm/markets", tags=["Polymarket"])
@limiter.limit("30/minute")
async def pm_markets(
    request: Request,
    id: str = Query(..., description="Comma-separated Gamma market IDs"),
):
    """
    Proxy for Polymarket Gamma API — returns live market data with CORS headers.
    Gamma API does not send Access-Control-Allow-Origin, so browser fetches from
    GitHub Pages are blocked; this endpoint forwards the request server-side.
    """
    if not _ID_RE.match(id):
        return JSONResponse({"detail": "id must be comma-separated integers"}, status_code=422)
    ids = id.split(",")
    if len(ids) > 50:
        return JSONResponse({"detail": "max 50 ids per request"}, status_code=422)

    # Gamma requires repeated params: ?id=111&id=222 (not comma-separated)
    params = [("id", i) for i in ids] + [("limit", str(len(ids)))]
    url = f"{_GAMMA_BASE}/markets"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, params=params, headers=_GAMMA_HEADERS)
            resp.raise_for_status()
        return JSONResponse(resp.json())
    except httpx.HTTPStatusError as exc:
        # Surface upstream failures as 502 rather than an opaque 500 — this proxy
        # is the browser's only path to Gamma.
        return JSONResponse(
            {"detail": f"Upstream Polymarket error: {exc.response.status_code}"},
            status_code=502,
        )
    except httpx.HTTPError:
        return JSONResponse({"detail": "Upstream Polymarket request failed"}, status_code=504)


# ── MCP server mount (only when Cognito OAuth is configured) ─────────────────
# The MCP protocol endpoint is mounted at /mcp. Its OAuth 2.0 protected-resource
# metadata (RFC 9728) must be served at the ORIGIN root so clients can discover
# the Authorization Server from the bare resource URL — mounting the whole
# sub-app would bury it under /mcp/.well-known/…, so the metadata routes are
# added to the main app directly while only the protocol endpoint is mounted.
if _mcp is not None:
    from mcp.server.auth.routes import create_protected_resource_routes

    # Which authorization server the protected-resource metadata points clients
    # at. With the DCR façade configured (human Claude-connector login), the
    # Oracle origin advertises itself as the AS so it can inject a
    # registration_endpoint Cognito lacks (see mcp_dcr.py); otherwise the metadata
    # points straight at the Cognito issuer (the M2M path needs no registration).
    if settings.dcr_enabled:
        from .mcp_dcr import create_dcr_routes

        _authorization_servers = [settings.mcp_as_issuer]
        for _route in create_dcr_routes(settings):
            app.router.routes.append(_route)
        logger.info("MCP DCR façade enabled (AS metadata + /register at origin)")
    else:
        _authorization_servers = [settings.cognito_issuer]

    for _route in create_protected_resource_routes(
        resource_url=settings.mcp_resource_url,
        authorization_servers=_authorization_servers,
        scopes_supported=[SCOPE_READ, SCOPE_FORECAST],
    ):
        app.router.routes.append(_route)
    app.mount("/mcp", _mcp.streamable_http_app())
    logger.info("Mounted MCP server at /mcp (OAuth RS via Cognito)")
