import asyncio
import ipaddress
import logging
import re
import socket
from contextlib import asynccontextmanager
from urllib.parse import urlparse

import httpx
from fastapi import Depends, FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from slowapi.errors import RateLimitExceeded

from .auth import verify_api_key
from .bayesoracle import compute_nodes
from .cache import forecast_cache
from .config import settings
from .forecaster import run_forecast
from .leaderboard import background_refresh_loop, get_leaderboard_data, leaderboard_size, refresh_cache
from tm.config import settings as _pipeline_settings
from tm.llm import complete_text_once
from .limiter import limiter
from .models import ForecastRequest, ForecastResponse, FetchUrlRequest, FetchUrlResponse, LlmRequest, LlmResponse, SearchRequest, SearchResponse, SearchHealthResponse
from .searcher import run_search, run_search_health

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    path = settings.resolved_leaderboard_path
    await refresh_cache(path)
    logger.info("Oracle API starting — leaderboard: %d sources, port: %d", leaderboard_size(), settings.port)
    refresh_task = asyncio.create_task(
        background_refresh_loop(path, settings.leaderboard_refresh_seconds)
    )
    yield
    # Shutdown
    refresh_task.cancel()
    try:
        await refresh_task
    except asyncio.CancelledError:
        pass
    logger.info("Oracle API shut down")


_CORS_ORIGIN = "https://komapc.github.io"
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
    version="0.1.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)

# Allow oracle-test.html on GitHub Pages to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://komapc.github.io",
        "https://bayes.daatan.com",
        "http://localhost:*",
        "http://127.0.0.1:*",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "x-api-key"],
)


@app.get("/", include_in_schema=False)
async def root():
    """Redirect to the interactive test console."""
    return RedirectResponse("https://komapc.github.io/retro/oracle-test.html")


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
    _: None = Depends(verify_api_key),
):
    """
    Return BayesOracle probabilities for the Israeli-politics DAG.

    Each node has a prior; children are computed from their parents via a
    fitted-intercept logistic CPT (exact enumeration over parent states), so
    baseline reproduces the priors and mutually-exclusive outcome groups stay on
    the probability simplex.  The graph lives in ``bayesoracle/graph_political.json``.
    Supply ``observations`` to lock specific nodes and see how the rest shifts.
    """
    obs: dict[str, float] = {}
    if observations:
        for part in observations.split(","):
            part = part.strip()
            if "=" not in part:
                continue
            node_id, _, val = part.partition("=")
            try:
                obs[node_id.strip().upper()] = float(val.strip())
            except ValueError:
                pass
    return {"nodes": compute_nodes(obs or None)}


@app.get("/leaderboard", tags=["Meta"])
async def leaderboard(_: None = Depends(verify_api_key)):
    """
    Return the live source credibility leaderboard, sorted by skill conservative score (μ − 3σ).
    Refreshed every N seconds from leaderboard.json (no restart required).
    """
    return {"sources": get_leaderboard_data(), "count": leaderboard_size()}


@app.get("/health", tags=["Meta"])
async def health():
    """Liveness probe — no auth required."""
    return {
        "status": "ok",
        "version": app.version,
        "leaderboard_sources": leaderboard_size(),
        "cache": {
            "enabled": forecast_cache.enabled,
            **forecast_cache.stats().as_dict(),
        },
    }


@app.post("/forecast", response_model=ForecastResponse, tags=["Forecast"])
@limiter.limit("10/minute")
async def forecast(
    request: Request,  # required by slowapi
    body: ForecastRequest,
    _: None = Depends(verify_api_key),
):
    """
    Given a binary question, return a calibrated probability distribution.

    The `mean` field is in stance space [-1, 1].
    Convert to probability [0, 1] with: `p = (mean + 1) / 2`
    """
    return await run_forecast(body)


@app.post("/search", response_model=SearchResponse, tags=["Search"])
@limiter.limit("60/minute")
async def search(
    request: Request,  # required by slowapi
    body: SearchRequest,
    _: None = Depends(verify_api_key),
):
    """
    Search for news articles using the full provider fallback chain.

    Tries: SerpAPI → Serper → Brave → BrightData → Nimbleway → ScrapingBee → DDG.
    DDG is skipped when the service is running on EC2.
    """
    return await run_search(body)


@app.get("/search/health", response_model=SearchHealthResponse, tags=["Search"])
async def search_health(_: None = Depends(verify_api_key)):
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
    _: None = Depends(verify_api_key),
):
    """
    Proxy LLM calls to Bedrock (via tm.llm → litellm) using the server's AWS creds.
    Accepts model + messages, returns the assistant's text content. ``model`` is a
    litellm ID and defaults to the server's configured Bedrock model; callers may
    override it (e.g. bedrock/amazon.nova-micro-v1:0).

    Uses the non-retrying complete_text_once: this is an interactive endpoint and
    daatan's caller times out at 60s, so we don't want the [30,60,120] backoff.
    """
    model = body.model or _pipeline_settings.extractor_model
    messages = [{"role": m.role, "content": m.content} for m in body.messages]
    try:
        content = await complete_text_once(
            model,
            messages=messages,
            max_tokens=1024,
            temperature=body.temperature,
            timeout=55,
        )
    except Exception as exc:
        logger.warning("llm proxy failed model=%s err=%s", model, exc)
        return JSONResponse({"detail": f"LLM call failed: {exc}"}, status_code=502)
    return LlmResponse(content=content, model=model)


def _is_safe_url(url: str) -> bool:
    """SSRF guard for the fetch proxy.

    Accept only http(s) URLs whose host resolves entirely to public IPs.
    Rejects loopback / private (RFC1918) / link-local (incl. the cloud
    metadata endpoint 169.254.169.254) / reserved / multicast addresses.
    All resolved addresses must be public — guards basic DNS rebinding.
    """
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    if parsed.scheme not in ("http", "https"):
        return False
    host = parsed.hostname
    if not host:
        return False
    try:
        infos = socket.getaddrinfo(host, None)
    except (socket.gaierror, UnicodeError, ValueError):
        return False
    if not infos:
        return False
    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0])
        except ValueError:
            return False
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            return False
    return True


@app.post("/fetch-url", response_model=FetchUrlResponse, tags=["IBI"])
@limiter.limit("30/minute")
async def fetch_url(request: Request, body: FetchUrlRequest):
    """
    Fetch an article URL and extract its text, title, and publication date.
    Uses trafilatura. Intentionally public (no x-api-key) so the IBI tool works
    without a key, but rate-limited and SSRF-guarded: rejects non-http(s) schemes
    and hosts that resolve to non-public addresses.
    """
    if not _is_safe_url(body.url):
        return JSONResponse(
            {"detail": "URL must be a public http(s) address"}, status_code=422
        )
    import trafilatura
    downloaded = trafilatura.fetch_url(body.url)
    if not downloaded:
        return JSONResponse({"detail": "Could not fetch URL"}, status_code=422)
    metadata = trafilatura.extract_metadata(downloaded)
    text = trafilatura.extract(downloaded, include_comments=False, include_tables=False) or ""
    date_str: str | None = None
    if metadata and metadata.date:
        # trafilatura returns date as string already
        date_str = str(metadata.date)[:10]
    return FetchUrlResponse(
        text=text,
        title=metadata.title if metadata else None,
        date=date_str,
        source=metadata.sitename if metadata else None,
    )


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
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(url, params=params, headers=_GAMMA_HEADERS)
        resp.raise_for_status()
    return JSONResponse(resp.json())
