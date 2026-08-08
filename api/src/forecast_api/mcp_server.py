"""MCP server for the Oracle — exposes forecasting + Polymarket tools to agents.

Mounted at /mcp by main.py (only when Cognito OAuth config is present; see
config.mcp_enabled). Tools call the Oracle's internal service functions in
process — no self-HTTP. Transport is stateless streamable-HTTP with JSON
responses, required because the API runs under gunicorn with multiple workers
and no sticky sessions.

Auth: OAuth 2.1 Resource Server (mcp_auth.CognitoTokenVerifier). The transport
enforces the global `oracle-mcp/read` floor; the two expensive tools additionally
assert `oracle-mcp/forecast` via require_scope().
"""

import logging
from typing import Optional

from mcp.server.fastmcp import FastMCP
from mcp.server.auth.settings import AuthSettings
from mcp.server.transport_security import TransportSecuritySettings

from tm.scorer import stance_to_prob

from .article_fetch import ArticleFetchError, fetch_and_extract
from .bayesoracle import compute_nodes, parse_node_observations
from .config import settings
from .forecaster import run_forecast
from .leaderboard import get_leaderboard_data, leaderboard_size, leaderboard_snapshot_date
from .mcp_auth import SCOPE_FORECAST, SCOPE_READ, CognitoTokenVerifier, require_scope
from .mcp_limiter import FETCH_LIMIT, FORECAST_LIMIT, MARKET_LIMIT, SEARCH_LIMIT, enforce
from .models import ForecastRequest, SearchRequest
from .polymarket_live import current_yes_price, is_binary_yesno, resolve_market, summarize_market
from .searcher import run_search

logger = logging.getLogger(__name__)

_NOT_ADVICE = "Informational only — not financial advice. This tool never places orders."


# ── response shaping ────────────────────────────────────────────────────────

def _source_brief(s) -> dict:
    return {
        "source_name": s.source_name,
        "url": s.url,
        "stance": round(s.stance, 3),
        "credibility_weight": round(s.credibility_weight, 3),
        "published_date": s.published_date,
    }


def _prob_ci(resp) -> list:
    """95% CI converted from stance space [-1,1] to probability space [0,1]."""
    return [round(stance_to_prob(resp.ci_low), 4), round(stance_to_prob(resp.ci_high), 4)]


def _confidence(resp) -> str:
    if resp.settled:
        return "high"
    lo, hi = _prob_ci(resp)
    width = hi - lo
    if resp.articles_used >= 5 and width <= 0.30:
        return "high"
    if width <= 0.50:
        return "medium"
    return "low"


def _forecast_payload(resp) -> dict:
    prob = None if resp.insufficient_data else round(stance_to_prob(resp.mean), 4)
    return {
        "question": resp.question,
        # Headline YES probability [0,1]. None when insufficient_data — render
        # "couldn't answer", NOT 0%. mean_stance is the raw stance space [-1,1].
        "probability": prob,
        "mean_stance": round(resp.mean, 4),
        "ci": _prob_ci(resp),
        "confidence": _confidence(resp),
        "articles_used": resp.articles_used,
        "settled": resp.settled,
        "insufficient_data": resp.insufficient_data,
        "reason": resp.reason,
        "provider": resp.provider,
        "sources": [_source_brief(s) for s in resp.sources],
    }


# ── tool implementations (registered onto the FastMCP instance in build_mcp) ──

async def forecast(question: str, max_articles: Optional[int] = None) -> dict:
    """Get the Oracle's calibrated probability for a binary (yes/no) question.

    Searches recent news, weights sources by historical credibility, and returns
    a probability in [0,1] under `probability` (the headline field). `mean_stance`
    is the raw internal stance in [-1,1]; `ci` is the 95% interval already in
    probability space. If `insufficient_data` is true, `probability` is null —
    say the Oracle couldn't answer rather than reporting 0%. This is the slow
    tool (a live news+LLM pipeline; tens of seconds, occasionally more).
    """
    require_scope(SCOPE_FORECAST)
    enforce(FORECAST_LIMIT, "forecast")
    resp = await run_forecast(ForecastRequest(question=question, max_articles=max_articles))
    return _forecast_payload(resp)


async def search_news(
    query: str,
    limit: int = 5,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> dict:
    """Search recent news via the Oracle's provider chain.

    Returns article title/url/snippet/source/date. `date_from`/`date_to` are
    ISO dates (YYYY-MM-DD). Evidence gathering — does not forecast.
    """
    enforce(SEARCH_LIMIT, "search_news")
    resp = await run_search(
        SearchRequest(query=query, limit=limit, date_from=date_from, date_to=date_to)
    )
    return resp.model_dump()


async def fetch_article(url: str) -> dict:
    """Fetch one article URL and extract clean text, title, date, and source.

    SSRF-guarded (public http(s) only). Use to read a specific source found via
    search_news.
    """
    enforce(FETCH_LIMIT, "fetch_article")
    try:
        return (await fetch_and_extract(url)).model_dump()
    except ArticleFetchError as exc:
        raise ValueError(exc.detail)


async def bayes_nodes(observations: str = "") -> dict:
    """BayesOracle probabilities for the Israeli-politics DAG.

    `observations` locks nodes: comma-separated `NODE=prob`, e.g.
    'ELECTIONS=0.95,TRUMP=0.70' (values clamped to [0,1]); unlisted nodes are
    computed from the graph. Useful for Israel-politics Polymarket markets.
    """
    obs = parse_node_observations(observations)
    return {"nodes": compute_nodes(obs or None)}


async def source_leaderboard() -> dict:
    """Source-credibility leaderboard (sorted by skill μ−3σ).

    A FROZEN snapshot (see `snapshot_date`), not a live board — this is the
    legacy vault, retained only as the fallback credibility source. It last
    changed on that date and only ever covered a handful of outlets, so an
    outlet's absence doesn't mean the Oracle distrusts it. Context for weighing
    the sources behind a forecast, not a claim about what the Oracle currently
    trusts.
    """
    return {
        "sources": get_leaderboard_data(),
        "count": leaderboard_size(),
        "snapshot_date": leaderboard_snapshot_date(),
        "note": (
            "Frozen vault snapshot, not regenerated since snapshot_date — retained as the "
            "credibility fallback, not a live ranking. See GET /leaderboard/resolution-shadow "
            "for the newer, resolution-informed board (small sample size)."
        ),
    }


async def polymarket_market(market: str) -> dict:
    """Look up a live Polymarket market (no forecast).

    `market` is a polymarket.com URL, an event/market slug, or a numeric Gamma
    market id. Returns question, outcomes, current prices (`yes_price` is the YES
    implied probability), volume, and end date.
    """
    enforce(MARKET_LIMIT, "polymarket_market")
    m = await resolve_market(market)
    if not m:
        raise ValueError(f"No Polymarket market found for {market!r}")
    return summarize_market(m)


async def polymarket_edge(market: str, edge_threshold: float = 0.05) -> dict:
    """Compare the Oracle's forecast to a Polymarket price and report the edge.

    `market` is a polymarket.com URL, slug, or Gamma id. Fetches the live YES
    price, runs the Oracle forecast on the market's question, and returns:
      - oracle_probability / market_probability (both [0,1] YES)
      - edge = oracle_probability − market_probability
      - suggested_side: "BUY YES" if edge > threshold, "BUY NO" if edge <
        −threshold, else "NO EDGE" ("NONE" when no estimate is possible)
      - confidence, ci, top_sources

    Only binary yes/no markets are supported. Slow (runs a full forecast).
    """
    require_scope(SCOPE_FORECAST)
    enforce(FORECAST_LIMIT, "polymarket_edge")
    m = await resolve_market(market)
    if not m:
        return {"market": market, "suggested_side": "NONE",
                "note": f"No Polymarket market found for {market!r}."}

    summary = summarize_market(m)
    question = summary.get("question") or ""
    base = {"market": summary.get("question"), "market_url": summary.get("url"), "note": _NOT_ADVICE}

    if not is_binary_yesno(m):
        return {**base, "suggested_side": "NONE",
                "note": f"{_NOT_ADVICE} Market is not a binary yes/no; edge not computed.",
                "market_data": summary}

    market_prob = current_yes_price(m)
    if market_prob is None:
        return {**base, "suggested_side": "NONE",
                "note": f"{_NOT_ADVICE} No current YES price available for this market.",
                "market_data": summary}

    resp = await run_forecast(ForecastRequest(question=question))
    if resp.insufficient_data:
        return {**base, "suggested_side": "NONE", "market_probability": round(market_prob, 4),
                "oracle_probability": None, "insufficient_data": True, "reason": resp.reason,
                "note": f"{_NOT_ADVICE} The Oracle couldn't form an estimate ({resp.reason})."}

    oracle_prob = stance_to_prob(resp.mean)
    edge = oracle_prob - market_prob
    if edge > edge_threshold:
        side = "BUY YES"
    elif edge < -edge_threshold:
        side = "BUY NO"
    else:
        side = "NO EDGE"

    return {
        **base,
        "oracle_probability": round(oracle_prob, 4),
        "market_probability": round(market_prob, 4),
        "edge": round(edge, 4),
        "edge_threshold": edge_threshold,
        "suggested_side": side,
        "confidence": _confidence(resp),
        "ci": _prob_ci(resp),
        "settled": resp.settled,
        "articles_used": resp.articles_used,
        "top_sources": [_source_brief(s) for s in resp.sources[:8]],
    }


_TOOLS = [
    forecast,
    search_news,
    fetch_article,
    bayes_nodes,
    source_leaderboard,
    polymarket_market,
    polymarket_edge,
]


def build_mcp() -> Optional[FastMCP]:
    """Construct the FastMCP server, or None when Cognito OAuth isn't configured.

    Returning None (config.mcp_enabled is false) makes main.py skip the /mcp
    mount, so a deploy without the Cognito env vars still boots the REST API.
    """
    if not settings.mcp_enabled:
        logger.info("MCP server disabled — set COGNITO_USER_POOL_ID to enable /mcp")
        return None

    verifier = CognitoTokenVerifier(
        issuer=settings.cognito_issuer,
        jwks_url=settings.cognito_jwks_url,
        allowed_client_ids=settings.cognito_allowed_client_id_set,
        resource_url=settings.mcp_resource_url,
    )
    mcp = FastMCP(
        "oracle",
        instructions=(
            "Tools from the TruthMachine Oracle: calibrated news-based forecasts, "
            "Polymarket edge analysis, news search, and source credibility."
        ),
        stateless_http=True,
        json_response=True,
        streamable_http_path="/",
        token_verifier=verifier,
        # Without this, the SDK auto-restricts the transport's DNS-rebinding guard
        # to localhost (the app binds 127.0.0.1), so authenticated calls arriving
        # through nginx as Host: oracle.daatan.com get a 421. Allow the real host.
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=settings.mcp_allowed_hosts,
            allowed_origins=settings.mcp_allowed_origins,
        ),
        auth=AuthSettings(
            issuer_url=settings.cognito_issuer,
            resource_server_url=settings.mcp_resource_url,
            required_scopes=[SCOPE_READ],
        ),
    )
    for fn in _TOOLS:
        mcp.tool()(fn)
    logger.info("MCP server enabled with %d tools at %s", len(_TOOLS), settings.mcp_resource_url)
    return mcp
