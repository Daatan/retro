"""
Core forecast logic — Phase 2: live pipeline integration.

Flow:
  1. search_articles(question) — Serper.dev → Brave → DDG fallback
  2. For each article (in parallel): gatekeeper → extractor
  3. Weight each source by credibility from leaderboard
  4. Aggregate: weighted mean stance + 95% CI → return ForecastResponse
"""
import asyncio
import hashlib
import logging
import math
import re
import time
from datetime import datetime
from typing import Optional
from urllib.parse import urlparse

import httpx
import litellm
import trafilatura

from tm.gatekeeper import check_is_prediction, PROMPT as GATEKEEPER_PROMPT
from tm.extractor import extract_predictions, PROMPT as EXTRACTOR_PROMPT
from tm.web_search import search_articles, SearchResult, get_last_search_provider, get_last_search_provider_chain
from tm.config import settings as _pipeline_settings

from .cache import forecast_cache, search_cache
from .leaderboard import get_credibility_weight
from .models import ArticleDebug, ArticleInput, DebugInfo, ForecastRequest, ForecastResponse, SourceSignal
from .config import settings

logger = logging.getLogger(__name__)

# In-flight deduplication: cache_key → asyncio.Event.
# When a second request arrives for a key that's already being processed,
# it waits on this event instead of launching a duplicate pipeline.
_inflight: dict[str, asyncio.Event] = {}


def _question_hash(question: str) -> str:
    """Short, non-reversible question tag used to correlate log lines."""
    return hashlib.sha256(question.strip().casefold().encode("utf-8")).hexdigest()[:12]


def _log_phase(
    phase: str,
    duration_ms: float,
    *,
    question: str,
    **extra: object,
) -> None:
    """
    Emit a structured single-line log for one phase of a forecast call.

    The line is key=value formatted so it is readable by humans and greppable
    by log aggregators (``journalctl``/CloudWatch) without a dedicated parser.
    Correlate related phases with ``question_hash``.
    """
    fields = {
        "event": "forecast_phase",
        "phase": phase,
        "duration_ms": round(duration_ms, 1),
        "question_hash": _question_hash(question),
        **extra,
    }
    logger.info(" ".join(f"{k}={v}" for k, v in fields.items()))

# Domain → leaderboard source_id mapping
_DOMAIN_MAP: dict[str, str] = {
    "timesofisrael.com": "toi",
    "haaretz.com": "haaretz",
    "jpost.com": "jpost",
    "ynetnews.com": "ynet",
    "ynet.co.il": "ynet",
    "israelhayom.com": "israel_hayom",
    "israelhayom.co.il": "israel_hayom",
    "globes.co.il": "globes",
    "en.globes.co.il": "globes",
    "maariv.co.il": "maariv",
    "calcalist.co.il": "calcalist",
    "walla.co.il": "walla",
    "news.walla.co.il": "walla",
    "mako.co.il": "mako",
    "kan.org.il": "kan",
    "13tv.co.il": "channel13",
    "reuters.com": "reuters",
    "bbc.com": "bbc",
    "aljazeera.com": "aljazeera",
    "cnn.com": "cnn",
    "bloomberg.com": "bloomberg",
    "wsj.com": "wsj",
    "ft.com": "ft",
    "apnews.com": "ap",
}


def _source_id_from_url(url: str) -> str:
    domain = re.sub(r"^www\.", "", urlparse(url).netloc)
    for key, sid in _DOMAIN_MAP.items():
        if domain == key or domain.endswith("." + key):
            return sid
    return domain  # fallback: raw domain as id


# Minimum extracted length before we trust the body over title+snippet.
# Real news leads are always >> 400 chars; values below this are almost
# always 404 stubs, paywall walls, or cookie-wall interstitials.
_MIN_ARTICLE_CHARS = 400

# Paywall / registration-wall phrases. Match is case-insensitive and
# substring-based. Only considered when extracted content is short — real
# articles may quote these phrases without being paywalled.
_PAYWALL_MARKERS: tuple[str, ...] = (
    "subscribe to continue",
    "subscribe to read",
    "sign in to continue",
    "sign in to read",
    "create a free account",
    "create an account to",
    "register to read",
    "this article is for subscribers",
    "log in to continue",
    "become a subscriber",
)


def _looks_like_paywall(text: str) -> bool:
    """True when a short body contains a subscription/registration CTA.

    We deliberately only check *short* bodies: a 5000-char article that
    merely quotes "subscribe to read" inside its prose is not a paywall.
    """
    low = text.lower()
    return any(marker in low for marker in _PAYWALL_MARKERS)


async def _distill_query(question: str) -> str:
    """Convert a long resolution-criterion question into 4-6 search keywords.

    Called only when verbatim search returns 0 results — adds ~200ms latency
    and a cheap Nova Micro call to unlock niche/Polymarket-style questions.
    Returns the original question on any error (preserves existing behaviour).
    """
    prompt = (
        "Extract 4-6 concise search keywords from this forecasting question. "
        "Output ONLY the keywords as a single line, space-separated. "
        "No explanations, no punctuation, no quotes.\n\n"
        f"Question: {question}"
    )
    try:
        kwargs: dict = dict(
            model=_pipeline_settings.gatekeeper_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=40,
            timeout=20,
        )
        if _pipeline_settings.model_api_base:
            kwargs["api_base"] = _pipeline_settings.model_api_base
            kwargs["api_key"] = _pipeline_settings.model_api_key
        if _pipeline_settings.aws_region:
            kwargs["aws_region_name"] = _pipeline_settings.aws_region
        resp = await litellm.acompletion(**kwargs)
        keywords = resp.choices[0].message.content.strip()
        if keywords:
            logger.info("query_distilled original=%r distilled=%r", question[:60], keywords)
            return keywords
    except Exception as exc:
        logger.warning("query distillation failed: %s", exc)
    return question


def _fetch_article_text(url: str, fallback: str) -> str:
    """Fetch full article body with trafilatura; return fallback on error.

    Upgraded from a naive ``httpx.get(...).text`` pipeline:

    - Non-2xx responses (404/403/paywall redirects) used to silently feed
      the gatekeeper an HTML error page. We now detect them via
      ``raise_for_status`` and fall back to title+snippet immediately.
    - Paywall / registration-wall stubs that trafilatura faithfully
      extracts (e.g. "Subscribe to read the full article…") previously
      passed the ``len(extracted) > len(fallback)`` check and became the
      "article content". We reject short extractions containing a known
      paywall marker.
    - Each fetch now logs its outcome at INFO so we can measure from
      production how often paywalls / 404s cost us an article.
    """
    outcome = "ok"
    status: int | None = None
    extracted_len = 0
    try:
        resp = httpx.get(
            url,
            timeout=6.0,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; TruthMachine/1.0)"},
        )
        status = resp.status_code
        resp.raise_for_status()
        extracted = trafilatura.extract(resp.text, include_comments=False, include_tables=False)
        if not extracted:
            outcome = "trafilatura_empty"
        else:
            extracted_len = len(extracted)
            if extracted_len < _MIN_ARTICLE_CHARS and _looks_like_paywall(extracted):
                outcome = "paywall_suspected"
            elif extracted_len <= len(fallback):
                # Fallback (title+snippet) is richer than the body — treat as
                # not helpful, keep the fallback. Common for link-only pages
                # and very short briefs.
                outcome = "extracted_too_short"
            else:
                logger.info(
                    "event=article_fetch outcome=ok url=%s status=%d extracted_len=%d",
                    url, status, extracted_len,
                )
                return extracted
    except httpx.HTTPStatusError as exc:
        outcome = "http_error"
        status = exc.response.status_code
    except Exception as exc:
        outcome = "fetch_error"
        logger.debug("Article fetch failed for %s: %s", url, exc)
    logger.info(
        "event=article_fetch outcome=%s url=%s status=%s extracted_len=%d using=fallback",
        outcome, url, status, extracted_len,
    )
    return fallback


def _truncate_article(text: str, max_chars: int) -> str:
    """
    Cap article body at ``max_chars``.

    News leads carry the thesis in the first ~2–3k chars; the remainder
    mostly burns LLM latency + tokens without improving stance extraction.
    Returns the original string untouched when already under the cap or when
    ``max_chars <= 0`` (truncation disabled).
    """
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars]


async def _process_article(
    result: SearchResult,
    question: str,
    *,
    max_article_chars: int,
    timings: list[dict],
    article_debugs: list[ArticleDebug],
) -> tuple[SearchResult, list] | None:
    """
    Run gatekeeper + extractor for one article.
    Fetches full article text via trafilatura; falls back to title+snippet.
    Appends per-phase durations to ``timings`` and an ArticleDebug to ``article_debugs``.
    """
    # Fallback text = title + snippet
    parts = [p for p in [result.title, result.snippet] if p and p.strip()]
    fallback = " — ".join(parts)
    if not fallback or len(fallback) < 20:
        return None

    # Use caller-supplied text if available; otherwise fetch via trafilatura.
    fetch_start = time.perf_counter()
    if result._prefetched_text:
        text = result._prefetched_text
        logger.info("event=article_fetch outcome=prefetched url=%s", result.url)
    else:
        text = await asyncio.to_thread(_fetch_article_text, result.url, fallback)
    fetch_ms = (time.perf_counter() - fetch_start) * 1000
    if not text:
        timings.append({"url": result.url, "fetch_ms": fetch_ms, "outcome": "empty_text"})
        article_debugs.append(ArticleDebug(url=result.url, outcome="empty_text", fetch_ms=round(fetch_ms, 1)))
        return None

    text = _truncate_article(text, max_article_chars)

    source_name = result.source or _source_id_from_url(result.url)
    article_date = result.published_date or datetime.now().strftime("%Y-%m-%d")

    gate_start = time.perf_counter()
    try:
        gate, gate_usage = await check_is_prediction(
            article_text=text,
            source_name=source_name,
            article_date=article_date,
            event_name=question,
        )
    except Exception as exc:
        logger.warning("Gatekeeper failed for %s: %s", result.url, exc)
        gate_ms = (time.perf_counter() - gate_start) * 1000
        timings.append({
            "url": result.url, "fetch_ms": fetch_ms,
            "gate_ms": gate_ms, "outcome": "gate_error",
        })
        article_debugs.append(ArticleDebug(
            url=result.url, outcome="gate_error",
            fetch_ms=round(fetch_ms, 1), gate_ms=round(gate_ms, 1),
        ))
        return None
    gate_ms = (time.perf_counter() - gate_start) * 1000

    if not gate.is_prediction:
        logger.info(
            "event=article_outcome outcome=gate_rejected url=%s reason=%r",
            result.url,
            (gate.reason or "")[:200],
        )
        timings.append({
            "url": result.url, "fetch_ms": fetch_ms, "gate_ms": gate_ms,
            "outcome": "gate_rejected",
        })
        article_debugs.append(ArticleDebug(
            url=result.url, outcome="gate_rejected",
            gate_passed=False,
            gate_reason=gate.reason,
            gate_prediction_count_estimate=gate.prediction_count_estimate,
            gate_tokens=gate_usage.get("total_tokens"),
            fetch_ms=round(fetch_ms, 1), gate_ms=round(gate_ms, 1),
        ))
        return None

    extract_start = time.perf_counter()
    try:
        extraction, extract_usage = await extract_predictions(
            article_text=text,
            source_name=source_name,
            article_date=article_date,
            event_name=question,
            event_description=question,
        )
    except Exception as exc:
        logger.warning("Extractor failed for %s: %s", result.url, exc)
        extract_ms = (time.perf_counter() - extract_start) * 1000
        timings.append({
            "url": result.url, "fetch_ms": fetch_ms, "gate_ms": gate_ms,
            "extract_ms": extract_ms, "outcome": "extract_error",
        })
        article_debugs.append(ArticleDebug(
            url=result.url, outcome="extract_error",
            gate_passed=True,
            gate_reason=gate.reason,
            gate_prediction_count_estimate=gate.prediction_count_estimate,
            gate_tokens=gate_usage.get("total_tokens"),
            fetch_ms=round(fetch_ms, 1), gate_ms=round(gate_ms, 1), extract_ms=round(extract_ms, 1),
        ))
        return None
    extract_ms = (time.perf_counter() - extract_start) * 1000

    if not extraction.predictions:
        timings.append({
            "url": result.url, "fetch_ms": fetch_ms, "gate_ms": gate_ms,
            "extract_ms": extract_ms, "outcome": "no_predictions",
        })
        article_debugs.append(ArticleDebug(
            url=result.url, outcome="no_predictions",
            gate_passed=True,
            gate_reason=gate.reason,
            gate_prediction_count_estimate=gate.prediction_count_estimate,
            gate_tokens=gate_usage.get("total_tokens"),
            extract_tokens=extract_usage.get("total_tokens"),
            total_tokens=(gate_usage.get("total_tokens", 0) + extract_usage.get("total_tokens", 0)) or None,
            fetch_ms=round(fetch_ms, 1), gate_ms=round(gate_ms, 1), extract_ms=round(extract_ms, 1),
        ))
        return None

    avg_stance = sum(p.stance for p in extraction.predictions) / len(extraction.predictions)
    avg_certainty = sum(p.certainty for p in extraction.predictions) / len(extraction.predictions)
    first_claim = (extraction.predictions[0].claim or "")[:160]
    logger.info(
        "event=article_outcome outcome=ok url=%s stance=%.3f certainty=%.3f n_preds=%d claim=%r",
        result.url, avg_stance, avg_certainty, len(extraction.predictions), first_claim,
    )
    timings.append({
        "url": result.url, "fetch_ms": fetch_ms, "gate_ms": gate_ms,
        "extract_ms": extract_ms, "outcome": "ok",
    })
    gate_tok = gate_usage.get("total_tokens", 0)
    ext_tok = extract_usage.get("total_tokens", 0)
    article_debugs.append(ArticleDebug(
        url=result.url, outcome="ok",
        gate_passed=True,
        gate_reason=gate.reason,
        gate_prediction_count_estimate=gate.prediction_count_estimate,
        gate_tokens=gate_tok or None,
        extract_tokens=ext_tok or None,
        total_tokens=(gate_tok + ext_tok) or None,
        fetch_ms=round(fetch_ms, 1), gate_ms=round(gate_ms, 1), extract_ms=round(extract_ms, 1),
    ))
    return (result, extraction.predictions)


async def run_forecast(req: ForecastRequest) -> ForecastResponse:
    limit = req.max_articles or settings.max_articles
    total_start = time.perf_counter()

    # Step 0a: forecast cache lookup.
    # When caller supplies articles, key includes an MD5 of sorted URLs so
    # two calls with the same question but different article sets don't collide.
    articles_hash: Optional[str] = None
    if req.articles:
        articles_hash = hashlib.md5(
            "|".join(sorted(a.url for a in req.articles)).encode()
        ).hexdigest()[:12]
    cache_key = forecast_cache.make_key(req.question, req.max_articles, articles_hash)
    cached = forecast_cache.get(cache_key)
    if cached is not None:
        _log_phase(
            "cache_hit",
            (time.perf_counter() - total_start) * 1000,
            question=req.question,
            articles_used=cached.articles_used,
        )
        return cached

    # Step 0b: in-flight deduplication.
    # If another coroutine is already processing this exact key, wait for it
    # and return its result rather than launching a duplicate pipeline.
    if cache_key in _inflight:
        logger.info(
            "event=inflight_wait question_hash=%s", _question_hash(req.question)
        )
        await _inflight[cache_key].wait()
        result = forecast_cache.get(cache_key)
        if result is not None:
            return result
        return _empty_response(req.question)

    event = asyncio.Event()
    _inflight[cache_key] = event

    try:
        return await _run_forecast_inner(req, cache_key, limit, total_start)
    finally:
        event.set()
        _inflight.pop(cache_key, None)


async def _run_forecast_inner(
    req: ForecastRequest,
    cache_key: str,
    limit: int,
    total_start: float,
) -> ForecastResponse:
    # Step 1: search (skipped when caller provides articles directly)
    search_start = time.perf_counter()
    search_provider: str
    provider_chain: list[str]
    # Strip leading emoji/markers the frontend may prepend (e.g. "🤖 Question…")
    # before any provider sees the query; supplementary-plane chars (U+10000+) cover
    # virtually all emoji while leaving ordinary punctuation and non-ASCII text intact.
    search_query = re.sub(r'^[\U00010000-\U0010FFFF\s]+', '', req.question).strip()
    if search_query != req.question:
        logger.info("Stripped leading markers from search query: %r → %r", req.question[:40], search_query[:40])
    if req.articles:
        search_results: list[SearchResult] = [
            SearchResult(
                title=a.title,
                url=a.url,
                snippet=a.snippet,
                source=a.source,
                published_date=a.published_date,
                _prefetched_text=a.text,
            )
            for a in req.articles
        ]
        search_provider = "caller"
        provider_chain = ["caller"]
        _log_phase(
            "search",
            (time.perf_counter() - search_start) * 1000,
            question=req.question,
            results=len(search_results),
            provider=search_provider,
        )
    else:
        # Check search cache before hitting provider APIs.
        search_key = search_cache.make_key(req.question, limit)
        cached_results = search_cache.get(search_key)
        if cached_results is not None:
            search_results = cached_results
            search_provider = "search_cache"
            provider_chain = ["search_cache"]
            _log_phase(
                "search",
                (time.perf_counter() - search_start) * 1000,
                question=req.question,
                results=len(search_results),
                provider=search_provider,
            )
        else:
            try:
                search_results = await asyncio.to_thread(
                    search_articles, search_query, limit
                )
            except Exception as exc:
                logger.error("Search failed: %s", exc)
                search_results = []
            # If verbatim question found nothing, try distilled keywords.
            if not search_results:
                distilled = await _distill_query(req.question)
                if distilled != req.question:
                    search_query = distilled
                    try:
                        search_results = await asyncio.to_thread(
                            search_articles, search_query, limit
                        )
                    except Exception as exc:
                        logger.error("Search (distilled) failed: %s", exc)
                        search_results = []
            search_provider = get_last_search_provider()
            provider_chain = get_last_search_provider_chain()
            _log_phase(
                "search",
                (time.perf_counter() - search_start) * 1000,
                question=req.question,
                results=len(search_results),
                provider=search_provider,
                distilled=search_query != req.question,
            )
            search_cache.set(search_key, search_results)

    if not search_results:
        logger.warning("No articles found for: %s", req.question[:80])
        _log_phase(
            "total",
            (time.perf_counter() - total_start) * 1000,
            question=req.question,
            articles_used=0,
            outcome="no_search_results",
        )
        resp = _empty_response(req.question)
        if req.debug:
            resp.debug = DebugInfo(
                search_query=search_query,
                search_provider=search_provider,
                search_provider_chain=provider_chain,
                gatekeeper_model=settings.gatekeeper_model,
                extractor_model=settings.extractor_model,
                articles_fetched=0,
                articles_gate_passed=0,
                articles_extracted=0,
                total_prompt_tokens=0,
                total_completion_tokens=0,
                total_tokens=0,
                per_article=[],
                gatekeeper_prompt=GATEKEEPER_PROMPT,
                extractor_prompt=EXTRACTOR_PROMPT,
            )
        return resp

    # Log the URLs that came back so we can trace exactly which articles each
    # downstream phase saw. The search query == the question (we don't rewrite
    # it), so question_hash + this line is enough to reconstruct the call.
    logger.info(
        "event=search_results count=%d question=%r urls=%s",
        len(search_results),
        req.question[:120],
        [r.url for r in search_results],
    )

    # Step 2: gatekeeper + extractor in parallel
    process_start = time.perf_counter()
    timings: list[dict] = []
    article_debugs: list[ArticleDebug] = []
    outcomes = await asyncio.gather(
        *[
            _process_article(
                r,
                req.question,
                max_article_chars=settings.max_article_chars,
                timings=timings,
                article_debugs=article_debugs,
            )
            for r in search_results
        ],
        return_exceptions=True,
    )
    process_ms = (time.perf_counter() - process_start) * 1000
    _log_phase(
        "articles_processed",
        process_ms,
        question=req.question,
        articles=len(search_results),
        ok=sum(1 for t in timings if t.get("outcome") == "ok"),
        avg_fetch_ms=_avg(timings, "fetch_ms"),
        avg_gate_ms=_avg(timings, "gate_ms"),
        avg_extract_ms=_avg(timings, "extract_ms"),
    )

    # Step 3: build per-source signals
    source_signals: list[SourceSignal] = []
    all_stances: list[float] = []
    all_weights: list[float] = []

    for result, outcome in zip(search_results, outcomes):
        if isinstance(outcome, Exception) or outcome is None:
            continue
        _, predictions = outcome

        source_id = _source_id_from_url(result.url)
        credibility = get_credibility_weight(source_id)
        avg_stance = sum(p.stance for p in predictions) / len(predictions)
        avg_certainty = sum(p.certainty for p in predictions) / len(predictions)
        weight = credibility * avg_certainty

        all_stances.append(avg_stance)
        all_weights.append(weight)

        source_signals.append(SourceSignal(
            source_id=source_id,
            source_name=result.source or source_id,
            url=result.url,
            stance=round(avg_stance, 3),
            certainty=round(avg_certainty, 3),
            credibility_weight=round(credibility, 3),
            claims=[p.claim for p in predictions if p.claim],
        ))

    if not all_stances:
        # Outcome histogram tells us *why* we got nothing — were articles
        # rejected by the gatekeeper, did extraction return empty, or did
        # fetch fail? Without this the warning is uninvestigatable.
        outcome_counts: dict[str, int] = {}
        for t in timings:
            key = str(t.get("outcome", "unknown"))
            outcome_counts[key] = outcome_counts.get(key, 0) + 1
        logger.warning(
            "No usable predictions extracted from %d articles (outcomes=%s)",
            len(search_results),
            outcome_counts,
        )
        _log_phase(
            "total",
            (time.perf_counter() - total_start) * 1000,
            question=req.question,
            articles_used=0,
            outcome="no_usable_predictions",
            **{f"n_{k}": v for k, v in outcome_counts.items()},
        )
        return _empty_response(req.question)

    # Step 4: weighted mean + 95% CI
    total_w = sum(all_weights)
    mean = sum(s * w for s, w in zip(all_stances, all_weights)) / total_w
    variance = sum(w * (s - mean) ** 2 for s, w in zip(all_stances, all_weights)) / total_w
    std = math.sqrt(variance)
    n = len(all_stances)
    sem = std / math.sqrt(n) if n > 1 else std
    ci_low = max(-1.0, mean - 1.96 * sem)
    ci_high = min(1.0, mean + 1.96 * sem)

    logger.info(
        "Forecast: mean=%.3f std=%.3f ci=[%.3f,%.3f] articles=%d",
        mean, std, ci_low, ci_high, n,
    )

    debug_info: Optional[DebugInfo] = None
    if req.debug:
        total_prompt_tok = sum(
            (d.gate_tokens or 0) + (d.extract_tokens or 0)
            for d in article_debugs
        )
        debug_info = DebugInfo(
            search_query=search_query,
            search_provider=search_provider,
            search_provider_chain=provider_chain,
            gatekeeper_model=settings.gatekeeper_model,
            extractor_model=settings.extractor_model,
            articles_fetched=len(search_results),
            articles_gate_passed=sum(1 for d in article_debugs if d.gate_passed),
            articles_extracted=n,
            total_prompt_tokens=sum(
                (d.gate_tokens or 0) + (d.extract_tokens or 0)
                for d in article_debugs
            ),
            total_completion_tokens=0,
            total_tokens=total_prompt_tok,
            per_article=article_debugs,
            gatekeeper_prompt=GATEKEEPER_PROMPT,
            extractor_prompt=EXTRACTOR_PROMPT,
        )

    response = ForecastResponse(
        question=req.question,
        mean=round(mean, 4),
        std=round(std, 4),
        ci_low=round(ci_low, 4),
        ci_high=round(ci_high, 4),
        articles_used=n,
        sources=source_signals,
        placeholder=False,
        debug=debug_info,
    )

    forecast_cache.set(cache_key, response)

    _log_phase(
        "total",
        (time.perf_counter() - total_start) * 1000,
        question=req.question,
        articles_used=n,
        outcome="ok",
    )

    return response


def _empty_response(question: str) -> ForecastResponse:
    """Return a maximally uncertain response when no usable articles are found."""
    return ForecastResponse(
        question=question,
        mean=0.0,
        std=0.0,
        ci_low=-0.2,
        ci_high=0.2,
        articles_used=0,
        sources=[],
        placeholder=True,
    )


def _avg(timings: list[dict], key: str) -> Optional[float]:
    """Mean of ``key`` across ``timings`` entries that carry it, rounded to 1 dp."""
    values = [t[key] for t in timings if key in t and t[key] is not None]
    if not values:
        return None
    return round(sum(values) / len(values), 1)
