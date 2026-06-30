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
import re
import time
from datetime import datetime
from typing import Optional
from urllib.parse import urlparse

import httpx
import trafilatura

from tm.gatekeeper import check_is_prediction, PROMPT as GATEKEEPER_PROMPT
from tm.extractor import extract_predictions, PROMPT as EXTRACTOR_PROMPT
from tm.web_search import search_articles, SearchResult, get_last_search_provider, get_last_search_provider_chain
from tm.config import settings as _pipeline_settings
from tm.llm import complete_text_once

from .aggregation import claim_weighted_stance, pool_sources, recency_weight
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
        # Non-retrying variant: this runs inside the latency-bounded /forecast
        # path, so it must not inherit complete_text's [30,60,120] backoff.
        keywords = (await complete_text_once(
            _pipeline_settings.gatekeeper_model,
            prompt,
            max_tokens=40,
            timeout=20,
        )).strip()
        if keywords:
            logger.info("query_distilled original=%r distilled=%r", question[:60], keywords)
            return keywords
    except Exception as exc:
        logger.warning("query distillation failed: %s", exc)
    return question


def _search_capturing(query: str, limit: int) -> tuple[list, str, list[str]]:
    """Run search_articles and capture the winning provider/chain *in the same
    thread*.

    ``get_last_search_provider()`` is thread-local; the forecaster runs search
    via ``asyncio.to_thread``, so reading the provider in the caller's thread
    always returned "none". Reading it here — inside the worker thread — fixes
    that, so ``debug.search_provider`` actually names the provider.
    """
    results = search_articles(query, limit)
    return results, get_last_search_provider(), get_last_search_provider_chain()


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


async def _process_article_bounded(
    result: SearchResult,
    question: str,
    *,
    max_article_chars: int,
    timings: list[dict],
    article_debugs: list[ArticleDebug],
    timeout_s: float,
) -> tuple[SearchResult, float, list] | None:
    """Run _process_article under a per-article wall-clock ceiling.

    Articles are processed in parallel, so one slow LLM call would otherwise
    stall the whole batch. On timeout we drop just this article (record it as a
    ``timeout`` outcome and return None) so the rest of the batch proceeds.
    """
    try:
        return await asyncio.wait_for(
            _process_article(
                result,
                question,
                max_article_chars=max_article_chars,
                timings=timings,
                article_debugs=article_debugs,
            ),
            timeout=timeout_s,
        )
    except asyncio.TimeoutError:
        logger.warning("event=article_timeout url=%s timeout_s=%s", result.url, timeout_s)
        timings.append({"url": result.url, "outcome": "timeout"})
        article_debugs.append(ArticleDebug(url=result.url, outcome="timeout"))
        return None


async def _process_article(
    result: SearchResult,
    question: str,
    *,
    max_article_chars: int,
    timings: list[dict],
    article_debugs: list[ArticleDebug],
) -> tuple[SearchResult, float, list] | None:
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

    # Certainty-weight the claims so a decisive claim isn't washed out by
    # tangential hedged ones (mirrors the cross-source weighting below).
    avg_stance = claim_weighted_stance(
        [p.stance for p in extraction.predictions],
        [p.certainty for p in extraction.predictions],
        [p.specificity for p in extraction.predictions],
    )
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
    return (result, gate.relevance_score, extraction.predictions)


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
        return _empty_response(req.question, reason="no_result")

    event = asyncio.Event()
    _inflight[cache_key] = event

    try:
        return await asyncio.wait_for(
            _run_forecast_inner(req, cache_key, limit, total_start),
            timeout=settings.forecast_timeout_seconds,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "event=forecast_timeout question_hash=%s timeout_s=%s",
            _question_hash(req.question),
            settings.forecast_timeout_seconds,
        )
        _log_phase(
            "total",
            (time.perf_counter() - total_start) * 1000,
            question=req.question,
            articles_used=0,
            outcome="timeout",
        )
        return _empty_response(req.question, reason="timeout")
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
    distilled_query: Optional[str] = None
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
            # Distill the natural-language question to keywords BEFORE searching.
            # The chain's keyword matchers (esp. GDELT, the usual winner) return
            # off-topic junk when fed a verbose question like "Will X happen by
            # 2027?"; distilling to e.g. "Russia Ukraine ceasefire" restores
            # relevance. _distill_query returns the original on any error.
            verbatim = search_query
            search_query = await _distill_query(verbatim)
            distilled = search_query != verbatim
            # Capture the distilled keywords now, before the verbatim fallback
            # below can overwrite search_query.
            distilled_query = search_query if distilled else None
            try:
                search_results, search_provider, provider_chain = await asyncio.to_thread(
                    _search_capturing, search_query, limit
                )
            except Exception as exc:
                logger.error("Search failed: %s", exc)
                search_results, search_provider, provider_chain = [], "none", []
            # Recall safety: if distilled keywords found nothing, retry verbatim.
            if not search_results and distilled:
                try:
                    search_results, search_provider, provider_chain = await asyncio.to_thread(
                        _search_capturing, verbatim, limit
                    )
                    search_query = verbatim
                except Exception as exc:
                    logger.error("Search (verbatim fallback) failed: %s", exc)
            _log_phase(
                "search",
                (time.perf_counter() - search_start) * 1000,
                question=req.question,
                results=len(search_results),
                provider=search_provider,
                distilled=distilled,
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
        resp = _empty_response(
            req.question,
            reason="no_search_results",
            articles_found=0,
            provider=search_provider,
            provider_chain=provider_chain,
            distilled_query=distilled_query,
        )
        if req.debug:
            resp.debug = DebugInfo(
                search_query=search_query,
                search_provider=search_provider,
                search_provider_chain=provider_chain,
                gatekeeper_model=_pipeline_settings.gatekeeper_model,
                extractor_model=_pipeline_settings.extractor_model,
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
            _process_article_bounded(
                r,
                req.question,
                max_article_chars=settings.max_article_chars,
                timings=timings,
                article_debugs=article_debugs,
                timeout_s=settings.per_article_timeout_seconds,
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

    # Step 3: build per-source signals.
    # Recency is measured against "now" so the latest reporting dominates as an
    # event resolves (stale pre-resolution coverage stops diluting the result).
    source_signals: list[SourceSignal] = []
    all_stances: list[float] = []
    all_weights: list[float] = []
    relevances: list[float] = []
    n_low_certainty = 0
    ref_date = datetime.now().strftime("%Y-%m-%d")

    for result, outcome in zip(search_results, outcomes):
        if isinstance(outcome, Exception):
            # Previously skipped silently — surface it so a systemic failure in
            # _process_article isn't invisible, and record it in the histogram.
            logger.warning("event=article_unhandled_error url=%s err=%r", result.url, outcome)
            timings.append({"url": result.url, "outcome": "unhandled_error"})
            continue
        if outcome is None:
            continue
        _, relevance, predictions = outcome

        source_id = _source_id_from_url(result.url)
        credibility = get_credibility_weight(source_id)
        # Layer A: certainty-weight the article's claims so a decisive claim
        # dominates tangential hedged ones instead of being washed out.
        avg_stance = claim_weighted_stance(
            [p.stance for p in predictions],
            [p.certainty for p in predictions],
            [p.specificity for p in predictions],
        )
        avg_certainty = sum(p.certainty for p in predictions) / len(predictions)
        # Certainty gate: drop sources whose claims are only hedged speculation
        # before they can pad the evidence mass or tug the pool. A pool of only such
        # sources then collapses to insufficient_data via the floors below. See
        # settings.certainty_floor (0.0 disables).
        if avg_certainty < settings.certainty_floor:
            n_low_certainty += 1
            continue
        # Layer B: down-weight older articles via exponential recency decay.
        article_date = result.published_date or None
        rweight = recency_weight(
            article_date,
            ref_date,
            settings.recency_half_life_days,
            floor=settings.recency_floor,
        )
        # Layer C: down-weight off-topic articles by the gatekeeper's graded
        # relevance, applied convexly (squared) so a confident-but-tangential
        # article (relevance ~0.5 → 0.25× pull) can't drag the pooled mean.
        weight = credibility * avg_certainty * rweight * (relevance ** 2)

        all_stances.append(avg_stance)
        all_weights.append(weight)
        relevances.append(relevance)

        source_signals.append(SourceSignal(
            source_id=source_id,
            source_name=result.source or source_id,
            url=result.url,
            stance=round(avg_stance, 3),
            certainty=round(avg_certainty, 3),
            credibility_weight=round(credibility, 3),
            claims=[p.claim for p in predictions if p.claim],
            published_date=article_date,
            recency_weight=round(rweight, 3),
            relevance_score=round(relevance, 3),
        ))

    # Aggregate relevance safety net: if every surviving article is off-topic,
    # the relevance² weights collapse and pool_sources would fall back to an
    # *unweighted* mean — re-admitting the off-topic articles at full strength.
    # Treat that as insufficient data instead. See settings.relevance_weight_floor.
    relevance_mass = sum(r * r for r in relevances)
    all_off_topic = bool(all_stances) and relevance_mass < settings.relevance_weight_floor

    # Decisiveness safety net: even with on-subject articles, a thin, low-certainty
    # pool produces a confident-looking ~50% from evidence that doesn't actually
    # bear on the claim (e.g. generic Musk news for "will Musk tweet about X?").
    # `all_weights` already folds credibility × certainty × recency × relevance²,
    # so their sum is the certainty-weighted evidence mass; below a floor we defer
    # to the caller's base rate instead of emitting a coin-flip. A genuinely
    # balanced ~50% backed by strong coverage has high mass and is unaffected.
    evidence_mass = sum(all_weights)
    no_decisive_signal = (
        bool(all_stances) and not all_off_topic
        and evidence_mass < settings.decisiveness_floor
    )

    if not all_stances or all_off_topic or no_decisive_signal:
        # Outcome histogram tells us *why* we got nothing — were articles
        # rejected by the gatekeeper, did extraction return empty, or did
        # fetch fail? Without this the warning is uninvestigatable.
        outcome_counts: dict[str, int] = {}
        for t in timings:
            key = str(t.get("outcome", "unknown"))
            outcome_counts[key] = outcome_counts.get(key, 0) + 1
        if n_low_certainty:
            outcome_counts["low_certainty"] = n_low_certainty
        if all_off_topic:
            outcome_counts["all_low_relevance"] = len(relevances)
            reason = "all_articles_off_topic"
        elif no_decisive_signal:
            outcome_counts["low_evidence_mass"] = len(all_weights)
            reason = "no_decisive_signal"
        elif not all_stances and n_low_certainty:
            # Every source that survived gatekeeper+extractor was dropped by the
            # certainty gate — the pool was all hedged speculation.
            reason = "all_low_certainty"
        else:
            reason = _reason_from_outcomes(outcome_counts)
        logger.warning(
            "No usable predictions extracted from %d articles (reason=%s outcomes=%s)",
            len(search_results),
            reason,
            outcome_counts,
        )
        _log_phase(
            "total",
            (time.perf_counter() - total_start) * 1000,
            question=req.question,
            articles_used=0,
            outcome="no_usable_predictions",
            reason=reason,
            **{f"n_{k}": v for k, v in outcome_counts.items()},
        )
        empty_debug: Optional[DebugInfo] = None
        if req.debug:
            empty_debug = DebugInfo(
                search_query=search_query,
                search_provider=search_provider,
                search_provider_chain=provider_chain,
                gatekeeper_model=_pipeline_settings.gatekeeper_model,
                extractor_model=_pipeline_settings.extractor_model,
                articles_fetched=len(search_results),
                articles_gate_passed=sum(1 for d in article_debugs if d.gate_passed),
                articles_extracted=0,
                total_prompt_tokens=0,
                total_completion_tokens=0,
                total_tokens=0,
                per_article=article_debugs,
                gatekeeper_prompt=GATEKEEPER_PROMPT,
                extractor_prompt=EXTRACTOR_PROMPT,
            )
        return _empty_response(
            req.question,
            reason=reason,
            articles_found=len(search_results),
            outcome_counts=outcome_counts,
            provider=search_provider,
            provider_chain=provider_chain,
            distilled_query=distilled_query,
            debug=empty_debug,
        )

    # Step 4: logit (log-odds) pooling + 95% CI.
    # Pooling in log-odds space (a logarithmic opinion pool) is robust to
    # outliers: a single dissenting source can't drag a confident consensus back
    # to the middle the way an arithmetic mean does. Combined with the per-claim
    # certainty weighting (Layer A) and recency weighting (Layer B), a decided
    # event reads decisive instead of the old wishy-washy ~0.5.
    n = len(all_stances)
    mean, std, ci_low, ci_high = pool_sources(
        all_stances, all_weights, clamp_eps=settings.logit_clamp
    )

    logger.info(
        "Forecast: mean=%.3f std=%.3f ci=[%.3f,%.3f] articles=%d (logit-pool)",
        mean, std, ci_low, ci_high, n,
    )

    # Per-article outcome histogram on the success path too, plus a low_relevance
    # tally so the admin dashboard can see how many sources were down-weighted.
    success_outcome_counts: dict[str, int] = {}
    for t in timings:
        key = str(t.get("outcome", "unknown"))
        success_outcome_counts[key] = success_outcome_counts.get(key, 0) + 1
    success_outcome_counts["low_relevance"] = sum(1 for r in relevances if r < 0.3)
    if n_low_certainty:
        success_outcome_counts["low_certainty"] = n_low_certainty

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
            gatekeeper_model=_pipeline_settings.gatekeeper_model,
            extractor_model=_pipeline_settings.extractor_model,
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
        articles_found=len(search_results),
        sources=source_signals,
        placeholder=False,
        outcome_counts=success_outcome_counts,
        provider=search_provider,
        provider_chain=provider_chain,
        distilled_query=distilled_query,
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


def _reason_from_outcomes(outcome_counts: dict[str, int]) -> str:
    """Pick the dominant failure reason from the per-article outcome histogram.

    Turns the (already-computed, previously-discarded) histogram into a single
    actionable label so an empty forecast says *why* — search returned junk vs
    the extractor is erroring vs fetches failed are very different problems.
    """
    if not outcome_counts:
        return "no_usable_predictions"
    errors = outcome_counts.get("gate_error", 0) + outcome_counts.get("extract_error", 0) + outcome_counts.get("unhandled_error", 0)
    total = sum(outcome_counts.values())
    if errors and errors >= total / 2:
        return "extraction_errors"
    if outcome_counts.get("gate_rejected", 0) >= total / 2:
        return "all_articles_off_topic"
    if outcome_counts.get("empty_text", 0) >= total / 2:
        return "all_fetches_failed"
    return "no_usable_predictions"


def _empty_response(
    question: str,
    *,
    reason: Optional[str] = None,
    articles_found: int = 0,
    outcome_counts: Optional[dict[str, int]] = None,
    provider: str = "",
    provider_chain: Optional[list[str]] = None,
    distilled_query: Optional[str] = None,
    debug: Optional[DebugInfo] = None,
) -> ForecastResponse:
    """Return a maximally uncertain response when no usable articles are found.

    Always carries ``insufficient_data=True`` and a ``reason`` so callers can
    distinguish 'couldn't answer (and why)' from a real 0.5 probability.
    ``provider``/``provider_chain`` surface which engine served (or failed to
    serve) the search, so an empty forecast still says where it looked.
    """
    return ForecastResponse(
        question=question,
        mean=0.0,
        std=0.0,
        ci_low=-0.2,
        ci_high=0.2,
        articles_used=0,
        sources=[],
        placeholder=True,
        insufficient_data=True,
        reason=reason,
        articles_found=articles_found,
        outcome_counts=outcome_counts or {},
        provider=provider,
        provider_chain=provider_chain or [],
        distilled_query=distilled_query,
        debug=debug,
    )


def _avg(timings: list[dict], key: str) -> Optional[float]:
    """Mean of ``key`` across ``timings`` entries that carry it, rounded to 1 dp."""
    values = [t[key] for t in timings if key in t and t[key] is not None]
    if not values:
        return None
    return round(sum(values) / len(values), 1)
