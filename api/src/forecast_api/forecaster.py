"""
Core forecast logic — Phase 2: live pipeline integration.

Flow:
  1. search_articles(question) — multi-provider chain (news-indexer first, then GDELT/Google-CSE/SerpAPI/…/DDG); see tm.web_search
  2. For each article (in parallel): gatekeeper → extractor
  3. Weight each source by credibility from leaderboard
  4. Aggregate: weighted mean stance + 95% CI → return ForecastResponse
"""
import asyncio
import hashlib
import logging
import re
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Sequence
from urllib.parse import urlparse

import httpx
import trafilatura

from tm.gatekeeper import (
    check_is_prediction,
    PROMPT_PREFIX as _GATEKEEPER_PROMPT_PREFIX,
    PROMPT_SUFFIX as _GATEKEEPER_PROMPT_SUFFIX,
)
from tm.extractor import (
    extract_predictions,
    enforce_anchor_provenance,
    enforce_deadline_arithmetic,
    enforce_interested_party_certainty,
    enforce_interested_party_stance_cap,
    enforce_precursor_cap,
    enforce_relative_date_resolution,
    enforce_settlement_event_date,
    enforce_winner_entity_consistency,
    flag_claim_stance_sign_conflicts,
    PROMPT_PREFIX as _EXTRACTOR_PROMPT_PREFIX,
    PROMPT_SUFFIX as _EXTRACTOR_PROMPT_SUFFIX,
)
from tm.models import GatekeeperOutput, PredictionExtraction
from tm.web_search import SearchResult, search_capturing
from tm.config import settings as _pipeline_settings
from tm.llm import complete_text_once_with_usage
from tm.net_guard import UnsafeURLError, safe_get

# gatekeeper.py/extractor.py each split their PROMPT into a cacheable PROMPT_PREFIX
# (fixed instructions) + PROMPT_SUFFIX (article/variable fields) so llm.py can mark
# the prefix as a Bedrock/Anthropic cache breakpoint. These two reconstruct the full,
# unformatted prompt text — byte-identical to the pre-split PROMPT — purely for the
# debug/introspection fields below (ForecastDebug.gatekeeper_prompt/extractor_prompt);
# nothing here re-enters an actual LLM call.
GATEKEEPER_PROMPT = _GATEKEEPER_PROMPT_PREFIX + _GATEKEEPER_PROMPT_SUFFIX
EXTRACTOR_PROMPT = _EXTRACTOR_PROMPT_PREFIX + _EXTRACTOR_PROMPT_SUFFIX

from .auth import ApiKeyClient
from .aggregation import (
    aggregate_pool,
    claim_weighted_stance,
    evidence_class_weight,
    recency_weight,
    relevance_weight,
    resolve_stance_certainty,
    settlement_grade,
)
from .cache import forecast_cache, search_cache
from .clustering import cluster_text_for_claims, cluster_texts_with_stats
from .dedup import dedupe_syndicated
from .leaderboard import get_credibility_weight
from .models import (
    ArticleDebug,
    ArticleInput,
    ClaimDetail,
    DebugInfo,
    ForecastRequest,
    ForecastResponse,
    PoolAggregateRequest,
    PoolAggregateResponse,
    SourceSignal,
    TokenUsage,
)
from .config import settings
from .settlement_verifier import SettlementVote, verify_settlement

logger = logging.getLogger(__name__)

# In-flight deduplication: cache_key → asyncio.Event.
# When a second request arrives for a key that's already being processed,
# it waits on this event instead of launching a duplicate pipeline.
# Per-worker on purpose (unlike forecast_cache, which is shared — retro#405):
# coalescing across processes needs real distributed locking, and the shared
# cache already catches the expensive half — the duplicate's result is stored
# for every later caller on either worker.
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


def build_claims_detail(predictions: list[PredictionExtraction]) -> list[ClaimDetail]:
    """Project this article's claims onto the wire's per-claim model (F1/F15,
    retro#364) — the layer the article's fused scalars are reduced from.

    Takes the POST-resolution claims (after the `enforce_*` chain and
    `resolve_stance_certainty()`), i.e. exactly the list the fusion below
    consumes, so `stance`/`certainty`/`evidence_weight`/`fact_signal` remain
    derivable from what is persisted. Persisting the extractor's raw output
    instead would create a second parallel truth that cannot reproduce the
    article's own vote.

    No filtering: unlike `SourceSignal.claims`, a claim with an empty summary
    is kept, because it still carried weight in the reduction.
    """
    return [
        ClaimDetail(
            claim=p.claim,
            quote=p.quote or None,
            stance=p.stance,
            certainty=p.certainty,
            specificity=p.specificity,
            prediction_type=p.prediction_type.value if p.prediction_type is not None else None,
            evidence_class=p.evidence_class,
            quantitative_estimate=p.quantitative_estimate,
            settled=p.settled,
            event_date=p.event_date,
            fact_signal=p.fact_signal,
            event_actors=p.event_actors,
            event_target=p.event_target,
            is_occurrence=p.is_occurrence,
            verified=p.verified,
        )
        for p in predictions
    ]


@dataclass(frozen=True)
class ArticleReduction:
    """The article-level scalars, as reduced FROM the per-claim layer.

    Item 3 of F1 (retro#364): the fused scalars are *derived* from
    ``claims_detail``, not computed beside it. Before this they were computed
    over the in-memory extraction list and the claims were projected onto the
    wire separately — two computations from one source, which is a parallel
    truth waiting to drift the moment either side is edited. Reducing from the
    persisted layer makes derivability structural: whatever a stored row
    contains is, by construction, what produced that row's numbers.

    Same formulas, same order of operations, same floats as before — this is a
    refactor. Any implementation of it that MOVES a number has quietly imported
    R1 (claim-level weighting), which is Phase 2 and gated on the shadow pool.
    """
    stance: float
    certainty: float
    evidence_weight: float
    evidence_class: Optional[str]
    settled: bool
    settlement_demoted: int
    settlement_event_date: Optional[str]
    quantitative_estimate: Optional[float]
    claims: list[str]
    fact_signal: Optional[float]
    event_actors: Optional[str]
    event_target: Optional[str]
    is_occurrence: Optional[bool]
    verified: Optional[bool]


def reduce_article(
    claims: list[ClaimDetail],
    *,
    settlement_min_stance: float,
    settlement_min_certainty: float,
    class_weights: dict,
    class_weight_default: float,
    class_weight_unclassified_cap: float,
) -> ArticleReduction:
    """Collapse one article's claims into the scalars the pool consumes.

    Five reductions over five different subsets, which is exactly why the
    per-claim layer had to survive:

    - ``stance``   — claim-weighted mean over the SETTLEMENT-GRADE claims if the
      article has any, else over all of them. A verdict must not be averaged
      down by the same article's colour quotes.
    - ``certainty`` / ``evidence_weight`` — means over ALL claims, including the
      ones the settlement subset excluded.
    - ``evidence_class`` — the most common non-null per-claim label, i.e. only
      the article's *representative* class; mixed-class articles are
      unattributable at this level by construction (which is what the
      per-claim field now records honestly).
    - ``fact_signal`` — claim-weighted mean over the same scored subset as
      ``stance``, but its qualifying facets ride from the single dominant
      (max |fact_signal|) claim so they stay internally coherent.

    Pure: takes claims and configuration, touches no globals, and is therefore
    replayable over persisted ``claims_detail`` rows — which is the whole point
    of keeping them (retroactive backtesting, R1 fitting, F3 attribution).
    """
    # Settlement-grade gate: the extractor's own stated rule, enforced in code.
    # A settled claim that fails it is demoted to ordinary evidence — it still
    # votes, it just cannot pin the estimate.
    settled_claims = [
        c for c in claims
        if c.settled and settlement_grade(
            c.stance, c.certainty,
            min_stance=settlement_min_stance,
            min_certainty=settlement_min_certainty,
        )
    ]
    demoted = sum(1 for c in claims if c.settled) - len(settled_claims)
    scored = settled_claims or claims

    stance = claim_weighted_stance(
        [c.stance for c in scored],
        [c.certainty for c in scored],
        [c.specificity for c in scored],
    )
    certainty = sum(c.certainty for c in claims) / len(claims)
    # S2 cutover: evidence-class weight replaces certainty as the linear factor
    # in the cross-article `weight`. Classified claims look up class_weight;
    # unclassified ones fall back to their own certainty, capped at the weakest
    # class (retro#366) — see evidence_class_weight().
    evidence_weight = sum(
        evidence_class_weight(
            c.evidence_class, c.certainty,
            weights=class_weights,
            default=class_weight_default,
            unclassified_cap=class_weight_unclassified_cap,
        )
        for c in claims
    ) / len(claims)

    # The credibility feedback loop (docs/ORACLE_VARIABLES.md §9) needs the
    # label, not just the resolved weight, to exclude opinion-class articles
    # from the resolution-outcome signal.
    labelled = [c.evidence_class for c in claims if c.evidence_class is not None]
    representative_class = Counter(labelled).most_common(1)[0][0] if labelled else None

    # fact_signal lane — SHADOW, parallel to stance, read by nothing in
    # aggregation. Mean-to-mean with stance so the offline fact-lane backtest
    # compares like with like; None when no scored claim carried a fact_signal.
    fact_claims = [c for c in scored if c.fact_signal is not None]
    if fact_claims:
        fact_signal = claim_weighted_stance(
            [c.fact_signal for c in fact_claims],
            [c.certainty for c in fact_claims],
            [c.specificity for c in fact_claims],
        )
        dominant = max(fact_claims, key=lambda c: abs(c.fact_signal))
        event_actors, event_target = dominant.event_actors, dominant.event_target
        is_occurrence, verified = dominant.is_occurrence, dominant.verified
    else:
        fact_signal = None
        event_actors = event_target = None
        is_occurrence = verified = None

    return ArticleReduction(
        stance=stance,
        certainty=certainty,
        evidence_weight=evidence_weight,
        evidence_class=representative_class,
        settled=bool(settled_claims),
        settlement_demoted=demoted,
        settlement_event_date=derive_settlement_event_date(settled_claims, stance),
        quantitative_estimate=next(
            (c.quantitative_estimate for c in claims if c.quantitative_estimate is not None), None
        ),
        claims=[c.claim for c in claims if c.claim],
        fact_signal=fact_signal,
        event_actors=event_actors,
        event_target=event_target,
        is_occurrence=is_occurrence,
        verified=verified,
    )


def derive_settlement_event_date(
    settled_preds: list[ClaimDetail],
    avg_stance: float,
) -> Optional[str]:
    """The article-level settlement anchor date, for SourceSignal.

    Among the article's settlement-grade claims, pick the ``event_date`` of the
    one that actually drives the article's settlement vote: same stance sign as
    the collapsed article stance (aggregate_pool reads the vote's direction from
    that sign), highest certainty first, earliest date on ties. None when no
    driving claim is dated — a positive settlement in that state has already
    been demoted by enforce_settlement_event_date, and a negative one may be
    legitimately undated (foreclosure by time expiring).
    """
    direction_positive = avg_stance >= 0
    dated = [
        p for p in settled_preds
        if p.event_date and (p.stance >= 0) == direction_positive
    ]
    if not dated:
        return None
    best = min(dated, key=lambda p: (-p.certainty, p.event_date))
    return best.event_date


def _cluster_text_of(s) -> Optional[str]:
    """The text one row is clustered on, from either a live ``SourceSignal`` or a
    caller-supplied ``PoolSourceInput`` — both carry ``claims_detail`` and a title-ish
    fallback, and both MUST resolve identically or a recompute would re-cluster what the
    live path already clustered (retro#404's band-table argument, applied to text)."""
    return cluster_text_for_claims(
        getattr(s, "claims_detail", None), getattr(s, "title", None),
    )


def _cluster_ids(texts: list[Optional[str]], question_hash: str) -> Optional[tuple[int, ...]]:
    """Cluster a pool's rows and log the echo structure (retro#355).

    Runs even when the discount is inert — that is the point. The decision to enable it
    has to rest on how much correlated evidence live pools actually carry, and nothing
    was measuring that. Logging it costs one pass over already-in-memory text.

    **The line is emitted for EVERY pool, including pools with no echo at all.** It
    previously fired only when some cluster reached size ≥2, which made a zero
    unreadable: a pool too small to compare, a pool of text-less legacy rows, and a pool
    of genuinely independent reporting all wrote the same nothing. Measured 2026-08-05,
    that produced exactly one line — a synthetic probe — across 180 ``/pool/aggregate``
    and 374 ``/forecast`` requests, so "no echo observed" could not be told apart from
    "nothing was ever observable". ``echoed_rows=0`` is now the discriminator, and
    ``textful``/``pairs``/``max_jaccard``/``hist`` carry the denominators and the
    near-misses needed to tune ``cluster_jaccard_threshold`` downward if the echo is
    sitting just under it.

    Unlike the clusters themselves, this measurement accrues at TRAFFIC rate rather than
    at resolution rate — it observes pool structure, not forecast accuracy — so it does
    not wait on the resolved-forecast backlog that gates enabling the discount (#403).
    """
    ids, stats = cluster_texts_with_stats(
        texts,
        threshold=settings.cluster_jaccard_threshold,
        shingle_size=settings.cluster_shingle_size,
    )
    sizes: dict[int, int] = {}
    for cid in ids:
        sizes[cid] = sizes.get(cid, 0) + 1
    echoed = {c: k for c, k in sizes.items() if k > 1}
    logger.info(
        "event=evidence_clusters question=%s rows=%d textful=%d pairs=%d clusters=%d "
        "largest=%d echoed_rows=%d max_jaccard=%.3f threshold=%.2f exponent=%.2f hist=%s",
        question_hash, stats.rows, stats.textful, stats.pairs, len(sizes),
        max(sizes.values()) if sizes else 0, sum(echoed.values()),
        stats.max_jaccard, settings.cluster_jaccard_threshold,
        settings.cluster_downweight_exponent,
        ",".join(str(c) for c in stats.histogram),
    )
    # Below two rows there is nothing to group; returning None keeps the discount path
    # untouched, exactly as before — only the logging above is new.
    if len(texts) < 2:
        return None
    return ids


def _settlement_votes(
    outlet: Optional[str], claims_detail: Optional[list[ClaimDetail]],
) -> list[SettlementVote]:
    """The settling claims of one article, as the match gate sees them.

    Only claims the extractor marked ``settled`` — an article's other claims are
    ordinary evidence and are not what the pin rests on. The verbatim ``quote``
    rides along because the whole failure class this gate exists for is visible
    in the quote and invisible in the numbers: on retro#388 the claim summaries
    read as the outcome while the quotes said the decider had *announced* it.

    Empty when ``claims_detail`` is absent — a legacy row, or a caller that
    hasn't been widened yet (F1/retro#364). The gate treats that as "cannot
    check" rather than "does not settle".
    """
    return [
        SettlementVote(
            outlet=outlet, claim=c.claim, quote=c.quote, event_date=c.event_date,
        )
        for c in (claims_detail or [])
        if c.settled
    ]


async def _apply_settlement_match_gate(
    agg,
    *,
    question: Optional[str],
    votes_for_index,
    rerun,
    settled_flags: Sequence[Optional[bool]],
):
    """Ask whether the settling facts ARE this claim's outcome (retro#388/#360).

    The verdict is logged either way; it acts only when
    ``settlement_verifier_enforce`` is set, which it has been since 2026-08-03
    (see ``settlement_verifier`` for the evidence, and for why the check is
    semantic rather than a field comparison).

    Enforcement re-runs the *same* ``aggregate_pool`` with the vetoed votes'
    ``settled`` flags cleared, rather than editing the pinned result in place.
    Those rows keep voting as ordinary evidence — a vetoed settlement is a
    demotion, not a deletion — and the published number is one the pooling code
    actually produced, so a recompute over the stored pool still reproduces it.
    """
    if agg is None or not agg.settled or not settings.settlement_verifier_enabled:
        return agg
    if not question:
        logger.info("event=settlement_verifier outcome=skipped reason=no_question")
        return agg

    votes: list[SettlementVote] = []
    for i in agg.settlement_vote_indices:
        votes.extend(votes_for_index(i))
    if not votes:
        logger.info("event=settlement_verifier outcome=skipped reason=no_claim_detail")
        return agg

    # The direction matters as much as the match: facts that decide the question
    # the OTHER way are not proof of the answer about to be published (the
    # France-World-Cup pin, found by replaying this gate over past pins).
    verdict = await verify_settlement(
        question, votes,
        model=settings.settlement_verifier_model or _pipeline_settings.extractor_model,
        timeout_s=settings.settlement_verifier_timeout_seconds,
        answer="YES" if agg.mean >= 0 else "NO",
    )
    logger.warning(
        "event=settlement_verifier settles=%s errored=%s enforced=%s votes=%d "
        "question=%s reason=%r",
        verdict.settles, verdict.errored,
        settings.settlement_verifier_enforce and not verdict.settles and not verdict.errored,
        len(votes), _question_hash(question), verdict.reason[:200],
    )
    if verdict.settles or verdict.errored or not settings.settlement_verifier_enforce:
        return agg

    vetoed = set(agg.settlement_vote_indices)
    cleared = [
        (False if i in vetoed else flag) for i, flag in enumerate(settled_flags)
    ]
    return rerun(cleared) or agg


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


async def _distill_query(question: str) -> tuple[str, dict]:
    """Convert a long resolution-criterion question into 4-6 search keywords.

    Called only when verbatim search returns 0 results — adds ~200ms latency
    and a cheap Nova Micro call to unlock niche/Polymarket-style questions.
    Returns ``(keywords, usage)``; the original question (and whatever usage the
    failed call reported, usually ``{}``) on any error — preserves existing
    behaviour, and the usage feeds the response's ``token_usage`` total.
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
        text, usage = await complete_text_once_with_usage(
            _pipeline_settings.gatekeeper_model,
            prompt,
            max_tokens=40,
            timeout=20,
        )
        keywords = text.strip()
        if keywords:
            logger.info("query_distilled original=%r distilled=%r", question[:60], keywords)
            return keywords, usage
        return question, usage
    except Exception as exc:
        logger.warning("query distillation failed: %s", exc)
    return question, {}


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
        resp = safe_get(
            url,
            timeout=6.0,
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
    except UnsafeURLError as exc:
        outcome = "blocked_unsafe_url"
        logger.warning("Blocked unsafe article URL %s: %s", url, exc)
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
    claim_deadline: str | None = None,
    claim_direction: str | None = None,
    prediction_id: str | None = None,
    usage_events: list[dict] | None = None,
) -> tuple[SearchResult, float, list, float | None, float | None] | None:
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
                claim_deadline=claim_deadline,
                claim_direction=claim_direction,
                prediction_id=prediction_id,
                usage_events=usage_events,
            ),
            timeout=timeout_s,
        )
    except asyncio.TimeoutError:
        logger.warning("event=article_timeout url=%s timeout_s=%s", result.url, timeout_s)
        timings.append({"url": result.url, "outcome": "timeout"})
        article_debugs.append(ArticleDebug(url=result.url, outcome="timeout"))
        return None


def _supplied_verdict(result: SearchResult) -> tuple[bool, float] | None:
    """The caller's gatekeeper verdict carried on the SearchResult (news-indexer's POST
    /relevance result, threaded through daatan), or None when absent/incomplete. Both
    fields must be present — a relevance without a pass/reject is not a usable verdict."""
    rel = getattr(result, "_supplied_relevance", None)
    is_pred = getattr(result, "_supplied_is_prediction", None)
    if rel is None or is_pred is None:
        return None
    return bool(is_pred), float(rel)


async def _process_article(
    result: SearchResult,
    question: str,
    *,
    max_article_chars: int,
    timings: list[dict],
    article_debugs: list[ArticleDebug],
    claim_deadline: str | None = None,
    claim_direction: str | None = None,
    prediction_id: str | None = None,
    usage_events: list[dict] | None = None,
) -> tuple[SearchResult, float, list, float | None, float | None] | None:
    """
    Run gatekeeper + extractor for one article.
    Fetches full article text via trafilatura; falls back to title+snippet.
    Appends per-phase durations to ``timings`` and an ArticleDebug to ``article_debugs``.
    Non-empty gate/extract usage dicts are appended to ``usage_events`` (when given)
    as soon as each call returns — including for articles later rejected, whose
    tokens were still spent — feeding ForecastResponse.token_usage.
    """
    # A t.me post is short-form (retro#297): mirrors news-indexer's rematch.py:_is_short_form,
    # which feeds the same hint to the gatekeeper on its /relevance rescue path — one line
    # duplicated on purpose rather than threading a flag through three repos.
    short_form = urlparse(result.url or "").netloc.lower().removeprefix("www.") == "t.me"
    # Caller-supplied language hint, threaded to the gatekeeper/extractor prompts (retro#417).
    language = getattr(result, "_language", None)

    # Fallback text = title + snippet
    parts = [p for p in [result.title, result.snippet] if p and p.strip()]
    fallback = " — ".join(parts)
    # Short-form posts are exempt from the 20-char floor (retro#417): the floor exists for
    # content-free search-result stubs, but a terse Telegram post IS the article — dropping
    # it here silently bypassed the _SHORT_FORM_OVERRIDE built for exactly that class. The
    # gatekeeper still rejects content-free posts without an LLM call (carries_proposition),
    # so only a truly-empty floor remains for them.
    min_fallback_chars = 5 if short_form else 20
    if not fallback or len(fallback) < min_fallback_chars:
        return None

    # Use caller-supplied text if available; otherwise fetch via trafilatura.
    fetch_start = time.perf_counter()
    if result._prefetched_text:
        text = result._prefetched_text
        logger.info("event=article_fetch outcome=prefetched url=%s", result.url)
    elif short_form:
        # Never origin-fetch t.me (retro#417): t.me serves a web preview that extracts to
        # near-nothing, so the fetch always lost the `extracted_len <= len(fallback)`
        # comparison in _fetch_article_text anyway — a wasted request plus a wasted
        # per-host throttle slot. news-indexer's rematch.py documents the same fact.
        text = fallback
        logger.info("event=article_fetch outcome=short_form_no_fetch url=%s", result.url)
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
    supplied = _supplied_verdict(result) if settings.reuse_supplied_relevance else None
    if supplied is not None:
        # Reuse the caller's gatekeeper verdict instead of re-judging. The SAME claim-aware
        # judge (tm.gatekeeper) already ran once upstream — in news-indexer's POST /relevance —
        # before this article was pushed. Only the duplicate RELEVANCE call is skipped; the
        # extractor below still runs (the Oracle is the only place stance/predictions are
        # produced). Design: news-indexer docs/MATCHING_ARCHITECTURE.md §3.
        is_pred, relevance_score = supplied
        gate = GatekeeperOutput(
            is_prediction=is_pred,
            reason="reused: gatekeeper verdict supplied by caller",
            relevance_score=relevance_score,
        )
        gate_usage = {}
        logger.info(
            "event=article_outcome outcome=gate_reused url=%s is_prediction=%s relevance=%.3f prediction_id=%s",
            result.url, is_pred, relevance_score, prediction_id or "",
        )
    else:
        try:
            gate, gate_usage = await check_is_prediction(
                article_text=text,
                source_name=source_name,
                article_date=article_date,
                event_name=question,
                # The override the floor exemption above exists for (retro#417): without it a
                # short t.me post survives the floor only to be rejected by the base prompt's
                # "under ~200 meaningful words" rule. Matches the /relevance rescue path.
                short_form=short_form,
                language=language,
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
        if usage_events is not None and gate_usage:
            usage_events.append(gate_usage)
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

    # The per-article relevance bar (retro#393). INERT at the shipped default of 0.0 — this
    # branch cannot be taken unless someone raises `forecast_relevance_bar`, because a
    # relevance of 0.0 is not < 0.0. It exists so that raising the bar is a config change,
    # and so the `low_relevance` counter below stops being decorative when it is raised.
    #
    # Placed AFTER the is_prediction check and BEFORE the extractor, which is the only
    # position that saves anything: the gatekeeper call is already paid for by here, and the
    # extractor is the expensive one.
    if settings.forecast_relevance_bar > 0 and (gate.relevance_score or 0.0) < settings.forecast_relevance_bar:
        logger.info(
            "event=article_outcome outcome=low_relevance url=%s relevance=%.2f bar=%.2f",
            result.url, gate.relevance_score or 0.0, settings.forecast_relevance_bar,
        )
        timings.append({
            "url": result.url, "fetch_ms": fetch_ms, "gate_ms": gate_ms,
            "outcome": "low_relevance",
        })
        article_debugs.append(ArticleDebug(
            url=result.url, outcome="low_relevance",
            gate_passed=True,
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
            claim_deadline=claim_deadline,
            short_form=short_form,
            language=language,
        )
        if usage_events is not None and extract_usage:
            usage_events.append(extract_usage)
        # Observability only (retro#298) — logs claim/stance sign mismatches on the
        # model's raw output, before any of the deterministic corrections below can
        # touch stance or settled. Never mutates.
        flag_claim_stance_sign_conflicts(extraction.predictions)
        # Before any date is compared, make sure the date itself is right: when the
        # article spoke in relative terms ("on Friday"), redo that calendar walk in
        # code — post-#267 the model still resolved the Knesset "Friday" to a
        # Saturday, and a ±1-day miss against a date ON the deadline flips the sign.
        extraction.predictions = enforce_relative_date_resolution(
            extraction.predictions, article_date,
        )
        # The model reports the date; arithmetic decides the sign. See
        # enforce_deadline_arithmetic — an LLM asked whether "Friday" beat a July 15
        # deadline answered +1.0/0.95 five times out of five, and Friday was July 17.
        extraction.predictions = enforce_deadline_arithmetic(
            extraction.predictions, claim_deadline, claim_direction,
        )
        # And a settlement vote must be anchored to a date the outcome occurred —
        # undated accomplished-fact language is historical background (the Netanyahu
        # false pin: the sitting coalition "settling" the NEXT election). Runs after
        # deadline arithmetic so settled-but-weak stances keep their sign correction.
        extraction.predictions = enforce_settlement_event_date(
            extraction.predictions, article_date,
        )
        # A fact that merely PRECEDES the event is capped at |0.3| by the prompt and
        # by 24.4% of live pool rows ignored it (retro#367). Magnitude is estimator
        # policy, so enforce it in code — before the fusion below reads fact_signal
        # for both the article mean and the dominant-claim facets.
        extraction.predictions = enforce_precursor_cap(extraction.predictions)
        # The strongest evidence class (4.0, and the only one that authorizes the
        # stance rewrite) must name a source whose figure could be checked —
        # otherwise one fabricated sentence buys it (retro#369). Runs before
        # resolve_stance_certainty below, so a demoted claim also stops rewriting
        # stance from its own number. Enforced since 2026-08-01.
        extraction.predictions = enforce_anchor_provenance(extraction.predictions)
        # And an interested party's UNVERIFIED assertion may not vote at full
        # magnitude: the prompt discounts only its certainty, which cancels under
        # normalization because a vote's location is stance alone (retro#368).
        # Runs after the demotion above so the log line carries the resolved class.
        extraction.predictions = enforce_interested_party_stance_cap(
            extraction.predictions,
        )
        # ...and the weight side of the same rule: the certainty ceiling the
        # prompt already promises, which 30.3% of live unverified rows exceed
        # (retro#378). Separate from the clamp above because stance is a vote's
        # location and certainty is its weight — different consequences, and the
        # two must move R8 cases separately attributably.
        extraction.predictions = enforce_interested_party_certainty(
            extraction.predictions,
        )
        # Deterministic winner-entity check (retro#401): for a two-named-actor
        # versus/sports question, does the dominant fact's actor→target dyad
        # (#313's facets, populated but never read until now) actually agree
        # with the stance sign it carries? Catches the retro#360 shape — "the
        # rival beat the subject" extracted as a positive stance FOR the
        # subject — without a second LLM call.
        extraction.predictions = enforce_winner_entity_consistency(
            extraction.predictions, question,
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
    # author_lean / author_lean_certainty are the byline author's OWN forecast
    # (retro #308/#309) — surfaced per-source for daatan's author-accuracy scoring
    # lane, kept entirely OUT of the estimate (never merged into stance/predictions).
    return (
        result,
        gate.relevance_score,
        extraction.predictions,
        extraction.author_lean,
        extraction.author_lean_certainty,
    )


async def run_forecast(
    req: ForecastRequest,
    client: Optional[ApiKeyClient] = None,
) -> ForecastResponse:
    limit = req.max_articles or settings.max_articles
    # Per-key ceiling (docs#57 item 1): clamps the search/extract limit AND the
    # caller-supplied articles list, which used to bypass every cap. Applied
    # before hashing/caching so the cache reflects the effective inputs.
    cap = client.max_articles if client else None
    if cap is not None:
        if limit > cap:
            logger.info(
                "event=per_key_cap client=%s limit=%d capped_to=%d",
                client.name, limit, cap,
            )
            limit = cap
        if req.articles and len(req.articles) > cap:
            logger.info(
                "event=per_key_cap client=%s supplied_articles=%d capped_to=%d",
                client.name, len(req.articles), cap,
            )
            req.articles = req.articles[:cap]
    total_start = time.perf_counter()

    # Step 0a: forecast cache lookup.
    # When caller supplies articles, key includes an MD5 of sorted URLs so
    # two calls with the same question but different article sets don't collide.
    articles_hash: Optional[str] = None
    if req.articles:
        articles_hash = hashlib.md5(
            "|".join(sorted(a.url for a in req.articles)).encode()
        ).hexdigest()[:12]
    claim_meta = f"{req.claim_direction or ''}|{req.claim_deadline or ''}" if (req.claim_direction or req.claim_deadline) else None
    # Keyed on the EFFECTIVE limit, not the raw request value — two keys with
    # different caps must not alias to the same cached response.
    cache_key = forecast_cache.make_key(req.question, limit, articles_hash, claim_meta)
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
    # LLM token spend of this run (docs#57 item 3): every usage dict the run's
    # LLM calls report lands here — distillation, gatekeeper, extractor —
    # summed into ForecastResponse.token_usage on EVERY exit path below,
    # debug or not.
    usage_events: list[dict] = []
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
                _supplied_relevance=a.relevance,
                _supplied_is_prediction=a.is_prediction,
                _language=a.language,
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
            search_query, distill_usage = await _distill_query(verbatim)
            if distill_usage:
                usage_events.append(distill_usage)
            distilled = search_query != verbatim
            # Capture the distilled keywords now, before the verbatim fallback
            # below can overwrite search_query.
            distilled_query = search_query if distilled else None
            try:
                search_results, search_provider, provider_chain = await asyncio.to_thread(
                    search_capturing, search_query, limit
                )
            except Exception as exc:
                logger.error("Search failed: %s", exc)
                search_results, search_provider, provider_chain = [], "none", []
            # Recall safety: if distilled keywords found nothing, retry verbatim.
            if not search_results and distilled:
                try:
                    search_results, search_provider, provider_chain = await asyncio.to_thread(
                        search_capturing, verbatim, limit
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
        # Only the distillation call (if any) ran by this point — still spend.
        resp.token_usage = TokenUsage.from_usages(usage_events)
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

    # Step 1b: collapse syndicated near-duplicates (one wire story re-hosted across
    # aggregators) to a single highest-credibility source, so it can't triple its
    # weight in the pool or pad the evidence mass. Done before fetch/extract to save
    # work too. Conservative (high title-similarity threshold) — different stories
    # on the same topic survive.
    n_before_dedupe = len(search_results)
    search_results = dedupe_syndicated(
        search_results,
        title_of=lambda r: r.title or "",
        url_of=lambda r: r.url or "",
        priority_of=lambda r: get_credibility_weight(_source_id_from_url(r.url)),
        threshold=settings.syndication_title_similarity,
    )
    if len(search_results) < n_before_dedupe:
        logger.info(
            "event=syndication_dedupe before=%d after=%d collapsed=%d",
            n_before_dedupe, len(search_results), n_before_dedupe - len(search_results),
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
                claim_deadline=req.claim_deadline,
                claim_direction=req.claim_direction,
                prediction_id=req.prediction_id,
                usage_events=usage_events,
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
    all_valve_weights: list[float] = []
    all_age_adjusted_weights: list[float] = []
    relevances: list[float] = []
    all_settled: list[bool] = []
    all_settlement_dates: list[Optional[str]] = []
    all_published_dates: list[Optional[str]] = []
    all_source_ids: list[Optional[str]] = []
    ref_date = datetime.now().strftime("%Y-%m-%d")
    # S2 cutover (retro docs/ORACLE_VARIABLES.md §5) — evidence_class now drives
    # `weight` below via evidence_class_weight(); this dict remains for
    # operational visibility into real-traffic classification coverage/mix.
    evidence_class_counts: dict[str, int] = {}

    for result, outcome in zip(search_results, outcomes):
        if isinstance(outcome, Exception):
            # Previously skipped silently — surface it so a systemic failure in
            # _process_article isn't invisible, and record it in the histogram.
            logger.warning("event=article_unhandled_error url=%s err=%r", result.url, outcome)
            timings.append({"url": result.url, "outcome": "unhandled_error"})
            continue
        if outcome is None:
            continue
        _, relevance, predictions, author_lean, author_lean_certainty = outcome
        for p in predictions:
            if p.evidence_class is not None:
                evidence_class_counts[p.evidence_class] = evidence_class_counts.get(p.evidence_class, 0) + 1
        # A source with an explicit quantitative_estimate has its stance/certainty
        # resolved from that figure rather than trusted verbatim from the extractor
        # — see resolve_stance_certainty() for why.
        resolved_predictions = []
        for p in predictions:
            # Settlement-grade claims keep the extractor's own stance/certainty:
            # a cited retrospective number in the same claim ("...defying models
            # that gave him 22%") must not flip an accomplished fact's direction.
            # A settled=true claim that doesn't clear the settlement_grade bar
            # (hedged, below-boundary) is not trusted as an accomplished fact —
            # it must still go through the same realignment as ordinary evidence,
            # or it keeps evidence_class_weight's cited_probability premium below
            # while carrying an unrealigned, possibly wrong-direction stance.
            if p.settled and settlement_grade(
                p.stance, p.certainty,
                min_stance=settings.settlement_min_claim_stance,
                min_certainty=settings.settlement_min_claim_certainty,
            ):
                resolved_predictions.append(p)
                continue
            stance, certainty = resolve_stance_certainty(
                p.stance, p.certainty, p.quantitative_estimate,
                evidence_class=p.evidence_class,
            )
            resolved_predictions.append(p.model_copy(update={"stance": stance, "certainty": certainty}))
        predictions = resolved_predictions

        source_id = _source_id_from_url(result.url)
        credibility = get_credibility_weight(source_id)
        # F1/F15 (retro#364): project the claims onto the wire's per-claim model
        # FIRST, then reduce the article's scalars from that layer. The claims
        # are no longer a by-product of the reduction — they are its input, so
        # what gets persisted is by construction what produced these numbers.
        #
        # Layer A: certainty-weight the article's claims so a decisive claim
        # dominates tangential hedged ones instead of being washed out.
        # Settlement claims (the outcome reported as an accomplished fact) go
        # further and *replace* the article's mixed claim set: a verdict must
        # not be averaged down by the same article's color quotes.
        # Layer A.5 (S2 cutover): evidence-class weight becomes the linear
        # factor in the cross-article `weight` below. A pool of only weak/hedged
        # sources still ends up below decisiveness_floor and surfaces as a
        # wide-CI low-confidence estimate (see the thin-evidence widening after
        # pooling), rather than being deleted and forcing an abstention on
        # on-topic coverage.
        claims_detail = build_claims_detail(predictions)
        reduction = reduce_article(
            claims_detail,
            settlement_min_stance=settings.settlement_min_claim_stance,
            settlement_min_certainty=settings.settlement_min_claim_certainty,
            class_weights=settings.evidence_class_weight,
            class_weight_default=settings.evidence_class_weight_default,
            class_weight_unclassified_cap=settings.evidence_class_weight_unclassified_cap,
        )
        if reduction.settlement_demoted:
            logger.info(
                "event=settlement_demoted url=%s demoted=%d (below stance/certainty gates)",
                result.url, reduction.settlement_demoted,
            )
        avg_stance = reduction.stance
        avg_certainty = reduction.certainty
        avg_evidence_weight = reduction.evidence_weight
        all_settled.append(reduction.settled)
        all_settlement_dates.append(reduction.settlement_event_date)
        # Layer B: down-weight older articles via exponential recency decay.
        article_date = result.published_date or None
        rweight = recency_weight(
            article_date,
            ref_date,
            settings.recency_half_life_days,
            floor=settings.recency_floor,
        )
        # Layer C: down-weight off-topic articles by the gatekeeper's relevance, so a
        # confident-but-tangential article (relevance ~0.5 → 0.25× pull) can't drag the
        # pooled mean. Read off RELEVANCE_BAND_WEIGHTS rather than squared inline: the
        # score is four band labels, not a gradient, so squaring it was arithmetic on a
        # categorical value (retro#394). The table is initialised to the squares, so this
        # is a no-op — it only moves the numbers somewhere they can be chosen.
        weight = credibility * avg_evidence_weight * rweight * relevance_weight(relevance)
        # The same product with recency UN-floored (retro#397, system-model §6.1).
        # recency_floor exists so an old row's voting influence never reaches
        # exactly zero; reusing it to decide whether we still know anything is
        # what stops an aging pool from ever fading out. Voting reads `weight`,
        # the abstention/CI valves read this.
        valve_weight = (
            credibility * avg_evidence_weight
            * recency_weight(article_date, ref_date, settings.recency_half_life_days, floor=0.0)
            * relevance_weight(relevance)
        )
        # Reporting-only twin of `weight` with recency's contribution forced to
        # neutral (retro#458 Phase 2) — same product as `weight` but with
        # `rweight` replaced by 1.0, i.e. "what would this row weigh if it had
        # not aged". Feeds PoolAggregateResult.age_adjusted_mass; nothing in
        # the pooling math reads it.
        age_adjusted_weight = credibility * avg_evidence_weight * relevance_weight(relevance)

        all_stances.append(avg_stance)
        all_weights.append(weight)
        all_valve_weights.append(valve_weight)
        all_age_adjusted_weights.append(age_adjusted_weight)
        relevances.append(relevance)
        all_published_dates.append(article_date)
        all_source_ids.append(source_id)

        source_signals.append(SourceSignal(
            source_id=source_id,
            source_name=result.source or source_id,
            url=result.url,
            stance=round(avg_stance, 3),
            certainty=round(avg_certainty, 3),
            credibility_weight=round(credibility, 3),
            claims=reduction.claims,
            published_date=article_date,
            recency_weight=round(rweight, 3),
            relevance_score=round(relevance, 3),
            settled=reduction.settled or None,
            quantitative_estimate=reduction.quantitative_estimate,
            evidence_weight=round(avg_evidence_weight, 3),
            evidence_class=reduction.evidence_class,
            settlement_event_date=reduction.settlement_event_date,
            author_lean=author_lean,
            author_lean_certainty=author_lean_certainty,
            fact_signal=round(reduction.fact_signal, 3) if reduction.fact_signal is not None else None,
            event_actors=reduction.event_actors,
            event_target=reduction.event_target,
            is_occurrence=reduction.is_occurrence,
            verified=reduction.verified,
            # F1/F15 (retro#364): the claims every scalar above was reduced
            # FROM, kept instead of discarded. Additive — nothing in
            # aggregation reads it.
            claims_detail=claims_detail,
        ))

    if evidence_class_counts:
        logger.info("event=evidence_class_weighted question=%s counts=%s", _question_hash(req.question), evidence_class_counts)

    # Steps 4-4b: relevance off-topic safety net, logit pooling, thin-evidence
    # CI widening, and the settlement override — all delegated to
    # aggregate_pool() (see aggregation.py) so a future recompute over an
    # accumulated evidence pool can never silently drift from what a fresh
    # run produces here.
    _pool_kwargs = dict(
        relevance_weight_floor=settings.relevance_weight_floor,
        decisiveness_floor=settings.decisiveness_floor,
        thin_evidence_ci_inflation=settings.thin_evidence_ci_inflation,
        defer_on_thin_evidence=settings.defer_on_thin_evidence,
        settlement_min_sources=settings.settlement_min_sources,
        settlement_stance=settings.settlement_stance,
        logit_clamp=settings.logit_clamp,
        pool_dispersion_floor=settings.pool_dispersion_floor,
        claim_direction=req.claim_direction,
        claim_deadline=req.claim_deadline,
        settlement_event_dates=all_settlement_dates,
        published_dates=all_published_dates,
        claim_created_at=req.claim_created_at,
        claim_archetype=req.claim_archetype,
        settlement_revalidate=settings.settlement_revalidate,
        settlement_post_deadline_grace_days=settings.settlement_post_deadline_grace_days,
        settlement_quality_floor=settings.settlement_quality_floor,
        cluster_ids=_cluster_ids(
            [_cluster_text_of(s) for s in source_signals], _question_hash(req.question),
        ),
        cluster_downweight_exponent=settings.cluster_downweight_exponent,
        valve_weights=all_valve_weights,
        age_adjusted_weights=all_age_adjusted_weights,
        source_ids=all_source_ids,
        max_source_share=settings.max_source_share,
    )
    agg = aggregate_pool(all_stances, all_weights, relevances, all_settled, **_pool_kwargs)
    if agg is not None:
        for idx, demotion_reason in agg.settlement_demotions:
            logger.warning(
                "event=settlement_vote_demoted reason=%s url=%s stance=%+.2f event_date=%s",
                demotion_reason, source_signals[idx].url, all_stances[idx],
                all_settlement_dates[idx],
            )
        if agg.settlement_suppressed:
            logger.warning(
                "event=settlement_suppressed reason=%s question=%s",
                agg.suppression_reason, _question_hash(req.question),
            )
        # The match gate (retro#388/#360) — shadow unless enforce is on. It runs
        # here rather than inside aggregate_pool because aggregation is a pure,
        # log-free, synchronous function and this is an LLM call; keeping the
        # split is what lets a recompute reproduce a live run exactly.
        agg = await _apply_settlement_match_gate(
            agg,
            question=req.question,
            votes_for_index=lambda i: _settlement_votes(
                source_signals[i].source_name, source_signals[i].claims_detail,
            ),
            rerun=lambda flags: aggregate_pool(
                all_stances, all_weights, relevances, flags, **_pool_kwargs,
            ),
            settled_flags=all_settled,
        )

    if agg is None or agg.insufficient_reason is not None:
        # Outcome histogram tells us *why* we got nothing — were articles
        # rejected by the gatekeeper, did extraction return empty, or did
        # fetch fail? Without this the warning is uninvestigatable.
        outcome_counts: dict[str, int] = {}
        for t in timings:
            key = str(t.get("outcome", "unknown"))
            outcome_counts[key] = outcome_counts.get(key, 0) + 1
        if agg is not None and agg.insufficient_reason == "all_articles_off_topic":
            outcome_counts["all_low_relevance"] = len(relevances)
            reason = "all_articles_off_topic"
        elif agg is not None and agg.insufficient_reason == "no_usable_weight":
            outcome_counts["zero_weight_pool"] = len(all_weights)
            reason = "no_usable_weight"
        elif agg is not None and agg.insufficient_reason == "no_decisive_signal":
            outcome_counts["low_evidence_mass"] = len(all_weights)
            reason = "no_decisive_signal"
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
        resp = _empty_response(
            req.question,
            reason=reason,
            articles_found=len(search_results),
            outcome_counts=outcome_counts,
            provider=search_provider,
            provider_chain=provider_chain,
            distilled_query=distilled_query,
            debug=empty_debug,
        )
        # The gate/extract calls were made and paid for even though no usable
        # forecast came out — an empty answer still reports its spend.
        resp.token_usage = TokenUsage.from_usages(usage_events)
        return resp

    n, mean, std, ci_low, ci_high, settled, evidence_mass, thin_evidence = (
        agg.n, agg.mean, agg.std, agg.ci_low, agg.ci_high, agg.settled,
        agg.evidence_mass, agg.thin_evidence,
    )
    if agg.settlement_suppressed:
        logger.info(
            "event=settlement_suppressed claim_direction=%s claim_deadline=%s",
            req.claim_direction, req.claim_deadline,
        )
    if agg.settled:
        logger.info(
            "event=settlement_override settled_sources=%d of=%d mean=%.3f",
            agg.settled_sources, n, mean,
        )

    logger.info(
        "Forecast: mean=%.3f std=%.3f ci=[%.3f,%.3f] articles=%d mass=%.3f valve_mass=%.4f settled=%s (logit-pool)",
        mean, std, ci_low, ci_high, n, evidence_mass, agg.valve_mass, settled,
    )

    # Per-article outcome histogram on the success path too, plus a low_relevance
    # tally so the admin dashboard can see how many sources were down-weighted.
    success_outcome_counts: dict[str, int] = {}
    for t in timings:
        key = str(t.get("outcome", "unknown"))
        success_outcome_counts[key] = success_outcome_counts.get(key, 0) + 1
    success_outcome_counts["low_relevance"] = sum(1 for r in relevances if r < 0.3)
    if thin_evidence:
        success_outcome_counts["thin_evidence"] = len(all_weights)

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
        settled=settled,
        # Which admission regime produced these sources (retro#393). 0.0 = no bar, this
        # path's historical behaviour; a caller persisting the rows can record it and
        # filter the pool retroactively instead of re-deriving which path admitted what.
        relevance_bar=settings.forecast_relevance_bar,
        token_usage=TokenUsage.from_usages(usage_events),
        debug=debug_info,
        evidence_mass=round(agg.evidence_mass, 4),
        n_eff=round(agg.n_eff, 4),
        age_adjusted_mass=round(agg.age_adjusted_mass, 4),
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


async def run_pool_aggregate(req: PoolAggregateRequest) -> PoolAggregateResponse:
    """Recompute a pooled estimate over a caller-supplied set of already-
    extracted per-source signals — no search, no LLM calls (retro
    docs/ORACLE_VARIABLES.md, recompute-over-pool). Reuses aggregate_pool()
    (aggregation.py), the exact same pooling math run_forecast() uses live,
    so a recompute over an accumulated evidence pool can never silently
    drift from what a fresh run of the same evidence would produce.

    Recency is recomputed fresh against "now" for each source's
    published_date, exactly like the live pipeline — an article decays
    further by the time of a LATER recompute even if nothing else about it
    changed, which is the whole point of recomputing over an accumulated
    pool rather than trusting a stale weight from first extraction.
    """
    ref_date = datetime.now().strftime("%Y-%m-%d")
    stances: list[float] = []
    weights: list[float] = []
    valve_weights: list[float] = []
    age_adjusted_weights: list[float] = []
    relevances: list[float] = []
    settled_flags: list[bool] = []
    settlement_dates: list[Optional[str]] = []
    published_dates: list[Optional[str]] = []
    source_ids: list[Optional[str]] = []
    # Whitelist, deliberately: only the eight scalars below reach the
    # estimator. PoolSourceInput also carries identity and per-claim data
    # (url / source_id / outlet / evidence_class / fact_signal + facets /
    # claims_detail — F1/F15, retro#364); reading any of them here would be an
    # estimator behaviour change smuggled in under a persistence PR. Claim-level
    # weighting (R1) remains the issue that gets to spend this data next;
    # claims_detail is already spent, deliberately and with its own R8 movement
    # report each time, by retro#355 (clustering, via _cluster_text_of below)
    # and retro#372 (the settlement count over those clusters). `source_id` is
    # spent the same way as of retro#458: read below (not merely persisted and
    # ignored) so `cap_source_mass` can group rows by outlet, but only ever
    # AS a grouping key — the identity string itself never enters the pooled
    # math, and the cap is inert at its shipped default like every other
    # widening on this list.
    for s in req.sources:
        rweight = recency_weight(
            s.published_date, ref_date,
            settings.recency_half_life_days, floor=settings.recency_floor,
        )
        # A row with no stored evidence_weight (legacy, pre-S2-cutover) falls back
        # to its certainty under the same R3 cap the live path applies to an
        # unclassified claim — the two are the same missing-data shape, and an
        # uncapped fallback would let the rows we know least about weigh most
        # (F10; 22 of 5729 prod COMPLETE rows, all above the cap, 2026-08-01).
        evidence_weight = (
            s.evidence_weight if s.evidence_weight is not None
            else min(s.certainty, settings.evidence_class_weight_unclassified_cap)
        )
        # Same band-weight table as the /forecast path above (retro#394) — the two must not
        # disagree about what a relevance score is worth, or a pool recompute would silently
        # re-weight the very rows the live path already weighted.
        weight = s.credibility_weight * evidence_weight * rweight * relevance_weight(s.relevance_score)
        # Un-floored twin of the same product — see the /forecast path (retro#397).
        valve_weight = (
            s.credibility_weight * evidence_weight
            * recency_weight(s.published_date, ref_date, settings.recency_half_life_days, floor=0.0)
            * relevance_weight(s.relevance_score)
        )
        # Reporting-only twin with recency forced to neutral — see the
        # /forecast path above (retro#458 Phase 2).
        age_adjusted_weight = s.credibility_weight * evidence_weight * relevance_weight(s.relevance_score)
        stances.append(s.stance)
        weights.append(weight)
        valve_weights.append(valve_weight)
        age_adjusted_weights.append(age_adjusted_weight)
        relevances.append(s.relevance_score)
        settled_flags.append(s.settled)
        settlement_dates.append(s.settlement_event_date)
        published_dates.append(s.published_date)
        source_ids.append(s.source_id)

    _pool_kwargs = dict(
        relevance_weight_floor=settings.relevance_weight_floor,
        decisiveness_floor=settings.decisiveness_floor,
        thin_evidence_ci_inflation=settings.thin_evidence_ci_inflation,
        defer_on_thin_evidence=settings.defer_on_thin_evidence,
        settlement_min_sources=settings.settlement_min_sources,
        settlement_stance=settings.settlement_stance,
        logit_clamp=settings.logit_clamp,
        pool_dispersion_floor=settings.pool_dispersion_floor,
        claim_direction=req.claim_direction,
        claim_deadline=req.claim_deadline,
        settlement_event_dates=settlement_dates,
        published_dates=published_dates,
        claim_created_at=req.claim_created_at,
        claim_archetype=req.claim_archetype,
        settlement_revalidate=settings.settlement_revalidate,
        settlement_post_deadline_grace_days=settings.settlement_post_deadline_grace_days,
        settlement_quality_floor=settings.settlement_quality_floor,
        # Same clusterer, same text derivation as the live path above — the two must
        # never disagree, or a recompute would re-weight rows /forecast already weighted.
        # This is the first estimator use of claims_detail, which run_pool_aggregate's
        # whitelist comment reserved for exactly this issue (retro#355).
        cluster_ids=_cluster_ids(
            [_cluster_text_of(s) for s in req.sources], _question_hash(req.question or ""),
        ),
        cluster_downweight_exponent=settings.cluster_downweight_exponent,
        valve_weights=valve_weights,
        age_adjusted_weights=age_adjusted_weights,
        source_ids=source_ids,
        max_source_share=settings.max_source_share,
    )
    agg = aggregate_pool(stances, weights, relevances, settled_flags, **_pool_kwargs)
    if agg is not None:
        for idx, demotion_reason in agg.settlement_demotions:
            logger.warning(
                "event=settlement_vote_demoted reason=%s source_index=%d stance=%+.2f event_date=%s",
                demotion_reason, idx, stances[idx], settlement_dates[idx],
            )
        if agg.settlement_suppressed:
            logger.warning("event=settlement_suppressed reason=%s sources=%d", agg.suppression_reason, agg.n)
        # Same match gate as the live path. `question` is optional on this
        # request precisely because the recompute path historically had no need
        # for the claim text; without it the gate cannot run and says so, rather
        # than guessing from the rows.
        agg = await _apply_settlement_match_gate(
            agg,
            question=req.question,
            votes_for_index=lambda i: _settlement_votes(
                req.sources[i].outlet or req.sources[i].url, req.sources[i].claims_detail,
            ),
            rerun=lambda flags: aggregate_pool(stances, weights, relevances, flags, **_pool_kwargs),
            settled_flags=settled_flags,
        )

    if agg is None:
        return PoolAggregateResponse(
            mean=0.0, std=0.0, ci_low=-0.2, ci_high=0.2, articles_used=0,
            settled=False, insufficient_data=True, reason="no_sources",
        )
    if agg.insufficient_reason is not None:
        return PoolAggregateResponse(
            mean=0.0, std=0.0, ci_low=-0.2, ci_high=0.2, articles_used=agg.n,
            settled=False, insufficient_data=True, reason=agg.insufficient_reason,
            # The pool the abstention judged unusable still had a shape — see
            # PoolAggregateResult's docstring on why evidence_mass/thin_evidence
            # ride along on an insufficient result (retro#458 Phase 2 extends
            # that to n_eff/age_adjusted_mass).
            evidence_mass=round(agg.evidence_mass, 4),
            n_eff=round(agg.n_eff, 4),
            age_adjusted_mass=round(agg.age_adjusted_mass, 4),
        )

    logger.info(
        "Pool aggregate: mean=%.3f std=%.3f ci=[%.3f,%.3f] articles=%d mass=%.3f valve_mass=%.4f settled=%s",
        agg.mean, agg.std, agg.ci_low, agg.ci_high, agg.n, agg.evidence_mass, agg.valve_mass, agg.settled,
    )
    return PoolAggregateResponse(
        mean=round(agg.mean, 4), std=round(agg.std, 4),
        ci_low=round(agg.ci_low, 4), ci_high=round(agg.ci_high, 4),
        articles_used=agg.n, settled=agg.settled,
        settlement_suppressed=agg.settlement_suppressed,
        settlement_suppression_reason=agg.suppression_reason,
        settlement_votes_demoted=len(agg.settlement_demotions),
        evidence_mass=round(agg.evidence_mass, 4),
        n_eff=round(agg.n_eff, 4),
        age_adjusted_mass=round(agg.age_adjusted_mass, 4),
    )


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
