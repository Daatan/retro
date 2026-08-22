"""
Multi-provider news search with fallback chain.
Used by the Oracle API (retro/api) and the pipeline (retro/pipeline).

Fallback order:
  1. GDELT Doc API                  (free, no key — primary; news-only, reliable dates)
  1b. GDELT BigQuery GKG            (free tier; historical coverage >3 months; entity-based)
  1c. Google Custom Search JSON     GOOGLE_CSE_API_KEY + GOOGLE_CSE_CX  (relevance-ranked;
                                    100 queries/day free then paid; intended primary once
                                    Google credits land — see _search_google_cse)
  2. SerpAPI (serpapi.com)          SERPAPI_API_KEY
  3. Serper.dev /news endpoint      SERPER_API_KEY
  4. Brave News Search              BRAVE_API_KEY
  4b. Tavily News Search            TAVILY_API_KEY       (after Brave: returns no published_date)
  4c. Newsdata.io                   NEWSDATA_API_KEY     (news API w/ dates; ahead of SERP scrapers)
  5. BrightData SERP API            BRIGHTDATA_API_KEY
  6. Nimbleway SERP API             NIMBLEWAY_API_KEY
  7. ScrapingBee Google Search      SCRAPINGBEE_API_KEY
  9. DataForSEO Google News         DATAFORSEO_API_KEY  (last-resort paid fallback)
 10. DuckDuckGo Lite                (free, no key)

GDELT rate limiting
-------------------
GDELT enforces 1 request / 10 s per IP. When the limit is exceeded it throttles
at the TCP/TLS layer — the handshake stalls for ~25 s before returning HTTP 429.
Because the API runs under multiple gunicorn workers (separate OS processes), a
simple module-level timestamp is not shared and concurrent workers can all pass
the in-process check and fire simultaneously.

The fix is a cross-process slot:
  * A flock(2) on /tmp/gdelt_ratelimit.lock serialises all workers through a
    shared critical section that reads/writes /tmp/gdelt_ratelimit.ts.
  * The lock is held only long enough to read the timestamp, optionally sleep the
    remaining window, and write the new "claimed at" time — not during the HTTP
    call itself.
  * If the remaining window is longer than _GDELT_SLOT_WAIT (1.5 s) the slot is
    considered busy and GDELT is skipped; the caller falls through to the next
    provider immediately rather than stalling the forecast pipeline.
  * HTTP 429 triggers a per-process 60 s cooldown (_GDELT_COOLDOWN_UNTIL) so we
    don't hammer an already-throttled IP every 10 s.

GDELT BigQuery (historical coverage)
-------------------------------------
The GDELT Doc API covers only a 3-month rolling window. For historical research
(duel scoring, retro analysis) we fall back to the GDELT GKG table in BigQuery:
  gdelt-bq.gdeltv2.gkg

Key differences from the Doc API:
  - Entity-based matching only (V2Persons, V2Locations, V2Organizations, AllNames)
  - No article titles — synthesized from the URL path slug
  - No full-text search — queries with only generic nouns will miss many articles
  - No rate limit; GCP free tier = 1 TB/month of query data

Requires a GCP service-account JSON key stored in AWS Secrets Manager at:
  daatan/gcp-service-account-key

The key is loaded once at startup (same _secret() pattern as other providers) and
used to construct a BigQuery client. When the secret is absent the provider is
silently skipped. Queries target gkg_partitioned (DAY-partitioned by ingestion
time) so _PARTITIONTIME pruning works correctly.

All keys are loaded from the environment first, then from AWS Secrets Manager
(daatan/* namespace) as a fallback. See _secret().

Usage:
    from tm.web_search import search_articles

    results = search_articles(
        "Gaza ceasefire negotiations site:timesofisrael.com",
        limit=5,
        date_from=datetime(2024, 1, 1),
        date_to=datetime(2024, 1, 14),
    )
    for r in results:
        print(r.url, r.title)
"""

import fcntl
import logging
import os
import re
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, fields as _dataclass_fields, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlencode

import httpx
from ddgs import DDGS

from .net_guard import safe_get

try:
    import trafilatura as _trafilatura
    from bs4 import BeautifulSoup as _BeautifulSoup
    _SNIPPET_LIBS_AVAILABLE = True
except ImportError:
    _SNIPPET_LIBS_AVAILABLE = False

try:
    import json as _json
    from google.cloud import bigquery as _bigquery
    from google.oauth2 import service_account as _sa
    _BQ_AVAILABLE = True
except ImportError:
    _BQ_AVAILABLE = False

logger = logging.getLogger(__name__)

# Thread-local: stores the provider name that served the last search_articles()
# call in this thread, and the full chain of providers attempted.
# Read via get_last_search_provider() / get_last_search_provider_chain() after the call returns.
_provider_local = threading.local()


def get_last_search_provider() -> str:
    """Return the provider that served the most recent search_articles() call in this thread."""
    return getattr(_provider_local, "name", "none")


def get_last_search_provider_chain() -> list[str]:
    """Return the ordered list of providers attempted in the most recent search_articles() call."""
    return list(getattr(_provider_local, "chain", []))


def _secret(env_var: str, ssm_name: str) -> Optional[str]:
    """Return env var if set, otherwise fetch from SSM Parameter Store.

    Migrated off Secrets Manager per docs#122 (free SSM standard SecureString vs.
    ~$0.40/secret/month). Fails open by design (a provider with no key is simply
    unavailable, not a startup error) — but that means a renamed/missing parameter
    silently disables a provider with no other signal. Logged at WARNING, and
    summarized by _log_unresolved_secrets(), so it shows up without debug logging
    enabled.
    """
    val = os.environ.get(env_var)
    if val:
        return val
    try:
        import boto3
        client = boto3.client("ssm", region_name="eu-central-1")
        val = client.get_parameter(Name=ssm_name, WithDecryption=True)["Parameter"]["Value"].strip()
        logger.info("Loaded %s from SSM Parameter Store", ssm_name)
        return val
    except Exception as e:
        logger.warning(
            "Could not load %s from SSM Parameter Store (env %s not set either): %s "
            "— this provider is silently disabled",
            ssm_name, env_var, e,
        )
        return None


DATAFORSEO_API_KEY: Optional[str] = _secret("DATAFORSEO_API_KEY", "/retro/prod/secrets/DATAFORSEO_API_KEY")
SERPAPI_API_KEY: Optional[str] = _secret("SERPAPI_API_KEY", "/retro/prod/secrets/SERPAPI_API_KEY")
SERPER_API_KEY: Optional[str] = _secret("SERPER_API_KEY", "/retro/prod/secrets/SERPER_API_KEY")
BRAVE_API_KEY: Optional[str] = _secret("BRAVE_API_KEY", "/retro/prod/secrets/BRAVE_API_KEY")
BRIGHTDATA_API_KEY: Optional[str] = _secret("BRIGHTDATA_API_KEY", "/retro/prod/secrets/BRIGHTDATA_API_KEY")
NIMBLEWAY_API_KEY: Optional[str] = _secret("NIMBLEWAY_API_KEY", "/retro/prod/secrets/NIMBLEWAY_API_KEY")
SCRAPINGBEE_API_KEY: Optional[str] = _secret("SCRAPINGBEE_API_KEY", "/retro/prod/secrets/SCRAPINGBEE_API_KEY")
NEWSDATA_API_KEY: Optional[str] = _secret("NEWSDATA_API_KEY", "/retro/prod/secrets/NEWSDATA_API_KEY")
TAVILY_API_KEY: Optional[str] = _secret("TAVILY_API_KEY", "/retro/prod/secrets/TAVILY_API_KEY")
# Google Programmable Search (Custom Search JSON API). Needs BOTH an API key and a
# search-engine id (cx). Inert until both are configured — see _search_google_cse.
GOOGLE_CSE_API_KEY: Optional[str] = _secret("GOOGLE_CSE_API_KEY", "/retro/prod/secrets/GOOGLE_CSE_API_KEY")
GOOGLE_CSE_CX: Optional[str] = _secret("GOOGLE_CSE_CX", "/retro/prod/secrets/GOOGLE_CSE_CX")
GCP_SA_KEY_JSON: Optional[str] = _secret("GCP_SA_KEY_JSON", "/retro/prod/secrets/GCP_SA_KEY_JSON")

# news-indexer — local semantic index (https://scrapper.daatan.com)
NEWS_INDEXER_URL: Optional[str] = _secret("NEWS_INDEXER_URL", "/retro/prod/secrets/NEWS_INDEXER_URL")
NEWS_INDEXER_API_KEY: Optional[str] = _secret("NEWS_INDEXER_API_KEY", "/retro/prod/secrets/NEWS_INDEXER_API_KEY")


def _log_unresolved_secrets() -> None:
    """Startup/refresh summary of which provider secrets failed to resolve.

    _secret()'s fail-open behavior means a missing key silently disables a
    provider with no other signal — this makes that visible in one line
    instead of requiring someone to notice 13 separate per-key log lines.
    """
    keys = {
        "/retro/prod/secrets/DATAFORSEO_API_KEY": DATAFORSEO_API_KEY,
        "/retro/prod/secrets/SERPAPI_API_KEY": SERPAPI_API_KEY,
        "/retro/prod/secrets/SERPER_API_KEY": SERPER_API_KEY,
        "/retro/prod/secrets/BRAVE_API_KEY": BRAVE_API_KEY,
        "/retro/prod/secrets/BRIGHTDATA_API_KEY": BRIGHTDATA_API_KEY,
        "/retro/prod/secrets/NIMBLEWAY_API_KEY": NIMBLEWAY_API_KEY,
        "/retro/prod/secrets/SCRAPINGBEE_API_KEY": SCRAPINGBEE_API_KEY,
        "/retro/prod/secrets/NEWSDATA_API_KEY": NEWSDATA_API_KEY,
        "/retro/prod/secrets/TAVILY_API_KEY": TAVILY_API_KEY,
        "/retro/prod/secrets/GOOGLE_CSE_API_KEY": GOOGLE_CSE_API_KEY,
        "/retro/prod/secrets/GOOGLE_CSE_CX": GOOGLE_CSE_CX,
        "/retro/prod/secrets/GCP_SA_KEY_JSON": GCP_SA_KEY_JSON,
    }
    missing = [name for name, value in keys.items() if not value]
    if missing:
        logger.warning(
            "%d/%d search-provider secrets unresolved, providers silently disabled: %s",
            len(missing), len(keys), ", ".join(missing),
        )


_log_unresolved_secrets()

_KEY_LOADED_AT: float = time.time()

# Cached BigQuery client — created lazily on first use, invalidated on key refresh.
_BQ_CLIENT: Optional[object] = None


def _get_bq_client() -> object:
    """Return a cached BigQuery client, creating it from GCP_SA_KEY_JSON if needed."""
    global _BQ_CLIENT
    if _BQ_CLIENT is not None:
        return _BQ_CLIENT
    if not _BQ_AVAILABLE:
        raise RuntimeError("google-cloud-bigquery not installed")
    if not GCP_SA_KEY_JSON:
        raise RuntimeError("GCP_SA_KEY_JSON / daatan/gcp-service-account-key not configured")
    info = _json.loads(GCP_SA_KEY_JSON)
    creds = _sa.Credentials.from_service_account_info(
        info,
        scopes=["https://www.googleapis.com/auth/bigquery"],
    )
    _BQ_CLIENT = _bigquery.Client(credentials=creds, project=info["project_id"])
    return _BQ_CLIENT
_KEY_MAX_AGE_SECONDS: float = 86400.0  # 24h


def _refresh_keys_if_stale() -> None:
    """Re-fetch all search API keys from Secrets Manager if >24h old.

    Keys are loaded once at module import. Long-running processes (the batch
    pipeline can run for days) would use stale keys after rotation. This
    function is called at the top of search_articles() to catch that case.
    """
    global DATAFORSEO_API_KEY, SERPAPI_API_KEY, SERPER_API_KEY, BRAVE_API_KEY
    global BRIGHTDATA_API_KEY, NIMBLEWAY_API_KEY, SCRAPINGBEE_API_KEY, NEWSDATA_API_KEY
    global TAVILY_API_KEY, GOOGLE_CSE_API_KEY, GOOGLE_CSE_CX
    global GCP_SA_KEY_JSON, _BQ_CLIENT, _KEY_LOADED_AT
    if time.time() - _KEY_LOADED_AT < _KEY_MAX_AGE_SECONDS:
        return
    logger.info("Refreshing search API keys from SSM Parameter Store (>24h since last fetch)")
    DATAFORSEO_API_KEY = _secret("DATAFORSEO_API_KEY", "/retro/prod/secrets/DATAFORSEO_API_KEY")
    SERPAPI_API_KEY = _secret("SERPAPI_API_KEY", "/retro/prod/secrets/SERPAPI_API_KEY")
    SERPER_API_KEY = _secret("SERPER_API_KEY", "/retro/prod/secrets/SERPER_API_KEY")
    BRAVE_API_KEY = _secret("BRAVE_API_KEY", "/retro/prod/secrets/BRAVE_API_KEY")
    BRIGHTDATA_API_KEY = _secret("BRIGHTDATA_API_KEY", "/retro/prod/secrets/BRIGHTDATA_API_KEY")
    NIMBLEWAY_API_KEY = _secret("NIMBLEWAY_API_KEY", "/retro/prod/secrets/NIMBLEWAY_API_KEY")
    SCRAPINGBEE_API_KEY = _secret("SCRAPINGBEE_API_KEY", "/retro/prod/secrets/SCRAPINGBEE_API_KEY")
    NEWSDATA_API_KEY = _secret("NEWSDATA_API_KEY", "/retro/prod/secrets/NEWSDATA_API_KEY")
    TAVILY_API_KEY = _secret("TAVILY_API_KEY", "/retro/prod/secrets/TAVILY_API_KEY")
    GOOGLE_CSE_API_KEY = _secret("GOOGLE_CSE_API_KEY", "/retro/prod/secrets/GOOGLE_CSE_API_KEY")
    GOOGLE_CSE_CX = _secret("GOOGLE_CSE_CX", "/retro/prod/secrets/GOOGLE_CSE_CX")
    new_gcp = _secret("GCP_SA_KEY_JSON", "/retro/prod/secrets/GCP_SA_KEY_JSON")
    if new_gcp != GCP_SA_KEY_JSON:
        GCP_SA_KEY_JSON = new_gcp
        _BQ_CLIENT = None  # force client rebuild with new credentials
    _KEY_LOADED_AT = time.time()
    _log_unresolved_secrets()



_DDG_LAST_CALL: float = 0.0
DDG_MIN_INTERVAL = 2.0

# Curated credible news domains for the last-resort "trusted sites" fallback
# (_search_trusted_sites). Mirrors gnews_ingest.SOURCES_CONFIG; duplicated here
# rather than imported because gnews_ingest imports this module (would be a
# circular import). TODO: unify to one source-of-truth in a neutral tm module.
TRUSTED_SEARCH_DOMAINS: list[str] = [
    "reuters.com", "apnews.com", "bbc.com", "nytimes.com", "ft.com",
    "theguardian.com", "washingtonpost.com", "bloomberg.com", "axios.com",
    "aljazeera.com", "timesofisrael.com", "jpost.com", "haaretz.com",
    "haaretz.co.il", "en.globes.co.il", "ynetnews.com", "israelhayom.com",
    "news.walla.co.il", "mako.co.il", "maariv.co.il", "13tv.co.il", "calcalist.co.il",
]
_TRUSTED_SITES_BATCH_SIZE = 5   # domains OR'd into one DDG query
_TRUSTED_SITES_MAX_BATCHES = 3  # cap DDG calls (bounded vs DDG_MIN_INTERVAL + /forecast budget)

# ── GDELT cross-process rate limiting ────────────────────────────────────────
GDELT_MIN_INTERVAL = 10.0   # documented GDELT rate limit: 1 req / 10 s
_GDELT_SLOT_WAIT   = 1.5    # max seconds we're willing to wait for a slot before skipping
_GDELT_COOLDOWN_UNTIL: float = 0.0   # per-process; epoch time until which GDELT is skipped after a 429
_GDELT_DOC_BROKEN_UNTIL: float = 0.0  # epoch; GDELT Doc skipped after repeated connection failures
_GDELT_DOC_FAIL_COUNT: int = 0        # consecutive connection failures (resets on success/empty result)
_GDELT_DOC_FAIL_THRESHOLD: int = 2    # open circuit after this many consecutive failures
_GDELT_DOC_BREAK_SECS: float = 3600.0 # 1-hour circuit break

_GDELT_LOCK_PATH = Path(tempfile.gettempdir()) / "gdelt_ratelimit.lock"
_GDELT_TS_PATH   = Path(tempfile.gettempdir()) / "gdelt_ratelimit.ts"

_SUBDOMAIN_STRIP_RE = re.compile(r"^(?:feeds|rss|m|amp|mobile)\.")


class _GDELTSlotBusy(Exception):
    """Raised by _gdelt_acquire_slot when the cross-process slot is not available."""


def _gdelt_acquire_slot() -> None:
    """Claim the next available GDELT request slot, enforced across all OS processes.

    Acquires an exclusive flock on _GDELT_LOCK_PATH, reads the timestamp of the
    last claimed request from _GDELT_TS_PATH, sleeps for any remaining window, then
    writes the current time to claim the slot before releasing the lock.

    Raises _GDELTSlotBusy if:
      - the remaining window exceeds _GDELT_SLOT_WAIT (slot too fresh — skip GDELT
        rather than stalling the forecast pipeline), or
      - the lock itself cannot be acquired within _GDELT_SLOT_WAIT (another worker
        is mid-claim).

    Falls back to no-op if file operations fail (e.g. read-only /tmp); callers
    are then responsible for their own rate limiting.
    """
    deadline = time.time() + _GDELT_SLOT_WAIT
    try:
        lock_fd = open(_GDELT_LOCK_PATH, "w")
        try:
            # Spin-acquire with deadline so we don't stall indefinitely.
            while True:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.time() >= deadline:
                        raise _GDELTSlotBusy("another worker holds the GDELT lock")
                    time.sleep(0.05)

            # Lock held — read the last claimed timestamp.
            last_ts = 0.0
            if _GDELT_TS_PATH.exists():
                try:
                    last_ts = float(_GDELT_TS_PATH.read_text().strip())
                except ValueError:
                    pass

            wait = GDELT_MIN_INTERVAL - (time.time() - last_ts)
            if wait > _GDELT_SLOT_WAIT:
                raise _GDELTSlotBusy(
                    f"GDELT slot in use, next available in {wait:.1f}s"
                )

            if wait > 0:
                logger.debug("GDELT rate-limit: waiting %.2fs (cross-process)", wait)
                time.sleep(wait)

            # Claim the slot.
            _GDELT_TS_PATH.write_text(str(time.time()))

        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            lock_fd.close()

    except _GDELTSlotBusy:
        raise
    except OSError as exc:
        # File operations unavailable — silently degrade to no rate limiting.
        logger.debug("GDELT file lock unavailable (%s): rate limit unenforced", exc)


_DATAFORSEO_QUOTA_EXHAUSTED: bool = False
_SERPAPI_QUOTA_EXHAUSTED: bool = False
_BRAVE_QUOTA_EXHAUSTED: bool = False
_SERPER_QUOTA_EXHAUSTED: bool = False
_BRIGHTDATA_QUOTA_EXHAUSTED: bool = False
_NIMBLEWAY_QUOTA_EXHAUSTED: bool = False
_SCRAPINGBEE_QUOTA_EXHAUSTED: bool = False
_NEWSDATA_QUOTA_EXHAUSTED: bool = False
_TAVILY_QUOTA_EXHAUSTED: bool = False
_GOOGLE_CSE_QUOTA_EXHAUSTED: bool = False

_QUOTA_STATE_PATH = Path(tempfile.gettempdir()) / "quota_exhausted.json"

# Provider name (as persisted) -> the module global holding its flag.
_QUOTA_FLAG_GLOBALS: dict[str, str] = {
    "dataforseo":  "_DATAFORSEO_QUOTA_EXHAUSTED",
    "serpapi":     "_SERPAPI_QUOTA_EXHAUSTED",
    "serper":      "_SERPER_QUOTA_EXHAUSTED",
    "brave":       "_BRAVE_QUOTA_EXHAUSTED",
    "brightdata":  "_BRIGHTDATA_QUOTA_EXHAUSTED",
    "nimbleway":   "_NIMBLEWAY_QUOTA_EXHAUSTED",
    "scrapingbee": "_SCRAPINGBEE_QUOTA_EXHAUSTED",
    "newsdata":    "_NEWSDATA_QUOTA_EXHAUSTED",
    "tavily":      "_TAVILY_QUOTA_EXHAUSTED",
    "google_cse":  "_GOOGLE_CSE_QUOTA_EXHAUSTED",
}

# A quota flag is sticky on purpose — once a provider answers "out of credit" we must
# not re-ask it on every search. But it must not be *permanent*: provider quotas reset
# on their own schedule (Google CSE daily, Serper/SerpAPI monthly), and any transient
# 401/429 trips the same flag. Nothing ever cleared it, so a single bad response
# disabled a provider for the life of the host — and, because the flag is persisted,
# for every process started afterwards. Re-probe once a flag is older than this.
_QUOTA_TTL_SECONDS: float = 86400.0  # 24h

# Provider name -> epoch seconds when its flag was last set. Stamped by
# _persist_quota_state(), so the many sites that just assign the global keep working.
_QUOTA_SET_AT: dict[str, float] = {}


def _load_quota_state() -> None:
    """Seed in-process quota flags from disk so a gunicorn SIGHUP reload doesn't forget exhausted providers.

    Accepts both the current `{provider: {"exhausted": bool, "set_at": float}}` shape and
    the legacy `{provider: bool}` one. A legacy entry carries no timestamp, so it is stamped
    as first-seen *now*: it still expires, just a full TTL from the first load rather than
    immediately. Every True flag therefore always has a timestamp, and nothing stays
    disabled forever.
    """
    now = time.time()
    try:
        data = _json.loads(_QUOTA_STATE_PATH.read_text())
    except Exception:
        return
    for name, attr in _QUOTA_FLAG_GLOBALS.items():
        entry = data.get(name, False)
        if isinstance(entry, dict):
            exhausted = bool(entry.get("exhausted", False))
            set_at = float(entry.get("set_at") or now)
        else:
            exhausted = bool(entry)
            set_at = now
        globals()[attr] = exhausted
        if exhausted:
            _QUOTA_SET_AT[name] = set_at
        else:
            _QUOTA_SET_AT.pop(name, None)


def _persist_quota_state() -> None:
    """Write current quota exhaustion flags to disk so they survive worker reloads.

    Stamps `set_at` for any flag that is newly True, so callers can keep setting the
    module global directly and still get a TTL.
    """
    now = time.time()
    payload = {}
    for name, attr in _QUOTA_FLAG_GLOBALS.items():
        exhausted = bool(globals()[attr])
        if exhausted:
            _QUOTA_SET_AT.setdefault(name, now)
        else:
            _QUOTA_SET_AT.pop(name, None)
        payload[name] = {"exhausted": exhausted, "set_at": _QUOTA_SET_AT.get(name, 0.0)}
    try:
        _QUOTA_STATE_PATH.write_text(_json.dumps(payload))
    except Exception:
        pass


def _expire_stale_quota_flags() -> None:
    """Clear any quota flag older than _QUOTA_TTL_SECONDS so the provider is retried.

    Costs at most one failed request per provider per TTL when a provider really is out
    of credit; in exchange, a provider that has been topped up (or was flagged by a
    transient error) comes back on its own instead of staying dark forever.
    """
    now = time.time()
    revived = []
    for name, attr in _QUOTA_FLAG_GLOBALS.items():
        if not globals()[attr]:
            _QUOTA_SET_AT.pop(name, None)
            continue
        # A flag set in-process (the callers just assign the global) has no timestamp
        # yet — stamp it now rather than reading it as epoch 0, which would expire the
        # flag on the very next sweep and defeat its purpose. Only _load_quota_state()
        # records an explicit 0.0, for the legacy on-disk booleans that carry no time.
        set_at = _QUOTA_SET_AT.setdefault(name, now)
        if now - set_at > _QUOTA_TTL_SECONDS:
            revived.append(name)
    if not revived:
        return
    for name in revived:
        globals()[_QUOTA_FLAG_GLOBALS[name]] = False
        _QUOTA_SET_AT.pop(name, None)
    logger.info("Quota flag TTL expired — re-enabling provider(s): %s", ", ".join(revived))
    _persist_quota_state()


_load_quota_state()


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    source: str = ""
    published_date: str = ""
    _prefetched_text: Optional[str] = field(default=None)
    # Caller-supplied gatekeeper verdict, carried through so _process_article can reuse it
    # instead of re-judging (see forecast_api.forecaster._supplied_verdict). Both must be
    # present to count as a verdict. Unused by the pipeline's own search providers.
    _supplied_relevance: Optional[float] = field(default=None)
    _supplied_is_prediction: Optional[bool] = field(default=None)
    # Caller-supplied language of the article text ("Hebrew", "ru", …), threaded to the
    # gatekeeper/extractor prompts as a hint (retro#417). Unused by the pipeline's own
    # search providers.
    _language: Optional[str] = field(default=None)


# The keys a remote provider payload is allowed to set, derived from the dataclass itself
# so it cannot drift as fields are added. Underscore-prefixed fields are caller-supplied
# in-process (prefetched text, gatekeeper verdict, language hint) and are deliberately
# NOT settable from a wire payload — `**h` could never have reached them anyway.
_SEARCH_RESULT_WIRE_FIELDS = frozenset(
    f.name for f in _dataclass_fields(SearchResult) if not f.name.startswith("_")
)


def _search_result_from_payload(h: dict) -> tuple["SearchResult", set[str]]:
    """Build a SearchResult from a provider's JSON object, ignoring keys we don't know.

    retro#459. The naive `SearchResult(**h)` makes every remote response a strict schema
    contract enforced by `TypeError` — so one added key on news-indexer's `/search` would
    take the whole free first-in-chain provider offline, push the Oracle onto paid SERP,
    and say so only in a WARNING line. news-indexer has already widened a *different*
    payload (`/context`) twice this way; daatan survived both because Zod strips unknown
    keys. This gives the search path the same tolerance.

    Returns the result plus the set of keys that were ignored, so the caller can log a
    real schema drift without failing on it.
    """
    unknown = set(h) - _SEARCH_RESULT_WIRE_FIELDS
    return SearchResult(**{k: v for k, v in h.items() if k in _SEARCH_RESULT_WIRE_FIELDS}), unknown


# ──────────────────────────────────────────────
# Date helpers shared across providers
# ──────────────────────────────────────────────

def _date_query_suffix(date_from: Optional[datetime], date_to: Optional[datetime]) -> str:
    """Append Google after:/before: operators for providers without native date params."""
    parts = []
    if date_from:
        parts.append(f"after:{date_from.strftime('%Y-%m-%d')}")
    if date_to:
        parts.append(f"before:{date_to.strftime('%Y-%m-%d')}")
    return (" " + " ".join(parts)) if parts else ""


def _google_tbs(date_from: Optional[datetime], date_to: Optional[datetime]) -> Optional[str]:
    """Google custom-date-range ``tbs`` value for either or both bounds.

    ``cdr:1`` with only ``cd_max:`` (or only ``cd_min:``) is valid Google
    syntax. A date_to-only window — the born-true pre-creation research leg —
    must NOT silently drop the restriction and answer a strictly historical
    query with this week's news (retro#562).
    """
    if not date_from and not date_to:
        return None
    parts = ["cdr:1"]
    if date_from:
        parts.append(f"cd_min:{date_from.month}/{date_from.day}/{date_from.year}")
    if date_to:
        parts.append(f"cd_max:{date_to.month}/{date_to.day}/{date_to.year}")
    return ",".join(parts)


_RELATIVE_DATE_RE = re.compile(
    r"^\s*(\d+)\s+(minute|hour|day|week|month|year)s?\s+ago\s*$", re.IGNORECASE
)
_RELATIVE_DATE_UNIT_DAYS = {
    "minute": 0, "hour": 0, "day": 1, "week": 7, "month": 30, "year": 365,
}


def _absolutize_relative_date(published: str, now: Optional[datetime] = None) -> str:
    """Convert a SERP relative date ('1 day ago', '3 weeks ago') to YYYY-MM-DD.

    SerpAPI/Serper return relative strings that ``_filter_by_date``'s %Y-%m-%d
    parse can't act on (ValueError → kept as benefit of the doubt), which made
    the post-filter inert on exactly the legs that needed it (retro#562).
    Anything unrecognized passes through unchanged, keeping the benefit of the
    doubt for genuinely undated or oddly-dated entries.
    """
    m = _RELATIVE_DATE_RE.match(published or "")
    if not m:
        return published
    n, unit = int(m.group(1)), m.group(2).lower()
    dt = (now or datetime.now()) - timedelta(days=n * _RELATIVE_DATE_UNIT_DAYS[unit])
    return dt.strftime("%Y-%m-%d")


def _filter_by_date(
    results: List["SearchResult"],
    date_from: Optional[datetime],
    date_to: Optional[datetime],
) -> List["SearchResult"]:
    """Post-filter results to the requested window using published_date.
    Relative dates are absolutized first; entries without a parseable date are
    kept (benefit of the doubt).
    """
    if not date_from and not date_to:
        return results
    out = []
    for r in results:
        if not r.published_date:
            out.append(r)
            continue
        try:
            d = datetime.strptime(
                _absolutize_relative_date(r.published_date)[:10], "%Y-%m-%d"
            )
            if date_from and d < date_from:
                continue
            if date_to and d > date_to:
                continue
            out.append(r)
        except ValueError:
            out.append(r)
    return out


# ──────────────────────────────────────────────
# Provider: DataForSEO Google News
# ──────────────────────────────────────────────

def _search_dataforseo(
    query: str,
    limit: int,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
) -> List[SearchResult]:
    global _DATAFORSEO_QUOTA_EXHAUSTED
    if not DATAFORSEO_API_KEY:
        raise RuntimeError("DATAFORSEO_API_KEY not set")

    task: dict = {
        "keyword": query,
        "language_code": "en",
        "location_code": 2840,  # United States
        "depth": min(limit, 100),
    }
    if date_from:
        task["date_from"] = date_from.strftime("%Y-%m-%d")
    if date_to:
        task["date_to"] = date_to.strftime("%Y-%m-%d")

    r = httpx.post(
        "https://api.dataforseo.com/v3/serp/google/news/live/advanced",
        headers={
            "Authorization": f"Basic {DATAFORSEO_API_KEY}",
            "Content-Type": "application/json",
        },
        json=[task],
        timeout=20,
    )
    r.raise_for_status()
    data = r.json()
    api_task = (data.get("tasks") or [{}])[0]
    task_code = api_task.get("status_code", 0)
    # 40101 = account suspended, 40201/40202/40203 = insufficient funds / billing
    if task_code in (40101, 40201, 40202, 40203):
        _DATAFORSEO_QUOTA_EXHAUSTED = True
        _persist_quota_state()
        raise RuntimeError(f"DataForSEO billing/quota error: status_code={task_code}")
    items = ((api_task.get("result") or [{}])[0].get("items")) or []
    results = []
    for item in items[:limit]:
        url = item.get("url", "")
        if not url:
            continue
        source = (item.get("source") or {}).get("name", "") or item.get("domain", "")
        # DataForSEO returns "timestamp" (e.g. "2024-09-20 14:32:00 +00:00"), not "date_published"
        raw_date = item.get("timestamp") or item.get("date_published", "")
        published_date = raw_date[:10] if raw_date else ""
        results.append(SearchResult(
            title=item.get("title", ""),
            url=url,
            snippet=item.get("snippet", ""),
            source=source,
            published_date=published_date,
        ))
    # DataForSEO date_from/date_to params are not reliably enforced by Google News;
    # post-filter to ensure we stay within the requested window.
    return _filter_by_date(results, date_from, date_to)


# ──────────────────────────────────────────────
# Provider: SerpAPI (serpapi.com) news
# ──────────────────────────────────────────────

def _search_serpapi_news(
    query: str,
    limit: int,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
) -> List[SearchResult]:
    global _SERPAPI_QUOTA_EXHAUSTED
    if not SERPAPI_API_KEY:
        raise RuntimeError("SERPAPI_API_KEY not set")

    # SerpAPI news (tbm=nws) doesn't support site: operator — strip it.
    # Results are filtered by domain by the caller anyway.
    import re as _re
    clean_query = _re.sub(r"\bsite:\S+\s*", "", query).strip()

    params: dict = {
        "q": clean_query,
        "tbm": "nws",
        "num": min(limit, 100),
        "api_key": SERPAPI_API_KEY,
    }
    tbs = _google_tbs(date_from, date_to)
    if tbs:
        params["tbs"] = tbs

    r = httpx.get(
        "https://serpapi.com/search.json",
        params=params,
        timeout=12,
    )
    if r.status_code == 429:
        body = r.text.lower()
        if "run out" in body or "quota" in body or "searches" in body:
            _SERPAPI_QUOTA_EXHAUSTED = True
            _persist_quota_state()
            raise RuntimeError("SerpAPI quota exhausted")
        raise RuntimeError("SerpAPI rate-limited (429)")
    r.raise_for_status()
    items = r.json().get("news_results", [])

    results = [
        SearchResult(
            title=item.get("title", ""),
            url=item.get("link", ""),
            snippet=item.get("snippet", ""),
            source=item.get("source", _extract_domain(item.get("link", ""))),
            published_date=_absolutize_relative_date(item.get("date", "")),
        )
        for item in items[:limit]
        if item.get("link")
    ]
    return _filter_by_date(results, date_from, date_to)


# ──────────────────────────────────────────────
# Provider: Serper.dev /news
# ──────────────────────────────────────────────

def _search_serper_news(
    query: str,
    limit: int,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
) -> List[SearchResult]:
    global _SERPER_QUOTA_EXHAUSTED
    if not SERPER_API_KEY:
        raise RuntimeError("SERPER_API_KEY not set")

    body: dict = {"q": query, "num": limit}
    # Serper tbs date range: cdr:1[,cd_min:M/D/YYYY][,cd_max:M/D/YYYY] — either
    # bound alone is valid (retro#562).
    tbs = _google_tbs(date_from, date_to)
    if tbs:
        body["tbs"] = tbs

    r = httpx.post(
        "https://google.serper.dev/news",
        json=body,
        headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
        timeout=10,
    )
    if r.status_code == 400 and "credits" in r.text.lower():
        _SERPER_QUOTA_EXHAUSTED = True
        _persist_quota_state()
        raise RuntimeError("Serper quota exhausted (no credits)")
    r.raise_for_status()
    data = r.json()
    items = data.get("news", [])

    results = [
        SearchResult(
            title=item.get("title", ""),
            url=item.get("link", ""),
            snippet=item.get("snippet", ""),
            source=_clean_source("", item.get("link", "")),
            published_date=_absolutize_relative_date(item.get("date", "")),
        )
        for item in items[:limit]
        if item.get("link")
    ]
    return _filter_by_date(results, date_from, date_to)


# ──────────────────────────────────────────────
# Provider: Google Programmable Search (Custom Search JSON API)
# ──────────────────────────────────────────────

def _cse_published_date(item: dict) -> str:
    """Best-effort publish date from a CSE result's pagemap metatags. '' if absent."""
    pagemap = item.get("pagemap") or {}
    metatags = (pagemap.get("metatags") or [{}])[0] if isinstance(pagemap.get("metatags"), list) else {}
    for key in ("article:published_time", "og:updated_time", "datepublished", "date"):
        val = metatags.get(key)
        if val:
            return str(val)[:10]
    return ""


def _search_google_cse(
    query: str,
    limit: int,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
) -> List[SearchResult]:
    """Google Custom Search JSON API (Programmable Search Engine).

    Requires BOTH GOOGLE_CSE_API_KEY and a search-engine id GOOGLE_CSE_CX; the
    caller (search_articles) only invokes this when both are set, so it is inert
    until configured. The engine (cx) is configured in Google's panel — set it to
    "search the entire web" or scope it to news sites.

    NOTE (unverified against the live API — no key until Google credits land):
    the response shape and 429 quota behaviour are assumed from CSE docs and need
    one live call to confirm before being trusted. Defensive parsing keeps a shape
    surprise degrading to empty results rather than raising.

    Caveats / follow-ups:
      - CSE returns at most 10 results per request; `limit > 10` under-delivers
        (pagination via `start` is a follow-up).
      - No server-side date filtering: a malformed `sort=date:r:…` makes CSE 400
        and silently kills the provider, so we send no date param and rely on
        `_filter_by_date` post-hoc. Date-aware CSE is a follow-up once testable.
      - Free tier is 100 queries/day, then HTTP 429 → quota-exhausted for the day.
    """
    global _GOOGLE_CSE_QUOTA_EXHAUSTED
    if not (GOOGLE_CSE_API_KEY and GOOGLE_CSE_CX):
        raise RuntimeError("GOOGLE_CSE_API_KEY / GOOGLE_CSE_CX not set")

    r = httpx.get(
        "https://www.googleapis.com/customsearch/v1",
        params={
            "key": GOOGLE_CSE_API_KEY,
            "cx": GOOGLE_CSE_CX,
            "q": query,
            "num": min(limit, 10),  # CSE hard cap is 10 per request
        },
        timeout=10,
    )
    if r.status_code == 429:
        _GOOGLE_CSE_QUOTA_EXHAUSTED = True
        _persist_quota_state()
        raise RuntimeError("Google CSE quota exhausted (429)")
    r.raise_for_status()
    data = r.json()

    results = []
    for item in (data.get("items") or [])[:limit]:
        link = item.get("link") or ""
        if not link:
            continue
        results.append(SearchResult(
            title=item.get("title", "") or "",
            url=link,
            snippet=item.get("snippet", "") or "",
            source=_clean_source(item.get("displayLink", "") or "", link),
            published_date=_cse_published_date(item),
        ))
    return _filter_by_date(results, date_from, date_to)


# ──────────────────────────────────────────────
# Provider: Brave News Search
# ──────────────────────────────────────────────

def _search_brave_news(
    query: str,
    limit: int,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
) -> List[SearchResult]:
    global _BRAVE_QUOTA_EXHAUSTED
    if not BRAVE_API_KEY:
        raise RuntimeError("BRAVE_API_KEY not set")

    # Brave has no arbitrary date-range param; inject Google-style operators into query.
    dated_query = query + _date_query_suffix(date_from, date_to)
    params: dict = {
        "q": dated_query,
        "count": min(limit, 20),
        "search_lang": "en",
        "country": "us",
    }

    r = httpx.get(
        "https://api.search.brave.com/res/v1/news/search",
        params=params,
        headers={
            "Accept": "application/json",
            "X-Subscription-Token": BRAVE_API_KEY,
        },
        timeout=10,
    )
    if r.status_code == 402:
        _BRAVE_QUOTA_EXHAUSTED = True
        _persist_quota_state()
        raise RuntimeError("Brave quota exhausted (402)")
    r.raise_for_status()

    items = r.json().get("results", [])
    results = [
        SearchResult(
            title=item.get("title", ""),
            url=item.get("url", ""),
            snippet=item.get("description", ""),
            source=_clean_source(item.get("meta_url", {}).get("hostname", ""), item.get("url", "")),
            published_date=item.get("age", ""),
        )
        for item in items[:limit]
        if item.get("url")
    ]
    return _filter_by_date(results, date_from, date_to)


# ──────────────────────────────────────────────
# Provider: BrightData SERP API
# ──────────────────────────────────────────────

def _search_brightdata(
    query: str,
    limit: int,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
) -> List[SearchResult]:
    global _BRIGHTDATA_QUOTA_EXHAUSTED
    if not BRIGHTDATA_API_KEY:
        raise RuntimeError("BRIGHTDATA_API_KEY not set")

    # Inject Google date operators since BrightData proxies Google.
    # NOTE: zone "serp_api1" returns empty body (zone not configured); keeping the
    # implementation correct for when the zone is fixed.
    dated_query = query + _date_query_suffix(date_from, date_to)
    search_url = "https://www.google.com/search?" + urlencode({"q": dated_query, "gl": "us", "hl": "en"})
    r = httpx.post(
        "https://api.brightdata.com/request",
        json={"zone": "serp_api1", "url": search_url, "format": "raw"},
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {BRIGHTDATA_API_KEY}"},
        timeout=20,
    )
    if r.status_code in (401, 402):
        _BRIGHTDATA_QUOTA_EXHAUSTED = True
        _persist_quota_state()
        raise RuntimeError(f"BrightData quota/auth error ({r.status_code})")
    r.raise_for_status()

    html = r.text
    if not html:
        return []

    # Google's CSS classes rotate; try several known variants
    for title_cls in ["LC20lb", "DKV0Md", "vvjwJb"]:
        result_pat = re.compile(
            rf'href="(https://[^"#]+)"[^>]*>[^<]*<h3[^>]*class="{title_cls}[^"]*">([^<]+)</h3>'
        )
        pairs = [(m.group(1), m.group(2)) for m in result_pat.finditer(html)]
        if pairs:
            break

    snippet_pat = re.compile(r'class="(?:VwiC3b|yXK7lf|MUxGbd)[^"]*"[^>]*>(.*?)</div>')
    snippets = [re.sub(r'<[^>]+>', '', m.group(1)).strip() for m in snippet_pat.finditer(html)]

    results = [
        SearchResult(
            title=title,
            url=url,
            snippet=snippets[i] if i < len(snippets) else "",
            source=_clean_source("", url),
        )
        for i, (url, title) in enumerate(pairs[:limit])
    ]
    return _filter_by_date(results, date_from, date_to)


# ──────────────────────────────────────────────
# Provider: Nimbleway SERP API
# ──────────────────────────────────────────────

def _search_nimbleway(
    query: str,
    limit: int,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
) -> List[SearchResult]:
    global _NIMBLEWAY_QUOTA_EXHAUSTED
    if not NIMBLEWAY_API_KEY:
        raise RuntimeError("NIMBLEWAY_API_KEY not set")

    dated_query = query + _date_query_suffix(date_from, date_to)
    r = httpx.post(
        "https://api.webit.live/api/v1/realtime/serp",
        json={"search_engine": "google_search", "country": "US", "query": dated_query, "parse": True},
        headers={"Authorization": f"Bearer {NIMBLEWAY_API_KEY}", "Content-Type": "application/json"},
        timeout=20,
    )
    if r.status_code == 402:
        _NIMBLEWAY_QUOTA_EXHAUSTED = True
        _persist_quota_state()
        raise RuntimeError("Nimbleway quota exhausted (402)")
    r.raise_for_status()

    data = r.json()
    if data.get("status") != "success":
        raise RuntimeError(f"Nimbleway error: {data.get('status')}")

    items = data.get("parsing", {}).get("entities", {}).get("OrganicResult", [])
    results = [
        SearchResult(
            title=item.get("title", ""),
            url=item.get("url", ""),
            snippet=item.get("snippet", ""),
            source=_clean_source(item.get("cleaned_domain") or "", item.get("url", "")),
        )
        for item in items[:limit]
        if item.get("url")
    ]
    return _filter_by_date(results, date_from, date_to)


# ──────────────────────────────────────────────
# Provider: ScrapingBee Google Search
# ──────────────────────────────────────────────

def _search_scrapingbee(
    query: str,
    limit: int,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
) -> List[SearchResult]:
    global _SCRAPINGBEE_QUOTA_EXHAUSTED
    if not SCRAPINGBEE_API_KEY:
        raise RuntimeError("SCRAPINGBEE_API_KEY not set")

    dated_query = query + _date_query_suffix(date_from, date_to)
    r = httpx.get(
        "https://app.scrapingbee.com/api/v1/store/google",
        params={"api_key": SCRAPINGBEE_API_KEY, "search": dated_query, "nb_results": limit},
        timeout=20,
    )
    if r.status_code == 402:
        _SCRAPINGBEE_QUOTA_EXHAUSTED = True
        _persist_quota_state()
        raise RuntimeError("ScrapingBee quota exhausted (402)")
    if r.status_code == 401:
        _SCRAPINGBEE_QUOTA_EXHAUSTED = True
        _persist_quota_state()
        raise RuntimeError("ScrapingBee unauthorized (401) — invalid key, disabling for session")
    r.raise_for_status()

    data = r.json()
    items = (
        data.get("news_results")
        or data.get("top_stories")
        or data.get("organic_results")
        or []
    )
    results = [
        SearchResult(
            title=item.get("title", ""),
            url=item.get("url", ""),
            snippet=item.get("description", ""),
            source=_clean_source(item.get("domain") or "", item.get("url", "")),
            published_date=item.get("date_utc") or item.get("date") or "",
        )
        for item in items[:limit]
        if item.get("url")
    ]
    return _filter_by_date(results, date_from, date_to)


# ──────────────────────────────────────────────
# Provider: GDELT Doc API (free, reliable dates)
# ──────────────────────────────────────────────

def _search_gdelt(
    query: str,
    limit: int,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
) -> List[SearchResult]:
    """Search GDELT Doc API.

    Acquires a cross-process rate-limit slot before issuing the request; raises
    _GDELTSlotBusy (not an error) when the slot is unavailable within 1.5 s.

    Connect timeout is intentionally short (2 s): when GDELT rate-limits an IP it
    throttles the TLS handshake to ~25 s before returning 429. Failing fast here
    lets the fallback chain proceed without stalling the pipeline.  A 429 response
    sets a per-process 60 s cooldown so we don't re-probe on every subsequent call.
    """
    # Raises _GDELTSlotBusy if the slot is not available within _GDELT_SLOT_WAIT.
    _gdelt_acquire_slot()

    # Translate Google-style site: operator to GDELT domain: filter.
    gdelt_query = re.sub(r"site:(\S+)", r"domain:\1", query)

    params: dict = {
        "query": gdelt_query,
        "mode": "artlist",
        "format": "json",
        "maxrecords": min(limit, 25),
        "sort": "DateDesc",
    }
    if date_from:
        params["startdatetime"] = date_from.strftime("%Y%m%d000000")
    if date_to:
        params["enddatetime"] = date_to.strftime("%Y%m%d235959")

    last_err: Exception = RuntimeError("GDELT: no attempts made")
    for attempt in range(2):  # 1 retry on connection failures; slot already claimed
        if attempt:
            time.sleep(3)
        try:
            r = httpx.get(
                "https://api.gdeltproject.org/api/v2/doc/doc",
                params=params,
                # Short connect timeout: GDELT stalls the TLS handshake when
                # rate-limiting rather than refusing the connection, so a long
                # timeout here causes multi-second hangs on every throttled call.
                timeout=httpx.Timeout(connect=2.0, read=10.0, write=2.0, pool=2.0),
            )
        except Exception as e:
            last_err = e
            logger.debug("GDELT attempt %d network error: %s", attempt + 1, e)
            continue

        if r.status_code == 429:
            global _GDELT_COOLDOWN_UNTIL
            _GDELT_COOLDOWN_UNTIL = time.time() + 60.0
            last_err = RuntimeError("GDELT 429: IP rate-limited; 60 s cooldown set")
            logger.warning(
                "GDELT returned 429 — IP is throttled. "
                "Cooldown until %.0f (%.0fs from now). "
                "Likely cause: concurrent workers bypassed the cross-process slot.",
                _GDELT_COOLDOWN_UNTIL,
                60.0,
            )
            break  # no point retrying a rate-limit hit

        if r.status_code != 200:
            last_err = RuntimeError(f"GDELT HTTP {r.status_code}")
            logger.debug("GDELT attempt %d: HTTP %d", attempt + 1, r.status_code)
            continue

        if not r.text.strip():
            last_err = RuntimeError("GDELT returned empty body")
            continue

        try:
            data = r.json()
        except Exception as e:
            last_err = RuntimeError(f"GDELT non-JSON body: {e}")
            continue

        articles = data.get("articles") or []
        results = []
        for art in articles:
            pub = art.get("seendate", "")
            try:
                pub_str = datetime.strptime(pub[:8], "%Y%m%d").strftime("%Y-%m-%d")
            except ValueError:
                pub_str = ""
            results.append(SearchResult(
                title=art.get("title", ""),
                url=art.get("url", ""),
                snippet="",  # GDELT artlist mode returns no body text
                source=art.get("domain", _extract_domain(art.get("url", ""))),
                published_date=pub_str,
            ))
        # GDELT occasionally leaks ±1 day past enddatetime; post-filter to be strict.
        filtered = _filter_by_date(results[:limit], date_from, date_to)
        logger.debug("GDELT returned %d articles (%d after date filter)", len(results), len(filtered))
        return filtered

    raise last_err


# ──────────────────────────────────────────────
# Provider: GDELT BigQuery GKG (historical, entity-based)
# ──────────────────────────────────────────────

# GDELT DOC API covers a rolling 3-month window. Beyond that, fall back to BQ.
_GDELT_DOC_WINDOW_DAYS = 90

# Cost fuse for BigQuery: a partition-pruned retro window scans ~10 GB; an
# accidental unpruned scan of the full ~15 TB GKG table would bill ~$80. Cap
# every job so such a query fails fast instead of running up a bill. 200 GB
# comfortably clears any legitimate multi-month window while blocking catastrophe.
_GDELT_BQ_MAX_BYTES_BILLED = 200 * 1024**3


def _slug_to_title(url: str) -> str:
    """Derive a human-readable title from a URL path slug.

    E.g. 'https://bbc.com/news/world-middle-east-67891234-netanyahu-ceasefire-deal'
    →    'Netanyahu Ceasefire Deal'
    """
    try:
        from urllib.parse import urlparse as _up
        path = _up(url).path.rstrip("/")
        slug = path.split("/")[-1]
        # Strip leading digits (article IDs like '67891234-')
        slug = re.sub(r'^\d+-', '', slug)
        words = slug.replace("-", " ").replace("_", " ").split()
        # Title-case non-stopword segments; keep short segments as-is
        _stops = {"a", "an", "the", "and", "or", "of", "in", "on", "at", "to", "for", "with"}
        return " ".join(w.capitalize() if w.lower() not in _stops else w for w in words if w)
    except Exception:
        return ""


def _extract_bq_terms(query: str) -> list[str]:
    """Extract search terms for BigQuery entity matching from a natural-language query.

    Returns a list of strings to REGEXP_CONTAINS against AllNames/V2Persons/V2Locations.
    Prioritises multi-word proper-noun phrases; falls back to significant single words.
    Empty list means no useful terms could be extracted.
    """
    # Strip search operators
    clean = re.sub(r'\b(?:site|after|before|domain):\S+', '', query)
    clean = re.sub(r'\b(?:OR|AND|NOT)\b', ' ', clean)

    # Multi-word proper-noun phrases (consecutive title-cased words)
    phrases = re.findall(r'\b([A-Z][a-z]{1,}(?:\s+[A-Z][a-z]{1,})+)\b', clean)

    # Single proper nouns (title-cased, length ≥ 4 to skip 'The', 'And', etc.)
    singles = re.findall(r'\b([A-Z][a-z]{3,})\b', clean)
    _common = {
        "Israel", "Israeli", "Iran", "Iranian", "Gaza", "West", "Bank",
        "United", "States", "America", "American", "European", "Middle", "East",
    }
    # Keep all phrases; filter singles to proper nouns not in stopword set
    terms = list(dict.fromkeys(phrases + [s for s in singles if s not in _common]))
    return terms[:8]  # cap to keep SQL manageable


def _entity_regex_patterns(terms: List[str]) -> List[str]:
    """Build case-insensitive REGEXP_CONTAINS patterns for BigQuery entity terms.

    Returns one ``(?i)<escaped term>`` pattern per term, meant to be bound as a
    BigQuery ``ArrayQueryParameter`` rather than interpolated into SQL text.
    ``re.escape()`` only neutralises regex metacharacters (so the pattern matches
    the term literally) — it says nothing about SQL syntax, so the caller must
    still bind these as parameters, never splice them into the query string.

    Lives here rather than in ``gdelt_bq_ingest`` (where retro#440 first added it)
    because both BigQuery callers need it and the import already runs this way:
    ``gdelt_bq_ingest`` imports ``web_search``, not the reverse.
    """
    return [f"(?i){re.escape(t)}" for t in terms]


def _sql_domain_filter(domains: Optional[List[str]]) -> str:
    """Build an ``AND SourceCommonName IN (...)`` clause, or '' when no domains.

    Domains are normalised to bare hostnames (scheme/``www.``/path stripped) so a
    source's configured ``url`` (e.g. ``https://www.ynet.co.il``) matches GKG's
    ``SourceCommonName`` (``ynet.co.il``). Single quotes are stripped as a defensive
    measure — hostnames never contain them, so this can only reject bad input, not
    corrupt a real domain.
    """
    if not domains:
        return ""
    cleaned = []
    for d in domains:
        host = re.sub(r"^https?://", "", d.strip().lower())
        host = re.sub(r"^www\.", "", host).split("/")[0].replace("'", "")
        if host:
            cleaned.append(host)
    if not cleaned:
        return ""
    joined = ",".join(f"'{h}'" for h in dict.fromkeys(cleaned))
    return f"AND SourceCommonName IN ({joined})"


def _search_gdelt_bq(
    query: str,
    limit: int,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    domains: Optional[List[str]] = None,
    max_rows: Optional[int] = None,
) -> List[SearchResult]:
    """Search GDELT GKG via BigQuery for historical coverage beyond the DOC API 3-month window.

    Matches against V2Persons, V2Locations, V2Organizations, and AllNames using
    REGEXP_CONTAINS. Partition pruning via _PARTITIONTIME on gkg_partitioned keeps scan costs low.
    Article titles are synthesized from the URL slug since GKG stores no titles.

    ``domains`` (optional): restrict to these outlets via ``SourceCommonName IN (...)``.
    This is what makes per-source retro cells possible (the live Oracle chain leaves it
    None). It is *free*: the predicate adds no scanned columns, so a domain-filtered query
    costs the same bytes as an unfiltered one over the same window.

    ``max_rows`` (optional): override the SQL row cap (default ``min(limit*4, 100)``).
    ``LIMIT`` does not change bytes scanned in BigQuery, so a retro backfill can raise this
    to pull many outlets' URLs from a single per-event scan instead of paying for one scan
    per source. The live Oracle path leaves it None and keeps the original cap.
    """
    client = _get_bq_client()  # raises if not configured

    terms = _extract_bq_terms(query)
    if not terms:
        raise RuntimeError("GDELT BQ: no extractable entity terms in query")

    # Entity terms are BOUND, not spliced (retro#462, same fix as retro#440).
    # re.escape() neutralises regex metacharacters; it is not SQL escaping, and the
    # only thing that made the old f-string safe was _extract_bq_terms happening to
    # emit [A-Za-z] — an accidental property of its regexes, not a guarantee anyone
    # was maintaining. One UNNEST over a parameter array replaces the per-term
    # OR-chain and matches the four entity columns exactly as before.
    patterns = _entity_regex_patterns(terms)
    entity_conditions = """
        EXISTS (
            SELECT 1 FROM UNNEST(@patterns) AS pattern
            WHERE REGEXP_CONTAINS(COALESCE(V2Persons,''), pattern)
               OR REGEXP_CONTAINS(COALESCE(V2Locations,''), pattern)
               OR REGEXP_CONTAINS(COALESCE(V2Organizations,''), pattern)
               OR REGEXP_CONTAINS(COALESCE(AllNames,''), pattern)
        )
    """

    domain_filter = _sql_domain_filter(domains)

    # Use gkg_partitioned (DAY-partitioned by ingestion time) — not the legacy
    # unpartitioned `gkg` table which lacks _PARTITIONTIME support.
    if date_from:
        ts_from = date_from.strftime("%Y-%m-%d")
    else:
        from datetime import timedelta
        ts_from = (datetime.utcnow() - timedelta(days=_GDELT_DOC_WINDOW_DAYS)).strftime("%Y-%m-%d")
    ts_to = date_to.strftime("%Y-%m-%d") if date_to else datetime.utcnow().strftime("%Y-%m-%d")

    row_cap = max_rows if max_rows is not None else min(limit * 4, 100)

    sql = f"""
        SELECT
            DocumentIdentifier AS url,
            SourceCommonName   AS source,
            DATE               AS gkg_date
        FROM `gdelt-bq.gdeltv2.gkg_partitioned`
        WHERE _PARTITIONTIME BETWEEN TIMESTAMP(@date_from) AND TIMESTAMP(@date_to)
          AND ({entity_conditions})
          {domain_filter}
          AND DocumentIdentifier IS NOT NULL
          AND DocumentIdentifier != ''
        ORDER BY DATE DESC
        LIMIT {row_cap}
    """

    logger.debug("GDELT BQ query for %r (terms: %s)", query[:60], terms)
    job = client.query(
        sql,
        job_config=_bigquery.QueryJobConfig(
            maximum_bytes_billed=_GDELT_BQ_MAX_BYTES_BILLED,
            query_parameters=[
                _bigquery.ScalarQueryParameter("date_from", "STRING", ts_from),
                _bigquery.ScalarQueryParameter("date_to", "STRING", ts_to),
                _bigquery.ArrayQueryParameter("patterns", "STRING", patterns),
            ],
        ),
    )
    rows = list(job.result())

    seen_urls: set[str] = set()
    results = []
    for row in rows:
        url = row.url or ""
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)

        # DATE column is an integer: YYYYMMDDHHMMSS
        pub_str = ""
        try:
            pub_str = datetime.strptime(str(row.gkg_date)[:8], "%Y%m%d").strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            pass

        results.append(SearchResult(
            title=_slug_to_title(url),
            url=url,
            snippet="",
            source=row.source or _extract_domain(url),
            published_date=pub_str,
        ))
        if len(results) >= limit:
            break

    filtered = _filter_by_date(results, date_from, date_to)
    logger.debug("GDELT BQ returned %d rows, %d after dedup/date-filter", len(rows), len(filtered))
    return filtered


# ──────────────────────────────────────────────
# Provider: Tavily Search API
# ──────────────────────────────────────────────

def _search_tavily(
    query: str,
    limit: int,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
) -> List[SearchResult]:
    global _TAVILY_QUOTA_EXHAUSTED
    if not TAVILY_API_KEY:
        raise RuntimeError("TAVILY_API_KEY not set")

    # Tavily supports site: in the query string; keep it as-is.
    body: dict = {
        "query": query,
        "topic": "news",
        "max_results": min(limit, 20),
        "search_depth": "basic",  # 1 credit/call
        "include_answer": False,
    }
    if date_from:
        body["start_date"] = date_from.strftime("%Y-%m-%d")
    if date_to:
        body["end_date"] = date_to.strftime("%Y-%m-%d")

    r = httpx.post(
        "https://api.tavily.com/search",
        json=body,
        headers={"Authorization": f"Bearer {TAVILY_API_KEY}", "Content-Type": "application/json"},
        timeout=15,
    )
    if r.status_code == 432:
        # Tavily-specific: plan usage limit exceeded (non-standard HTTP code)
        _TAVILY_QUOTA_EXHAUSTED = True
        _persist_quota_state()
        raise RuntimeError("Tavily usage limit exceeded (432) — disabling for session")
    if r.status_code in (401, 403, 429):
        body_text = r.text.lower()
        if any(x in body_text for x in ("quota", "credit", "limit", "exceeded", "invalid")):
            _TAVILY_QUOTA_EXHAUSTED = True
            _persist_quota_state()
            raise RuntimeError(f"Tavily quota/key error ({r.status_code}) — disabling for session")
        raise RuntimeError(f"Tavily HTTP {r.status_code}")
    r.raise_for_status()

    items = r.json().get("results", [])
    results = [
        SearchResult(
            title=item.get("title", ""),
            url=item.get("url", ""),
            snippet=item.get("content", ""),
            source=_clean_source("", item.get("url", "")),
            published_date=_date_from_url(item.get("url", "")),
        )
        for item in items[:limit]
        if item.get("url")
    ]
    return _filter_by_date(results, date_from, date_to)


# ──────────────────────────────────────────────
# Provider: DuckDuckGo Lite (free fallback)
# ──────────────────────────────────────────────

def _search_ddg_news(
    query: str,
    limit: int,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
) -> List[SearchResult]:
    global _DDG_LAST_CALL
    elapsed = time.time() - _DDG_LAST_CALL
    if elapsed < DDG_MIN_INTERVAL:
        time.sleep(DDG_MIN_INTERVAL - elapsed)

    # Use d.news() (DDG /news.js → Bing backend) — works from EC2 datacenter IPs.
    # d.text() uses Yahoo which blocks AWS IPs; d.news() does not.
    results = []
    with DDGS() as d:
        for item in d.news(query, max_results=limit):
            url = item.get("url", "")
            results.append(
                SearchResult(
                    title=item.get("title", ""),
                    url=url,
                    snippet=item.get("body", ""),
                    source=item.get("source") or _extract_domain(url),
                    published_date=item.get("date", "")[:10] if item.get("date") else "",
                )
            )
    _DDG_LAST_CALL = time.time()
    return _filter_by_date(results[:limit], date_from, date_to)


# ──────────────────────────────────────────────
# Last-resort fallback: trusted-site DDG search
# ──────────────────────────────────────────────

def _search_trusted_sites(
    query: str,
    limit: int,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
) -> List[SearchResult]:
    """Quota-free last resort: re-query DDG restricted to a curated list of
    credible news domains (batched ``site:`` OR groups). Runs only when the rest
    of the chain returned nothing, so it biases results toward trusted sources
    instead of falling to low-relevance GDELT BigQuery.

    Bounded: at most ``_TRUSTED_SITES_MAX_BATCHES`` DDG calls; stops early once it
    has ``limit`` results. A failing batch is skipped, not fatal.
    """
    # Drop any caller-supplied site: filter so we don't double-restrict.
    base = re.sub(r"\bsite:\S+\s*", "", query).strip()
    collected: List[SearchResult] = []
    seen: set = set()
    for i in range(_TRUSTED_SITES_MAX_BATCHES):
        batch = TRUSTED_SEARCH_DOMAINS[i * _TRUSTED_SITES_BATCH_SIZE:(i + 1) * _TRUSTED_SITES_BATCH_SIZE]
        if not batch:
            break
        site_clause = " OR ".join(f"site:{d}" for d in batch)
        batch_query = f"{base} ({site_clause})"
        try:
            for r in _search_ddg_news(batch_query, limit, date_from, date_to):
                if r.url and r.url not in seen:
                    seen.add(r.url)
                    collected.append(r)
        except Exception as e:
            logger.warning("trusted_sites batch %d failed: %s", i, e)
        if len(collected) >= limit:
            break
    return collected[:limit]


# ──────────────────────────────────────────────
# Provider: Newsdata.io
# ──────────────────────────────────────────────

def _search_newsdata_io(
    query: str,
    limit: int,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
) -> List[SearchResult]:
    global _NEWSDATA_QUOTA_EXHAUSTED
    if not NEWSDATA_API_KEY:
        raise RuntimeError("NEWSDATA_API_KEY not set")

    params: dict = {
        "apikey": NEWSDATA_API_KEY,
        "q": query,
        "language": "en",
        "size": min(limit, 50),
    }
    if date_from:
        params["from_date"] = date_from.strftime("%Y-%m-%d")
    if date_to:
        params["to_date"] = date_to.strftime("%Y-%m-%d")

    r = httpx.get("https://newsdata.io/api/1/latest", params=params, timeout=10)
    if r.status_code == 429:
        _NEWSDATA_QUOTA_EXHAUSTED = True
        _persist_quota_state()
        raise RuntimeError("Newsdata.io quota exhausted (rate limit)")
    r.raise_for_status()
    data = r.json()
    if data.get("status") != "success":
        code = (data.get("results") or {}).get("code", "")
        if code in ("AccessDenied", "RateLimitExceeded"):
            _NEWSDATA_QUOTA_EXHAUSTED = True
            _persist_quota_state()
        raise RuntimeError(f"Newsdata.io error: {data}")

    items = data.get("results") or []
    results = [
        SearchResult(
            title=item.get("title") or "",
            url=item.get("link") or "",
            snippet=item.get("description") or "",
            source=_clean_source(item.get("source_id") or "", item.get("link") or ""),
            published_date=(item.get("pubDate") or "")[:10],
        )
        for item in items[:limit]
        if item.get("link")
    ]
    return _filter_by_date(results, date_from, date_to)


# ──────────────────────────────────────────────
# Snippet enrichment (opt-in scraping for GDELT results)
# ──────────────────────────────────────────────

_SNIPPET_UA = "Mozilla/5.0 (compatible; TruthMachine/1.0)"
_SNIPPET_TIMEOUT = httpx.Timeout(connect=3.0, read=7.0, write=3.0, pool=3.0)


def _fetch_snippet(url: str) -> str:
    """Fetch *url* and return first ~600 chars of article text. Returns '' on any failure."""
    if not _SNIPPET_LIBS_AVAILABLE:
        return ""
    try:
        r = safe_get(url, headers={"User-Agent": _SNIPPET_UA}, timeout=_SNIPPET_TIMEOUT)
        if r.status_code != 200 or "text/html" not in r.headers.get("content-type", ""):
            return ""
        text = _trafilatura.extract(r.text, include_comments=False, include_tables=False)
        if not text:
            soup = _BeautifulSoup(r.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "aside", "iframe", "figure"]):
                tag.extract()
            for sel in ["article", "main", ".article-body", ".article-content", ".post-content",
                        '[class*="article"]', '[class*="story"]']:
                el = soup.select_one(sel)
                if el:
                    candidate = el.get_text(separator=" ", strip=True)
                    if len(candidate) > 200:
                        text = candidate
                        break
            if not text:
                text = soup.get_text(separator=" ", strip=True)
        return text[:600].strip() if text else ""
    except Exception:
        return ""


def enrich_snippets(results: List[SearchResult], timeout: float = 8.0) -> List[SearchResult]:
    """Fetch and fill snippets for results that have none, up to *timeout* seconds total.

    Uses up to 8 parallel workers. Articles that timeout or fail keep an empty snippet.
    Only has an effect when trafilatura and beautifulsoup4 are installed.
    """
    to_enrich = [i for i, r in enumerate(results) if not r.snippet]
    if not to_enrich or not _SNIPPET_LIBS_AVAILABLE:
        return results

    fetched: dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_fetch_snippet, results[i].url): i for i in to_enrich}
        try:
            for fut in as_completed(futures, timeout=timeout):
                idx = futures[fut]
                try:
                    fetched[idx] = fut.result()
                except Exception:
                    fetched[idx] = ""
        except TimeoutError:
            logger.warning("enrich_snippets: %ss wall hit, %d URLs incomplete",
                           timeout, sum(1 for f in futures if not f.done()))
            for fut, idx in futures.items():
                if idx not in fetched:
                    fetched[idx] = ""

    out = list(results)
    for idx, snippet in fetched.items():
        out[idx] = replace(out[idx], snippet=snippet)
    return out


# ──────────────────────────────────────────────
# Public API — tries providers in order
# ──────────────────────────────────────────────

def _warm_news_indexer(results: List[SearchResult]) -> None:
    """Fire-and-forget: feed paid-provider results back into the news-indexer on-demand cache
    (`POST /enqueue`) so a future identical query is served locally with no SERP cost. No-op
    when the result came from news_indexer itself (already indexed), nothing was found, or the
    indexer isn't configured. Runs in a daemon thread and never raises — /search is on the
    Oracle latency path, so warming must not add latency or fail the search."""
    try:
        if not (NEWS_INDEXER_URL and NEWS_INDEXER_API_KEY):
            return
        if get_last_search_provider() in ("news_indexer", "none"):
            return
        urls = [r.url for r in results if getattr(r, "url", None)][:10]
        if not urls:
            return

        def _post() -> None:
            try:
                httpx.post(
                    f"{NEWS_INDEXER_URL}/enqueue",
                    json={"urls": urls},
                    headers={"x-api-key": NEWS_INDEXER_API_KEY},
                    timeout=httpx.Timeout(connect=2.0, read=3.0, write=2.0, pool=2.0),
                )
            except Exception as exc:
                logger.debug("news_indexer warm failed (non-fatal): %s", exc)

        threading.Thread(target=_post, name="ni-warm", daemon=True).start()
    except Exception as exc:  # warming must never perturb the search result
        logger.debug("news_indexer warm skipped (non-fatal): %s", exc)


def search_articles(
    query: str,
    limit: int = 10,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
) -> List[SearchResult]:
    """Search wrapper: run the provider chain, then warm the news-indexer cache with any
    paid-provider results before returning. See `_search_articles_chain` for the chain itself."""
    results = _search_articles_chain(query, limit, date_from=date_from, date_to=date_to)
    _warm_news_indexer(results)
    return results


def search_capturing(
    query: str,
    limit: int = 10,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
) -> tuple[List[SearchResult], str, list[str]]:
    """Run search_articles() and capture the winning provider/chain *in the same
    thread* the search ran in.

    ``get_last_search_provider()``/``get_last_search_provider_chain()`` are
    thread-local; callers typically run search via ``asyncio.to_thread``, so
    reading the provider back in the calling (event-loop) thread always
    returns stale/"none" state. Call this whole function inside the worker
    thread instead (e.g. ``asyncio.to_thread(search_capturing, query, limit)``)
    so the provider read happens where the search actually ran.
    """
    results = search_articles(query, limit, date_from, date_to)
    return results, get_last_search_provider(), get_last_search_provider_chain()


def _search_articles_chain(
    query: str,
    limit: int = 10,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
) -> List[SearchResult]:
    """
    Search for news articles matching *query*, returning up to *limit* results.

    Tries providers in order, skipping any without a configured key or with
    an exhausted quota flag set for this process lifetime:
      GDELT → SerpAPI → Serper.dev → Brave → Tavily → Newsdata.io → BrightData → Nimbleway → ScrapingBee → DataForSEO → DDG

    DDG is tried last; if AWS IPs are blocked by DDG/Yahoo it fails and is logged.

    Args:
        query:     Search string. Include `site:domain.com` to restrict to a source.
        limit:     Max results to return.
        date_from: Optional start of date window.
        date_to:   Optional end of date window.

    Returns:
        List of SearchResult(title, url, snippet, source, published_date).
    """
    global _GDELT_DOC_FAIL_COUNT, _GDELT_DOC_BROKEN_UNTIL
    _refresh_keys_if_stale()
    _expire_stale_quota_flags()
    _provider_local.name = "none"
    _provider_local.chain = []

    # 0. news-indexer — local semantic index; first-in-chain, no SERP cost.
    #    Returns [] (gating check runs server-side) → fall through to GDELT/SERP.
    #    Inert when NEWS_INDEXER_URL / NEWS_INDEXER_API_KEY are not configured.
    if NEWS_INDEXER_URL and NEWS_INDEXER_API_KEY:
        _provider_local.chain.append("news_indexer")
        _t0 = time.perf_counter()
        try:
            r = httpx.get(
                f"{NEWS_INDEXER_URL}/search",
                params={"q": query, "limit": limit},
                headers={"x-api-key": NEWS_INDEXER_API_KEY},
                timeout=httpx.Timeout(connect=2.0, read=8.0, write=2.0, pool=2.0),
            )
            r.raise_for_status()
            _parsed = [_search_result_from_payload(h) for h in r.json()]
            hits = [res for res, _ in _parsed]
            _unknown = set().union(*(u for _, u in _parsed)) if _parsed else set()
            if _unknown:
                # Tolerated, not fatal — but this IS the upstream widening its contract,
                # and it should be visible before the ignored field turns out to matter.
                logger.warning(
                    "news_indexer: ignoring unknown /search field(s) %s — SearchResult may "
                    "need widening", sorted(_unknown),
                )
            # news-indexer's /search accepts no date params and only holds recent
            # articles, so a caller's window was silently a no-op on this leg while
            # every SERP leg honored it — backward-looking dated queries (resolution
            # research, MCP search_news) got answered with this week's coverage
            # (retro#559; the Brent-$100 resolution failure). Post-filter like the
            # DataForSEO leg does; zero survivors fall through to the providers that
            # can actually scope by date.
            if hits and (date_from or date_to):
                _pre = len(hits)
                hits = _filter_by_date(hits, date_from, date_to)
                if len(hits) != _pre:
                    logger.info(
                        "news_indexer: %d/%d hits outside requested window "
                        "[%s, %s], dropped: %s",
                        _pre - len(hits), _pre,
                        date_from.date() if date_from else "-inf",
                        date_to.date() if date_to else "now", query[:60],
                    )
            if hits:
                _provider_local.name = "news_indexer"
                logger.debug(
                    "news_indexer: %d hits %dms: %s",
                    len(hits), int((time.perf_counter() - _t0) * 1000), query[:60],
                )
                return hits
            logger.debug(
                "news_indexer: below threshold %dms: %s",
                int((time.perf_counter() - _t0) * 1000), query[:60],
            )
        except Exception as exc:
            # ERROR, not WARNING (retro#459). Losing this provider is not a degraded
            # nicety: it silently moves every retro caller onto paid SERP, swapping the
            # local semantic index for GDELT keyword matching. No counter moves when it
            # happens, so the log line is the only signal there is — it has to be one
            # that can alarm.
            logger.error(
                "news_indexer failed, falling back to paid providers %dms: %s",
                int((time.perf_counter() - _t0) * 1000), exc,
            )

    # 1. GDELT Doc API (free, no key) — primary; news-only, reliable dates, 3-month window
    _provider_local.chain.append("gdelt")
    _gdelt_cooldown_remaining = _GDELT_COOLDOWN_UNTIL - time.time()
    _gdelt_broken_remaining = _GDELT_DOC_BROKEN_UNTIL - time.time()
    if _gdelt_broken_remaining > 0:
        logger.info(
            "GDELT Doc skipped: circuit open for %.0fs more (repeated connection failures)",
            _gdelt_broken_remaining,
        )
    elif _gdelt_cooldown_remaining > 0:
        logger.info(
            "GDELT skipped: 429 cooldown active for %.0fs more", _gdelt_cooldown_remaining
        )
    else:
        _t0 = time.perf_counter()
        try:
            results = _search_gdelt(query, limit, date_from, date_to)
            if results:
                _GDELT_DOC_FAIL_COUNT = 0
                _provider_local.name = "gdelt"
                return results
            logger.debug("gdelt empty %dms: %s", int((time.perf_counter() - _t0) * 1000), query[:60])
            _GDELT_DOC_FAIL_COUNT = 0  # empty result is not a connection failure
        except _GDELTSlotBusy as e:
            logger.info("GDELT skipped (slot busy): %s", e)
        except Exception as e:
            logger.warning("gdelt failed %dms: %s", int((time.perf_counter() - _t0) * 1000), e)
            _GDELT_DOC_FAIL_COUNT += 1
            if _GDELT_DOC_FAIL_COUNT >= _GDELT_DOC_FAIL_THRESHOLD:
                _GDELT_DOC_BROKEN_UNTIL = time.time() + _GDELT_DOC_BREAK_SECS
                logger.warning(
                    "GDELT Doc: %d consecutive failures — circuit open for %.0f min",
                    _GDELT_DOC_FAIL_COUNT,
                    _GDELT_DOC_BREAK_SECS / 60,
                )

    # GDELT BigQuery GKG. It is the historical specialist (full-history entity
    # match), but its results are URL-slug-titled and recency-ranked, i.e.
    # low-relevance for a topical query. So it runs EARLY only for genuinely
    # historical queries (older than the Doc API's ~3-month window), where the
    # SERP providers have poor coverage; for live/recent queries it is a LAST
    # resort (see below, after DataForSEO). Running it early for live queries
    # short-circuited the chain with off-topic articles that the relevance
    # gatekeeper then discarded — leaving forecasts with no usable articles.
    _bq_tried = False

    def _try_gdelt_bq() -> Optional[List[SearchResult]]:
        nonlocal _bq_tried
        if not GCP_SA_KEY_JSON or _bq_tried:
            return None
        _bq_tried = True
        _provider_local.chain.append("gdelt_bq")
        _t0 = time.perf_counter()
        try:
            results = _search_gdelt_bq(query, limit, date_from, date_to)
            if results:
                _provider_local.name = "gdelt_bq"
                return results
            logger.debug("gdelt_bq empty %dms: %s", int((time.perf_counter() - _t0) * 1000), query[:60])
        except Exception as e:
            logger.warning("gdelt_bq failed %dms: %s", int((time.perf_counter() - _t0) * 1000), e)
        return None

    # 1b. Early BigQuery — historical queries only (Doc API covers ~3 months).
    if date_from is not None and (datetime.utcnow() - date_from).days > _GDELT_DOC_WINDOW_DAYS:
        _bq_results = _try_gdelt_bq()
        if _bq_results:
            return _bq_results

    # 1c. Google Custom Search — relevance-ranked, independent quota. Placed right
    # after GDELT so it is the primary keyed provider once configured (intended to
    # be the workhorse when Google startup credits supply paid quota; on the free
    # 100/day tier it will 429 and fall through). Inert until both keys are set.
    if GOOGLE_CSE_API_KEY and GOOGLE_CSE_CX and not _GOOGLE_CSE_QUOTA_EXHAUSTED:
        _provider_local.chain.append("google_cse")
        _t0 = time.perf_counter()
        try:
            results = _search_google_cse(query, limit, date_from, date_to)
            if results:
                _provider_local.name = "google_cse"
                return results
            logger.debug("google_cse empty %dms: %s", int((time.perf_counter() - _t0) * 1000), query[:60])
        except Exception as e:
            logger.warning("google_cse failed %dms: %s", int((time.perf_counter() - _t0) * 1000), e)

    # 2. SerpAPI
    if SERPAPI_API_KEY and not _SERPAPI_QUOTA_EXHAUSTED:
        _provider_local.chain.append("serpapi")
        _t0 = time.perf_counter()
        try:
            results = _search_serpapi_news(query, limit, date_from, date_to)
            if results:
                _provider_local.name = "serpapi"
                return results
            logger.debug("serpapi empty %dms: %s", int((time.perf_counter() - _t0) * 1000), query[:60])
        except Exception as e:
            logger.warning("serpapi failed %dms: %s", int((time.perf_counter() - _t0) * 1000), e)

    # 3. Serper.dev news
    if SERPER_API_KEY and not _SERPER_QUOTA_EXHAUSTED:
        _provider_local.chain.append("serper")
        _t0 = time.perf_counter()
        try:
            results = _search_serper_news(query, limit, date_from, date_to)
            if results:
                _provider_local.name = "serper"
                return results
            logger.debug("serper empty %dms: %s", int((time.perf_counter() - _t0) * 1000), query[:60])
        except Exception as e:
            logger.warning("serper failed %dms: %s", int((time.perf_counter() - _t0) * 1000), e)

    # 4. Brave News (returns published_date; higher free quota than Tavily)
    if BRAVE_API_KEY and not _BRAVE_QUOTA_EXHAUSTED:
        _provider_local.chain.append("brave")
        _t0 = time.perf_counter()
        try:
            results = _search_brave_news(query, limit, date_from, date_to)
            if results:
                _provider_local.name = "brave"
                return results
            logger.debug("brave empty %dms: %s", int((time.perf_counter() - _t0) * 1000), query[:60])
        except Exception as e:
            logger.warning("brave failed %dms: %s", int((time.perf_counter() - _t0) * 1000), e)

    # 4b. Tavily news (after Brave: no published_date in API response; 1 credit/call)
    if TAVILY_API_KEY and not _TAVILY_QUOTA_EXHAUSTED:
        _provider_local.chain.append("tavily")
        _t0 = time.perf_counter()
        try:
            results = _search_tavily(query, limit, date_from, date_to)
            if results:
                _provider_local.name = "tavily"
                return results
            logger.debug("tavily empty %dms: %s", int((time.perf_counter() - _t0) * 1000), query[:60])
        except Exception as e:
            logger.warning("tavily failed %dms: %s", int((time.perf_counter() - _t0) * 1000), e)

    # 4c. Newsdata.io — dedicated news API (published dates), ahead of the generic SERP scrapers.
    if NEWSDATA_API_KEY and not _NEWSDATA_QUOTA_EXHAUSTED:
        _provider_local.chain.append("newsdata")
        _t0 = time.perf_counter()
        try:
            results = _search_newsdata_io(query, limit, date_from, date_to)
            if results:
                _provider_local.name = "newsdata"
                return results
            logger.debug("newsdata empty %dms: %s", int((time.perf_counter() - _t0) * 1000), query[:60])
        except Exception as e:
            logger.warning("newsdata failed %dms: %s", int((time.perf_counter() - _t0) * 1000), e)

    # 5. BrightData SERP API
    if BRIGHTDATA_API_KEY and not _BRIGHTDATA_QUOTA_EXHAUSTED:
        _provider_local.chain.append("brightdata")
        _t0 = time.perf_counter()
        try:
            results = _search_brightdata(query, limit, date_from, date_to)
            if results:
                _provider_local.name = "brightdata"
                return results
            logger.debug("brightdata empty %dms: %s", int((time.perf_counter() - _t0) * 1000), query[:60])
        except Exception as e:
            logger.warning("brightdata failed %dms: %s", int((time.perf_counter() - _t0) * 1000), e)

    # 6. Nimbleway SERP API
    if NIMBLEWAY_API_KEY and not _NIMBLEWAY_QUOTA_EXHAUSTED:
        _provider_local.chain.append("nimbleway")
        _t0 = time.perf_counter()
        try:
            results = _search_nimbleway(query, limit, date_from, date_to)
            if results:
                _provider_local.name = "nimbleway"
                return results
            logger.debug("nimbleway empty %dms: %s", int((time.perf_counter() - _t0) * 1000), query[:60])
        except Exception as e:
            logger.warning("nimbleway failed %dms: %s", int((time.perf_counter() - _t0) * 1000), e)

    # 7. ScrapingBee Google Search
    if SCRAPINGBEE_API_KEY and not _SCRAPINGBEE_QUOTA_EXHAUSTED:
        _provider_local.chain.append("scrapingbee")
        _t0 = time.perf_counter()
        try:
            results = _search_scrapingbee(query, limit, date_from, date_to)
            if results:
                _provider_local.name = "scrapingbee"
                return results
            logger.debug("scrapingbee empty %dms: %s", int((time.perf_counter() - _t0) * 1000), query[:60])
        except Exception as e:
            logger.warning("scrapingbee failed %dms: %s", int((time.perf_counter() - _t0) * 1000), e)

    # 9. DataForSEO (paid — last-resort fallback only)
    if DATAFORSEO_API_KEY and not _DATAFORSEO_QUOTA_EXHAUSTED:
        _provider_local.chain.append("dataforseo")
        _t0 = time.perf_counter()
        try:
            results = _search_dataforseo(query, limit, date_from, date_to)
            if results:
                _provider_local.name = "dataforseo"
                return results
            logger.debug("dataforseo empty %dms: %s", int((time.perf_counter() - _t0) * 1000), query[:60])
        except Exception as e:
            logger.warning("dataforseo failed %dms: %s", int((time.perf_counter() - _t0) * 1000), e)

    # 10. DuckDuckGo (free, no key)
    _provider_local.chain.append("ddg")
    _t0 = time.perf_counter()
    try:
        results = _search_ddg_news(query, limit, date_from, date_to)
        if results:
            _provider_local.name = "ddg"
            return results
        logger.debug("ddg empty %dms: %s", int((time.perf_counter() - _t0) * 1000), query[:60])
    except Exception as e:
        logger.warning("ddg failed %dms: %s", int((time.perf_counter() - _t0) * 1000), e)

    # 10b. Trusted-sites fallback — quota-free, last resort. When the whole chain
    # came back empty, re-query DDG restricted to a curated list of credible news
    # domains. Biases to trusted sources and beats the low-relevance gdelt_bq junk
    # that would otherwise serve. Always available (no key); only reached when
    # everything above returned nothing.
    _provider_local.chain.append("trusted_sites")
    _t0 = time.perf_counter()
    try:
        results = _search_trusted_sites(query, limit, date_from, date_to)
        if results:
            _provider_local.name = "trusted_sites"
            return results
        logger.debug("trusted_sites empty %dms: %s", int((time.perf_counter() - _t0) * 1000), query[:60])
    except Exception as e:
        logger.warning("trusted_sites failed %dms: %s", int((time.perf_counter() - _t0) * 1000), e)

    # 11. GDELT BigQuery — absolute last resort for live/recent queries, AFTER DDG.
    # Its URL-slug, recency-ranked results are low-relevance, so it must not
    # short-circuit the chain before a real news provider (DDG returns relevant
    # results, incl. Hebrew). It runs here only when everything else came back
    # empty. (Skipped if already tried as the historical early path.)
    _bq_results = _try_gdelt_bq()
    if _bq_results:
        return _bq_results

    logger.error("All search providers exhausted — no articles found for: %s", query[:60])
    return []


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _extract_domain(url: str) -> str:
    try:
        return re.sub(r"^www\.", "", __import__("urllib.parse", fromlist=["urlparse"]).urlparse(url).netloc)
    except Exception:
        return ""


_URL_DATE_RE = re.compile(r"[/_-](\d{4})[/_-](\d{1,2})[/_-](\d{1,2})(?:[/_\-.]|$)")


def _date_from_url(url: str) -> str:
    """Extract YYYY-MM-DD from common news URL date patterns (/2026/05/28/ etc.). Returns '' if not found."""
    m = _URL_DATE_RE.search(url)
    if not m:
        return ""
    try:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 2000 <= y <= 2100 and 1 <= mo <= 12 and 1 <= d <= 31:
            return f"{y:04d}-{mo:02d}-{d:02d}"
    except Exception:
        pass
    return ""


def _clean_source(source: str, url: str = "") -> str:
    """Normalise provider source strings: strip noisy subdomains (feeds., rss., m., amp.)."""
    if not source:
        source = _extract_domain(url) if url else ""
    if not source:
        return ""
    # Proper names (contain space) and bare IDs (no dot): leave as-is
    if " " in source or "." not in source:
        return source
    cleaned = source.lower()
    while _SUBDOMAIN_STRIP_RE.match(cleaned):
        cleaned = _SUBDOMAIN_STRIP_RE.sub("", cleaned)
    return cleaned
