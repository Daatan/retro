import hmac
import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from fastapi import Header, HTTPException, status
from .config import settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ApiKeyClient:
    """Resolved caller identity for a REST request.

    ``max_articles`` is a hard per-key ceiling on how many articles a forecast
    may fetch/extract — including caller-supplied ``articles``, which previously
    bypassed every cap (docs#57: staging shared the prod key and burned
    unmetered LLM spend). ``None`` = uncapped.
    """

    name: str
    max_articles: Optional[int] = None


_DEFAULT_CLIENT = ApiKeyClient(name="default")


@lru_cache(maxsize=8)
def _named_keys(raw: str) -> tuple[tuple[str, ApiKeyClient], ...]:
    """Parse ``ORACLE_API_KEYS`` into ``((key, client), …)``.

    Fail-closed on purpose: a malformed value logs an error and yields no named
    keys, so a config typo can only produce 401s for the named keys — never an
    uncapped bypass. The primary ``ORACLE_API_KEY`` is unaffected either way.
    """
    if not raw:
        return ()
    try:
        entries = []
        for name, entry in json.loads(raw).items():
            key = entry["key"]
            cap = entry.get("max_articles")
            if not isinstance(key, str) or not key:
                raise ValueError(f"client {name!r}: 'key' must be a non-empty string")
            if cap is not None and (not isinstance(cap, int) or cap < 1):
                raise ValueError(f"client {name!r}: 'max_articles' must be a positive int")
            entries.append((key, ApiKeyClient(name=name, max_articles=cap)))
        return tuple(entries)
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        logger.error("ORACLE_API_KEYS is unparseable (%s); named keys disabled", exc)
        return ()


async def verify_api_key(x_api_key: str = Header(...)) -> ApiKeyClient:
    # Constant-time compare so response timing can't leak the key prefix length.
    if hmac.compare_digest(x_api_key, settings.oracle_api_key):
        return _DEFAULT_CLIENT
    for key, client in _named_keys(settings.oracle_api_keys):
        if hmac.compare_digest(x_api_key, key):
            return client
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid API key",
    )
