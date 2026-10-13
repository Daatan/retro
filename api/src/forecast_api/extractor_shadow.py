"""Extractor model shadow (retro#921): run a candidate extractor model next to the live one.

Why: Claude Haiku 5.5 is reported ~9x cheaper than Haiku 4.5 and, on the offline A/B harness,
within noise on quality but a different *shape* (less selective, lower |stance|, lower
`fact_signal` fill). Offline cases cannot say what that does to real traffic, so this logs a
side-by-side per live article and changes nothing else.

Contract (same shape as the Jev shadows, `jev_shadow.py`):
  * Log-only. The shadow's output never reaches the response, the pool, the memo or any
    `enforce_*` step; it is compared on the RAW extractor output of both models.
  * Fire-and-forget. A background task with its own timeout; the live request does not wait
    for it, and every failure is swallowed into `err=` on the log line.
  * Off by default: `extractor_shadow_model` empty means nothing runs.
  * No shared state: the shadow call passes ``memo=None`` and ``cache_coordinator=None`` so it
    cannot write the extraction memo or perturb the live prompt-cache coordination.

One line per article: ``event=extractor_shadow`` (see ``summarize`` for the fields).
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

_SETTLED_FIELDS = ("settled",)


def summarize(extraction: Any) -> dict[str, Any]:
    """Shape of one extraction for side-by-side comparison. Pure."""
    preds = list(getattr(extraction, "predictions", None) or [])
    stances = [abs(float(p.stance)) for p in preds if getattr(p, "stance", None) is not None]
    with_fs = [p for p in preds if getattr(p, "fact_signal", None) is not None]
    return {
        "n": len(preds),
        "max_abs_stance": round(max(stances), 3) if stances else 0.0,
        "mean_abs_stance": round(sum(stances) / len(stances), 3) if stances else 0.0,
        "settled": sum(1 for p in preds if getattr(p, "settled", False)),
        "fs_n": len(with_fs),
        "net_stance": round(sum(float(p.stance) for p in preds if getattr(p, "stance", None) is not None), 3),
    }


def format_line(*, url: str, prediction_id: Optional[str], live_model: str, shadow_model: str,
                live: dict[str, Any], shadow: Optional[dict[str, Any]], usage: Optional[dict],
                ms: float, err: Optional[str]) -> str:
    """The single log line. Fields are ``key=value`` with no spaces in values."""
    parts = [
        "event=extractor_shadow",
        f"url={url}",
        f"prediction_id={prediction_id or ''}",
        f"live_model={live_model}",
        f"shadow_model={shadow_model}",
        f"live_n={live['n']}",
        f"live_max_abs_stance={live['max_abs_stance']}",
        f"live_mean_abs_stance={live['mean_abs_stance']}",
        f"live_settled={live['settled']}",
        f"live_fs_n={live['fs_n']}",
        f"live_net_stance={live['net_stance']}",
    ]
    if shadow is not None:
        parts += [
            f"shadow_n={shadow['n']}",
            f"shadow_max_abs_stance={shadow['max_abs_stance']}",
            f"shadow_mean_abs_stance={shadow['mean_abs_stance']}",
            f"shadow_settled={shadow['settled']}",
            f"shadow_fs_n={shadow['fs_n']}",
            f"shadow_net_stance={shadow['net_stance']}",
        ]
    u = usage or {}
    parts += [
        f"shadow_prompt_tokens={u.get('prompt_tokens', '')}",
        f"shadow_completion_tokens={u.get('completion_tokens', '')}",
        f"shadow_cache_read={u.get('cache_read_input_tokens', '')}",
        f"shadow_ms={ms:.0f}",
        f"err={err or ''}",
    ]
    return " ".join(parts)


async def run_shadow(
    *,
    extract: Callable[..., Any],
    shadow_model: str,
    live_model: str,
    live_extraction: Any,
    call_kwargs: dict[str, Any],
    url: str,
    prediction_id: Optional[str],
    timeout_s: float,
) -> None:
    """Run the candidate model once and log the comparison. Never raises."""
    live = summarize(live_extraction)
    started = time.perf_counter()
    shadow: Optional[dict[str, Any]] = None
    usage: Optional[dict] = None
    err: Optional[str] = None
    try:
        out, usage = await asyncio.wait_for(
            extract(**call_kwargs, model=shadow_model, memo=None, cache_coordinator=None),
            timeout=timeout_s,
        )
        shadow = summarize(out)
    except asyncio.TimeoutError:
        err = "timeout"
    except Exception as exc:  # noqa: BLE001 - a shadow must never surface an error
        # Type alone hid an AccessDenied behind InstructorRetryException (2026-10-13); add a
        # short single-token message so the readout can tell IAM from schema from timeout.
        msg = "_".join(str(exc).split())[:160]
        err = f"{type(exc).__name__}:{msg}" if msg else type(exc).__name__
    ms = (time.perf_counter() - started) * 1000
    try:
        logger.info(format_line(url=url, prediction_id=prediction_id, live_model=live_model,
                                shadow_model=shadow_model, live=live, shadow=shadow,
                                usage=usage, ms=ms, err=err))
    except Exception:  # noqa: BLE001
        logger.exception("event=extractor_shadow_log_failed")


# Strong references to in-flight shadow tasks: asyncio keeps only weak ones, and a
# garbage-collected task is silently cancelled mid-call.
_INFLIGHT: set[asyncio.Task] = set()


def fire_extractor_shadow(**kwargs: Any) -> None:
    """Schedule ``run_shadow`` without awaiting it. Safe to call from a request path."""
    try:
        task = asyncio.get_running_loop().create_task(run_shadow(**kwargs))
    except RuntimeError:  # no running loop: nothing to do, shadows are optional
        return
    _INFLIGHT.add(task)
    task.add_done_callback(_INFLIGHT.discard)
