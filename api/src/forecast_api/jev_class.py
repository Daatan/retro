"""Jev as the `evidence_class` corrector (retro#851).

Haiku writes every claim; Jev re-reads each claim's quote and picks its evidence class. The
live `jev_shadow` pass 2 already asks Jev for a class, but on the sentence alone — and a
sentence cut out of its article is misread: a poll figure that doesn't name the poll ("the
bloc gained a seat, rising to 55") reads as a plain fact, an official's boast as reporting.
Here Jev sees the whole article.

Measured 2026-09-27 against a blind two-labeler gold (post-retro#876 live pairs, 25 articles,
205 claims) and the 2026-09-22 held-out 40:

    Haiku                                157/205   26/40
    Jev, sentence only (live shadow)     171/205   30/40
    Jev, sentence + whole article        183/205   34/40   <- this module

The class criteria are the shadow's (`EVIDENCE_CLASSES`) unchanged: every rewording tried
scored higher in-sample and lower on the held-out set (best in-sample 189 -> 26/40). One call
per claim, not one per article: asking all of an article's quoted sentences in one request
scored 32/40.

Every failure keeps Haiku's class: a quote not found in the text, a Jev error, a timeout, no
key. With `jev_class_enforce` off it only logs `event=jev_class`.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from types import SimpleNamespace
from typing import Optional, Sequence

import httpx

from forecast_api.jev_shadow import (
    API_URL, EVIDENCE_CLASSES, _TASKS, _ask, _top_choice, locate_quote, resolve_api_key, segment,
)

logger = logging.getLogger(__name__)

CLASS_QUESTION = {"evidence_class": {
    "type": "choice",
    "instructions": ("Which evidence class is the claim in `sentence`? `article` is the full text "
                     "it comes from — read the sentence in that context."),
    "criteria": EVIDENCE_CLASSES,
}}


async def jev_evidence_classes(
    *,
    text: str,
    question: str,
    quotes: Sequence[Optional[str]],
    api_key: str = "",
    api_url: str = API_URL,
    max_sentences: int = 400,
    timeout_s: float = 3.0,
    transport: Optional[httpx.AsyncBaseTransport] = None,
) -> dict:
    """Jev's class for each quote: `classes` and `p` are parallel to `quotes`, None / 0.0
    where the quote is not in the text or its call failed. Also `tok_in`, `ms`, `err_n`,
    or `skip`/`err` for the whole article. Never raises."""
    t0 = time.perf_counter()
    out: dict = {"classes": [None] * len(quotes), "p": [0.0] * len(quotes)}
    try:
        norm_text, spans = segment(text)
        sentences = [norm_text[s:e] for s, e in spans]
        if not sentences or len(sentences) > max_sentences:
            out["skip"] = "no_sentences" if not sentences else "too_long"
            return out
        api_key = api_key or await asyncio.to_thread(resolve_api_key)
        if not api_key:
            out["skip"] = "no_key"
            return out
        article = " ".join(sentences)
        jobs: list[tuple[int, str]] = []
        for k, quote in enumerate(quotes):
            idx = locate_quote(norm_text, spans, quote or "")
            if idx:
                jobs.append((k, " ".join(sentences[i] for i in idx)))
        if not jobs:
            out["skip"] = "no_quotes"
            return out
        async with httpx.AsyncClient(timeout=timeout_s, transport=transport) as client:
            resps = await asyncio.wait_for(asyncio.gather(*[
                _ask(client, api_key, {"sentence": s, "related_event": question, "article": article},
                     CLASS_QUESTION, api_url)
                for _, s in jobs
            ], return_exceptions=True), timeout=timeout_s)
        tok_in = err_n = 0
        for (k, _), r in zip(jobs, resps):
            if isinstance(r, BaseException):
                err_n += 1
                continue
            tok_in += r.get("usage", {}).get("input_tokens", 0)
            label, p = _top_choice((r.get("answers") or {}).get("evidence_class"))
            if label in EVIDENCE_CLASSES:
                out["classes"][k], out["p"][k] = label, round(p, 3)
        out.update(tok_in=tok_in, err_n=err_n)
    except Exception as exc:  # never let Jev affect /forecast
        out["err"] = repr(exc)[:200]
    finally:
        out["ms"] = round((time.perf_counter() - t0) * 1000)
    return out


def apply_jev_classes(predictions: list, result: dict, *, enforce: bool, url: str = "") -> list:
    """Log Jev's class beside Haiku's for every claim; with `enforce`, replace Haiku's class
    wherever Jev returned one. Rows: [haiku, jev, jev_p]."""
    classes = result.get("classes") or []
    probs = result.get("p") or []
    rows = []
    changed = 0
    for k, p in enumerate(predictions):
        jev = classes[k] if k < len(classes) else None
        rows.append([p.evidence_class, jev, probs[k] if jev else 0.0])
        if jev is not None and jev != p.evidence_class:
            changed += 1
            if enforce:
                p.evidence_class = jev
    payload = {"url": url, "mode": "enforce" if enforce else "shadow", "changed": changed,
               "rows": rows, "ms": result.get("ms")}
    for key in ("tok_in", "err_n", "skip", "err"):
        if key in result:
            payload[key] = result[key]
    logger.info("event=jev_class payload=%s", json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    return predictions


def fire_jev_class_shadow(predictions: list, **kwargs) -> Optional[asyncio.Task]:
    """Shadow mode: run in the background and only log. `predictions` is snapshotted now, so
    later enforce_* rewrites don't leak into the Haiku side of the log."""
    snap = [SimpleNamespace(evidence_class=p.evidence_class) for p in predictions]
    url = kwargs.pop("url", "")

    async def _run() -> None:
        apply_jev_classes(snap, await jev_evidence_classes(**kwargs), enforce=False, url=url)

    try:
        task = asyncio.get_running_loop().create_task(_run())
    except RuntimeError:
        return None
    _TASKS.add(task)
    task.add_done_callback(_TASKS.discard)
    return task
