"""Jev shadow extraction (retro#840) — SHADOW ONLY, read by nothing.

TypeSafe's Jev (System One API) returns typed judgments instead of generated text, and the
extractor's `quote` field is selection ("Exact sentence(s) from the article"), not
generation. So a Jev extractor is: code segments the article into sentences, pass 1 asks one
`noul` per sentence ("does this sentence bear on the question?"), pass 2 scores stance /
settled / claim_strength on each candidate sentence. Offline on 150 real pairs against Haiku
v15 (2026-09-22): pass-1 top sentence matched a Haiku-quoted sentence 63% of the time
(keyword baseline 39%), stance sign agreed on 84% of Haiku's 422 quotes, at ~1/7 the cost.

This module runs that design next to the live Haiku extraction and logs one
`event=jev_shadow` line per article with both sides, so the cutover decision is made on live
traffic rather than a 150-article sample. It never touches the response: it is fired as a
background task after the extractor returns, every error is swallowed and logged, and it is
off unless JEV_SHADOW_ENABLED is set. The key is TYPESAFE_API_KEY or, failing that, the SSM
SecureString `/retro/prod/secrets/TYPESAFE_API_KEY` (same store and fallback as the search
providers' keys); a missing key logs `skip=no_key` on every article rather than going quiet.

Two known gaps the log is built to measure, not to paper over:
- Jev ignores negation INSIDE a score ("X will NOT happen" is scored as X), but detects a
  negated question reliably (43/44 offline). The line carries `neg` so analysis can flip, or
  count how many live questions are negated at all.
- The probability-weighted stance never reaches ±1 while the argmax level over-saturates;
  each `cand` row carries both, so the mapping is chosen on live data.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import unicodedata
from typing import Optional, Sequence

import httpx

logger = logging.getLogger(__name__)

API_URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"

# Same segmentation as the offline evaluation: Latin/Hebrew/Arabic sentence ends, newlines.
_SPLIT = re.compile(r'(?<=[.!?׃۔؟])\s+|\n+')
_WS = re.compile(r"\s+")
_HEB_GERSHAYIM = re.compile(r'(?<=[א-ת])"(?=[א-ת])')
_QUOTES = dict.fromkeys(map(ord, '"“”„«»״″'), '"')
_QUOTES.update(dict.fromkeys(map(ord, "'‘’׳′"), "'"))
_TRIM = ' ….,;:"\'–—-'
_MIN_SENTENCE_CHARS = 16

STANCE_VALUES = (-1.0, -0.7, -0.3, 0.0, 0.3, 0.7, 1.0)
STANCE_LEVELS = [
    "The sentence reports the related event, exactly as written, as definitively NOT happening or now impossible (a rival already won, the subject was eliminated, the deadline passed, the opposite occurred)",
    "Strong direct evidence AGAINST the related event as written: a result or figure that contradicts it, the decisive actor blocking it, a clear reversal",
    "Weak evidence AGAINST: a setback, an obstacle, a non-terminal loss, a sign it is receding",
    "No directional signal about THIS event: background, process or scheduling, a different actor or target, or unrelated content",
    "Weak evidence FOR: a capability, stated intent, threat, expectation, a 'favourite' label, one early stage cleared, a precondition — not the event itself",
    "Strong direct evidence FOR: the event itself is imminent or nearly all stages are cleared, a figure that meets the threshold, corroborated movement toward exactly this outcome",
    "The sentence reports the related event, exactly as written, as having HAPPENED, to this exact actor and target",
]
STANCE_RULES = [
    "Score the related event exactly as written. If it is negated ('X will NOT happen'), evidence that X is approaching or happened counts AGAINST it.",
    "Capability, intent, threats, expectations, preparations, and success against a DIFFERENT target are preconditions: at most weak evidence, never the event.",
    "An office title used only to name someone says nothing about their tenure.",
    "Compare numbers against the event's threshold. Tone is not direction: alarming wording that affirms the event is evidence FOR it.",
    "In multi-stage contests a 'favourite' label or one cleared stage is weak evidence; elimination is definitive.",
    "A cited probability from a named model, market or poll: a low percentage is evidence against, a high one for.",
]
STRENGTH_VALUES = (0.1, 0.3, 0.5, 0.7, 0.95)
STRENGTH_LEVELS = [
    "Speculative: casual personal opinion, rumour, or an interested party's own unverified claim",
    "Hedged: an expectation, a 'favourite' framing, a stated intent or threat, a trend without a level",
    "Moderate: a concrete development whose bearing on the event is partial",
    "Firm: a specific attributed figure, result or cleared stage, or a named model's estimate",
    "Absolute: reported as an accomplished, unhedged fact",
]

# Strong refs to in-flight tasks: asyncio keeps only weak references, so an unreferenced
# fire-and-forget task can be garbage-collected mid-flight.
_TASKS: set[asyncio.Task] = set()
# Negation verdict per question — one small request per distinct question, not per article.
_NEG_CACHE: dict[str, float] = {}
_NEG_CACHE_MAX = 1024
KEY_SSM_NAME = "/retro/prod/secrets/TYPESAFE_API_KEY"
# Resolved key per process: SSM is asked once, not per article.
_KEY: list[Optional[str]] = []


def resolve_api_key(configured: str = "") -> Optional[str]:
    """Configured key, else SSM (blocking boto3 call — run it off the event loop)."""
    if configured:
        return configured
    if not _KEY:
        from tm.web_search import _secret
        _KEY.append(_secret("TYPESAFE_API_KEY", KEY_SSM_NAME))
    return _KEY[0]


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s or "")
    s = _HEB_GERSHAYIM.sub('"', s).translate(_QUOTES)
    return _WS.sub(" ", s).strip()


def segment(text: str) -> tuple[str, list[tuple[int, int]]]:
    """Normalised text + [(start, end)] of each sentence long enough to carry a claim."""
    t = _norm(text)
    spans: list[tuple[int, int]] = []
    pos = 0
    for piece in _SPLIT.split(t):
        if not piece:
            continue
        i = t.find(piece, pos)
        if i < 0:
            continue
        pos = i + len(piece)
        if len(piece.strip()) >= _MIN_SENTENCE_CHARS:
            spans.append((i, pos))
    return t, spans


def locate_quote(text: str, spans: Sequence[tuple[int, int]], quote: str) -> list[int]:
    """Indices of the sentences a quote overlaps — how a Haiku quote is compared to Jev's
    per-sentence selection. Tolerates an elided middle (both 40-char ends real). [] when the
    quote is not in the text (Haiku paraphrases ~5% of quotes)."""
    q = _norm(quote).strip(_TRIM)
    if not q:
        return []
    a = text.find(q)
    b = a + len(q)
    if a < 0 and len(q) > 80:
        h = text.find(q[:40])
        t = text.find(q[-40:], h + 40) if h >= 0 else -1
        if h >= 0 and t > h:
            a, b = h, t + 40
    if a < 0:
        return []
    return [k for k, (s, e) in enumerate(spans) if s < b and e > a]


def _expected(ans: dict, values: Sequence[float]) -> float:
    return sum(p * values[int(k)] for k, p in ans["probabilities"].items())


def _nonzero(ans: dict, values: Sequence[float]) -> tuple[float, float]:
    """(expectation over the directional levels only, p(no-signal level)). The plain
    expectation lets the no-signal mass drag every stance toward 0 (Jev |stance| 0.16 vs
    Haiku 0.42 on live pairs); split, p0 says WHETHER there is a signal and the rest says
    which way and how hard — same-sign MAE to Haiku 0.27 -> 0.19 (retro#845)."""
    zero = values.index(0.0)
    probs = {int(k): p for k, p in ans["probabilities"].items()}
    p0 = probs.get(zero, 0.0)
    rest = 1.0 - p0
    ex = sum(p * values[k] for k, p in probs.items() if k != zero) / rest if rest > 1e-9 else 0.0
    return ex, p0


def _argmax(ans: dict, values: Sequence[float]) -> float:
    probs = ans["probabilities"]
    return values[int(max(probs, key=lambda k: probs[k]))]


async def _ask(client: httpx.AsyncClient, api_key: str, state, questions: dict,
               api_url: str = API_URL) -> dict:
    r = await client.post(
        api_url,
        headers={"Authorization": f"Bearer {api_key}"},
        json={"state": state, "model": MODEL, "questions": questions},
    )
    r.raise_for_status()
    return r.json()


def _selection_questions(n: int) -> dict:
    return {
        f"s{i}": {
            "type": "noul",
            "instructions": f"Does `sentences[{i}]` make a claim bearing on `question`?",
            "criteria": {"true": "It asserts or reports something that moves the question",
                         "false": "It does not"},
        }
        for i in range(n)
    }


def _scoring_questions() -> dict:
    return {
        "stance": {"type": "score",
                   "instructions": {"question": "How does `sentence` bear on whether `related_event` happens?",
                                    "rules": STANCE_RULES},
                   "criteria": STANCE_LEVELS},
        "strength": {"type": "score",
                     "instructions": "How strongly and unhedgedly does `sentence` assert what it asserts?",
                     "criteria": STRENGTH_LEVELS},
        "settled": {"type": "noul",
                    "instructions": "Does `sentence` report the outcome of `related_event` as an accomplished fact — it already happened, or became permanently impossible — rather than a prediction or progress toward it?"},
        # retro#847: Jev's "no signal" conflates "none" with "needed context it cannot see".
        # These two separated the cases 0.85 / 0.81 AUC offline (15 labelled cases).
        "topic": {"type": "noul",
                  "instructions": "Is `sentence` about the same subject as `related_event` — the same actors, institution, place or quantity?"},
        "refs": {"type": "noul",
                 "instructions": "Does `sentence` depend on text outside it to be understood — unresolved 'it/this/that', an unnamed speaker, or a condition/plan introduced elsewhere?"},
    }


_NEG_QUESTION = {"neg": {
    "type": "noul",
    "instructions": "Is `related_event` phrased as something NOT happening (a negated outcome), so that the event it names happening would make it false?",
}}


async def run_jev_shadow(
    *,
    text: str,
    question: str,
    url: str,
    haiku_predictions: Sequence[dict],
    api_key: str = "",
    api_url: str = API_URL,
    select_bar: float = 0.3,
    min_top: int = 3,
    max_candidates: int = 25,
    max_sentences: int = 400,
    timeout_s: float = 30.0,
    transport: Optional[httpx.AsyncBaseTransport] = None,
) -> Optional[dict]:
    """Run both Jev passes on one article and log `event=jev_shadow`. Returns the payload
    (for tests); never raises."""
    t0 = time.perf_counter()
    payload: dict = {"url": url}
    try:
        norm_text, spans = segment(text)
        sentences = [norm_text[s:e] for s, e in spans]
        payload["n"] = len(sentences)
        payload["haiku"] = [
            [locate_quote(norm_text, spans, p.get("quote") or ""),
             p.get("stance"), p.get("settled"), p.get("claim_strength")]
            for p in haiku_predictions
        ]
        if not sentences or len(sentences) > max_sentences:
            payload["skip"] = "no_sentences" if not sentences else "too_long"
            return payload
        api_key = api_key or await asyncio.to_thread(resolve_api_key)
        if not api_key:
            payload["skip"] = "no_key"
            return payload
        tok_in = 0
        async with httpx.AsyncClient(timeout=timeout_s, transport=transport) as client:
            neg = _NEG_CACHE.get(question)
            sel_coro = _ask(client, api_key, {"sentences": sentences, "question": question},
                            _selection_questions(len(sentences)), api_url)
            if neg is None:
                sel, neg_resp = await asyncio.gather(
                    sel_coro, _ask(client, api_key, {"related_event": question}, _NEG_QUESTION, api_url))
                tok_in += neg_resp.get("usage", {}).get("input_tokens", 0)
                neg = neg_resp["answers"]["neg"]["noul"]
                if len(_NEG_CACHE) >= _NEG_CACHE_MAX:
                    _NEG_CACHE.pop(next(iter(_NEG_CACHE)))
                _NEG_CACHE[question] = neg
            else:
                sel = await sel_coro
            tok_in += sel.get("usage", {}).get("input_tokens", 0)
            nouls = [sel["answers"][f"s{i}"]["noul"] for i in range(len(sentences))]
            order = sorted(range(len(nouls)), key=lambda i: (-nouls[i], i))
            cand = [i for k, i in enumerate(order) if k < min_top or nouls[i] >= select_bar][:max_candidates]
            # Every Haiku-quoted sentence gets a Jev verdict too (outside the cap), so a veto on
            # Haiku's claims can be measured claim by claim (retro#847).
            cand += sorted({q[0] for q, *_ in payload["haiku"] if q} - set(cand))
            scored = await asyncio.gather(*[
                _ask(client, api_key, {"sentence": sentences[i], "related_event": question}, _scoring_questions(), api_url)
                for i in cand
            ])
        payload["neg"] = round(neg, 3)
        payload["max_noul"] = round(max(nouls), 3)
        payload["top"] = [[i, round(nouls[i], 3)] for i in order[:8]]
        cands = []
        for i, resp in zip(cand, scored):
            tok_in += resp.get("usage", {}).get("input_tokens", 0)
            a = resp["answers"]
            nz, p0 = _nonzero(a["stance"], STANCE_VALUES)
            cands.append([
                i, round(nouls[i], 3),
                round(_expected(a["stance"], STANCE_VALUES), 3), _argmax(a["stance"], STANCE_VALUES),
                round(a["settled"]["noul"], 3),
                round(_expected(a["strength"], STRENGTH_VALUES), 3),
                round(nz, 3), round(p0, 3),
                round(a["topic"]["noul"], 3), round(a["refs"]["noul"], 3),
            ])
        # cand rows: [sentence_idx, noul, stance_expected, stance_argmax, settled, claim_strength,
        #             stance_nonzero, p_no_signal, topic, refs]
        payload["cand"] = cands
        payload["tok_in"] = tok_in
    except Exception as exc:  # shadow: never let Jev affect /forecast
        payload["err"] = repr(exc)[:200]
    finally:
        payload["ms"] = round((time.perf_counter() - t0) * 1000)
        logger.info("event=jev_shadow payload=%s", json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    return payload


def fire_jev_shadow(**kwargs) -> Optional[asyncio.Task]:
    """Schedule run_jev_shadow in the background, holding a strong reference to the task."""
    try:
        task = asyncio.get_running_loop().create_task(run_jev_shadow(**kwargs))
    except RuntimeError:
        return None
    _TASKS.add(task)
    task.add_done_callback(_TASKS.discard)
    return task
