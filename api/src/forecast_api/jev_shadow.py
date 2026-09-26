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
import hashlib
import json
import logging
import re
import time
import unicodedata
from datetime import date
from typing import Optional, Sequence

import httpx

from .jev_dates import intervals as date_intervals

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

# retro#851: evidence_class as a Jev `choice`, shadow only. Wording is verbatim from the
# offline blind-gold run (Jev 30/40 vs Haiku 26/40, sentence-only state) — rewording was 0 for 4
# on the stance side, so these words are the measured artefact and must not be tuned in place.
EVIDENCE_CLASSES = {
    "reported_fact": "reports a concrete event, action, decision or result as having happened",
    "cited_probability": "cites an explicit probability of the event from a model, market or poll",
    "cited_share": "cites a vote share, seat count, price or other measured figure",
    "reporting": "sourced reporting about intentions, plans, talks or expectations (not yet a fact)",
    "opinion": "someone's opinion, prediction or assessment with no privileged access",
}

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


_WD = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


def _fmt_interval(a, b) -> str:
    return f"{a.isoformat()} ({_WD[a.weekday()]})" if a == b else f"{a.isoformat()} to {b.isoformat()}"


def _parse_day(value) -> Optional[date]:
    try:
        return date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


def date_questions(cands: Sequence[tuple], pub: date) -> dict:
    """retro#873: `dated` noul + a `when` choice over code-found candidates, the publication
    date and `none`. Wording as measured offline (settled claims: Jev = Haiku 35/37, both
    disagreements Haiku's pub-date substitution)."""
    crit = {f"c{k}": f'"{e}" in the text = {_fmt_interval(a, b)}' for k, (a, b, e) in enumerate(cands)}
    crit["pub"] = (f"the publication date {_fmt_interval(pub, pub)}: the text reports the event as "
                   "today's news without naming another date")
    crit["none"] = "the text gives no date or time for this event"
    return {
        "dated": {"type": "noul", "instructions": "Does `context` say WHEN the event reported in `sentence` happened or is scheduled to happen — a date, a weekday, a month, or a relative time such as 'yesterday' or 'last week'?"},
        "when": {"type": "choice", "instructions": "When did (or will) the event reported in `sentence` happen? `publication_date` is when the article was published. Pick the option that dates THIS event, not another event mentioned nearby.",
                 "criteria": crit},
    }


def _top_choice(ans: dict) -> tuple[Optional[str], float]:
    """(label, probability) of a `choice` answer's most likely option; (None, 0.0) if absent."""
    probs = (ans or {}).get("probabilities") or {}
    if not probs:
        return None, 0.0
    label = max(probs, key=probs.get)
    return label, float(probs[label])


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
        "evidence_class": {"type": "choice",
                           "instructions": "Which evidence class is the claim in `sentence`?",
                           "criteria": EVIDENCE_CLASSES},
    }


_NEG_QUESTION = {"neg": {
    "type": "noul",
    "instructions": "Is `related_event` phrased as something NOT happening (a negated outcome), so that the event it names happening would make it false?",
}}


_SCRIPT_RANGES = (("he", "֐", "׿"), ("ar", "؀", "ۿ"), ("cyr", "Ѐ", "ӿ"))


def script_of(text: str) -> str:
    """Dominant script of the letters in `text` — `he`/`ar`/`cyr`/`latin`/`other`. The gate
    log carries it so the skip threshold can be read per language (retro#850): Hebrew
    selection quality is the open question, and the caller's language hint is often absent."""
    counts = {k: 0 for k, *_ in _SCRIPT_RANGES}
    latin = total = 0
    for ch in text:
        if not ch.isalpha():
            continue
        total += 1
        if ch.isascii():
            latin += 1
            continue
        for k, lo, hi in _SCRIPT_RANGES:
            if lo <= ch <= hi:
                counts[k] += 1
                break
    if not total:
        return "other"
    best = max(counts, key=counts.get)
    if counts[best] > latin and counts[best] / total >= 0.2:
        return best
    return "latin" if latin / total >= 0.5 else "other"


async def jev_pass1(
    *,
    text: str,
    question: str,
    api_key: str = "",
    api_url: str = API_URL,
    max_sentences: int = 400,
    timeout_s: float = 30.0,
    transport: Optional[httpx.AsyncBaseTransport] = None,
) -> dict:
    """Pass 1 alone: segment the article and ask one `noul` per sentence (plus the cached
    per-question negation verdict). Returns a dict with `sentences`, `spans`, `norm_text`,
    `nouls`, `neg`, `max_noul`, `tok_in`, `ms` — or `skip`/`err` instead of `nouls`. Never
    raises. `run_jev_shadow` accepts it as `pass1` so the skip-gate (retro#850) and the
    shadow share one selection call per article."""
    t0 = time.perf_counter()
    out: dict = {}
    try:
        norm_text, spans = segment(text)
        sentences = [norm_text[s:e] for s, e in spans]
        out.update(norm_text=norm_text, spans=spans, sentences=sentences)
        if not sentences or len(sentences) > max_sentences:
            out["skip"] = "no_sentences" if not sentences else "too_long"
            return out
        api_key = api_key or await asyncio.to_thread(resolve_api_key)
        if not api_key:
            out["skip"] = "no_key"
            return out
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
        out.update(nouls=nouls, neg=neg, max_noul=max(nouls), tok_in=tok_in)
    except Exception as exc:  # never let Jev affect /forecast
        out["err"] = repr(exc)[:200]
    finally:
        out["ms"] = round((time.perf_counter() - t0) * 1000)
    return out


def evaluate_jev_gate(pass1: Optional[dict], threshold: float) -> tuple[bool, Optional[float]]:
    """(would_skip, max_noul). Fails open: no pass-1 result, a skip or an error → (False, None),
    so an unreachable Jev never costs an article its Haiku extraction."""
    if not pass1 or "nouls" not in pass1:
        return False, None
    max_noul = float(pass1["max_noul"])
    return max_noul < threshold, max_noul


_SIMPLIFY_SYSTEM = (
    "Rewrite the forecasting question into its CORE EVENT form, for use as the target of a "
    "sentence-relevance filter. Keep the subject (actors, place, quantity) and the event. Drop "
    "deadlines and dates, hedges, parenthetical qualifications and subordinate conditions. "
    "Remove negation: 'X will not pass the threshold' becomes 'X passes the threshold', because "
    "polarity is irrelevant to a relevance filter. Answer with the rewrite alone, 4 to 10 words, "
    "no quotes, no preamble, no explanation."
)
_SIMPLIFIED: dict[str, str] = {}


def question_fingerprint(question: str) -> str:
    """Stable 8-hex id for a question, the `q8` field of `event=jev_gate` (retro#849)."""
    return hashlib.sha256(question.encode("utf-8")).hexdigest()[:8] if question else ""


async def simplify_question(question: str, *, model: str, timeout_s: float = 15.0,
                            completer=None) -> str:
    """Question rewritten to its core event, cached per question. `""` on any failure.

    Never raises: the A/B it feeds is measurement, so a rewrite that errors, times out or
    comes back empty simply means no comparison is logged for that article. Questions are
    few and long-lived, so in practice this is one cheap call per question, ever.
    """
    q = (question or "").strip()
    if not q:
        return ""
    key = question_fingerprint(q)
    if key in _SIMPLIFIED:
        return _SIMPLIFIED[key]
    try:
        if completer is None:                       # imported lazily: keeps this module importable alone
            from tm.llm import complete_text_once_with_usage as completer  # type: ignore
        text, _usage = await asyncio.wait_for(
            completer(model, q, system=_SIMPLIFY_SYSTEM, max_tokens=60, temperature=0.0),
            timeout=timeout_s,
        )
    except Exception:                               # noqa: BLE001 - measurement only, fail quiet
        return ""
    first = (text or "").strip().splitlines()[0] if (text or "").strip() else ""
    out = first.strip().strip('"').strip()      # first line, then quotes: a quote can sit mid-text
    out = out.strip("'").strip()
    if not out or len(out) > 300:
        return ""
    _SIMPLIFIED[key] = out
    return out


def fire_jev_pass1(**kwargs) -> Optional[asyncio.Task]:
    """Schedule jev_pass1 in the background (strong ref held); None outside an event loop."""
    try:
        task = asyncio.get_running_loop().create_task(jev_pass1(**kwargs))
    except RuntimeError:
        return None
    _TASKS.add(task)
    task.add_done_callback(_TASKS.discard)
    return task


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
    pass1: Optional[dict] = None,
    article_date: Optional[str] = None,
) -> Optional[dict]:
    """Run both Jev passes on one article and log `event=jev_shadow`. Returns the payload
    (for tests); never raises. A `pass1` result from `jev_pass1` (same text and question) is
    reused instead of re-asking the selection questions."""
    t0 = time.perf_counter()
    payload: dict = {"url": url}
    try:
        if pass1 is None or pass1.get("norm_text") is None:
            pass1 = await jev_pass1(text=text, question=question, api_key=api_key, api_url=api_url,
                                    max_sentences=max_sentences, timeout_s=timeout_s, transport=transport)
        norm_text, spans, sentences = pass1["norm_text"], pass1["spans"], pass1["sentences"]
        payload["n"] = len(sentences)
        payload["haiku"] = [
            [locate_quote(norm_text, spans, p.get("quote") or ""),
             p.get("stance"), p.get("settled"), p.get("claim_strength"), p.get("evidence_class"),
             p.get("event_date")]
            for p in haiku_predictions
        ]
        if "skip" in pass1:
            payload["skip"] = pass1["skip"]
            return payload
        if "err" in pass1:
            payload["err"] = pass1["err"]
            return payload
        api_key = api_key or await asyncio.to_thread(resolve_api_key)
        if not api_key:
            payload["skip"] = "no_key"
            return payload
        tok_in = pass1["tok_in"]
        neg, nouls = pass1["neg"], pass1["nouls"]
        async with httpx.AsyncClient(timeout=timeout_s, transport=transport) as client:
            order = sorted(range(len(nouls)), key=lambda i: (-nouls[i], i))
            cand = [i for k, i in enumerate(order) if k < min_top or nouls[i] >= select_bar][:max_candidates]
            # Every Haiku-quoted sentence gets a Jev verdict too (outside the cap), so a veto on
            # Haiku's claims can be measured claim by claim (retro#847).
            cand += sorted({q[0] for q, *_ in payload["haiku"] if q} - set(cand))
            # retro#873: date shadow on Haiku-SETTLED claims only — on unsettled claims Jev
            # dates the utterance ("X said on Thursday"), not the event. Context = quote ±2
            # sentences (the whole article added nothing offline); candidates are found and
            # resolved in code against the publication date, Jev only picks.
            pub = _parse_day(article_date)
            date_jobs = []
            for h, row in enumerate(payload["haiku"]):
                idx = row[0]
                if pub is None or not idx or row[2] is not True:
                    continue
                lo, hi = max(0, min(idx) - 2), min(len(sentences), max(idx) + 3)
                window = " ".join(sentences[lo:hi])
                dcands = date_intervals(window, pub)
                date_jobs.append((h, dcands, _ask(
                    client, api_key,
                    {"sentence": " ".join(sentences[i] for i in idx), "context": window,
                     "publication_date": pub.isoformat()},
                    date_questions(dcands, pub), api_url)))
            scored, dated = await asyncio.gather(
                asyncio.gather(*[
                    _ask(client, api_key, {"sentence": sentences[i], "related_event": question}, _scoring_questions(), api_url)
                    for i in cand
                ], return_exceptions=True),
                asyncio.gather(*[job for _, _, job in date_jobs], return_exceptions=True),
            )
        payload["neg"] = round(neg, 3)
        payload["max_noul"] = round(max(nouls), 3)
        payload["top"] = [[i, round(nouls[i], 3)] for i in order[:8]]
        cands = []
        # One failed scoring call (e.g. a 429 among ~30 concurrent) drops that row only, not
        # the article: counted in `score_err`, the rest of the payload is still logged.
        score_err = sum(isinstance(r, BaseException) for r in scored)
        if score_err == len(scored) and scored:
            raise next(r for r in scored if isinstance(r, BaseException))
        if score_err:
            payload["score_err"] = score_err
        for i, resp in zip(cand, scored):
            if isinstance(resp, BaseException):
                continue
            tok_in += resp.get("usage", {}).get("input_tokens", 0)
            a = resp["answers"]
            nz, p0 = _nonzero(a["stance"], STANCE_VALUES)
            ec, ec_p = _top_choice(a.get("evidence_class"))
            cands.append([
                i, round(nouls[i], 3),
                round(_expected(a["stance"], STANCE_VALUES), 3), _argmax(a["stance"], STANCE_VALUES),
                round(a["settled"]["noul"], 3),
                round(_expected(a["strength"], STRENGTH_VALUES), 3),
                round(nz, 3), round(p0, 3),
                round(a["topic"]["noul"], 3), round(a["refs"]["noul"], 3),
                ec, round(ec_p, 3),
            ])
        # cand rows: [sentence_idx, noul, stance_expected, stance_argmax, settled, claim_strength,
        #             stance_nonzero, p_no_signal, topic, refs, evidence_class, evidence_class_p]
        # haiku rows: [sentence_idxs, stance, settled, claim_strength, evidence_class, event_date]
        # (event_date raw, before enforce_relative_date_resolution) — evidence_class is Haiku's
        # RAW class, snapshotted before enforce_anchor_provenance can demote cited_probability;
        # the shipped class is recoverable offline with tm.extractor._names_allowlisted_source.
        payload["cand"] = cands
        # dates rows: [haiku_row, haiku_event_date, jev_start, jev_end, jev_pick, jev_pick_p,
        #              dated_noul, n_candidates] — jev_start/end null for pick "none"; a failed
        #              date call logs its error string in place of jev_pick.
        date_rows = []
        for (h, cands_h, _), resp in zip(date_jobs, dated):
            if isinstance(resp, BaseException):
                date_rows.append([h, payload["haiku"][h][5], None, None, "err:" + type(resp).__name__, 0.0, None, len(cands_h)])
                continue
            tok_in += resp.get("usage", {}).get("input_tokens", 0)
            a = resp["answers"]
            pick, pick_p = _top_choice(a.get("when"))
            if pick == "pub":
                iv = (pub, pub)
            elif pick and pick.startswith("c") and pick[1:].isdigit() and int(pick[1:]) < len(cands_h):
                iv = cands_h[int(pick[1:])][:2]
            else:
                iv = (None, None)
            date_rows.append([h, payload["haiku"][h][5],
                              iv[0].isoformat() if iv[0] else None, iv[1].isoformat() if iv[1] else None,
                              pick, round(pick_p, 3), round(a["dated"]["noul"], 3), len(cands_h)])
        if date_rows:
            payload["dates"] = date_rows
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
