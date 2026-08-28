"""Threshold-shape detection for batch extractor routing (retro#688).

WHY THIS EXISTS AND WHAT IT IS NOT
----------------------------------
retro#688 asks to route ``claim_archetype = threshold`` questions to Haiku 4.5 in the
batch lane. **The batch lane has no ``claim_archetype``.** That field lives on the live
Oracle's ``ForecastRequest`` and its own description says where it comes from:

    "Temporal archetype from the caller's claim classifier."

i.e. daatan classifies its claims and sends the label over the wire on ``/forecast``.
retro contains no code that derives one, and the batch lane has no caller to send it:
its unit of work is an ``(event, source, article)`` triple built in ``orchestrator.py``
from ``data/events/*.json``, whose schema is id / name / outcome / outcome_date /
search_keywords / llm_referee_criteria / predictive_window_days / category / tags. No
archetype, and no question either — batch events are curated retroactive event
descriptions ("Shekel drops below 4.0 NIS/USD"), not the binary questions the live lane
answers. ``runner.py`` already documents the same asymmetry for
``enforce_deadline_arithmetic``: "retroactive events don't pose a single binary question
with a direction the way a live /forecast request does."

So this module supplies the missing input, and it is deliberately NARROW:

* It answers one question — *does a number decide this event?* — and returns a bool.
  It is not a four-way archetype classifier and must not grow into one. Reproducing
  daatan's scheduled/diffuse/threshold/none taxonomy here would create a second
  classifier free to drift from the one that actually labels claims, with nothing
  comparing them.
* Its output is used for MODEL ROUTING ONLY. It is never written to a row, never
  compared against a live ``claim_archetype``, and nothing downstream reads it.
  A misclassification costs money or quality on one extraction; it cannot corrupt data.

WHAT COUNTS AS THRESHOLD-SHAPED
-------------------------------
Both halves must be present:

1. a **magnitude** — digits carrying a unit, currency, percentage or scale word
   ("$100/barrel", "4.75%", "15,000", "$1 trillion", "100K"), and
2. a **comparison or attainment cue** — the event turns on the number being crossed
   ("exceeds", "drops below", "reaches", "raises ... to", "top 5", a trailing "+").

Requiring both is what separates the class from "merely contains a digit". On the live
91-event corpus that distinction is doing real work:

    included   "Brent crude oil exceeds $100/barrel"        magnitude + "exceeds"
               "Bank of Israel raises interest rate to 4.75%"  magnitude + "raises…to"
               "Israeli tech layoffs exceed 15,000 cumulative"
    excluded   "Google agrees to acquire Wiz for $32B"      magnitude, no cue — the
                                                            number names the deal, it
                                                            does not decide it
               "UK PM Liz Truss resigns after 45 days"      duration, not a threshold
               "Netanyahu forms new government (6th …)"     ordinal identifier
               "Israel strikes Iran's nuclear facilities (April 2024 …)"   a date

Bare years, ordinals and "after N days/years" durations are excluded explicitly; each
one is a real member of the corpus that a naive digit test would sweep in.
"""

from __future__ import annotations

import re
from typing import Optional

from .config import settings as _settings
from .config import Settings

# ── magnitude ────────────────────────────────────────────────────────────────
# A number that carries a unit, and is therefore a quantity rather than a label.
# Written as alternatives rather than one regex so each clause stays readable and
# individually testable.
_MAGNITUDE_PATTERNS = (
    # currency-led: $100, $1 trillion, $23B, €70
    r"[$€£₪]\s?\d[\d,.]*\s*(?:k|m|bn?|tn?|trillion|billion|million|thousand)?\b",
    # percentage: 15%, 4.75%, 50%+
    r"\d[\d,.]*\s?%",
    # scale-suffixed: 100K, 15,000, $32B handled above, 100M
    r"\b\d[\d,.]*\s?(?:k|m|bn|tn)\b",
    # spelled scale: 1 trillion, 23 billion
    r"\b\d[\d,.]*\s+(?:trillion|billion|million|thousand)\b",
    # unit-bearing: 4.0 NIS/USD, 100/barrel, 15,000 cumulative is caught by the
    # grouped-thousands clause below
    r"\b\d[\d,.]*\s*/\s*\w+",
    r"\b\d+\.\d+\s+[A-Z]{3}\b",
    # grouped thousands: 15,000
    r"\b\d{1,3}(?:,\d{3})+\b",
)
_MAGNITUDE_RE = re.compile("|".join(_MAGNITUDE_PATTERNS), re.IGNORECASE)

# ── comparison / attainment cue ──────────────────────────────────────────────
# The event turns on the number being crossed, reached or ranked.
_CUE_RE = re.compile(
    r"\b("
    r"exceed(?:s|ed|ing)?|surpass(?:es|ed|ing)?|top(?:s|ped|ping)?|cross(?:es|ed|ing)?"
    r"|above|below|under|over|beyond"
    r"|at\s+least|at\s+most|more\s+than|less\s+than|fewer\s+than|no\s+more\s+than"
    r"|reach(?:es|ed|ing)?|hit(?:s|ting)?|climb(?:s|ed|ing)?|rise(?:s|n)?|ris(?:es|ing)"
    r"|drop(?:s|ped|ping)?|fall(?:s|en|ing)?|declin(?:e|es|ed|ing)|sink(?:s|ing)?"
    r"|raise(?:s|d)?|cut(?:s|ting)?|lower(?:s|ed|ing)?"
    r")\b",
    re.IGNORECASE,
)
# "top 5", "top-10" — rank cues carry their own magnitude, which is a bare integer and
# so deliberately not in _MAGNITUDE_PATTERNS.
_RANK_RE = re.compile(r"\btop[\s-]?\d+\b", re.IGNORECASE)
# A trailing "+" turns a plain number into a floor: "100K+", "4.5%+", "15,000+".
_OPEN_ENDED_RE = re.compile(r"\d[\d,.]*\s?(?:%|k|m|bn|tn)?\s?\+")

# ── exclusions ───────────────────────────────────────────────────────────────
# Stripped BEFORE the magnitude test so their digits cannot satisfy it.
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_ORDINAL_RE = re.compile(r"\b\d+(?:st|nd|rd|th)\b", re.IGNORECASE)
_DURATION_RE = re.compile(
    r"\bafter\s+\d[\d,.]*\s*(?:second|minute|hour|day|week|month|year)s?\b",
    re.IGNORECASE,
)


def is_threshold_shaped(text: Optional[str]) -> bool:
    """True when a numeric threshold decides ``text``.

    Deliberately conservative: an event with no number, or with a number that merely
    labels something (a year, an ordinal, a deal size, a duration), is False. False is
    the safe answer — it routes to the configured default model, i.e. today's behaviour.
    """
    if not text:
        return False

    stripped = _DURATION_RE.sub(" ", text)
    stripped = _YEAR_RE.sub(" ", stripped)
    stripped = _ORDINAL_RE.sub(" ", stripped)

    if _RANK_RE.search(stripped):
        return True
    if not _MAGNITUDE_RE.search(stripped):
        return False
    return bool(_CUE_RE.search(stripped) or _OPEN_ENDED_RE.search(stripped))


def select_extractor_model(
    event_name: Optional[str],
    settings: Optional[Settings] = None,
) -> Optional[str]:
    """The extractor model this batch event should use, or None for the configured default.

    None — not ``settings.extractor_model`` — is the "no opinion" answer, because that is
    what ``extract_predictions(model=...)`` already means (retro#652): "override this call
    only; None keeps the configured global". Returning None rather than echoing the global
    keeps one meaning of "unrouted" instead of two.

    Off unless ``threshold_extractor_model`` is set, so merging retro#688 changes nothing
    on its own — see the setting's comment in config.py.
    """
    s = settings or _settings
    override = (s.threshold_extractor_model or "").strip()
    if not override:
        return None
    return override if is_threshold_shaped(event_name) else None
