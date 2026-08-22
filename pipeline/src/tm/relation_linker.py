"""Signed relation typing between candidate forecast pairs (retro#574).

The linker's candidate-pair step (cosine >= 0.75-0.85 + shared tag, daatan's
``findSimilarForecasts``) is out of scope here — this module only answers, given two
claim texts, what the LOGICAL relationship between them is. See ``PairRelationOutput``
in models.py for the signed enum this returns and retro#574 for the motivating audit
(8 of 110 high-similarity ACTIVE pairs were incoherent, 7 of those 8 because the
existing claim_direction+deadline rule cannot see negation/implication/complement).

Deliberately NOT wired to anything yet: no cron, no candidate-pair fetch from daatan,
no persistence. Those are separate follow-ups once this classifier is proven accurate
(see the eval harness in tests/test_relation_linker.py).
"""

import asyncio

from instructor.core.exceptions import InstructorRetryException

from .config import settings
from .llm import complete_structured
from .models import PairRelationOutput

# Split into a fixed, cacheable PROMPT_PREFIX (identical on every call, no .format()
# placeholders) and a PROMPT_SUFFIX carrying the two claim texts — same rationale as
# gatekeeper.py/extractor.py's split.
PROMPT_PREFIX = """\
You are a relation-typing classifier for a forecasting system. You are given two
independent forecasting questions (claim A and claim B) about real-world events, and
must judge the LOGICAL relationship between the propositions they assert — not their
topical similarity.

Two propositions can look similar (same actors, same topic) while asserting OPPOSITE
things ("Country X keeps troops in Y" vs "Country X withdraws from Y" are about the
same underlying fact, negated) — do not let topical closeness make you miss this.

Classify the pair into exactly one relation_type:

- **alias** — A and B assert the SAME underlying proposition, possibly phrased with the
  same polarity (paraphrases) or opposite polarity (one asserts the negation of the
  other, e.g. "the ceasefire holds" vs "the ceasefire collapses"). Set polarity
  accordingly.
- **nested** — one proposition is a strict logical SUBSET of the other: whenever the
  narrower one is true, the broader one MUST also be true, but not vice versa. Classic
  cases: (a) the same event at a later deadline is a superset of the event at an
  earlier deadline ("X happens by 2029" is a subset of "X happens by 2030"); (b) a
  conjunction is a subset of one of its conjuncts ("X invades AND occupies Y" is a
  subset of "X invades Y"); (c) a stricter numeric threshold is a subset of a looser
  one ("price >= 200" is a subset of "price >= 130"). Set direction to a_to_b if A is
  the narrower (subset) side, b_to_a if B is.
- **complement** — A and B describe DIFFERENT, mutually exclusive outcomes of the same
  underlying situation (at most one can be true), but neither is simply the logical
  negation of the other and neither is a subset of the other — e.g. two different named
  individuals each becoming the same office-holder, two different candidate winners of
  the same race. Direction does not apply (leave null).
- **implies** — A being true logically forces B to be true (or false), but B does NOT
  symmetrically constrain A the way a subset would — typically a causal/definitional
  precondition ("a country ORDERS a deployment" is implied by "the deployment HAPPENS",
  since it cannot happen without having been ordered) or a stated policy implication
  ("a government maintains long-term occupation" implies "that government has NOT
  withdrawn" — note the implied side can be negated; set polarity=opposite when it is).
  Set direction to a_to_b if A is the side doing the implying (the trigger/cause),
  b_to_a otherwise.
- **independent** — A and B are merely topically related (same actor, region, or story)
  but neither's truth value logically constrains the other's. Most same-topic pairs are
  this. Direction and polarity do not apply.

For every relation_type, also set:
- polarity: "same" if the relationship holds between A and B exactly as stated;
  "opposite" if it only holds once you mentally negate one side (this is the field that
  catches "H happens" vs "H does not happen" being missed as unrelated instead of typed
  as an alias-by-negation). Leave null for independent.
- quote_a / quote_b: the shortest verbatim span from each claim's own text that
  justifies your judgment (usually just the key predicate — "withdraws", "maintains
  presence", "by 2029"). Leave null only if the whole (short) claim text is itself the
  justification.
- reason: one sentence.

Worked examples (illustrative only, not from the real question set):
1. A="Person X remains in office through the end of the year." B="Person X leaves
   office before the end of the year." -> alias, polarity=opposite (same underlying
   fact — whether X is in office at year end — negated).
2. A="Candidate P wins the general election." B="Candidate Q wins the general
   election." -> complement, polarity=same (different people, at most one wins, but Q
   winning is not simply "P does not win" — a third candidate could win).
3. A="Company C's stock closes above $50 on any day this year." B="Company C's stock
   closes above $30 on any day this year." -> nested, direction=a_to_b, polarity=same
   (crossing the higher bar guarantees crossing the lower one).
4. A="The central bank raises interest rates at its next meeting." B="The central
   bank's policy statement signals tightening." -> implies, direction=a_to_b,
   polarity=same (a rate hike entails a tightening signal; the reverse is not
   guaranteed).
5. A="A wildfire is reported in the region this month." B="The regional governor is
   re-elected next year." -> independent.
"""

PROMPT_SUFFIX = """\
Claim A: {claim_a}
Claim B: {claim_b}
"""


# Nova Lite's MD_JSON mode is measurably unreliable on this schema: empirically (see
# eval_relation_linker.py's run log / PR description) it sometimes echoes back a
# JSON-Schema-shaped object (`{"properties": {...}}`, the retro#306 envelope quirk
# GatekeeperOutput already guards against — or, seen here for the first time, a bare
# `{"description": ..., "type": "object"}` schema echo with no instance data at all,
# which no amount of local unwrapping can repair) instead of an instance of
# PairRelationOutput. complete_structured's shared `max_retries=1` gives instructor
# zero self-correction attempts, so a single call's failure rate on this schema was
# observed north of 50% in manual runs. This retries the WHOLE call (not just local
# parsing) up to _MAX_ATTEMPTS times before giving up, isolated to this module so it
# doesn't change retry behaviour for gatekeeper/extractor's simpler schemas.
_MAX_ATTEMPTS = 4
_RETRY_DELAY_S = 2


async def classify_relation(claim_a: str, claim_b: str):
    """One structured-output call typing the (claim_a, claim_b) pair. Returns
    ``(PairRelationOutput, usage)`` — same shape as gatekeeper.check_is_prediction /
    extractor calls. Cheap tier (extractor_model, not gatekeeper_model): this task
    requires real entailment/negation reasoning, closer to extraction than to a binary
    relevance gate, so it gets the richer of the two cheap tiers.

    Retries on a malformed-output failure (see _MAX_ATTEMPTS above) — NOT on a
    genuine rate-limit error, which complete_structured's own retry_on_rate_limit
    wrapper already handles with its own backoff schedule before this ever sees it.
    """
    prompt = PROMPT_SUFFIX.format(claim_a=claim_a, claim_b=claim_b)
    last_exc = None
    for attempt in range(_MAX_ATTEMPTS):
        try:
            return await complete_structured(
                settings.extractor_model, PairRelationOutput, prompt, max_tokens=300, timeout=60,
                cached_prefix=PROMPT_PREFIX,
            )
        except InstructorRetryException as exc:
            last_exc = exc
            if attempt < _MAX_ATTEMPTS - 1:
                await asyncio.sleep(_RETRY_DELAY_S)
    raise last_exc
