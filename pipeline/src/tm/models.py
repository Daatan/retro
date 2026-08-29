import json as _json
import logging
from pydantic import (
    AliasChoices, BaseModel, ConfigDict, Field, model_serializer, model_validator,
)
from pydantic import ValidationError as _ValidationError
from typing import Literal, Optional, Any
from enum import Enum

logger = logging.getLogger(__name__)


# --- LLM Output Schemas ---

def _unwrap_properties_envelope(data: Any) -> Any:
    """Nova Lite (MD_JSON mode) intermittently wraps its structured output in a
    spurious top-level {"properties": {...}} envelope instead of returning the
    flat schema fields directly, which fails Pydantic validation outright. Unwrap
    it before field validation runs. See retro#306."""
    if isinstance(data, dict) and data.keys() == {"properties"} and isinstance(data["properties"], dict):
        return data["properties"]
    return data


def _coerce_nested_json_string(data: Any, key: str) -> Any:
    """Parse a nested object the model double-serialized as a JSON string.

    `ExtractionOutput._deserialize_string_predictions` already handles this one
    level up — some models in TOOLS mode emit a nested object as a string rather
    than an object. `reader_confidence` (retro#681) is the first nested field
    INSIDE a prediction, so it needs the same treatment.

    Deliberately narrow: a string that does not parse to a dict is left exactly
    as it came, so Pydantic raises on it. Swallowing it into None would turn a
    model that answers badly into a model that looks like it did not answer,
    which is precisely the failure mode a shadow field's fill rate is supposed
    to make visible.
    """
    if not isinstance(data, dict):
        return data
    raw = data.get(key)
    if not isinstance(raw, str):
        return data
    try:
        parsed = _json.loads(raw)
    except (ValueError, TypeError):
        return data
    if isinstance(parsed, dict):
        data = {**data, key: parsed}
    return data


class GatekeeperOutput(BaseModel):
    is_prediction: bool
    reason: str
    prediction_count_estimate: int = Field(default=0, ge=0)
    relevance_score: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="Graded topic relevance of the article to the event [0,1]; "
                    "its square multiplies the source's aggregation weight. When "
                    "a caller/model OMITS it entirely, the default is is_prediction- "
                    "dependent: 1.0 (neutral) on a passing gate, 0.0 on a rejection — "
                    "so an unscored rejection can never read as relevant to a caller "
                    "(e.g. the /relevance endpoint response, or ledger analysis) that "
                    "doesn't also check is_prediction. An EXPLICITLY supplied score is "
                    "never overridden, at any value — a graded near-miss on a "
                    "rejection (e.g. relevance=0.1) is meaningful and must survive.",
    )

    @model_validator(mode="before")
    @classmethod
    def _unwrap_envelope(cls, data: Any) -> Any:
        return _unwrap_properties_envelope(data)

    @model_validator(mode="after")
    def _default_relevance_to_zero_on_unscored_rejection(self) -> "GatekeeperOutput":
        if not self.is_prediction and "relevance_score" not in self.model_fields_set:
            self.relevance_score = 0.0
        return self


class PredictionType(str, Enum):
    binary = "binary"
    continuous = "continuous"
    range = "range"
    trend = "trend"


# How firmly the EXTRACTOR stands behind its own reading of a span (retro#681).
#
# Rationale lives in comments, NOT in a docstring, and that is load-bearing rather than
# stylistic: Pydantic copies a model's docstring into its JSON schema `description`, and
# that schema is the LLM tool definition sent on every extraction call. `PredictionExtraction`
# and `ExtractionOutput` both keep their rationale up here for the same reason and both
# carry a zero-length class description. An earlier draft of this class used a docstring
# and shipped 1,283 characters of the paragraphs below to the model on every call — 18% of
# the whole ExtractionOutput schema, larger than any single field description in it.
#
# The other half of the `certainty` split (retro#680): `claim_strength` is the SOURCE's
# commitment to the claim, this is the READER's confidence in its own interpretation of it.
# One number carried both until Oracle 1.5 Phase 1, and the conflation was billed to the
# source — retro#664's Kenya case is the evidence: on the flat span "retained the Central
# Bank Rate at 8.75 percent" Haiku returned 0.70 and Nova Lite 0.30/stance 0.00. The 0.30 is
# the reader's confusion filed as the source's hedge.
#
# Deliberately NOT a scalar (plan §4.1): verbalised LLM confidence clusters at 0.8–0.9
# whatever the input, so a float would record the model's register, not its difficulty. The
# three-level `level` plus a named `trap` is what a model can actually answer — and the trap
# names are the classes for which detectors already exist (retro#657 negation, the PR#671
# numeric cases, `stance_tone_conflation.json`, the dyad facets), so every self-flag is
# checkable against an independent detector from day one.
#
# EXPERIMENTAL, shadow: populated and persisted, read by nothing. Phase 4 is where `low`
# rows get down-weighted and excluded from the credibility bill, and where settlement gains
# its second bar (`level == "high"`).
class ReaderConfidence(BaseModel):
    level: Literal["high", "medium", "low"] = Field(
        description="How confident YOU are that another careful reader would extract the "
                    "same stance sign from this span: 'high' the span says what it says; "
                    "'medium' the direction is clear but you had to resolve something (a "
                    "comparison, a referent, a date) to reach it; 'low' another careful "
                    "reader could plausibly read it differently. NOT the source's hedging — "
                    "that is claim_strength.",
    )
    trap: Optional[Literal[
        "negation", "numeric_comparison", "entity_or_event_mismatch",
        "tone_vs_content", "inference_needed", "conflicting_signals",
    ]] = Field(
        default=None,
        description="Which ONE of the known reading traps applied to THIS span, or omit when "
                    "none did: 'negation' the meaning turns on a not/no/fails-to/remains-below; "
                    "'numeric_comparison' the direction came from comparing numbers, not from "
                    "the words; 'entity_or_event_mismatch' the span is about a neighbouring "
                    "actor, target or arena and had to be carried across; 'tone_vs_content' the "
                    "tone points one way and the factual content the other; 'inference_needed' "
                    "the span does not address the related event directly and reaching it took "
                    "a reasoning step; 'conflicting_signals' the span carries two indications "
                    "pointing opposite ways. A trap does not force a low level.",
    )


def _drop_malformed_reader_confidence(data: Any) -> Any:
    """Never let the shadow field cost a real prediction (retro#681).

    `ReaderConfidence` is strict — `level` is required and both enums are
    closed — because a half-answer is not data. But `complete_structured` runs
    instructor with `max_retries=1`: if the model still returns e.g.
    `{"trap": "negation"}` with no level, the whole `ExtractionOutput` raises
    and the article is dropped from the forecast. Paying a real article for a
    field nothing reads is the wrong trade, so a malformed value is discarded
    here and the prediction stands.

    The drop is LOGGED rather than silent. A field harvested to be measured
    must not be able to make "the model answered badly" look identical to "the
    model did not answer" — that is the only distinction its fill rate exists
    to draw. Grep `event=reader_confidence_malformed` to count them.

    Strictness is kept where it belongs: constructing `ReaderConfidence`
    directly, or validating a stored `ClaimDetail`, still raises.
    """
    if not isinstance(data, dict):
        return data
    raw = data.get("reader_confidence")
    if raw is None:
        return data
    try:
        ReaderConfidence.model_validate(raw)
    except (_ValidationError, TypeError):
        logger.warning("event=reader_confidence_malformed value=%r", raw)
        return {**data, "reader_confidence": None}
    return data


def _drop_out_of_enum(data: Any, key: str, allowed: frozenset[str]) -> Any:
    """The flat-enum version of the guard above, for retro#686's shadow fields.

    Same trade, same reasoning: `report_kind` and `consensus_view` are read by
    nothing yet, and `complete_structured` runs instructor with `max_retries=1`,
    so a model that keeps answering `"levels"` or `"expects yes"` would raise out
    of `ExtractionOutput` and drop a real article from a real forecast. A new
    enum is exactly where that happens — the model has never been asked for
    these names before, and the models most likely to garble them are the ones
    this field is being harvested to study.

    Logged, never silent, for the reason `reader_confidence` is: a shadow field
    exists to be counted, and a silent coercion to None makes "answered badly"
    indistinguishable from "did not answer". Grep `event=<field>_malformed`.

    Deliberately NOT applied to `evidence_class`, `facet` or
    `fact_signal_absent_reason`. Those are read by live consumers, where a
    dropped value changes a forecast rather than a fill rate — relaxing them is
    a behaviour change that belongs to whoever owns those fields, not to this
    one.
    """
    if not isinstance(data, dict):
        return data
    raw = data.get(key)
    if raw is None:
        return data
    # The isinstance guard is load-bearing, not defensive: a model that answers
    # with a list or an object hands `in` an unhashable value, and the raise
    # would come from the guard that exists to prevent raises.
    if isinstance(raw, str) and raw in allowed:
        return data
    logger.warning("event=%s_malformed value=%r", key, raw)
    return {**data, key: None}


_REPORT_KIND_VALUES = frozenset({"level", "change"})
_CONSENSUS_VIEW_VALUES = frozenset({"expects_yes", "expects_no", "divided"})


class PredictionExtraction(BaseModel):
    # `claim_strength` was named `certainty` until Oracle 1.5 Phase 1 (retro#680). The
    # elicitation text is unchanged — only the name moved, so the number this field carries
    # is the same one the within-article mean has always used. The old name is accepted on
    # input (`validation_alias` below) so stored rows and any not-yet-updated producer still
    # parse; the wire keeps emitting BOTH names for one schema cycle (api ClaimDetail /
    # SourceSignal / PoolSourceInput). The rename exists because "certainty" invited a second
    # reading — the *reader's* confidence in its own interpretation — which is a different
    # quantity and gets its own field (`reader_confidence`, retro#681). retro#664's Kenya case
    # is the evidence: an unhedged span ("retained the Central Bank Rate at 8.75 percent")
    # scored 0.30 because the reader was unsure, not because the source hedged.
    model_config = ConfigDict(populate_by_name=True)

    quote: str = Field(description="Exact sentence(s) from the article containing the prediction")
    claim: str = Field(description="One-sentence neutral summary in English")
    stance: float = Field(ge=-1.0, le=1.0, description="Directional outlook: -1=event won't happen, +1=event will happen")
    claim_strength: float = Field(
        ge=0.0, le=1.0,
        validation_alias=AliasChoices("claim_strength", "certainty"),
        description="Linguistic confidence: 0=very hedged, 1=absolute",
    )

    @model_serializer(mode="wrap")
    def _emit_certainty_alias(self, handler) -> dict[str, Any]:
        """Serialize `claim_strength` under BOTH names for one schema cycle.

        `orchestrator.py` dumps this model straight into the atlas article JSON,
        and `utils.py`/`backtest.py` validate those rows on the literal key
        `certainty` — dropping it would make every new row score as broken rather
        than fail loudly. A serializer (not a `computed_field`) is what does this:
        it leaves the VALIDATION schema untouched, which is the schema instructor
        sends to the model, so the extractor is still asked for exactly one name.
        """
        data = handler(self)
        if "claim_strength" in data:
            data["certainty"] = data["claim_strength"]
        return data
    settled: Optional[bool] = Field(default=None, description="True when the source reports the outcome as an accomplished fact (event occurred, or became permanently impossible) — not a prediction, however confident. A POSITIVE settlement (event occurred) must be accompanied by event_date; one without a parseable event_date, or dated after the article itself, is demoted to ordinary evidence in code (enforce_settlement_event_date). A NEGATIVE settlement (became impossible) carries the FORECLOSING event's date in event_date when the article dates it — the rival's win, the elimination, the death that made the outcome impossible; leave it empty when the impossibility comes only from time expiring or the foreclosure is undated")
    quantitative_estimate: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="An explicit modeled/poll/market PROBABILITY the source cites FOR THE "
                    "RELATED EVENT ITSELF (not a proxy stage), as a probability in [0,1]. "
                    "Null when the source has no such cited figure — qualitative "
                    "'favorite'/'front-runner' framing without a stated number does not "
                    "count, and neither does a vote share, seat count, or poll share (those "
                    "are cited_share evidence, not probabilities of the event; retro#362).",
    )
    evidence_class: Optional[Literal[
        "reported_fact", "cited_probability", "cited_share", "reporting", "opinion",
    ]] = Field(
        default=None,
        description="The kind of evidence this claim is, independent of stance/claim_strength "
                    "(S2, retro docs/ORACLE_VARIABLES.md §5). LOAD-BEARING since the S2 "
                    "weight cutover: keys the cross-article evidence_class_weight lookup, "
                    "and only cited_probability authorizes the quantitative_estimate "
                    "stance rewrite (resolve_stance_certainty, retro#362). "
                    "Omit entirely rather than guessing when none fits cleanly.",
    )
    # --- fact_signal lane (EXPERIMENTAL, shadow — Phase 2 of the author-scoring redesign) ---
    # The fact-lane counterpart to `stance`: what the REPORTED FACTS alone imply about the
    # event, un-fused from the author's assertion/framing. Populated in shadow alongside
    # stance; not yet consumed by any estimator. See the FACT_SIGNAL prompt section.
    fact_signal: Optional[float] = Field(
        default=None, ge=-1.0, le=1.0,
        description="EXPERIMENTAL, shadow — what the REPORTED FACTS alone imply about the "
                    "related event, separate from any assertion, quoted opinion, or framing: "
                    "+1 the facts establish it happened / is happening, -1 the facts establish "
                    "it will not or cannot, 0 the facts bear on it but point neither way. A "
                    "precursor/precondition/escalation is capped at |0.3|; a fact about a "
                    "DIFFERENT actor-target pair than the claim is context only (near 0), never "
                    "settlement; a merely CLAIMED (unverified) event is down-weighted. Null when "
                    "the prediction rests on opinion/advocacy with no reported fact bearing on "
                    "the event.",
    )
    fact_signal_absent_reason: Optional[Literal[
        "no_fact_found", "contrary_below_anchor", "opinion",
    ]] = Field(
        default=None,
        description="Required whenever fact_signal is omitted, so the null itself stays "
                    "honest (retro#471) — a consumer must be able to tell 'nothing found' "
                    "from 'something found that points the other way': "
                    "'no_fact_found' — nothing in the article bears on the event's "
                    "occurrence either way; "
                    "'contrary_below_anchor' — a reported fact DOES point against the "
                    "event but is too weak, ambiguous, or off-dyad to anchor a graded "
                    "negative value (most contrary facts should still be graded per the "
                    "NEGATIVE PRECURSORS rule — reserve this for the genuine remainder); "
                    "'opinion' — the claim rests on opinion, advocacy, or expectation with "
                    "no reported fact bearing on the event. Omit entirely when fact_signal "
                    "is present.",
    )
    facet: Optional[Literal["announcement", "denial", "neither"]] = Field(
        default=None,
        description="EXPERIMENTAL, shadow (retro#354 D2a/D2c) — whether the reported fact "
                    "behind fact_signal ANNOUNCES the event happening/happened, DENIES it "
                    "will/did happen, or is NEITHER (bears on the event without asserting "
                    "either polarity, e.g. a precursor or a decider's on-record statement "
                    "that doesn't itself confirm or deny). Lets a future magnitude refit "
                    "(D2b/D3) compare |fact_signal| separately for announcement vs. denial "
                    "claims. Omit when fact_signal is None.",
    )
    event_actors: Optional[str] = Field(
        default=None,
        description="EXPERIMENTAL, shadow — WHO acts in the reported fact behind fact_signal "
                    "(the acting subject(s), e.g. 'United States', 'Likud'). Recorded so the "
                    "estimator can check the fact's actor-target pair against the claim's and "
                    "demote a wrong-dyad fact. Omit when fact_signal is null or no actor applies.",
    )
    event_target: Optional[str] = Field(
        default=None,
        description="EXPERIMENTAL, shadow — the TARGET/object of the action in the reported "
                    "fact behind fact_signal (e.g. 'Iran', 'the Knesset'). With event_actors "
                    "this is the fact's dyad, for the actor-pair check. Omit when fact_signal "
                    "is null or no target applies.",
    )
    is_occurrence: Optional[bool] = Field(
        default=None,
        description="EXPERIMENTAL, shadow — True when the reported fact IS the related event "
                    "itself occurring (or its definitive outcome); False when it is only a "
                    "precursor/precondition/escalation/capability that precedes the event. "
                    "Omit when fact_signal is null.",
    )
    verified: Optional[bool] = Field(
        default=None,
        description="EXPERIMENTAL, shadow — True when the reported fact is independently "
                    "reported as having happened; False when it is only CLAIMED by an "
                    "interested/belligerent party and not independently confirmed. Omit when "
                    "fact_signal is null.",
    )
    event_date: Optional[str] = Field(
        default=None,
        description="ISO date (YYYY-MM-DD) on which the article says the RELATED EVENT "
                    "itself occurs or occurred. Resolve relative references ('on Friday', "
                    "'tomorrow', 'next week') against the article's date to an absolute "
                    "calendar date. The date of the event in the claim — NOT of any adjacent "
                    "or downstream event. For a NEGATIVE settlement, the date of the "
                    "FORECLOSING event that made the outcome impossible (see settled). "
                    "Omit entirely when the article states no date for it.",
    )
    event_date_reference: Optional[str] = Field(
        default=None,
        description="The article's VERBATIM relative expression behind event_date ('on "
                    "Friday', 'yesterday', 'tomorrow'), copied unchanged so code can redo "
                    "the calendar arithmetic and audit the resolution "
                    "(enforce_relative_date_resolution). Omit when the article states the "
                    "absolute date outright.",
    )
    # Not requested from LLM; kept Optional for backward compat with existing atlas entries
    sentiment: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    specificity: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    hedge_ratio: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    conditionality: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    magnitude: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    time_horizon: Optional[str] = Field(default=None)
    time_horizon_days: Optional[int] = Field(default=None)
    prediction_type: Optional[PredictionType] = Field(default=None)
    source_authority: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    # --- Conditional fields (v1.1, Phase 1 capture plan; see conditionals.md §4.4 + conditional-capture-phase1.md §3) ---
    # PRE-RESOLUTION: recorded BEFORE enforce_* chain (unlike other PredictionExtraction fields which are post-resolution)
    is_conditional: Optional[bool] = Field(
        default=None,
        description="True when the claim is conditional on an antecedent; the gate for step 4 (attenuation)"
    )
    antecedent_text: Optional[str] = Field(
        default=None,
        description="Verbatim 'if'-clause from the article, original language; unrecoverable if not captured now"
    )
    antecedent_text_en: Optional[str] = Field(
        default=None,
        description="Antecedent as standalone English proposition, stated positively (v1.1). "
                    "Negation lives in antecedent_polarity. THE ONLY FIELD used for embedding/linking (§3.4)"
    )
    antecedent_polarity: Optional[bool] = Field(
        default=None,
        description="False for 'if X does NOT happen', True/None for affirmative form"
    )
    relation: Optional[str] = Field(
        default=None,
        description="How antecedent relates to consequent: 'raises'/'lowers' (evidential), "
                    "'requires'/'precludes' (logical), 'unclear'. Plain string so new values don't fail callers"
    )
    strength: Optional[str] = Field(
        default=None,
        description="Source's stated strength when no explicit probability: 'certain'/'likely'/'possible'/'unlikely'. "
                    "Plain string to match relation enum pattern"
    )
    stated_probability: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="P(consequent|antecedent) when source explicitly states a number. Null otherwise"
    )
    is_counterfactual: Optional[bool] = Field(
        default=None,
        description="True for 'had X not happened' — past-directed, different epistemic object than conditional"
    )
    speaker: Optional[str] = Field(
        default=None,
        description="Who asserted the conditional: outlet name or quoted analyst. For attribution"
    )
    # --- Reader confidence (Oracle 1.5 Phase 1, retro#681) ---
    # Appended at the TAIL on purpose. retro#680 measured that perturbing the middle of this
    # schema costs Nova Lite the whole fact_signal block (fill 42% → 25%); a new field added
    # after every existing one leaves their relative order — and the text the model reads
    # around them — untouched.
    reader_confidence: Optional[ReaderConfidence] = Field(
        default=None,
        description="EXPERIMENTAL, shadow (retro#681) — how confident YOU are in your own "
                    "reading of this span, as {level, trap}. The reader's confidence, NOT the "
                    "source's commitment to the claim: that is claim_strength, and the two are "
                    "independent (a flat categorical sentence you had to work to interpret is "
                    "high claim_strength with a low level). Set it on every prediction.",
    )
    # retro#686 (unparked from #673 §2). A flat two-member enum, not a scalar: #673's own
    # caveat is that every new graded field is a fresh place for the #394 pathology, where a
    # scalar collapses onto a handful of band labels anyway. One bit cannot collapse.
    report_kind: Optional[Literal["level", "change"]] = Field(
        default=None,
        # Deliberately terse: the REPORT_KIND prose block carries the rule and the worked
        # test. Every char here is billed on every call and the schema is ~27% of the
        # extractor prompt (retro#700), so the schema says WHICH field, not HOW to decide it.
        description="EXPERIMENTAL, shadow (retro#686) — the standing situation ('level') or a "
                    "step in it ('change'). See REPORT_KIND; omit when neither fits.",
    )

    @model_validator(mode="before")
    @classmethod
    def _unwrap_envelope(cls, data: Any) -> Any:
        data = _coerce_nested_json_string(_unwrap_properties_envelope(data), "reader_confidence")
        data = _drop_malformed_reader_confidence(data)
        return _drop_out_of_enum(data, "report_kind", _REPORT_KIND_VALUES)


class ExtractionOutput(BaseModel):
    predictions: list[PredictionExtraction]
    author_lean: Optional[float] = Field(
        default=None, ge=-1.0, le=1.0,
        description="The BYLINE author's / outlet's OWN forecast of the related event, for "
                    "scoring the author's accuracy later — SEPARATE from the evidence "
                    "`predictions` and NOT used in the event estimate. +1 = the author "
                    "themselves expects the event to happen, -1 = the author expects it will "
                    "NOT happen, 0 = the author explicitly weighs both sides and commits to "
                    "neither. Null (omit) when the byline author only reports facts or relays "
                    "other people's views without endorsing a direction. A view held by a "
                    "QUOTED third party is that source's, not the byline's — never record it "
                    "here.")
    author_lean_certainty: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="How firmly the byline author commits to author_lean (0 = heavily hedged, "
                    "1 = emphatic). Null when author_lean is null.")
    # retro#686 (unparked from #673's "predicted consensus"). Article-level, next to
    # author_lean, because it is a property of the article's reporting rather than of any one
    # claim — and deliberately the same shape question as author_lean so the two stay
    # comparable: author_lean is what the AUTHOR thinks, this is what the author says EVERYONE
    # ELSE thinks. Categorical rather than a signed float for #394's reason (see report_kind);
    # the consumer needs a sign it can disagree with the pool about, not a magnitude.
    consensus_view: Optional[Literal["expects_yes", "expects_no", "divided"]] = Field(
        default=None,
        # Terse for the same reason as report_kind. The one clause kept here is the
        # never-your-own-view guard: it is the field's whole failure mode, its kill criterion
        # (>20% of non-null rows carrying the model's own view), and the one thing a reader of
        # the schema alone would otherwise get wrong.
        description="EXPERIMENTAL, shadow (retro#686) — what the ARTICLE says OTHERS expect "
                    "for the event; never your own view, never the byline author's (that is "
                    "author_lean). See CONSENSUS_VIEW; omit when the article does not say.")

    @model_validator(mode="before")
    @classmethod
    def _deserialize_string_predictions(cls, data: Any) -> Any:
        """
        Some models (in TOOLS mode) double-serialize nested objects as strings.
        Handles two observed variants:
          1. Valid JSON strings:   '{"quote": "...", ...}'
          2. YAML-style strings:   'quote: ... source_authority: 0.8'
        """
        data = _unwrap_properties_envelope(data)
        data = _drop_out_of_enum(data, "consensus_view", _CONSENSUS_VIEW_VALUES)
        if isinstance(data, dict) and "predictions" in data:
            preds = data["predictions"]
            if not isinstance(preds, list):
                return data
            parsed = []
            for p in preds:
                if not isinstance(p, str):
                    parsed.append(p)
                    continue
                # Try JSON first
                try:
                    parsed.append(_json.loads(p))
                    continue
                except _json.JSONDecodeError:
                    pass
                # Try YAML-style "key: value" lines
                try:
                    import yaml as _yaml
                    obj = _yaml.safe_load(p)
                    if isinstance(obj, dict):
                        parsed.append(obj)
                        continue
                except Exception:
                    pass
                # Give up — keep as-is and let Pydantic report the error
                parsed.append(p)
            data["predictions"] = parsed
        return data


class CellSignal(BaseModel):
    """
    Aggregated signal for one (event, source) cell.
    Computed from all predictions across all articles for the cell.
    Continuous metrics are weighted mean (weight = certainty × specificity when available).
    Categorical fields use majority vote. Median for time_horizon_days.
    Optional fields are None when all contributing predictions lacked that field.
    """
    claim_count:      int
    stance:           float
    certainty:        float
    sentiment:        Optional[float]
    specificity:      Optional[float]
    hedge_ratio:      Optional[float]
    conditionality:   Optional[float]
    magnitude:        Optional[float]
    source_authority: Optional[float]
    time_horizon:     Optional[str]
    time_horizon_days: Optional[int]
    prediction_type:  Optional[str]
    quotes:           list[str]
    claims:           list[str]


# --- Matrix Progress Tracking ---

class CellStatus(str, Enum):
    pending = "pending"
    in_progress = "in_progress"
    done = "done"
    failed = "failed"
    no_predictions = "no_predictions"


# Visual representation per status
CELL_CHAR: dict[CellStatus, str] = {
    CellStatus.pending: "░",
    CellStatus.in_progress: "▒",
    CellStatus.done: "▓",
    CellStatus.failed: "✗",
    CellStatus.no_predictions: "·",
}

CELL_COLOR: dict[CellStatus, str] = {
    CellStatus.pending: "white",
    CellStatus.in_progress: "yellow",
    CellStatus.done: "green",
    CellStatus.failed: "red",
    CellStatus.no_predictions: "bright_black",
}


class MatrixCell(BaseModel):
    event_id: str
    source_id: str
    status: CellStatus = CellStatus.pending
    prediction_count: int = 0
    error: Optional[str] = None


class MatrixState(BaseModel):
    cells: dict[str, MatrixCell] = {}  # key: "event_id:source_id"
    last_updated: Optional[str] = None

    def key(self, event_id: str, source_id: str) -> str:
        return f"{event_id}:{source_id}"

    def get(self, event_id: str, source_id: str) -> MatrixCell:
        k = self.key(event_id, source_id)
        if k not in self.cells:
            self.cells[k] = MatrixCell(event_id=event_id, source_id=source_id)
        return self.cells[k]

    def set_status(self, event_id: str, source_id: str, status: CellStatus, **kwargs) -> None:
        cell = self.get(event_id, source_id)
        cell.status = status
        for k, v in kwargs.items():
            setattr(cell, k, v)

    def stats(self) -> dict[str, int]:
        counts: dict[str, int] = {s.value: 0 for s in CellStatus}
        for cell in self.cells.values():
            counts[cell.status.value] += 1
        return counts


# --- Negative-result extraction markers (Daatan/docs#57 item 2) ---
#
# vault2/extractions/{hash}_{event}_{v}.json historically only existed for
# articles that PASSED the gatekeeper — a gate-rejected article left no trace,
# so the batch loop (which runs with --retry-empty every 5 minutes) re-ran the
# gatekeeper LLM on the same rejected articles every cycle. The orchestrator now
# writes a marker file for those negative outcomes, same shape as a positive
# extraction but with `"status"` set and `"extraction": null`. Every consumer
# that globs the extractions dir must filter markers out via is_negative_marker.
# Positive files written after this change carry `"status": "done"`; positive
# files written before it have no `status` field at all — both are non-markers.

NEGATIVE_MARKER_STATUSES = frozenset({"gate_rejected", "no_predictions"})


def is_negative_marker(ext_data: Any) -> bool:
    """True when a vault2/extractions JSON payload is a negative-result marker
    (gatekeeper rejected the article, or extraction yielded nothing) rather
    than a real extraction. Old positive files without a `status` field and
    new ones with `"status": "done"` both return False."""
    return isinstance(ext_data, dict) and ext_data.get("status") in NEGATIVE_MARKER_STATUSES
