import json as _json
from pydantic import BaseModel, Field, model_validator
from typing import Literal, Optional, Any
from enum import Enum


# --- LLM Output Schemas ---

def _unwrap_properties_envelope(data: Any) -> Any:
    """Nova Lite (MD_JSON mode) intermittently wraps its structured output in a
    spurious top-level {"properties": {...}} envelope instead of returning the
    flat schema fields directly, which fails Pydantic validation outright. Unwrap
    it before field validation runs. See retro#306."""
    if isinstance(data, dict) and data.keys() == {"properties"} and isinstance(data["properties"], dict):
        return data["properties"]
    return data


class GatekeeperOutput(BaseModel):
    is_prediction: bool
    reason: str
    prediction_count_estimate: int = Field(default=0, ge=0)
    relevance_score: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="Graded topic relevance of the article to the event [0,1]; "
                    "its square multiplies the source's aggregation weight. "
                    "Defaults to 1.0 (neutral) when a caller/model omits it.",
    )

    @model_validator(mode="before")
    @classmethod
    def _unwrap_envelope(cls, data: Any) -> Any:
        return _unwrap_properties_envelope(data)


class PredictionType(str, Enum):
    binary = "binary"
    continuous = "continuous"
    range = "range"
    trend = "trend"


class PredictionExtraction(BaseModel):
    quote: str = Field(description="Exact sentence(s) from the article containing the prediction")
    claim: str = Field(description="One-sentence neutral summary in English")
    stance: float = Field(ge=-1.0, le=1.0, description="Directional outlook: -1=event won't happen, +1=event will happen")
    certainty: float = Field(ge=0.0, le=1.0, description="Linguistic confidence: 0=very hedged, 1=absolute")
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
        description="The kind of evidence this claim is, independent of stance/certainty "
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

    @model_validator(mode="before")
    @classmethod
    def _unwrap_envelope(cls, data: Any) -> Any:
        return _unwrap_properties_envelope(data)


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
