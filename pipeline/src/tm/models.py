import json as _json
from pydantic import BaseModel, Field, model_validator
from typing import Literal, Optional, Any
from enum import Enum


# --- LLM Output Schemas ---

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
    settled: Optional[bool] = Field(default=None, description="True when the source reports the outcome as an accomplished fact (event occurred, or became permanently impossible) — not a prediction, however confident. A POSITIVE settlement (event occurred) must be accompanied by event_date; one without a parseable event_date, or dated after the article itself, is demoted to ordinary evidence in code (enforce_settlement_event_date). A negative settlement (became impossible) carries no event_date — the event never occurred")
    quantitative_estimate: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="An explicit modeled probability, poll number, or market price the "
                    "source cites FOR THE RELATED EVENT ITSELF (not a proxy stage), as a "
                    "probability in [0,1]. Null when the source has no such cited figure — "
                    "qualitative 'favorite'/'front-runner' framing without a stated number "
                    "does not count.",
    )
    evidence_class: Optional[Literal[
        "reported_fact", "cited_probability", "cited_share", "reporting", "opinion",
    ]] = Field(
        default=None,
        description="EXPERIMENTAL, shadow-classification only (S2, retro "
                    "docs/ORACLE_VARIABLES.md §5) — not yet used to weight anything. "
                    "The kind of evidence this claim is, independent of stance/certainty. "
                    "Omit entirely rather than guessing when none fits cleanly.",
    )
    event_date: Optional[str] = Field(
        default=None,
        description="ISO date (YYYY-MM-DD) on which the article says the RELATED EVENT "
                    "itself occurs or occurred. Resolve relative references ('on Friday', "
                    "'tomorrow', 'next week') against the article's date to an absolute "
                    "calendar date. The date of the event in the claim — NOT of any adjacent "
                    "or downstream event. Omit entirely when the article states no date for it.",
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


class ExtractionOutput(BaseModel):
    predictions: list[PredictionExtraction]

    @model_validator(mode="before")
    @classmethod
    def _deserialize_string_predictions(cls, data: Any) -> Any:
        """
        Some models (in TOOLS mode) double-serialize nested objects as strings.
        Handles two observed variants:
          1. Valid JSON strings:   '{"quote": "...", ...}'
          2. YAML-style strings:   'quote: ... source_authority: 0.8'
        """
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
