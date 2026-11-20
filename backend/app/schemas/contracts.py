"""Public contracts shared by specialists and the future API."""
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Probability = Annotated[float, Field(ge=0, le=1)]
Box = tuple[float, float, float, float]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Confidence(Contract):
    value: Probability
    calibrated: bool = False
    note: str

    @model_validator(mode="after")
    def label_uncalibrated(self):
        if not self.calibrated and "uncalibrated" not in self.note.lower():
            raise ValueError("Uncalibrated confidence must be labelled uncalibrated")
        return self


class Metadata(Contract):
    model_name: str
    version: str
    runtime_ms: float = Field(ge=0)
    device: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    mode: Literal["stub", "real"]
    execution_type: Literal["live", "stub", "pre-computed"] | None = None
    demo_info: dict[str, Any] = Field(default_factory=dict)
    fallback_reason: str | None = None
    confidence_reason: str | None = None
    validation_status: str | None = None
    warnings: list[str] = Field(default_factory=list)


class GroundingOutput(Contract):
    overlay_path: str | None = None
    score_calibration: Literal["uncalibrated"] = "uncalibrated"
    boxes: list[Box]
    labels: list[str]
    scores: list[Probability]

    @model_validator(mode="after")
    def validate_boxes(self):
        if not len(self.boxes) == len(self.labels) == len(self.scores):
            raise ValueError("Boxes, labels, and scores must have matching lengths")
        for x1, y1, x2, y2 in self.boxes:
            if not (0 <= x1 < x2 and 0 <= y1 < y2):
                raise ValueError("Boxes must use ordered, nonnegative absolute pixels")
        return self


class FusionOutput(Contract):
    label: str
    confidence: Probability
    top3: list[tuple[str, Probability]] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def validate_ranking(self):
        if len({label for label, _ in self.top3}) != 3:
            raise ValueError("top3 labels must be distinct")
        if self.top3 != sorted(self.top3, key=lambda item: item[1], reverse=True):
            raise ValueError("top3 must be sorted by probability")
        if self.top3[0] != (self.label, self.confidence):
            raise ValueError("Prediction must match top-ranked class")
        return self


class VLMOutput(Contract):
    text: str


class ChangeOutput(Contract):
    caption: str


class StandardResult(Contract):
    answer: str
    evidence: GroundingOutput | FusionOutput | None = None
    confidence: Confidence | None = None
    metadata: Metadata

    @model_validator(mode="after")
    def explain_missing_confidence(self):
        if self.confidence is None and self.metadata.confidence_reason != "not available":
            raise ValueError('Missing confidence requires reason "not available"')
        return self


class VLMInput(Contract):
    image_path: str
    task: Literal["qa", "caption"] = "qa"


class GroundingInput(Contract):
    image_path: str
    text_prompt: str = Field(min_length=1)
    box_threshold: Probability = 0.3


class ChangeInput(Contract):
    before_path: str
    after_path: str


class FusionInput(Contract):
    optical_path: str
    sar_path: str


SpecialistInput = VLMInput | GroundingInput | ChangeInput | FusionInput


class QueryRequest(Contract):
    model_id: str
    inputs: SpecialistInput
    query: str = ""
    options: dict[str, Any] = Field(default_factory=dict)


class QueryResponse(Contract):
    request_id: str
    result: StandardResult
    trace_path: str | None = None
