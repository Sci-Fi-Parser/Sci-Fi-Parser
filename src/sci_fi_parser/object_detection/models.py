"""JSON-serializable models shared by the OCR/CV component stages."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StageModel(BaseModel):
    """Base model with strict assignment and JSON-friendly serialization."""

    model_config = ConfigDict(validate_assignment=True)


class OcrBox(StageModel):
    """Canonical top-left-origin OCR box."""

    left: float
    top: float
    right: float
    bottom: float

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.bottom - self.top

    @property
    def center_x(self) -> float:
        return (self.left + self.right) / 2.0

    @property
    def center_y(self) -> float:
        return (self.top + self.bottom) / 2.0


class ParsedNumber(StageModel):
    """Numeric interpretation kept separate from the OCR text."""

    value: float
    parsed_text: str
    is_percentage: bool = False
    magnitude_suffix: Literal["K", "M", "B"] | None = None
    correction_applied: bool = False


class OcrToken(StageModel):
    """One canonical OCR observation."""

    token_id: str
    original_text: str
    normalized_text: str
    confidence: float = Field(ge=0.0, le=1.0)
    box: OcrBox
    parsed_number: ParsedNumber | None = None


class OcrOutput(StageModel):
    """Canonical output produced immediately after PaddleOCR."""

    schema_version: Literal["ocr-token-v1"] = "ocr-token-v1"
    tokens: list[OcrToken] = Field(default_factory=list)


AxisKind = Literal["y_column", "x_band"]
TokenRole = Literal["y_tick", "x_label", "other"]
ChartType = Literal["bar_chart", "line_chart"]


class AxisCandidate(StageModel):
    """Scored row or column hypothesis retained for diagnosis."""

    candidate_id: str
    kind: AxisKind
    token_ids: list[str]
    score: float = Field(ge=0.0, le=1.0)
    score_components: dict[str, float]
    bounds: OcrBox
    parsed_values: dict[str, ParsedNumber] = Field(default_factory=dict)
    rejection_reasons: list[str] = Field(default_factory=list)


class RoleAssignment(StageModel):
    token_id: str
    role: TokenRole
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


class AxisPairDiagnostic(StageModel):
    """Ownership resolution and scoring evidence for one x/y pair."""

    y_candidate_id: str
    x_candidate_id: str
    accepted: bool
    shared_token_ids: list[str] = Field(default_factory=list)
    effective_x_token_ids: list[str] = Field(default_factory=list)
    pair_score: float | None = None
    rejection_reason: str | None = None


class RoleInferenceResult(StageModel):
    """Joint axis-label inference, including alternatives and abstention."""

    y_candidates: list[AxisCandidate] = Field(default_factory=list)
    x_candidates: list[AxisCandidate] = Field(default_factory=list)
    selected_y_candidate_id: str | None = None
    selected_x_candidate_id: str | None = None
    assignments: list[RoleAssignment] = Field(default_factory=list)
    pair_diagnostics: list[AxisPairDiagnostic] = Field(default_factory=list)
    thresholds: dict[str, float] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0)
    abstention_reason: str | None = None
    x_abstention_reason: str | None = None
    y_abstention_reason: str | None = None
    winner_runner_up_margin: float | None = None


class CalibrationLabel(StageModel):
    token_id: str
    text: str
    value: float
    pixel_y: float
    residual_value: float
    residual_pixels: float


class LeaveOneOutDiagnostic(StageModel):
    token_id: str
    fit_available: bool
    predicted_value: float | None = None
    residual_value: float | None = None
    residual_pixels: float | None = None


class ExtrapolationLimits(StageModel):
    pixel_y_min: float
    pixel_y_max: float
    value_at_pixel_y_min: float
    value_at_pixel_y_max: float


class YCalibrationResult(StageModel):
    """Linear value = slope * pixel_y + intercept calibration evidence."""

    succeeded: bool = False
    slope: float | None = None
    intercept: float | None = None
    inliers: list[CalibrationLabel] = Field(default_factory=list)
    outliers: list[CalibrationLabel] = Field(default_factory=list)
    visible_labeled_range: tuple[float, float] | None = None
    pixel_span: float = 0.0
    inlier_threshold_pixels: float | None = None
    residual_value_mae: float | None = None
    residual_pixel_mae: float | None = None
    residual_pixel_max: float | None = None
    leave_one_out: list[LeaveOneOutDiagnostic] = Field(default_factory=list)
    extrapolation_limits: ExtrapolationLimits | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    failure_reason: str | None = None


class ElementEvidence(StageModel):
    """Serializable geometry that can help anchor chart structure."""

    element_id: str
    kind: Literal["bar"]
    box: OcrBox


class OcrCvStageResult(StageModel):
    """Chart-neutral OCR, element, role, and calibration evidence."""

    schema_version: Literal["ocr-cv-stage-v2"] = "ocr-cv-stage-v2"
    chart_type: ChartType
    image_width: int
    image_height: int
    ocr: OcrOutput
    initial_elements: list[ElementEvidence] = Field(default_factory=list)
    roles: RoleInferenceResult
    calibration: YCalibrationResult
    stage_runtime_ms: dict[str, float] = Field(default_factory=dict)
    unsupported_scope: list[str] = Field(
        default_factory=lambda: [
            "line-series extraction",
            "horizontal bars",
            "grouped bars",
            "stacked bars",
            "log scales",
        ]
    )
