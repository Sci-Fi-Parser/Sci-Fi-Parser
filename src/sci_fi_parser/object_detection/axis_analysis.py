"""Composition of OCR normalization evidence, role inference, and calibration."""

from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np

from .computer_vision.calibration import calibrate_y_axis
from .computer_vision.roles import infer_axis_roles
from .models import ChartType, ElementEvidence, OcrBox, OcrCvStageResult, OcrOutput


def _serialize_elements(initial_elements: Sequence) -> list[ElementEvidence]:
    evidence = []
    for index, element in enumerate(initial_elements):
        box = element.bbox if hasattr(element, "bbox") else element
        evidence.append(
            ElementEvidence(
                element_id=f"bar-{index}",
                kind="bar",
                box=OcrBox(
                    left=float(box.x),
                    top=float(box.y),
                    right=float(box.right),
                    bottom=float(box.bottom),
                ),
            )
        )
    return evidence


def analyze_ocr_cv(
    image: np.ndarray,
    ocr: OcrOutput,
    chart_type: ChartType,
    initial_elements: Sequence = (),
) -> OcrCvStageResult:
    """Build chart-neutral axis and calibration evidence without invoking a VLM."""

    height, width = image.shape[:2]
    started = time.perf_counter()
    role_started = time.perf_counter()
    roles = infer_axis_roles(ocr, (width, height), initial_elements)
    role_ms = (time.perf_counter() - role_started) * 1000.0
    calibration_started = time.perf_counter()
    calibration = calibrate_y_axis(ocr, roles, height)
    calibration_ms = (time.perf_counter() - calibration_started) * 1000.0
    return OcrCvStageResult(
        chart_type=chart_type,
        image_width=width,
        image_height=height,
        ocr=ocr,
        initial_elements=_serialize_elements(initial_elements),
        roles=roles,
        calibration=calibration,
        stage_runtime_ms={
            "role_inference": role_ms,
            "calibration": calibration_ms,
            "total_without_ocr": (time.perf_counter() - started) * 1000.0,
        },
    )
