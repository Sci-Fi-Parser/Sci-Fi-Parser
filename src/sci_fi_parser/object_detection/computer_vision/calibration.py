"""Robust linear y-axis calibration with explicit abstention diagnostics."""

from __future__ import annotations

import math

import numpy as np

from ..models import (
    CalibrationLabel,
    ExtrapolationLimits,
    LeaveOneOutDiagnostic,
    OcrOutput,
    RoleInferenceResult,
    YCalibrationResult,
)


def _fit(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float]:
    slope, intercept = np.polyfit(xs, ys, 1)
    return float(slope), float(intercept)


def _labels(token_ids, tokens, values, slope: float, intercept: float) -> list[CalibrationLabel]:
    result = []
    for token_id in token_ids:
        token = tokens[token_id]
        value = values[token_id].value
        residual_value = value - (slope * token.box.center_y + intercept)
        residual_pixels = residual_value / abs(slope) if slope else math.inf
        result.append(
            CalibrationLabel(
                token_id=token_id,
                text=token.original_text,
                value=value,
                pixel_y=token.box.center_y,
                residual_value=residual_value,
                residual_pixels=residual_pixels,
            )
        )
    return result


def calibrate_y_axis(
    ocr: OcrOutput,
    roles: RoleInferenceResult,
    image_height: int,
) -> YCalibrationResult:
    """Fit value = slope * pixel_y + intercept from the selected y-label column."""

    selected = next(
        (
            candidate
            for candidate in roles.y_candidates
            if candidate.candidate_id == roles.selected_y_candidate_id
        ),
        None,
    )
    if selected is None:
        return YCalibrationResult(failure_reason=roles.abstention_reason or "no selected y-label column")

    token_by_id = {token.token_id: token for token in ocr.tokens}
    ids = [token_id for token_id in selected.token_ids if token_id in selected.parsed_values]
    if len(ids) < 2:
        return YCalibrationResult(failure_reason="at least two numeric y labels are required")
    xs = np.asarray([token_by_id[token_id].box.center_y for token_id in ids], dtype=float)
    values = np.asarray([selected.parsed_values[token_id].value for token_id in ids], dtype=float)
    if len(set(xs.tolist())) < 2 or len(set(values.tolist())) < 2:
        return YCalibrationResult(failure_reason="y labels do not contain two distinct positions and values")

    median_height = float(np.median([token_by_id[token_id].box.height for token_id in ids]))
    threshold_pixels = max(2.0, median_height * 0.75, image_height * 0.015)
    models: list[tuple[int, float, np.ndarray, float, float]] = []
    for first in range(len(ids) - 1):
        for second in range(first + 1, len(ids)):
            if xs[first] == xs[second]:
                continue
            slope = float((values[second] - values[first]) / (xs[second] - xs[first]))
            intercept = float(values[first] - slope * xs[first])
            if slope >= 0:
                continue
            residual_pixels = np.abs(values - (slope * xs + intercept)) / abs(slope)
            inliers = residual_pixels <= threshold_pixels
            models.append(
                (int(np.sum(inliers)), float(np.mean(residual_pixels[inliers])), inliers, slope, intercept)
            )

    if not models:
        return YCalibrationResult(
            inlier_threshold_pixels=threshold_pixels,
            failure_reason="all y-label pairs imply a non-negative slope",
        )
    _, _, inlier_mask, _, _ = max(models, key=lambda model: (model[0], -model[1]))
    minimum_inliers = 2 if len(ids) <= 3 else math.ceil(len(ids) * 0.67)
    if int(np.sum(inlier_mask)) < minimum_inliers:
        return YCalibrationResult(
            inlier_threshold_pixels=threshold_pixels,
            failure_reason="numeric labels disagree materially with a linear scale",
        )
    if len(ids) == 3 and int(np.sum(inlier_mask)) == 2:
        inlier_confidence = [
            token_by_id[token_id].confidence for token_id, keep in zip(ids, inlier_mask, strict=True) if keep
        ]
        outlier_confidence = next(
            token_by_id[token_id].confidence
            for token_id, keep in zip(ids, inlier_mask, strict=True)
            if not keep
        )
        if outlier_confidence >= min(inlier_confidence) - 0.10:
            return YCalibrationResult(
                inlier_threshold_pixels=threshold_pixels,
                failure_reason="three high-confidence labels do not agree on a linear scale",
            )

    slope, intercept = _fit(xs[inlier_mask], values[inlier_mask])
    if slope >= 0:
        return YCalibrationResult(
            inlier_threshold_pixels=threshold_pixels,
            failure_reason="fitted calibration has a non-negative slope",
        )
    residual_values = values - (slope * xs + intercept)
    residual_pixels = residual_values / abs(slope)
    inlier_ids = [token_id for token_id, keep in zip(ids, inlier_mask, strict=True) if keep]
    outlier_ids = [token_id for token_id, keep in zip(ids, inlier_mask, strict=True) if not keep]

    leave_one_out = []
    for index, token_id in enumerate(ids):
        keep = np.arange(len(ids)) != index
        if int(np.sum(keep)) < 2 or len(set(xs[keep].tolist())) < 2:
            leave_one_out.append(LeaveOneOutDiagnostic(token_id=token_id, fit_available=False))
            continue
        loo_slope, loo_intercept = _fit(xs[keep], values[keep])
        predicted = loo_slope * xs[index] + loo_intercept
        residual = values[index] - predicted
        leave_one_out.append(
            LeaveOneOutDiagnostic(
                token_id=token_id,
                fit_available=True,
                predicted_value=float(predicted),
                residual_value=float(residual),
                residual_pixels=float(residual / abs(loo_slope)) if loo_slope else None,
            )
        )

    inlier_xs = xs[inlier_mask]
    inlier_values = values[inlier_mask]
    pixel_span = float(np.ptp(inlier_xs))
    pixel_min = max(0.0, float(np.min(inlier_xs) - 0.25 * pixel_span))
    pixel_max = min(float(image_height), float(np.max(inlier_xs) + 0.25 * pixel_span))
    inlier_residual_pixels = np.abs(residual_pixels[inlier_mask])
    residual_score = math.exp(-float(np.mean(inlier_residual_pixels)) / max(threshold_pixels, 1.0))
    ordered_values = inlier_values[np.argsort(inlier_xs)]
    monotonicity = float(np.mean(np.diff(ordered_values) < 0)) if len(ordered_values) > 1 else 0.0
    confidence = (
        0.25 * min(1.0, len(inlier_ids) / 5.0)
        + 0.30 * min(1.0, pixel_span / max(image_height * 0.35, 1.0))
        + 0.25 * monotonicity
        + 0.20 * residual_score
    )
    return YCalibrationResult(
        succeeded=True,
        slope=slope,
        intercept=intercept,
        inliers=_labels(inlier_ids, token_by_id, selected.parsed_values, slope, intercept),
        outliers=_labels(outlier_ids, token_by_id, selected.parsed_values, slope, intercept),
        visible_labeled_range=(float(np.min(inlier_values)), float(np.max(inlier_values))),
        pixel_span=pixel_span,
        inlier_threshold_pixels=threshold_pixels,
        residual_value_mae=float(np.mean(np.abs(residual_values[inlier_mask]))),
        residual_pixel_mae=float(np.mean(inlier_residual_pixels)),
        residual_pixel_max=float(np.max(inlier_residual_pixels)),
        leave_one_out=leave_one_out,
        extrapolation_limits=ExtrapolationLimits(
            pixel_y_min=pixel_min,
            pixel_y_max=pixel_max,
            value_at_pixel_y_min=slope * pixel_min + intercept,
            value_at_pixel_y_max=slope * pixel_max + intercept,
        ),
        confidence=max(0.0, min(1.0, confidence)),
    )
