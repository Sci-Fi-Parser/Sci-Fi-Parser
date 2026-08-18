"""Joint geometric inference of OCR x-label and y-label roles."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from ..models import (
    AxisCandidate,
    AxisPairDiagnostic,
    OcrBox,
    OcrOutput,
    ParsedNumber,
    RoleAssignment,
    RoleInferenceResult,
)
from ..normalization import parse_numeric_token


def _bounds(tokens) -> OcrBox:
    return OcrBox(
        left=min(token.box.left for token in tokens),
        top=min(token.box.top for token in tokens),
        right=max(token.box.right for token in tokens),
        bottom=max(token.box.bottom for token in tokens),
    )


def _candidate_groups(items, coordinate, tolerance: float, minimum: int) -> list[list]:
    groups: dict[tuple[str, ...], list] = {}
    for seed in items:
        selected = [item for item in items if abs(coordinate(item) - coordinate(seed)) <= tolerance]
        if len(selected) >= minimum:
            groups[tuple(sorted(item.token_id for item in selected))] = selected
    return list(groups.values())


def _element_box(element):
    if hasattr(element, "bbox"):
        return element.bbox
    return element.box if hasattr(element, "box") else element


def _axis_multiplier(tokens, all_tokens, values: dict[str, ParsedNumber], width: int) -> float | None:
    bounds = _bounds(tokens)
    token_ids = {token.token_id for token in tokens}
    upper_label_center = float(np.quantile([token.box.center_y for token in tokens], 0.30))
    if max(abs(parsed.value) for parsed in values.values()) > 100:
        return None
    options = []
    for token in all_tokens:
        if token.token_id in token_ids:
            continue
        parsed = token.parsed_number or parse_numeric_token(token.normalized_text, cautious_corrections=True)
        if parsed is None or parsed.value < 100:
            continue
        explicit = "e" in token.normalized_text.lower() or parsed.magnitude_suffix is not None
        horizontally_near = token.box.left <= bounds.right + width * 0.15
        vertically_near_top = token.box.center_y <= upper_label_center
        if explicit and horizontally_near and vertically_near_top:
            options.append(parsed.value)
    return max(options) if options else None


def _score_y_candidate(
    tokens,
    values: dict[str, ParsedNumber],
    width: int,
    height: int,
    element_anchors,
    all_tokens,
) -> AxisCandidate:
    tokens = list(tokens)
    for token in tokens:
        parsed = values[token.token_id]
        explicit_multiplier = "e" in token.normalized_text.lower() or parsed.magnitude_suffix is not None
        others = [other for other in tokens if other.token_id != token.token_id]
        if (
            explicit_multiplier
            and parsed.value >= 100
            and len(others) >= 2
            and token.box.center_y <= min(other.box.center_y for other in others)
            and token.box.left >= max(other.box.right for other in others)
        ):
            tokens = others
            values = {token_id: value for token_id, value in values.items() if token_id != token.token_id}
            break
    multiplier = _axis_multiplier(tokens, all_tokens, values, width)
    if multiplier is not None:
        values = {
            token_id: parsed.model_copy(
                update={
                    "value": parsed.value * multiplier,
                    "parsed_text": f"{parsed.parsed_text} * {multiplier:g}",
                }
            )
            for token_id, parsed in values.items()
        }
    ordered = sorted(tokens, key=lambda token: token.box.center_y)
    ys = np.asarray([token.box.center_y for token in ordered], dtype=float)
    vals = np.asarray([values[token.token_id].value for token in ordered], dtype=float)
    span = float(np.ptp(ys))
    if span < 1e-6:
        slope = 0.0
        residual_px = math.inf
    else:
        slope, intercept = np.polyfit(ys, vals, 1)
        predicted = slope * ys + intercept
        residual_px = float(np.mean(np.abs(vals - predicted)) / max(abs(float(slope)), 1e-12))
    adjacent = np.diff(vals)
    monotonicity = float(np.mean(adjacent < 0)) if adjacent.size else 0.0
    median_height = float(np.median([token.box.height for token in tokens]))
    align_tolerance = max(6.0, width * 0.018, median_height * 1.5)
    right_alignment = math.exp(-float(np.std([token.box.right for token in tokens])) / align_tolerance)
    center_alignment = math.exp(-float(np.std([token.box.center_x for token in tokens])) / align_tolerance)
    alignment = max(right_alignment, center_alignment)
    span_score = min(1.0, span / max(height * 0.35, 1.0))
    linearity = math.exp(-residual_px / max(median_height, height * 0.015, 1.0))
    unique_score = min(1.0, len(set(vals.tolist())) / 4.0)
    differences = np.abs(np.diff(vals))
    regularity = 1.0
    if len(differences) > 1 and float(np.mean(differences)) > 0:
        regularity = 1.0 / (1.0 + float(np.std(differences) / np.mean(differences)))

    if element_anchors:
        boundary = max(token.box.right for token in tokens)
        elements_on_right = float(
            np.mean([_element_box(element).center_x > boundary for element in element_anchors])
        )
    else:
        elements_on_right = max(0.0, min(1.0, 1.0 - _bounds(tokens).right / max(width * 0.55, 1.0)))

    components = {
        "alignment": alignment,
        "vertical_span": span_score,
        "monotonicity": monotonicity,
        "linear_fit": linearity,
        "tick_regularity": regularity,
        "elements_on_right": elements_on_right,
        "distinct_values": unique_score,
    }
    if multiplier is not None:
        components["axis_multiplier"] = multiplier
    score = (
        0.17 * alignment
        + 0.18 * span_score
        + 0.20 * monotonicity
        + 0.18 * linearity
        + 0.07 * regularity
        + 0.15 * elements_on_right
        + 0.05 * unique_score
    )
    reasons = []
    if slope >= 0:
        reasons.append("numeric values do not decrease down the image")
        score *= 0.2
    if span < height * 0.12:
        reasons.append("numeric labels have insufficient vertical span")
    if len(set(vals.tolist())) < 2:
        reasons.append("fewer than two distinct numeric values")
        score = 0.0
    return AxisCandidate(
        candidate_id="",
        kind="y_column",
        token_ids=[token.token_id for token in ordered],
        score=max(0.0, min(1.0, score)),
        score_components=components,
        bounds=_bounds(tokens),
        parsed_values=values,
        rejection_reasons=reasons,
    )


def _score_x_candidate(tokens, width: int, height: int, element_anchors) -> AxisCandidate:
    centers = [token.box.center_x for token in tokens]
    horizontal_span = min(1.0, float(np.ptp(centers)) / max(width * 0.45, 1.0)) if len(tokens) > 1 else 0.15
    y_center = float(np.median([token.box.center_y for token in tokens]))
    lower_position = max(0.0, min(1.0, (y_center / max(height, 1) - 0.45) / 0.45))
    if element_anchors:
        bottoms = np.asarray([_element_box(element).bottom for element in element_anchors], dtype=float)
        below_elements = float(np.mean([token.box.center_y >= float(np.median(bottoms)) for token in tokens]))
        element_centers = np.asarray(
            [_element_box(element).center_x for element in element_anchors], dtype=float
        )
        nearest = [float(np.min(np.abs(element_centers - center))) for center in centers]
        anchor_match = math.exp(-float(np.mean(nearest)) / max(width * 0.08, 1.0))
    else:
        below_elements = lower_position
        anchor_match = min(1.0, len(tokens) / 4.0)
    count_score = min(1.0, len(tokens) / 4.0)
    score = (
        0.30 * horizontal_span
        + 0.25 * lower_position
        + 0.25 * below_elements
        + 0.15 * anchor_match
        + 0.05 * count_score
    )
    reasons = []
    if y_center < height * 0.45:
        reasons.append("label band is too high in the image")
    if horizontal_span < 0.15 and len(tokens) > 1:
        reasons.append("label band has insufficient horizontal span")
    return AxisCandidate(
        candidate_id="",
        kind="x_band",
        token_ids=[token.token_id for token in sorted(tokens, key=lambda token: token.box.center_x)],
        score=max(0.0, min(1.0, score)),
        score_components={
            "horizontal_span": horizontal_span,
            "lower_position": lower_position,
            "below_elements": below_elements,
            "element_anchor_match": anchor_match,
            "label_count": count_score,
        },
        bounds=_bounds(tokens),
        rejection_reasons=reasons,
    )


def infer_axis_roles(
    ocr: OcrOutput,
    image_size: tuple[int, int],
    element_anchors: Sequence = (),
) -> RoleInferenceResult:
    """Evaluate numeric columns and label bands together, retaining alternatives."""

    width, height = image_size
    tokens = ocr.tokens
    if not tokens:
        reason = "OCR returned no tokens"
        return RoleInferenceResult(
            confidence=0.0,
            abstention_reason=reason,
            x_abstention_reason=reason,
            y_abstention_reason=reason,
        )

    median_height = float(np.median([max(token.box.height, 1.0) for token in tokens]))
    y_tolerance = max(6.0, width * 0.018, median_height * 1.5)
    numeric = []
    parsed_by_id = {}
    for token in tokens:
        parsed = token.parsed_number or parse_numeric_token(token.normalized_text, cautious_corrections=True)
        if parsed is not None:
            numeric.append(token)
            parsed_by_id[token.token_id] = parsed

    y_groups = _candidate_groups(numeric, lambda token: token.box.right, y_tolerance, 2)
    y_groups += _candidate_groups(numeric, lambda token: token.box.center_x, y_tolerance, 2)
    unique_y_groups = {tuple(sorted(token.token_id for token in group)): group for group in y_groups}.values()
    y_candidates = [
        _score_y_candidate(
            group,
            {token.token_id: parsed_by_id[token.token_id] for token in group},
            width,
            height,
            element_anchors,
            tokens,
        )
        for group in unique_y_groups
    ]
    y_candidates.sort(key=lambda candidate: candidate.score, reverse=True)
    for index, candidate in enumerate(y_candidates):
        candidate.candidate_id = f"y-{index}"

    row_tolerance = max(5.0, height * 0.018, median_height * 0.75)
    x_groups = _candidate_groups(tokens, lambda token: token.box.center_y, row_tolerance, 1)
    x_candidates = [_score_x_candidate(group, width, height, element_anchors) for group in x_groups]
    x_candidates.sort(key=lambda candidate: candidate.score, reverse=True)
    for index, candidate in enumerate(x_candidates):
        candidate.candidate_id = f"x-{index}"

    token_by_id = {token.token_id: token for token in tokens}
    scored_pairs = []
    pair_diagnostics = []
    for y_candidate in y_candidates:
        y_ids = set(y_candidate.token_ids)
        for x_candidate in x_candidates:
            shared_ids = y_ids.intersection(x_candidate.token_ids)
            effective_x_ids = [token_id for token_id in x_candidate.token_ids if token_id not in y_ids]
            rejection_reason = None
            if len(shared_ids) > 1:
                rejection_reason = "x/y candidates share more than one token"
            elif shared_ids:
                shared_id = next(iter(shared_ids))
                bottommost_y = max(
                    y_candidate.token_ids,
                    key=lambda token_id: token_by_id[token_id].box.center_y,
                )
                if shared_id != bottommost_y:
                    rejection_reason = "shared token is not the bottommost y tick"
                elif not effective_x_ids:
                    rejection_reason = "shared y endpoint leaves no x-only evidence"
                elif not any(
                    token_by_id[token_id].box.center_x > y_candidate.bounds.right
                    for token_id in effective_x_ids
                ):
                    rejection_reason = "x-only evidence does not extend right of the y-label column"
            if rejection_reason is not None:
                pair_diagnostics.append(
                    AxisPairDiagnostic(
                        y_candidate_id=y_candidate.candidate_id,
                        x_candidate_id=x_candidate.candidate_id,
                        accepted=False,
                        shared_token_ids=sorted(shared_ids),
                        effective_x_token_ids=effective_x_ids,
                        rejection_reason=rejection_reason,
                    )
                )
                continue

            effective_x = _score_x_candidate(
                [token_by_id[token_id] for token_id in effective_x_ids],
                width,
                height,
                element_anchors,
            )
            effective_x.candidate_id = x_candidate.candidate_id
            vertical_order = 1.0 if effective_x.bounds.top >= y_candidate.bounds.top else 0.5
            side_order = 1.0 if y_candidate.bounds.right <= effective_x.bounds.right else 0.6
            pair_score = (
                0.55 * y_candidate.score
                + 0.35 * effective_x.score
                + 0.05 * vertical_order
                + 0.05 * side_order
            )
            pair_diagnostics.append(
                AxisPairDiagnostic(
                    y_candidate_id=y_candidate.candidate_id,
                    x_candidate_id=x_candidate.candidate_id,
                    accepted=True,
                    shared_token_ids=sorted(shared_ids),
                    effective_x_token_ids=effective_x.token_ids,
                    pair_score=pair_score,
                )
            )
            scored_pairs.append((pair_score, y_candidate, effective_x))

    scored_pairs.sort(key=lambda pair: pair[0], reverse=True)
    best_pair = scored_pairs[0] if scored_pairs else None
    winner_runner_up_margin = (
        best_pair[0] - scored_pairs[1][0] if best_pair is not None and len(scored_pairs) > 1 else None
    )

    selected_y = best_pair[1] if best_pair and best_pair[1].score >= 0.52 else None
    selected_x = best_pair[2] if best_pair and selected_y is not None and best_pair[2].score >= 0.35 else None
    if selected_y is None and y_candidates and y_candidates[0].score >= 0.52:
        selected_y = y_candidates[0]
    if selected_x is not None:
        x_candidates = [
            selected_x if candidate.candidate_id == selected_x.candidate_id else candidate
            for candidate in x_candidates
        ]

    selected_y_ids = set(selected_y.token_ids if selected_y else [])
    selected_x_ids = set(selected_x.token_ids if selected_x else [])
    assignments = []
    for token in tokens:
        if token.token_id in selected_y_ids:
            assert selected_y is not None
            assignments.append(
                RoleAssignment(
                    token_id=token.token_id,
                    role="y_tick",
                    confidence=selected_y.score,
                    reason=f"member of selected {selected_y.candidate_id}",
                )
            )
        elif token.token_id in selected_x_ids:
            assert selected_x is not None
            assignments.append(
                RoleAssignment(
                    token_id=token.token_id,
                    role="x_label",
                    confidence=selected_x.score,
                    reason=f"member of selected {selected_x.candidate_id}",
                )
            )
        else:
            assignments.append(
                RoleAssignment(token_id=token.token_id, role="other", confidence=0.5, reason="not selected")
            )

    confidence = 0.65 * selected_y.score if selected_y else 0.0
    if selected_x is not None:
        confidence += 0.35 * selected_x.score
    y_abstention = (
        None if selected_y is not None else "no numeric column passed the y-axis evidence threshold"
    )
    if selected_x is not None:
        x_abstention = None
    elif not scored_pairs:
        x_abstention = "no x/y pair passed endpoint ownership checks"
    else:
        x_abstention = "no x-label band passed the x-only evidence threshold"
    abstention = "; ".join(reason for reason in (y_abstention, x_abstention) if reason) or None
    return RoleInferenceResult(
        y_candidates=y_candidates,
        x_candidates=x_candidates,
        selected_y_candidate_id=selected_y.candidate_id if selected_y else None,
        selected_x_candidate_id=selected_x.candidate_id if selected_x else None,
        assignments=assignments,
        pair_diagnostics=pair_diagnostics,
        thresholds={
            "y_group_alignment_pixels": y_tolerance,
            "x_group_alignment_pixels": row_tolerance,
            "selected_y_score_min": 0.52,
            "selected_x_score_min": 0.35,
        },
        confidence=max(0.0, min(1.0, confidence)),
        abstention_reason=abstention,
        x_abstention_reason=x_abstention,
        y_abstention_reason=y_abstention,
        winner_runner_up_margin=winner_runner_up_margin,
    )
