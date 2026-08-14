"""Benchmark structural-line detection with synthetic truth or real charts."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2

from sci_fi_parser.object_detection.computer_vision.axis_candidates import (
    CandidateDetection,
    detect_vertical_candidates,
)
from sci_fi_parser.object_detection.computer_vision.config import CvConfig
from sci_fi_parser.object_detection.computer_vision.geometry import interval_overlap
from sci_fi_parser.object_detection.computer_vision.lines import detect_directional_lines, detect_merged_lines
from sci_fi_parser.schema import ImageSet

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
LINE_CONFIG_FIELDS = (
    "canny_threshold1",
    "canny_threshold2",
    "hough_threshold",
    "max_line_gap",
    "angle_tolerance_degrees",
    "line_min_length_ratio",
    "line_min_length_pixels",
    "line_merge_angle_tolerance_degrees",
    "line_collinearity_tolerance_pixels",
    "line_along_gap_tolerance_pixels",
    "directional_line_min_length_ratio",
    "directional_line_min_length_pixels",
)
Detector = Callable[[Any], list[Any]]
MINIMUM_TARGET_COVERAGE = 0.5
COORDINATE_TOLERANCE_RATIO = 0.005
COORDINATE_TOLERANCE_PIXELS = 2.0


@dataclass(slots=True)
class LineTarget:
    kind: str
    orientation: str
    p1: tuple[float, float]
    p2: tuple[float, float]


def _image_paths(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path] if input_path.suffix.lower() in IMAGE_SUFFIXES else []
    if input_path.is_dir():
        return sorted(
            path for path in input_path.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
    raise FileNotFoundError(input_path)


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _counts(detection: CandidateDetection) -> dict[str, int]:
    return {
        "hough_horizontal": sum(line.orientation == "horizontal" for line in detection.hough_lines),
        "hough_vertical": len(detection.hough_vertical),
        "morphology_horizontal": sum(line.orientation == "horizontal" for line in detection.morphology_lines),
        "morphology_vertical": len(detection.morphology_vertical),
        "fused_vertical": len(detection.candidates),
        "fused_from_both": sum(candidate.sources == "both" for candidate in detection.candidates),
    }


def _line_coordinates(line: Any) -> tuple[float, float, float]:
    if line.orientation == "horizontal":
        return (line.p1[1] + line.p2[1]) / 2, *sorted((line.p1[0], line.p2[0]))
    return (line.p1[0] + line.p2[0]) / 2, *sorted((line.p1[1], line.p2[1]))


def _targets(geometry: object) -> list[LineTarget] | None:
    if not isinstance(geometry, dict) or "y_axis" not in geometry:
        return None
    targets = []
    axis = geometry["y_axis"]
    if axis.get("visible", False):
        targets.append(LineTarget("y_axis", "vertical", tuple(axis["p1_px"]), tuple(axis["p2_px"])))
    for line in geometry.get("horizontal_gridlines", []):
        targets.append(LineTarget("gridline", "horizontal", tuple(line["p1_px"]), tuple(line["p2_px"])))
    return targets


def _score(lines: list[Any], targets: list[LineTarget], shape: tuple[int, ...]) -> dict[str, Any]:
    possible = []
    for target_index, target in enumerate(targets):
        target_cross, target_start, target_end = _line_coordinates(target)
        target_length = target_end - target_start
        dimension = shape[0] if target.orientation == "horizontal" else shape[1]
        tolerance = max(COORDINATE_TOLERANCE_PIXELS, dimension * COORDINATE_TOLERANCE_RATIO)
        for candidate_index, line in enumerate(lines):
            if line.orientation != target.orientation or target_length <= 0:
                continue
            cross, start, end = _line_coordinates(line)
            coverage = interval_overlap(start, end, target_start, target_end) / target_length
            error = abs(cross - target_cross)
            if error <= tolerance and coverage >= MINIMUM_TARGET_COVERAGE:
                possible.append((error, -coverage, target_index, candidate_index))
    matched_targets: set[int] = set()
    matched_candidates: set[int] = set()
    matches = []
    for error, negative_coverage, target_index, candidate_index in sorted(possible):
        if target_index in matched_targets or candidate_index in matched_candidates:
            continue
        matched_targets.add(target_index)
        matched_candidates.add(candidate_index)
        matches.append(
            {
                "kind": targets[target_index].kind,
                "coordinate_error_px": error,
                "coverage": -negative_coverage,
                "candidate_index": candidate_index,
            }
        )
    return {
        "target_count": len(targets),
        "matched_count": len(matches),
        "recall": len(matches) / len(targets) if targets else None,
        "matches": matches,
        "by_kind": {
            kind: {
                "target_count": sum(target.kind == kind for target in targets),
                "matched_count": sum(match["kind"] == kind for match in matches),
            }
            for kind in ("y_axis", "gridline")
        },
    }


def _draw_evidence(
    image: Any,
    lines: list[Any],
    targets: list[LineTarget],
    score: dict[str, Any],
) -> Any:
    if image.ndim == 2:
        overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        overlay = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    else:
        overlay = image.copy()
    for target in targets:
        cv2.line(overlay, tuple(map(round, target.p1)), tuple(map(round, target.p2)), (255, 255, 0), 3)
    matched = {match["candidate_index"] for match in score["matches"]}
    for index, line in enumerate(lines):
        cv2.line(overlay, line.p1, line.p2, (0, 190, 0) if index in matched else (0, 120, 255), 2)
    return overlay


def run_line_benchmark(
    image_set: ImageSet,
    truth_by_image_id: dict[str, Any],
    output_path: Path,
    evidence_dir: Path | None = None,
) -> dict[str, Any]:
    """Score all line detector variants against structural geometry."""
    detectors: dict[str, Detector] = {
        "hough": detect_merged_lines,
        "directional_morphology": detect_directional_lines,
        "fused_vertical": lambda image: list(detect_vertical_candidates(image).candidates),
    }
    rows: list[dict[str, Any]] = []
    if evidence_dir:
        evidence_dir.mkdir(parents=True, exist_ok=True)
    for image_id, truth in truth_by_image_id.items():
        path = image_set.get_image_path(image_id)
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        targets = _targets(truth.geometry)
        if image is None or targets is None:
            continue
        results: dict[str, dict[str, Any]] = {}
        for name, detector in detectors.items():
            started = time.perf_counter()
            lines = detector(image)
            runtime_ms = (time.perf_counter() - started) * 1000
            score = _score(lines, targets, image.shape)
            if evidence_dir:
                cv2.imwrite(
                    str(evidence_dir / f"{path.stem}_{name}.png"),
                    _draw_evidence(image, lines, targets, score),
                )
            results[name] = {"runtime_ms": runtime_ms, "candidate_count": len(lines), "score": score}
        rows.append({"image": path.name, "detectors": results})
    aggregates: dict[str, dict[str, Any]] = {}
    for name in detectors:
        detector_rows = [row["detectors"][name] for row in rows]
        target_count = sum(row["score"]["target_count"] for row in detector_rows)
        matched_count = sum(row["score"]["matched_count"] for row in detector_rows)
        by_kind = {}
        for kind in ("y_axis", "gridline"):
            kind_targets = sum(row["score"]["by_kind"][kind]["target_count"] for row in detector_rows)
            kind_matches = sum(row["score"]["by_kind"][kind]["matched_count"] for row in detector_rows)
            by_kind[kind] = {
                "target_count": kind_targets,
                "matched_count": kind_matches,
                "recall": kind_matches / kind_targets if kind_targets else None,
            }
        aggregates[name] = {
            "image_count": len(detector_rows),
            "recall": matched_count / target_count if target_count else None,
            "by_kind": by_kind,
            "runtime_ms_mean": (
                sum(row["runtime_ms"] for row in detector_rows) / len(detector_rows)
                if detector_rows
                else None
            ),
            "candidate_count_mean": (
                sum(row["candidate_count"] for row in detector_rows) / len(detector_rows)
                if detector_rows
                else None
            ),
        }
    report = {
        "benchmark": "structural_line_detection",
        "configuration": asdict(CvConfig()),
        "detectors": aggregates,
        "images": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def run_benchmark(paths: list[Path], output_path: Path, runs: int = 1) -> dict:
    """Measure runtime and candidate burden; unlabeled input has no accuracy score."""
    if runs < 1:
        raise ValueError("runs must be at least 1")
    config = CvConfig()
    rows: list[dict[str, Any]] = []
    for image_path in paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            rows.append({"image": str(image_path), "error": "unreadable image"})
            continue
        timings = []
        detection = None
        for _ in range(runs):
            started = time.perf_counter()
            detection = detect_vertical_candidates(image, config)
            timings.append((time.perf_counter() - started) * 1000)
        assert detection is not None
        rows.append(
            {
                "image": str(image_path),
                "width": image.shape[1],
                "height": image.shape[0],
                "runtime_ms_mean": sum(timings) / len(timings),
                "runtime_ms_min": min(timings),
                "counts": _counts(detection),
            }
        )

    successful = [row for row in rows if "counts" in row]
    timings = [row["runtime_ms_mean"] for row in successful]
    candidate_counts = [row["counts"]["fused_vertical"] for row in successful]
    report = {
        "benchmark": "experimental_structural_line_detection",
        "note": "Unlabeled images report runtime and candidate counts, not accuracy.",
        "configuration": {key: value for key, value in asdict(config).items() if key in LINE_CONFIG_FIELDS},
        "runs_per_image": runs,
        "summary": {
            "image_count": len(rows),
            "successful_image_count": len(successful),
            "runtime_ms_mean": sum(timings) / len(timings) if timings else None,
            "runtime_ms_p95": _percentile(timings, 0.95),
            "fused_vertical_candidates_mean": (
                sum(candidate_counts) / len(candidate_counts) if candidate_counts else None
            ),
        },
        "images": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, help="unlabeled image file or directory")
    parser.add_argument("--data", type=Path, help="synthetic dataset with images/ and truth.jsonl")
    parser.add_argument("--dataset", choices=("synthetic", "benetech"), default="synthetic")
    parser.add_argument("--output", type=Path, default=Path("reports/line_benchmark.json"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--runs", type=int, default=1, help="timed runs per image")
    parser.add_argument("--no-evidence", action="store_true")
    args = parser.parse_args()
    if bool(args.input) == bool(args.data):
        parser.error("provide either input or --data")
    if args.data:
        from sci_fi_parser.benchmark.bench_pipeline import load_inputs

        inputs = load_inputs(args.data, args.limit, args.dataset)
        evidence_dir = None if args.no_evidence else args.output.parent / f"{args.output.stem}_evidence"
        report = run_line_benchmark(
            inputs.image_set,
            inputs.truth_by_image_id,
            args.output,
            evidence_dir,
        )
        print(json.dumps(report["detectors"], indent=2))
        print(f"report: {args.output}")
        return
    try:
        assert args.input is not None
        paths = _image_paths(args.input)
    except FileNotFoundError as error:
        parser.error(str(error))
    if args.limit is not None:
        paths = paths[: args.limit]
    if not paths:
        parser.error("no images found")
    if args.runs < 1:
        parser.error("--runs must be at least 1")

    report = run_benchmark(paths, args.output, args.runs)
    print(json.dumps(report["summary"], indent=2))
    print(f"report: {args.output}")


if __name__ == "__main__":
    main()
