"""Reusable overlays and batch review output for OCR/CV stage evidence."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path

import cv2
import numpy as np

from sci_fi_parser.object_detection.axis_analysis import analyze_ocr_cv
from sci_fi_parser.object_detection.computer_vision.bars import (
    BarDetectionDiagnostics,
    detect_bars_with_diagnostics,
)
from sci_fi_parser.object_detection.models import ChartType, OcrCvStageResult, OcrOutput
from sci_fi_parser.object_detection.normalization import normalize_ocr_output

COLORS = {
    "raw": (160, 160, 160),
    "y_tick": (40, 70, 230),
    "x_label": (50, 180, 60),
    "other": (150, 150, 150),
    "candidate_y": (220, 80, 220),
    "candidate_x": (0, 180, 220),
    "inlier": (30, 190, 30),
    "outlier": (20, 20, 230),
    "bar": (230, 140, 20),
    "shared_y": (255, 0, 255),
    "rejected": (30, 30, 230),
    "saturation": (20, 180, 240),
    "grayscale": (180, 180, 180),
    "outline": (240, 80, 180),
}
CORE_VIEWS = (
    "raw_ocr",
    "roles",
    "axis_candidates",
    "calibration",
)
BAR_VIEWS = (
    "initial_bars",
    "bar_saturation_mask",
    "bar_grayscale_mask",
    "bar_outline_mask",
    "bar_contours",
)
VIEWS = CORE_VIEWS + BAR_VIEWS + ("combined",)


def _rectangle(canvas: np.ndarray, box, color, thickness: int = 2) -> None:
    left = box.left if hasattr(box, "left") else box.x
    top = box.top if hasattr(box, "top") else box.y
    cv2.rectangle(
        canvas,
        (round(left), round(top)),
        (round(box.right), round(box.bottom)),
        color,
        thickness,
    )


def _label(canvas: np.ndarray, text: str, x: float, y: float, color, scale: float = 0.42) -> None:
    cv2.putText(
        canvas,
        text,
        (round(x), max(13, round(y))),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        1,
        cv2.LINE_AA,
    )


def render_overlay(
    image: np.ndarray,
    result: OcrCvStageResult,
    view: str,
    bar_diagnostics: BarDetectionDiagnostics | None = None,
) -> np.ndarray:
    """Render one named view from the same structured stage result."""

    if view not in VIEWS:
        raise ValueError(f"unknown OCR/CV debug view {view!r}; expected one of {VIEWS}")
    canvas = image.copy()
    mask_views = {
        "bar_saturation_mask": "saturation_mask",
        "bar_grayscale_mask": "grayscale_mask",
        "bar_outline_mask": "outline_mask",
    }
    if view in mask_views:
        mask = getattr(bar_diagnostics, mask_views[view], None) if bar_diagnostics else None
        if mask is None:
            canvas = np.zeros_like(image)
            _label(canvas, "mask not used", 8, 18, COLORS["raw"], 0.5)
        else:
            canvas = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        return canvas
    assignments = {assignment.token_id: assignment for assignment in result.roles.assignments}
    tokens = {token.token_id: token for token in result.ocr.tokens}

    if view == "raw_ocr":
        for token in result.ocr.tokens:
            _rectangle(canvas, token.box, COLORS["raw"], 1)
            _label(
                canvas,
                f"{token.original_text} {token.confidence:.2f}",
                token.box.left,
                token.box.top - 3,
                COLORS["raw"],
            )

    if view in {"roles", "combined"}:
        for token in result.ocr.tokens:
            assignment = assignments[token.token_id]
            color = COLORS[assignment.role]
            _rectangle(canvas, token.box, color, 2 if assignment.role != "other" else 1)
            if view == "roles":
                _label(
                    canvas,
                    f"{assignment.role}: {token.original_text}",
                    token.box.left,
                    token.box.bottom + 13,
                    color,
                )

        selected_pair = next(
            (
                pair
                for pair in result.roles.pair_diagnostics
                if pair.accepted
                and pair.y_candidate_id == result.roles.selected_y_candidate_id
                and pair.x_candidate_id == result.roles.selected_x_candidate_id
            ),
            None,
        )
        if selected_pair is not None:
            for token_id in selected_pair.shared_token_ids:
                token = tokens[token_id]
                _rectangle(canvas, token.box, COLORS["shared_y"], 4)
                _label(
                    canvas,
                    "shared endpoint -> y_tick",
                    token.box.right + 4,
                    token.box.bottom,
                    COLORS["shared_y"],
                )

    if view in {"axis_candidates", "combined"}:
        for candidate in result.roles.y_candidates:
            selected = candidate.candidate_id == result.roles.selected_y_candidate_id
            if view == "combined" and not selected:
                continue
            _rectangle(canvas, candidate.bounds, COLORS["candidate_y"], 3 if selected else 1)
            _label(
                canvas,
                f"{candidate.candidate_id} y={candidate.score:.2f}",
                candidate.bounds.left,
                candidate.bounds.top - 5,
                COLORS["candidate_y"],
            )
        for candidate in result.roles.x_candidates:
            selected = candidate.candidate_id == result.roles.selected_x_candidate_id
            if view == "combined" and not selected:
                continue
            _rectangle(canvas, candidate.bounds, COLORS["candidate_x"], 3 if selected else 1)
            _label(
                canvas,
                f"{candidate.candidate_id} x={candidate.score:.2f}",
                candidate.bounds.left,
                candidate.bounds.bottom + 14,
                COLORS["candidate_x"],
            )

    if view in {"calibration", "combined"}:
        for label in result.calibration.inliers:
            token = tokens[label.token_id]
            _rectangle(canvas, token.box, COLORS["inlier"], 3)
            fitted = label.value - label.residual_value
            _label(
                canvas,
                f"{label.value:g} fit={fitted:g} err={label.residual_pixels:.1f}px",
                token.box.right + 4,
                token.box.center_y,
                COLORS["inlier"],
            )
        for label in result.calibration.outliers:
            token = tokens[label.token_id]
            _rectangle(canvas, token.box, COLORS["outlier"], 3)
            _label(
                canvas,
                f"outlier {label.value:g} err={label.residual_pixels:.1f}px",
                token.box.right + 4,
                token.box.center_y,
                COLORS["outlier"],
            )
        status = (
            f"y={result.calibration.slope:.6g}*px+{result.calibration.intercept:.6g} "
            f"confidence={result.calibration.confidence:.2f}"
            if result.calibration.succeeded
            else f"calibration abstained: {result.calibration.failure_reason}"
        )
        _label(
            canvas,
            status,
            8,
            18,
            COLORS["inlier"] if result.calibration.succeeded else COLORS["outlier"],
            0.5,
        )

    if view in {"initial_bars", "combined"}:
        for element in result.initial_elements:
            _rectangle(canvas, element.box, COLORS[element.kind], 2)
            _label(
                canvas,
                element.element_id,
                element.box.left,
                element.box.top - 4,
                COLORS[element.kind],
            )
    if view == "bar_contours" and bar_diagnostics is not None:
        for contour in bar_diagnostics.contours:
            color = COLORS[contour.source] if contour.accepted else COLORS["rejected"]
            _rectangle(canvas, contour.bbox, color, 2 if contour.accepted else 1)
            status = contour.source if contour.accepted else ", ".join(contour.rejection_reasons)
            if contour.deduplicated_to:
                status = f"duplicate -> {contour.deduplicated_to}"
            _label(canvas, status, contour.bbox.x, contour.bbox.y - 3, color)
    return canvas


def write_debug_artifacts(
    image: np.ndarray,
    result: OcrCvStageResult,
    output_dir: Path,
    sample_id: str,
    source_path: Path,
    bar_diagnostics: BarDetectionDiagnostics | None = None,
) -> dict:
    """Write all milestone 2-4 views and a complete JSON sidecar."""

    sample_dir = output_dir / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    overlays = {}
    views = CORE_VIEWS + (BAR_VIEWS if result.chart_type == "bar_chart" else ()) + ("combined",)
    for view in views:
        path = sample_dir / f"{view}.png"
        if not cv2.imwrite(str(path), render_overlay(image, result, view, bar_diagnostics)):
            raise OSError(f"failed to write debug overlay: {path}")
        overlays[view] = str(path.relative_to(output_dir))

    deferred_views = {
        "plot_region": "milestone 5 not implemented",
        "category_links": "milestone 6 not implemented",
        "estimated_values": "milestone 6 not implemented",
    }
    if result.chart_type == "bar_chart":
        deferred_views["refined_bars"] = "milestone 5 not implemented"
    else:
        deferred_views["line_series"] = "line-series extraction not implemented"
    sidecar = {
        "source_image": str(source_path),
        "available_views": list(views),
        "deferred_views": deferred_views,
        "result": result.model_dump(mode="json"),
        "bar_detection": bar_diagnostics.to_dict() if bar_diagnostics is not None else None,
    }
    sidecar_path = sample_dir / "evidence.json"
    sidecar_path.write_text(
        json.dumps(sidecar, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return {
        "sample_id": sample_id,
        "source_image": str(source_path),
        "sidecar": str(sidecar_path.relative_to(output_dir)),
        "overlays": overlays,
        "role_confidence": result.roles.confidence,
        "role_abstention": result.roles.abstention_reason,
        "calibration_confidence": result.calibration.confidence,
        "calibration_failure": result.calibration.failure_reason,
    }


def _write_html(output_dir: Path, records: list[dict]) -> Path:
    cards = []
    for record in records:
        images = "".join(
            f'<figure><img loading="lazy" src="{html.escape(path)}">'
            f"<figcaption>{html.escape(view)}</figcaption></figure>"
            for view, path in record["overlays"].items()
        )
        cards.append(
            f"<section><h2>{html.escape(record['sample_id'])}</h2>"
            f"<p>roles={record['role_confidence']:.3f}; calibration={record['calibration_confidence']:.3f}; "
            f'<a href="{html.escape(record["sidecar"])}">JSON evidence</a></p><div>{images}</div></section>'
        )
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>OCR/CV stage review</title>
<style>body{{font:14px sans-serif;margin:20px;background:#eee}}
section{{background:white;padding:14px;margin:0 0 20px}}
section>div{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:10px}}figure{{margin:0}}
img{{max-width:100%;border:1px solid #999}}figcaption{{font-weight:bold}}</style></head><body>
<h1>OCR/CV stage visual review</h1>{"".join(cards)}</body></html>"""
    path = output_dir / "index.html"
    path.write_text(page, encoding="utf-8")
    return path


def _image_paths(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
    return [path for path in sorted(input_path.rglob("*")) if path.suffix.lower() in extensions]


def _sample_id(path: Path, root: Path) -> str:
    relative = path.name if root.is_file() else str(path.relative_to(root))
    safe = "".join(character if character.isalnum() or character in "-_" else "_" for character in relative)
    digest = hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:8]
    return f"{safe}_{digest}"


def _load_cached_ocr(directory: Path, image_path: Path) -> OcrOutput:
    path = directory / f"{image_path.stem}.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    if "tokens" in raw:
        return OcrOutput.model_validate(raw)
    if "text" in raw:
        entries = [entry for entry in raw["text"] if "text" in entry and "polygon" in entry]
        labels = [entry["text"] for entry in entries]
        confidences = [1.0] * len(entries)
        boxes = [
            [[entry["polygon"][f"x{index}"], entry["polygon"][f"y{index}"]] for index in range(4)]
            for entry in entries
        ]
        return normalize_ocr_output(labels, confidences, boxes)
    return normalize_ocr_output(raw.get("labels"), raw.get("confidence"), raw.get("bbox"))


def run_debug_batch(
    input_path: Path,
    output_dir: Path,
    *,
    chart_type: ChartType = "bar_chart",
    limit: int | None = None,
    ocr_json_dir: Path | None = None,
) -> Path:
    """Run the component stage over images and generate an HTML review index."""

    paths = _image_paths(input_path)
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise ValueError(f"no supported images found under {input_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    ocr_engine = None
    if ocr_json_dir is None:
        from sci_fi_parser.object_detection.ocr import Ocr

        ocr_engine = Ocr()

    records = []
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"could not read image: {path}")
        if ocr_json_dir is not None:
            ocr = _load_cached_ocr(ocr_json_dir, path)
        else:
            assert ocr_engine is not None
            ocr_engine.read_image(image)
            ocr = ocr_engine.run_ocr()
        bar_result = detect_bars_with_diagnostics(image) if chart_type == "bar_chart" else None
        initial_elements = bar_result.bars if bar_result is not None else ()
        result = analyze_ocr_cv(image, ocr, chart_type, initial_elements)
        records.append(
            write_debug_artifacts(
                image,
                result,
                output_dir,
                _sample_id(path, input_path),
                path,
                bar_result.diagnostics if bar_result is not None else None,
            )
        )

    manifest = output_dir / "manifest.json"
    manifest.write_text(json.dumps(records, indent=2, allow_nan=False), encoding="utf-8")
    return _write_html(output_dir, records)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images", type=Path, required=True, help="image file or recursively scanned directory"
    )
    parser.add_argument("--out", type=Path, default=Path("reports/ocr_cv_stage/debug"))
    parser.add_argument(
        "--chart-type",
        choices=("bar_chart", "line_chart"),
        default="bar_chart",
        help="chart family; line charts skip bar detection and bar-specific overlays",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--ocr-json-dir",
        type=Path,
        help="optional cached <image-stem>.json files containing canonical or legacy OCR output",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    index = run_debug_batch(
        args.images,
        args.out,
        chart_type=args.chart_type,
        limit=args.limit,
        ocr_json_dir=args.ocr_json_dir,
    )
    print(f"OCR/CV review index: {index}")


if __name__ == "__main__":
    main()
