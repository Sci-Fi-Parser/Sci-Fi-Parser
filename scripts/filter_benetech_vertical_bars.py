#!/usr/bin/env python3
"""Copy extracted Benetech charts of one type with data-series labels."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def has_xy_data_series(annotation: dict[str, Any]) -> bool:
    data_series = annotation.get("data-series")
    if not isinstance(data_series, list):
        return False

    return any(isinstance(point, dict) and "x" in point and "y" in point for point in data_series)


def is_wanted_annotation(annotation: dict[str, Any], chart_type: str = "vertical_bar") -> bool:
    return (
        annotation.get("source") == "extracted"
        and annotation.get("chart-type") == chart_type
        and has_xy_data_series(annotation)
    )


def find_image(images_dir: Path, stem: str) -> Path | None:
    for extension in IMAGE_EXTENSIONS:
        image_path = images_dir / f"{stem}{extension}"
        if image_path.exists():
            return image_path
    return None


def copy_matching_files(
    input_dir: Path,
    output_dir: Path,
    dry_run: bool,
    chart_type: str = "vertical_bar",
) -> None:
    annotations_dir = input_dir / "annotations"
    images_dir = input_dir / "images"
    output_annotations_dir = output_dir / "annotations"
    output_images_dir = output_dir / "images"

    if not annotations_dir.is_dir():
        raise FileNotFoundError(f"Missing annotations directory: {annotations_dir}")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing images directory: {images_dir}")

    scanned = 0
    matched = 0
    copied_images = 0
    missing_images = 0
    malformed = 0

    if not dry_run:
        output_annotations_dir.mkdir(parents=True, exist_ok=True)
        output_images_dir.mkdir(parents=True, exist_ok=True)

    for annotation_path in sorted(annotations_dir.glob("*.json")):
        scanned += 1

        try:
            annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            malformed += 1
            continue

        if not isinstance(annotation, dict) or not is_wanted_annotation(annotation, chart_type):
            continue

        matched += 1
        image_path = find_image(images_dir, annotation_path.stem)

        if image_path is None:
            missing_images += 1
            continue

        copied_images += 1

        if dry_run:
            continue

        shutil.copy2(annotation_path, output_annotations_dir / annotation_path.name)
        shutil.copy2(image_path, output_images_dir / image_path.name)

    print(f"Scanned annotations: {scanned}")
    print(f"Matched annotations: {matched}")
    print(f"Copied image/annotation pairs: {copied_images}")
    print(f"Missing images: {missing_images}")
    print(f"Malformed annotations: {malformed}")

    if dry_run:
        print("Dry run only; no files were copied.")
    else:
        print(f"Output written to: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy extracted Benetech chart annotations of one type that contain "
            "data-series x/y values, plus their corresponding images."
        )
    )
    parser.add_argument(
        "--chart-type",
        choices=("vertical_bar", "line"),
        default="vertical_bar",
        help="Benetech chart-type value to select.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("train"),
        help="Input train-like directory containing annotations/ and images/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("train_extracted_vertical_bars"),
        help="Output directory to create with annotations/ and images/.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print counts without copying files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    copy_matching_files(args.input, args.output, args.dry_run, args.chart_type)


if __name__ == "__main__":
    main()
