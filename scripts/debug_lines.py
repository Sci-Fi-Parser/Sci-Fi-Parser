"""Generate raw and merged structural-line overlays for chart images."""

import argparse
from pathlib import Path

import cv2

from sci_fi_parser.object_detection.computer_vision.line_debug import (
    write_line_detection_overlays,
)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def iter_image_paths(paths: list[Path]) -> list[Path]:
    """Expand file and directory arguments into a list of image paths."""
    images: list[Path] = []

    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)

        if path.is_dir():
            images.extend(
                sorted(
                    p
                    for p in path.rglob("*")
                    if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
                )
            )
        elif path.is_file():
            images.append(path)
        else:
            raise ValueError(f"Unsupported path: {path}")

    return images


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "images",
        nargs="+",
        type=Path,
        help="One or more image files and/or directories containing images.",
    )
    parser.add_argument("--output", type=Path, default=Path("output/line_debug"))
    args = parser.parse_args()

    try:
        image_paths = iter_image_paths(args.images)
    except (FileNotFoundError, ValueError) as e:
        parser.error(str(e))

    if not image_paths:
        parser.error("No images found.")

    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"Skipping unreadable image: {image_path}")
            continue

        raw_path, merged_path = write_line_detection_overlays(
            image,
            args.output,
            image_path.stem,
        )

        print(raw_path)
        print(merged_path)


if __name__ == "__main__":
    main()
