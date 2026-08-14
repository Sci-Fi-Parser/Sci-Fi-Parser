"""Generate raw and merged structural-line overlays for chart images."""

import argparse
from pathlib import Path

import cv2

from sci_fi_parser.object_detection.computer_vision.line_debug import (
    write_line_detection_overlays,
)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def iter_image_paths(paths: list[Path]) -> list[Path]:
    images: list[Path] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        if path.is_dir():
            images.extend(
                sorted(
                    candidate
                    for candidate in path.rglob("*")
                    if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES
                )
            )
        elif path.is_file():
            images.append(path)
        else:
            raise ValueError(f"unsupported path: {path}")
    return images


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path, help="image files or directories")
    parser.add_argument("--output", type=Path, default=Path("output/line_debug"))
    parser.add_argument("--morphology", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    try:
        image_paths = iter_image_paths(args.images)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    if args.limit is not None:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        parser.error("no images found")

    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"skipping unreadable image: {image_path}")
            continue
        raw_path, result_path = write_line_detection_overlays(
            image,
            args.output,
            image_path.stem,
            morphology=args.morphology,
        )
        print(raw_path)
        print(result_path)


if __name__ == "__main__":
    main()
