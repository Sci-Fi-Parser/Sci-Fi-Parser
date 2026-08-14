"""Generate Y-axis evidence reports for chart folders and classified-label sidecars."""

from __future__ import annotations

import argparse
import fnmatch
import html
import json
from dataclasses import fields
from pathlib import Path

import cv2

from sci_fi_parser.object_detection.computer_vision.axis import NumericAxisLabel
from sci_fi_parser.object_detection.computer_vision.axis_debug import write_y_axis_debug_outputs
from sci_fi_parser.object_detection.computer_vision.axis_ocr import numeric_labels_from_ocr
from sci_fi_parser.object_detection.computer_vision.bars import (
    BarCandidate,
    BoundingBox,
    detect_bars,
)
from sci_fi_parser.object_detection.computer_vision.config import YAxisConfig

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _image_paths(paths: list[Path], pattern: str) -> list[Path]:
    images = []
    for path in paths:
        if path.is_dir():
            images.extend(
                candidate
                for candidate in path.rglob("*")
                if candidate.is_file()
                and candidate.suffix.lower() in IMAGE_SUFFIXES
                and fnmatch.fnmatch(candidate.name, pattern)
            )
        elif path.is_file():
            images.append(path)
        else:
            raise FileNotFoundError(path)
    return sorted(set(images))


def _load_config(path: Path | None) -> YAxisConfig:
    if path is None:
        return YAxisConfig()
    values = json.loads(path.read_text(encoding="utf-8"))
    valid = {item.name for item in fields(YAxisConfig)}
    if unknown := set(values) - valid:
        raise ValueError(f"unknown Y-axis config keys: {sorted(unknown)}")
    return YAxisConfig(**values)


def _load_sidecar(path: Path | None) -> tuple[list[NumericAxisLabel], list[BarCandidate]]:
    if path is None or not path.exists():
        return [], []
    raw = json.loads(path.read_text(encoding="utf-8"))
    labels = [NumericAxisLabel(float(item["value"]), tuple(item["bbox"])) for item in raw.get("labels", [])]
    bars = []
    for item in raw.get("bars", []):
        left, top, right, bottom = item["bbox"]
        bars.append(
            BarCandidate(BoundingBox(round(left), round(top), round(right - left), round(bottom - top)))
        )
    return labels, bars


def _cached_ocr(image, cache_path: Path, refresh: bool, ocr_holder: list) -> dict:
    if cache_path.exists() and not refresh:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    if not ocr_holder:
        from sci_fi_parser.object_detection.ocr import Ocr

        ocr_holder.append(Ocr())
    ocr_holder[0].read_image(image)
    result = ocr_holder[0].run_ocr()
    serializable = {
        key: value.tolist() if hasattr(value, "tolist") else value for key, value in result.items()
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(serializable), encoding="utf-8")
    return serializable


def _bar_image(image):
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def _write_index(output: Path, rows: list[dict]) -> None:
    cards = []
    for row in rows:
        images = "".join(
            f'<figure><img src="{html.escape(path)}"><figcaption>{label}</figcaption></figure>'
            for label, path in row["images"]
        )
        cards.append(f"<section><h2>{html.escape(row['stem'])}: {row['status']}</h2>{images}</section>")
    document = (
        '<!doctype html><meta charset="utf-8"><title>Y-axis review</title>'
        "<style>body{font:14px sans-serif;margin:20px;background:#eee}"
        "section{background:white;padding:12px;margin:16px 0}"
        "figure{display:inline-block;width:24%;margin:0 .5%}"
        "img{max-width:100%;max-height:420px}figcaption{text-align:center}</style>" + "".join(cards)
    )
    (output / "index.html").write_text(document, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("output/y_axis_review"))
    parser.add_argument("--labels-dir", type=Path, help="directory containing <image-stem>.json sidecars")
    parser.add_argument("--ocr-cache", type=Path, default=Path("temp/y_axis_ocr_cache"))
    parser.add_argument("--refresh-ocr", action="store_true")
    parser.add_argument("--no-ocr", action="store_true", help="inspect structural candidates only")
    parser.add_argument("--no-bars", action="store_true", help="skip automatic bar detection")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--image-pattern", default="*")
    parser.add_argument("--candidate")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    try:
        image_paths = _image_paths(args.images, args.image_pattern)
        config = _load_config(args.config)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    if args.limit is not None:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        parser.error("no images found")

    args.output.mkdir(parents=True, exist_ok=True)
    rows, ocr_holder = [], []
    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"skipping unreadable image: {image_path}")
            continue
        sidecar = args.labels_dir / f"{image_path.stem}.json" if args.labels_dir else None
        if sidecar is not None and sidecar.exists():
            labels, bars = _load_sidecar(sidecar)
        else:
            cache_path = args.ocr_cache / f"{image_path.stem}.json"
            ocr_result = (
                {"labels": [], "confidence": [], "bbox": []}
                if args.no_ocr
                else _cached_ocr(image, cache_path, args.refresh_ocr, ocr_holder)
            )
            labels = numeric_labels_from_ocr(ocr_result, config.numeric_ocr_min_confidence)
            bars = []
        if not bars and not args.no_bars:
            bars = detect_bars(_bar_image(image))
        result, paths = write_y_axis_debug_outputs(
            image,
            labels,
            bars,
            args.output,
            image_path.stem,
            axis_config=config,
            candidate_id=args.candidate,
        )
        rows.append(
            {
                "stem": image_path.stem,
                "status": result.selected_candidate.id if result.selected_candidate else "ABSTAINED",
                "images": [
                    ("sources", paths[0].name),
                    ("fusion", paths[1].name),
                    ("evidence", paths[2].name),
                    ("selection", paths[3].name),
                ],
            }
        )
        print(f"{image_path}: {rows[-1]['status']}")
    _write_index(args.output, rows)
    print(f"review: {args.output / 'index.html'}")


if __name__ == "__main__":
    main()
