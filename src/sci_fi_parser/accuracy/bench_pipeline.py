"""Staged benchmark pipeline skeleton.

This module keeps the pipeline shape explicit while delegating the final scoring
stage to :mod:`sci_fi_parser.accuracy.benchmark`. The early stages mirror the
real pipeline shape and are mostly pass-throughs until their implementations are
ready to benchmark.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from sci_fi_parser.accuracy import benchmark
from sci_fi_parser.data_pipeline import OCRSet, ImageSet, PdfSet
from sci_fi_parser.vlm.pipeline import _ocr_suffix
from sci_fi_parser.vlm.vlm_config import VLMProfile, load_profile


@dataclass(slots=True)
class PipelineInputs:
    truth: dict
    images: list[str]
    img_dir: Path


def load_inputs(data: Path, limit: int | None = None) -> PipelineInputs:
    """Load the current synthetic truth and image list."""
    truth = benchmark.load_truth(data)
    images = sorted(truth)[:limit] if limit else sorted(truth)
    return PipelineInputs(truth=truth, images=images, img_dir=data / "images")


def bench_image_extraction(data: Path, limit: int | None = None) -> tuple(ImageSet(), PdfSet()):
    """Placeholder for a future real image-extraction benchmark stage."""
    from sci_fi_parser.image_extraction.pipeline import start_extraction
    image_set = ImageSet()
    pdf_set = PdfSet()
    start_extraction(data, image_set, pdf_set, None)
    return image_set, pdf_set


def bench_classification(inputs: PipelineInputs) -> None:
    """Placeholder for a future chart-classification stage."""
    return None


def bench_ocr_and_cv(inputs: PipelineInputs) -> OCRSet:
    """Run the real OCR/CV pipeline stage and return its normal OCRSet output."""
    from sci_fi_parser.cv.pipeline import extract_ocr_data, format_ocr_output

    ocr_set = OCRSet()
    raw_data = extract_ocr_data(inputs.img_dir)
    for data in raw_data:
        output_string = format_ocr_output(data)
        ocr_set.add(data.image_name, output_string)
    return ocr_set


def bench_vlm_output(*, inputs: PipelineInputs, out: Path, extractor_name: str,
                     profile: VLMProfile | None, seed: int,
                     ocr_set: OCRSet | None,
                     print_summary: bool = True) -> dict:
    """Run the existing VLM/value benchmark as the final pipeline stage."""
    prompt_suffixes = None
    if ocr_set is not None:
        prompt_suffixes = {
            name: _ocr_suffix(ocr_set.get(name)) for name in inputs.images
        }
    return benchmark.run_benchmark(
        data=inputs.img_dir.parent,
        out=out,
        extractor_name=extractor_name,
        profile=profile,
        seed=seed,
        limit=len(inputs.images),
        prompt_suffixes=prompt_suffixes,
        print_summary=print_summary,
    )


def run_pipeline(*, data: Path, out: Path, extractor_name: str = "noisy-oracle",
                 profile: VLMProfile | None = None, seed: int = 0,
                 limit: int | None = None, ocr_cv: bool = False,
                 print_summary: bool = True) -> dict:
    inputs = load_inputs(data, limit)
    ocr_set = bench_ocr_and_cv(inputs) if ocr_cv else None
    return bench_vlm_output(
        inputs=inputs,
        out=out,
        extractor_name=extractor_name,
        profile=profile,
        seed=seed,
        ocr_set=ocr_set,
        print_summary=print_summary,
    )


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True,
                    help="synthetic dataset dir (images/ + labels.jsonl)")
    ap.add_argument("--out", type=Path, default=Path("reports/pipeline"))
    ap.add_argument("--extractor", default="noisy-oracle",
                    help="noisy-oracle | ollama | ollama:<model> | api | api:<model>")
    ap.add_argument("--vlm-config", type=Path, default=None,
                    help="VLM profile TOML (default: config/vlm.toml if present)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="only first N charts")
    ap.add_argument("--ocr-cv", action="store_true",
                    help="run current OCR/CV components and pass their output to the VLM")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    run_pipeline(
        data=args.data,
        out=args.out,
        extractor_name=args.extractor,
        profile=benchmark._resolve_profile(args.vlm_config),
        seed=args.seed,
        limit=args.limit,
        ocr_cv=args.ocr_cv,
    )


if __name__ == "__main__":
    main()
