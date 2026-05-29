"""VLM stage of the extraction pipeline.

Reads OCR text from an :class:`OCRSet`, runs the VLM on each image with that
text appended to the prompt, and stores the resulting ChartData in a
:class:`VLMSet`. The set types and the offloader live in
:mod:`sci_fi_parser.data_pipeline`.
"""

from __future__ import annotations

from pathlib import Path

from sci_fi_parser.data_pipeline import OCRSet, VLMSet
from sci_fi_parser.vlm.vlm import build_vlm
from sci_fi_parser.vlm.vlm_config import load_profile

_IMAGE_GLOBS = ("*.png", "*.jpg", "*.jpeg")


def _images_in(folder: Path) -> list[Path]:
    """Top-level PNG/JPG/JPEG in ``folder``, sorted by filename."""
    return sorted(p for pat in _IMAGE_GLOBS for p in folder.glob(pat))

def _ocr_suffix(ocr_text: str) -> str:
    """Format OCR text as a prompt-context block (empty in -> empty out)."""
    if not ocr_text.strip():
        return ""
    return (
        "Additional text recognised from the page by OCR (treat as a hint, "
        "not gospel -- prefer what you actually see on the chart):\n"
        f"{ocr_text}"
    )


def start_vlm(target: Path, ocr_set: OCRSet, vlm_set: VLMSet,
              vlm_config: Path = Path("config/vlm.toml")) -> None:
    """For each image in ``target``, look up its OCR text in ``ocr_set``,
    append that to the VLM prompt, run the model, and store the resulting
    ChartData (as a JSON-ready dict) in ``vlm_set`` under the image name.
    """
    vlm = build_vlm(load_profile(vlm_config))
    for img in _images_in(target):
        suffix = _ocr_suffix(ocr_set.get(img.name))
        data = vlm.extract(img, prompt_suffix=suffix)
        vlm_set.add(img.name, data.model_dump())
