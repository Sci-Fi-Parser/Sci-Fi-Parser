"""Two-stage extraction pipeline: OCR (shadow) -> VLM -> offload.

Each stage writes into its own filename-keyed *set*:

  - :class:`OCRSet`  --  image name -> OCR text
  - :class:`VLMSet`  --  image name -> ChartData as JSON-ready dict

VLM reads from the OCR set to enrich its prompt; the offloader takes only
the VLM set. The OCR backend is a shadow stub for now -- swap the body of
:func:`start_ocr` to plug in Tesseract / EasyOCR; nothing else changes.
"""

from __future__ import annotations

import json
from pathlib import Path

from cv.ocr import Ocr
import cv.bars
import cv2

from sci_fi_parser.accuracy.vlm import OllamaVLM
from sci_fi_parser.accuracy.vlm_config import load_profile

_IMAGE_GLOBS = ("*.png", "*.jpg", "*.jpeg")


def _images_in(folder: Path) -> list[Path]:
    """Top-level PNG/JPG/JPEG in ``folder``, sorted by filename."""
    return sorted(p for pat in _IMAGE_GLOBS for p in folder.glob(pat))


# --------------------------------------------------------------------------- #
# Set classes -- name-keyed payload stores, one per pipeline stage
# --------------------------------------------------------------------------- #
class OCRSet:
    """image filename -> OCR text. Populated by :func:`start_ocr`."""

    def __init__(self) -> None:
        self._data: dict[str, str] = {}

    def add(self, name: str, text: str) -> None:
        self._data[name] = text

    def get(self, name: str) -> str:
        return self._data.get(name, "")

    def __len__(self) -> int:
        return len(self._data)


class VLMSet:
    """image filename -> ChartData payload. Populated by :func:`start_vlm`."""

    def __init__(self) -> None:
        self._data: dict[str, dict] = {}

    def add(self, name: str, payload: dict) -> None:
        self._data[name] = payload

    def items(self):
        return self._data.items()

    def __len__(self) -> int:
        return len(self._data)


# --------------------------------------------------------------------------- #
# Pipeline stages
# --------------------------------------------------------------------------- #
def start_ocr(folder: str, ocr_set: dict):
    path = Path(folder).glob("*.jpg")
    ocr = Ocr()
    for image in path:
        image_array = cv2.imread(image)
        bar_candidates = bars.detect_bars(image_array)

        ocr.read_image(image_array)
        ocr_res = ocr.run_ocr()
        ocr_set.add(image , (bar_candidates, ocr_res))

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
    vlm = OllamaVLM(load_profile(vlm_config))
    for img in _images_in(target):
        suffix = _ocr_suffix(ocr_set.get(img.name))
        data = vlm.extract(img, prompt_suffix=suffix)
        vlm_set.add(img.name, data.model_dump())


def data_offloader(vlm_set: VLMSet, output: Path) -> None:
    """Write each entry of ``vlm_set`` as one JSON file under ``output``.
    Filename rule: ``<image_stem>.json`` (e.g. ``foo.png`` -> ``foo.json``).
    """
    output.mkdir(parents=True, exist_ok=True)
    for name, payload in vlm_set.items():
        (output / f"{Path(name).stem}.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8")
