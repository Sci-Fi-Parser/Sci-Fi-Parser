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


# --------------------------------------------------------------------------- #
# Set classes -- name-keyed payload stores, one per pipeline stage
# --------------------------------------------------------------------------- #
class OCRSet:
    """image filename -> OCR text. Populated by :func:`start_ocr`."""

    def __init__(self) -> None:
        self._data: dict[str, str] = {}

    def add(self, name: str, text: str) -> None:
        self._data[name] = text

    def items(self):
        return self._data.items()

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

    def get(self, name: str) -> str:
        return self._data.get(name, "")

    def __len__(self) -> int:
        return len(self._data)


class ImageSet:
    """"""
    def __init__(self) -> None:
        """Key: image_id, dict: metadata etc..."""
        self._data: dict[str, dict] = {}

    def add(self, id: str, payload: dict) -> None:
        self._data[id] = payload

    def items(self):
        return self._data.items()

    def get(self, name: str) -> str:
        return self._data.get(name, "")

    def __len__(self) -> int:
        return len(self._data)


class PdfSet:
    """Key: pdf_id, dict: metadata"""
    def __init__(self) -> None:
        """Key: pdf_id, dict: metadata etc..."""
        self._data: dict[str, dict] = {}

    def add(self, id: str, payload: dict) -> None:
        self._data[id] = payload

    def items(self):
        return self._data.items()

    def get(self, name: str) -> str:
        return self._data.get(name, "")

    def __len__(self) -> int:
        return len(self._data)

# --------------------------------------------------------------------------- #
# Pipeline stages
# --------------------------------------------------------------------------- #

def data_offloader(vlm_set: VLMSet, output: Path) -> None:
    """Write each entry of ``vlm_set`` as one JSON file under ``output``.
    Filename rule: ``<image_stem>.json`` (e.g. ``foo.png`` -> ``foo.json``).
    """
    output.mkdir(parents=True, exist_ok=True)
    for name, payload in vlm_set.items():
        (output / f"{Path(name).stem}.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8")
