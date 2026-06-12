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


_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


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
    """Canonical image records accumulated by pipeline stages."""

    def __init__(self) -> None:
        """Key: image_id, dict: metadata etc..."""
        self._data: dict[str, dict] = {}

    def add(self, id: str, payload: dict) -> None:
        self._data[id] = payload

    def items(self):
        return self._data.items()

    def get(self, id: str) -> str:
        return self._data.get(id, "")

    def __len__(self) -> int:
        return len(self._data)

    def add_extracted_image(
        self,
        image_id: str,
        image_path: Path,
        extraction_metadata: dict | None = None,
    ) -> None:
        """Add an extracted image using the standard ImageSet record shape."""
        record = self._empty_record(image_path)
        record["metadata"]["extraction"].update(extraction_metadata or {})
        self._data[image_id] = record

    def filter_by_type(self, chart_type: str, limit: int = 100) -> list[str]:
        """Return images classified as ``chart_type``, optionally capped by limit."""
        filtered = []
        if limit <= 0:
            return filtered
        for image_id, payload in self.items():
            result = payload.get("classification", {}).get("result", {})
            if result.get("selected_type") != chart_type:
                continue
            filtered.append(image_id)
            if len(filtered) >= limit:
                break
        return filtered

    def get_image_path(self, image_id: str) -> Path:
        """Get Path object for a image from its id"""
        return self._data[image_id]["output"]["path"]

    @staticmethod
    def _empty_record(image_path: Path) -> dict:
        return {
            "metadata": {
                "extraction": {},
                "classification": {},
                "ocrcv": {},
                "vlm": {},
            },
            "output": {
                "path": image_path
            },
            "classification": {
                "result": {},
                "raw": {},
            },
            "ocrcv": {
                "result": {},
                "raw": {},
            },
            "vlm": {
                "result": {},
                "raw": {},
            },
        }

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
