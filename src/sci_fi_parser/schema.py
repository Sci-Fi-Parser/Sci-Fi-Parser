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
from typing import Any

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

    def __repr__(self) -> str:
        return f"ImageSet({self._data!r})"

    def __str__(self) -> str:
        return str(self.__repr__)

    def __iter__(self):
        return iter(self._data)

    def add_extracted_image(
        self,
        image_id: str,
        image_path: Path,
        extraction_metadata: dict | None = None,
    ) -> None:
        """Add an image using the standard ImageSet record shape."""
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

    # Image Path
    def get_image_path(self, image_id: str) -> Path:
        """Get Path object for an image from its id"""
        return self._data[image_id]["output"]["path"]

    # Classification
    def add_classification_result(self, image_id: str, result) -> None:
        """Add the classification result to the set."""
        record = self._data[image_id]
        record.setdefault("classification", {})
        record["classification"]["result"] = result

    def add_classification_raw(self, image_id: str, raw) -> None:
        """Add the raw classification scores to the set."""
        record = self._data[image_id]
        record.setdefault("classification", {})
        record["classification"]["raw"] = raw

    def get_classification_result(self, image_id: str):
        """Get classification result for an image from its id."""
        return self._data[image_id]["classification"]["result"]

    def get_classification_raw(self, image_id: str):
        """Get raw classification scores for an image from its id."""
        return self._data[image_id]["classification"]["raw"]

    # OCR/CV
    def add_ocrcv_raw(self, image_id: str, result: dict) -> None:
        """Add raw result data from OCR/CV pipeline."""
        record = self._data[image_id]
        record.setdefault("ocrcv", {})
        record["ocrcv"]["raw"] = result

    def add_ocrcv_result(self, image_id: str, result: str) -> None:
        """Add contextual text from OCR/CV to be passed to VLM."""
        record = self._data[image_id]
        record.setdefault("ocrcv", {})
        record["ocrcv"]["result"] = result

    def get_ocrcv_result(self, image_id: str) -> str:
        """Get OCR/CV result."""
        return self._data[image_id]["ocrcv"]["result"]

    def get_ocrcv_raw(self, image_id: str) -> dict:
        """Get raw OCR/CV result."""
        return self._data[image_id]["ocrcv"]["raw"]

    # VLM
    def add_vlm_result(self, image_id: str, vlm_data: dict[str, Any]):
        """Add the VLM parsed data to the set"""
        record = self._data[image_id]
        record.setdefault("vlm", {})
        record["vlm"]["result"] = vlm_data

    def add_vlm_result_raw(self, image_id: str, vlm_raw_data: dict[str, Any]):
        """Add the VLM raw data to the set"""
        record = self._data[image_id]
        record.setdefault("vlm", {})
        record["vlm"]["raw"] = vlm_raw_data

    def get_vlm_result(self, image_id: str) -> dict[str, Any]:
        """Get VLM result for an from its id"""
        return self._data[image_id]["vlm"]["result"]

    def get_vlm_raw(self, image_id: str) -> dict[str, Any]:
        """Get VLM result for an from its id"""
        return self._data[image_id]["vlm"]["raw"]

    @staticmethod
    def _empty_record(image_path: Path) -> dict:
        return {
            "metadata": {
                "extraction": {},
                "classification": {},
                "ocrcv": {},
                "vlm": {},
            },
            "output": {"path": image_path},
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
        (output / f"{Path(name).stem}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
