"""This module contains objects to store data throughout the pipeline.

Classes:
    ImageSet
    PdfSet

ImageSet stores per-image records for extraction, classification, OCR/CV, and VLM stages.
PdfSet stores per-PDF metadata such as file name and page count.
"""

from __future__ import annotations

from collections.abc import ItemsView, Iterator
from pathlib import Path
from typing import Any, NotRequired, TypedDict


class Metadata(TypedDict):
    extraction: dict[str, Any]
    classification: dict[str, Any]
    ocrcv: dict[str, Any]
    vlm: dict[str, Any]
    benchmark: NotRequired[dict[str, Any]]


class Output(TypedDict):
    path: Path


class Classification(TypedDict):
    result: str
    raw: dict[str, Any]


class OcrCv(TypedDict):
    result: str
    raw: dict[str, Any]


class Vlm(TypedDict):
    result: dict[str, Any]
    raw: dict[str, Any]


class ImageRecord(TypedDict):
    metadata: Metadata
    output: Output
    classification: Classification
    ocrcv: OcrCv
    vlm: Vlm


class ImageSet:
    """Canonical image records accumulated by pipeline stages."""

    def __init__(self) -> None:
        """Key: image_id, dict: metadata etc..."""
        self._data: dict[str, ImageRecord] = {}

    def add(self, id: str, payload: ImageRecord) -> None:
        self._data[id] = payload

    def items(self) -> ItemsView[str, ImageRecord]:
        return self._data.items()

    def get(self, id: str) -> ImageRecord:
        return self._data[id]

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"ImageSet({self._data!r})"

    def __str__(self) -> str:
        return str(self.__repr__)

    def __iter__(self) -> Iterator[str]:
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

    def filter_by_type(self, chart_types: list[str], limit: int | None = None) -> list[str]:
        """Return images classified as ``chart_type``, optionally capped by limit."""
        filtered: list[str] = []
        if limit and limit <= 0:
            return filtered

        for image_id, payload in self.items():
            result = payload.get("classification", {}).get("result", {})
            if result not in chart_types:
                continue
            filtered.append(image_id)
            if limit and len(filtered) >= limit:
                break
        return filtered

    # Image Path
    def get_image_path(self, image_id: str) -> Path:
        """Get Path object for an image from its id"""
        return self._data[image_id]["output"]["path"]

    # Classification
    def add_classification_result(self, image_id: str, result: str) -> None:
        """Add the classification result to the set."""
        record = self._data[image_id]
        record["classification"]["result"] = result

    def add_classification_raw(self, image_id: str, raw: dict[str, Any]) -> None:
        """Add the raw classification scores to the set."""
        record = self._data[image_id]
        record["classification"]["raw"] = raw

    def get_classification_result(self, image_id: str) -> str:
        """Get classification result for an image from its id."""
        return self._data[image_id]["classification"]["result"]

    def get_classification_raw(self, image_id: str) -> dict[str, Any]:
        """Get raw classification scores for an image from its id."""
        return self._data[image_id]["classification"]["raw"]

    # OCR/CV
    def add_ocrcv_raw(self, image_id: str, result: dict[str, Any]) -> None:
        """Add raw result data from OCR/CV pipeline."""
        record = self._data[image_id]
        record["ocrcv"]["raw"] = result

    def add_ocrcv_result(self, image_id: str, result: str) -> None:
        """Add contextual text from OCR/CV to be passed to VLM."""
        record = self._data[image_id]
        record["ocrcv"]["result"] = result

    def get_ocrcv_result(self, image_id: str) -> str:
        """Get OCR/CV result."""
        return self._data[image_id]["ocrcv"]["result"]

    def get_ocrcv_raw(self, image_id: str) -> dict[str, Any]:
        """Get raw OCR/CV result."""
        return self._data[image_id]["ocrcv"]["raw"]

    # VLM
    def add_vlm_result(self, image_id: str, vlm_data: dict[str, Any]) -> None:
        """Add the VLM parsed data to the set"""
        record = self._data[image_id]
        record["vlm"]["result"] = vlm_data

    def add_vlm_result_raw(self, image_id: str, vlm_raw_data: dict[str, Any]) -> None:
        """Add the VLM raw data to the set"""
        record = self._data[image_id]
        record["vlm"]["raw"] = vlm_raw_data

    def get_vlm_result(self, image_id: str) -> dict[str, Any]:
        """Get VLM result for an image from its id"""
        return self._data[image_id]["vlm"]["result"]

    def get_vlm_raw(self, image_id: str) -> dict[str, Any]:
        """Get VLM result for an image from its id"""
        return self._data[image_id]["vlm"]["raw"]

    @staticmethod
    def _empty_record(image_path: Path) -> ImageRecord:
        return {
            "metadata": {
                "extraction": {},
                "classification": {},
                "ocrcv": {},
                "vlm": {},
            },
            "output": {"path": image_path},
            "classification": {
                "result": "",
                "raw": {},
            },
            "ocrcv": {
                "result": "",
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

    def get(self, name: str) -> dict | str:
        return self._data.get(name, "")

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"PdfSet({self._data!r})"

    def __str__(self) -> str:
        return str(self.__repr__)
