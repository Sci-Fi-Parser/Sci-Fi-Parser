"""VLM stage of the extraction pipeline.

Reads OCR text from an :class:`OCRSet`, runs the VLM on each image with that
text appended to the prompt, and stores the resulting ChartData in a
:class:`VLMSet`. The set types and the offloader live in
:mod:`sci_fi_parser.data_pipeline`.
"""

from __future__ import annotations

from pathlib import Path

from sci_fi_parser.data_pipeline import OCRSet, VLMSet
from sci_fi_parser.vlm.vlm import OllamaVLM
from sci_fi_parser.vlm.vlm_config import load_profile

_IMAGE_GLOBS = ("*.png", "*.jpg", "*.jpeg")


def _images_in(folder: Path) -> list[Path]:
    """Top-level PNG/JPG/JPEG in ``folder``, sorted by filename."""
    return sorted(p for pat in _IMAGE_GLOBS for p in folder.glob(pat))

def _ocr_suffix(ocr_record: dict) -> str:
    labels = ocr_record["ocr_result"]["labels"]

    if not labels:
        return ""

    ocr_text = "\n".join(labels)

    return (
        "Additional text recognised from the page by OCR (treat as a hint, "
        "not gospel -- prefer what you actually see on the chart):\n"
        f"{ocr_text}"
    )


def start_vlm(target: Path, ocr_set: OCRSet, vlm_set: VLMSet, writer, run_id,
              vlm_config: Path = Path("config/vlm.toml")) -> None:
    """For each image in ``target``, look up its OCR text in ``ocr_set``,
    append that to the VLM prompt, run the model, and store the resulting
    ChartData (as a JSON-ready dict) in ``vlm_set`` under the image name.
    """
    vlm = OllamaVLM(load_profile(vlm_config))
    
    for image_name, ocr_record in ocr_set._data.items():
        print("ocr_record")
        print(ocr_record)
        
        chart_id = ocr_record["chart_id"]

        image_path = target / image_name
        suffix = _ocr_suffix(ocr_record)
        print("suffix:")
        print(suffix)
        
        data = vlm.extract(image_path, writer=writer, run_id=run_id, prompt_suffix=suffix)

        print("data")
        print(data)

        vlm_set.add(image_name, data.model_dump())
