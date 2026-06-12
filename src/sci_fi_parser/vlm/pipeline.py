"""VLM stage of the extraction pipeline.

Reads OCR text from an :class:`OCRSet`, runs the VLM on each image with that
text appended to the prompt, and stores the resulting ChartData in a
:class:`VLMSet`. The set types and the offloader live in
:mod:`sci_fi_parser.data_pipeline`.
"""

from __future__ import annotations

from pathlib import Path

from sci_fi_parser.data_pipeline import ImageSet
from sci_fi_parser.vlm.vlm import build_vlm
from sci_fi_parser.vlm.vlm_config import VLMProfile, load_profile


def start_vlm(image_set: ImageSet,
              profile: VLMProfile | Path) -> None:
    
    vlm = build_vlm(profile if isinstance(profile, VLMProfile)
                    else load_profile(profile))
    for img_id, img_data in image_set.items():
        ocr_result = img_data["ocrcv"]["result"]
        if ocr_result:
            suffix = ocr_result["text"]
        else:
            suffix = ""
        img_path = img_data["output"]["path"]
        data = vlm.extract(img_path, suffix)
        image_set.add(img_id, data.model_dump())
