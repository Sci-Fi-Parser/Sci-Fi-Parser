"""VLM stage of the extraction pipeline.

Reads OCR text from an :class:`OCRSet`, runs the VLM on each image with that
text appended to the prompt, and stores the resulting ChartData in a
:class:`VLMSet`. The set types and the offloader live in
:mod:`sci_fi_parser.data_pipeline`.
"""

from __future__ import annotations

import logging
from pathlib import Path

from sci_fi_parser.data_pipeline import ImageSet
from sci_fi_parser.vlm.vlm import build_vlm
from sci_fi_parser.vlm.vlm_config import VLMProfile, load_profile
from tqdm import tqdm

def start_vlm(image_set: ImageSet,
              profile: VLMProfile | Path) -> None:

    vlm = build_vlm(profile if isinstance(profile, VLMProfile)
                    else load_profile(profile))
    for image_id, imgage_data in tqdm(image_set.items()):
        ocr_result = image_set.get_ocrcv_result(image_id)
        imgage_path = image_set.get_image_path(image_id)
        try:
            parsed_data, raw_data = vlm.extract(imgage_path, ocr_result)
        except Exception as exc:
            logging.warning("VLM extraction failed for image_id=%s: %s", image_id, exc)
            continue
        image_set.add_vlm_result(image_id, parsed_data)
        image_set.add_vlm_result_raw(image_id, raw_data)
