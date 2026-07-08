from __future__ import annotations

import logging
from pathlib import Path

from tqdm import tqdm

from sci_fi_parser.schema import ImageSet
from sci_fi_parser.vlm.vlm import ChatCompletionsVLM
from sci_fi_parser.vlm.vlm_config import VLMProfile, load_profile


def start_vlm(image_set: ImageSet, profile: VLMProfile | Path) -> None:

    if not isinstance(profile, VLMProfile):
        profile = load_profile(profile)
    vlm = ChatCompletionsVLM(profile)
    for image_id, _ in tqdm(image_set.items()):
        ocr_result = image_set.get_ocrcv_result(image_id)
        imgage_path = image_set.get_image_path(image_id)
        try:
            parsed_data, raw_data = vlm.extract(imgage_path, ocr_result)
        except Exception as exc:
            logging.warning("VLM extraction failed for image_id=%s: %s", image_id, exc)
            continue
        image_set.add_vlm_result(image_id, parsed_data)
        image_set.add_vlm_result_raw(image_id, raw_data)
