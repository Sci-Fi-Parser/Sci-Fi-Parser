from __future__ import annotations

import logging
from pathlib import Path

from tqdm import tqdm

from sci_fi_parser.schema import ImageSet
from sci_fi_parser.vlm.vlm import ChatCompletionsVLM
from sci_fi_parser.vlm.vlm_config import VLMProfile, load_profile

_SUPPORTED_CHARTS = ["bar_chart"]


def start_vlm(image_set: ImageSet, profile: VLMProfile | Path) -> None:
    # TODO: DOCSTRING
    if not isinstance(profile, VLMProfile):
        profile = load_profile(profile)
    vlm = ChatCompletionsVLM(profile)
    image_ids = image_set.filter_by_type(_SUPPORTED_CHARTS)
    for image_id in tqdm(image_ids):
        if image_set.get_classification_result(image_id) not in _SUPPORTED_CHARTS:
            continue
        ocr_result = image_set.get_ocrcv_result(image_id)
        image_path = image_set.get_image_path(image_id)
        try:
            parsed_data, raw_data = vlm.extract(image_path, ocr_result)
        except Exception as exc:
            logging.warning("VLM extraction failed for image_id=%s: %s", image_id, exc)
            continue
        image_set.add_vlm_result(image_id, parsed_data)
        image_set.add_vlm_result_raw(image_id, raw_data)
