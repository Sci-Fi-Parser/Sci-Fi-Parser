from typing import List

from sci_fi_parser.classifier import image_classifier
from sci_fi_parser.data_pipeline import ImageSet
from pathlib import Path
import torch

IMAGE_SUFFIXES = (".png", ".jpg")

def start_classification(
        image_set: ImageSet
) -> None:
    """
    Iterates over the image set, classifies each image and adds the result.

    """
    classifier = image_classifier.DoclingClassifier()
    image_paths = [(im_id, image_set.get_image_path(im_id)) for im_id in image_set]
    for im_id, im_path in image_paths:
        if im_path.suffix.lower() in IMAGE_SUFFIXES:
            classification, scores = classifier.classify_image(im_path)
            image_set.add_classification_result(im_id, classification)
            image_set.add_classification_raw(im_id, scores)
