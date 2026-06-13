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
    TODO
    """
    classifier = image_classifier.ImageClassifier()
    image_paths = [(im_id, image_set.get_image(im_id)) for im_id in image_set]
    for im_id, im_path in image_paths:
        classification, scores = classifier.classify_image()
