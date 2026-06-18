from __future__ import annotations

from pathlib import Path

from sci_fi_parser.data_pipeline import ImageSet
from sci_fi_parser.image_extraction.image_parser import create_image_id


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def load_image_file(
    image_path: Path,
    image_set: ImageSet,
) -> None:
    """Add one existing image file to an ImageSet"""
    if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError(f"Input path must be an image file: {image_path}")

    metadata = {
        "extracted": False,
        "source_type": "input_image",
    }

    image_id = create_image_id()
    image_set.add_extracted_image(image_id, image_path, extraction_metadata=metadata)

def load_images_from_folder(
    folder_path: Path,
    image_set: ImageSet,
) -> None:
    """Load all supported image files from a folder into an ImageSet."""
    if not folder_path.is_dir():
        raise ValueError(f"Input path must be a directory: {folder_path}")

    for image_path in sorted(folder_path.iterdir()):
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES:
            load_image_file(
                image_path=image_path,
                image_set=image_set,
            )
