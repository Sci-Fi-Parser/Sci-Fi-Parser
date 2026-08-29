"""Perceptual hashing shared by PDF extraction and image file loading.

Lives in its own module so image-only workflows do not import a PDF backend.
"""

import imagehash
from PIL import Image


def create_hash(img: Image.Image) -> str:
    """Create an image hash shared by extracted and input images."""
    return str(imagehash.average_hash(img))
