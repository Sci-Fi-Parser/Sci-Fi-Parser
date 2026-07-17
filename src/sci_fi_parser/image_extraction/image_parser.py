"""PDF image parsing primitives.

This module parses a PyMuPDF document and returns in-memory data only.
It creates one PDF metadata entry and zero or more image entries for embedded raster
images and clustered vector drawings. All IDs and metadata values are strings; the
Pillow images are kept outside metadata so later pipeline stages can save them.
"""

import logging
from pathlib import Path
from typing import cast
from uuid import uuid4

import pymupdf
from PIL import Image

MAX_IMAGE_SIZE = 1000


def create_image_id() -> str:
    """Create an image id shared by extracted and input images."""
    return str(uuid4())


def start_parser(
    doc: pymupdf.Document,
) -> tuple[dict[str, dict[str, str]], dict[str, tuple[Image.Image, dict[str, str]]]]:
    """Parse a PyMuPDF document into PDF metadata and extracted image data.

    Args:
        doc (pymupdf.Document): An open PyMuPDF Document to parse.

    Returns:
        A tuple of pdf_data and image_data:
        - "pdf_data" (dict[pdf_id str, metadata dict]):
          PDF-level metadata entries keyed by string of UUID.
        - "image_data" (dict[image_id str, tuple(Image.Image, metadata dict)]):
          Pillow image and metadata keyed by UUID string
    """
    if not doc.name:
        raise ValueError("Document has no name; cannot build metadata")

    pdf_id = str(uuid4())
    pdf_data = {
        pdf_id: {
            "file_name": Path(doc.name).name,
            "page_count": str(doc.page_count),
        }
    }
    image_data: dict[str, tuple[Image.Image, dict[str, str]]] = {}

    extract_images(doc, image_data, pdf_id)
    extract_drawings(doc, image_data, pdf_id)

    return pdf_data, image_data


def extract_images(
    doc: pymupdf.Document,
    image_data: dict[str, tuple[Image.Image, dict[str, str]]],
    pdf_id: str,
) -> None:
    """Extract embedded raster images into ``image_data``.

    Each unique image xref is processed once for the whole document. For every image
    placement rectangle on the page, the clipped page region is rendered, downsized,
    and stored under a generated string image ID.
    """
    xref_seen = set()
    for page in doc.pages():
        for img in page.get_images():
            xref = img[0]
            if xref in xref_seen:
                continue
            xref_seen.add(xref)
            for rect in page.get_image_rects(xref):
                try:
                    pix = page.get_pixmap(dpi=300, clip=rect)
                    downsized_img = _downsize(pix)
                except (pymupdf.mupdf.FzErrorGeneric, RuntimeError) as e:
                    logging.warning("Failed to extract image xref=%s page=%s: %s", xref, page, e)
                    continue

                if downsized_img:
                    image_id = create_image_id()
                    image_data[image_id] = (
                        downsized_img,
                        {
                            "pdf_id": pdf_id,
                            "page_number": str(page.number + 1),
                            "source_type": "embedded_image",
                        },
                    )


def extract_drawings(
    doc: pymupdf.Document,
    image_data: dict[str, tuple[Image.Image, dict[str, str]]],
    pdf_id: str,
) -> None:
    """Extract clustered vector drawings into ``image_data``.

    PyMuPDF clusters page drawing commands with the current tolerance values and each
    cluster is rendered as a clipped pixmap. The stored metadata mirrors embedded image
    metadata style
    """
    for page in doc.pages():
        for drawing in page.cluster_drawings(x_tolerance=75, y_tolerance=75):
            try:
                pix = page.get_pixmap(
                    dpi=300,
                    clip=drawing,
                )
                img = _downsize(pix)
            except (pymupdf.mupdf.FzErrorGeneric, RuntimeError) as e:
                logging.warning("Failed to extract drawing page= %s: %s", page, e)
                continue

            if img:
                image_id = create_image_id()
                image_data[image_id] = (
                    img,
                    {
                        "pdf_id": pdf_id,
                        "page_number": str(page.number + 1),
                        "source_type": "vector_drawing",
                    },
                )


def _downsize(pix: pymupdf.Pixmap) -> Image.Image | None:
    """Convert a pixmap to a Pillow image and cap its longest side.

    The image is resized in place with Pillow's ``thumbnail`` method, preserving aspect
    ratio and limiting both dimensions to ``MAX_IMAGE_SIZE`` pixels.
    """
    if pix.width <= 0 or pix.height <= 0:
        return None
    img = cast(Image.Image, pix.pil_image())
    img.thumbnail((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE))
    return img


if __name__ == "__main__":
    from pprint import pprint

    pdf = input("Enter PDF: ")
    document = pymupdf.open(pdf)
    res = start_parser(document)
    pprint(res)
