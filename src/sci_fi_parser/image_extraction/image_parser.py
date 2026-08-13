"""PDF image parsing primitives.

This module parses a pypdfium2 document and returns in-memory data only.
It creates one PDF metadata entry and zero or more image entries for embedded raster
images and clustered vector drawings. All IDs and metadata values are strings; the
Pillow images are kept outside metadata so later pipeline stages can save them.

The source path is passed alongside the document because ``pypdfium2.PdfDocument``
carries no file name of its own.
"""

import hashlib
import logging
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image
from pypdfium2.raw import FPDF_PAGEOBJ_IMAGE, FPDF_PAGEOBJ_PATH

from sci_fi_parser.image_extraction.hashing import create_hash
from sci_fi_parser.image_extraction.vector_clustering import Box, cluster_boxes

MAX_IMAGE_SIZE = 1000
RENDER_DPI = 300
_RENDER_SCALE = RENDER_DPI / 72

# Merge distances for vector drawing clusters.
DRAWING_X_TOLERANCE = 75.0
DRAWING_Y_TOLERANCE = 75.0
# Clusters smaller than this are page rules and table borders rather than charts.
MIN_DRAWING_WIDTH = 75.0
MIN_DRAWING_HEIGHT = 75.0
# Boxes thinner than this render to an empty bitmap.
MIN_CLIP_SIZE = 1.0
# Clustering is quadratic in the number of disjoint boxes.
MAX_DRAWING_PATHS = 20_000

# Rotation maps a (left, bottom, right, top) crop onto the rendered canvas.
_CROP_BY_ROTATION = {
    0: lambda c: c,
    90: lambda c: (c[1], c[0], c[3], c[2]),
    180: lambda c: (c[2], c[3], c[0], c[1]),
    270: lambda c: (c[3], c[2], c[1], c[0]),
}


def start_parser(
    doc: pdfium.PdfDocument,
    source_path: Path,
    pdf_id: str,
) -> tuple[dict[str, dict[str, str]], dict[str, tuple[Image.Image, dict[str, str]]]]:
    """Parse a pypdfium2 document into PDF metadata and extracted image data.

    Args:
        doc (pdfium.PdfDocument): An open pypdfium2 document to parse.
        source_path (Path): Path the document was loaded from, used for metadata.
        pdf_id (str): Content hash identifying this PDF, used as the key
            for both pdf_data and any images extracted from it.

    Returns:
        A tuple of pdf_data and image_data:
        - "pdf_data" (dict[pdf_id str, metadata dict]):
          PDF-level metadata entries keyed by content hash.
        - "image_data" (dict[image_id str, tuple(Image.Image, metadata dict)]):
          Pillow image and metadata keyed by image hash
    """
    file_name = Path(source_path).name
    if not file_name:
        raise ValueError("Source path has no name; cannot build metadata")

    pdf_data = {
        pdf_id: {
            "file_name": file_name,
            "page_count": str(len(doc)),
        }
    }
    image_data: dict[str, tuple[Image.Image, dict[str, str]]] = {}

    extract_images(doc, image_data, pdf_id)
    extract_drawings(doc, image_data, pdf_id)

    return pdf_data, image_data


def extract_images(
    doc: pdfium.PdfDocument,
    image_data: dict[str, tuple[Image.Image, dict[str, str]]],
    pdf_id: str,
) -> None:
    """Extract embedded raster images into ``image_data``.

    Each unique image stream is processed once for the whole document. The page region
    covered by the image is rendered, downsized, and stored under its image hash.
    """
    streams_seen: set[str] = set()
    for page_number, page in _pages(doc):
        page_width, page_height = page.get_size()
        rotation = page.get_rotation()

        for image_object in page.get_objects(filter=(FPDF_PAGEOBJ_IMAGE,)):
            stream_id = hashlib.sha256(bytes(image_object.get_data())).hexdigest()
            if stream_id in streams_seen:
                continue
            streams_seen.add(stream_id)

            try:
                image = _render_clip(
                    page, _page_space_bounds(image_object), page_width, page_height, rotation
                )
            except (RuntimeError, ValueError) as e:
                logging.warning(
                    "Failed to extract image stream=%s page=%s: %s", stream_id[:12], page_number, e
                )
                continue

            if image:
                image_data[create_hash(image)] = (
                    image,
                    {
                        "pdf_id": pdf_id,
                        "page_number": str(page_number),
                        "source_type": "embedded_image",
                    },
                )


def extract_drawings(
    doc: pdfium.PdfDocument,
    image_data: dict[str, tuple[Image.Image, dict[str, str]]],
    pdf_id: str,
) -> None:
    """Extract clustered vector drawings into ``image_data``.

    Path bounding boxes are collected per page, boxes not fully inside the page are
    dropped, the rest are merged within the drawing tolerances, and clusters smaller
    than the minimum drawing size are discarded. Each surviving cluster is rendered as
    a clipped page region.
    """
    for page_number, page in _pages(doc):
        page_width, page_height = page.get_size()
        rotation = page.get_rotation()

        for cluster in _drawing_clusters(page, page_width, page_height, page_number):
            try:
                image = _render_clip(page, cluster, page_width, page_height, rotation)
            except (RuntimeError, ValueError) as e:
                logging.warning("Failed to extract drawing page=%s: %s", page_number, e)
                continue

            if image:
                image_data[create_hash(image)] = (
                    image,
                    {
                        "pdf_id": pdf_id,
                        "page_number": str(page_number),
                        "source_type": "vector_drawing",
                    },
                )


def _pages(doc: pdfium.PdfDocument):
    """Yield ``(page_number, page)`` pairs, closing each page once it has been used."""
    for index in range(len(doc)):
        page = doc[index]
        try:
            yield index + 1, page
        finally:
            page.close()


def _drawing_clusters(
    page: pdfium.PdfPage,
    page_width: float,
    page_height: float,
    page_number: int,
) -> list[Box]:
    """Return the vector drawing clusters on a page worth rendering."""
    boxes = [
        box
        for box in (_page_space_bounds(obj) for obj in page.get_objects(filter=(FPDF_PAGEOBJ_PATH,)))
        if box[0] >= 0 and box[1] >= 0 and box[2] <= page_width and box[3] <= page_height
    ]

    if len(boxes) > MAX_DRAWING_PATHS:
        logging.warning(
            "Skipping drawing clustering on page %s: %s paths exceeds %s",
            page_number,
            len(boxes),
            MAX_DRAWING_PATHS,
        )
        return []

    return [
        cluster
        for cluster in cluster_boxes(boxes, DRAWING_X_TOLERANCE, DRAWING_Y_TOLERANCE)
        if cluster[2] - cluster[0] > MIN_DRAWING_WIDTH and cluster[3] - cluster[1] > MIN_DRAWING_HEIGHT
    ]


def _page_space_bounds(obj: pdfium.PdfObject) -> Box:
    """Return object bounds as ``(left, bottom, right, top)`` in unrotated page space.

    Objects nested in Form XObjects report bounds in the coordinate space of the form,
    so each enclosing form's matrix is applied to lift them into page space.
    """
    left, bottom, right, top = obj.get_bounds()
    corners = [(left, bottom), (right, bottom), (right, top), (left, top)]

    container = obj.container
    while container is not None:
        m = container.get_matrix()
        corners = [(m.a * x + m.c * y + m.e, m.b * x + m.d * y + m.f) for x, y in corners]
        container = container.container

    xs = [corner[0] for corner in corners]
    ys = [corner[1] for corner in corners]
    return min(xs), min(ys), max(xs), max(ys)


def _render_crop(box: Box, rendered_width: float, rendered_height: float, rotation: int) -> Box:
    """Convert page-space bounds into the amount to cut from each side when rendering.

    ``PdfPage.render`` takes a crop as insets from (left, bottom, right, top) rather
    than a rectangle, and applies it after rotation. Negative insets are silently
    ignored by pdfium, so they are clamped rather than allowed to shift the region.
    """
    left, bottom, right, top = box
    if rotation in (90, 270):
        page_width, page_height = rendered_height, rendered_width
    else:
        page_width, page_height = rendered_width, rendered_height

    crop = (left, bottom, page_width - right, page_height - top)
    crop = _CROP_BY_ROTATION[rotation](crop)
    return (max(0.0, crop[0]), max(0.0, crop[1]), max(0.0, crop[2]), max(0.0, crop[3]))


def _render_clip(
    page: pdfium.PdfPage,
    box: Box,
    page_width: float,
    page_height: float,
    rotation: int,
) -> Image.Image | None:
    """Render the page region covered by ``box`` and downsize it."""
    if box[2] - box[0] < MIN_CLIP_SIZE or box[3] - box[1] < MIN_CLIP_SIZE:
        return None

    crop = _render_crop(box, page_width, page_height, rotation)
    bitmap = page.render(scale=_RENDER_SCALE, crop=crop)
    return _downsize(bitmap.to_pil())


def _downsize(image: Image.Image) -> Image.Image | None:
    """Cap the longest side of an image.

    The image is resized in place with Pillow's ``thumbnail`` method, preserving aspect
    ratio and limiting both dimensions to ``MAX_IMAGE_SIZE`` pixels.
    """
    if image.width <= 0 or image.height <= 0:
        return None
    image.thumbnail((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE))
    return image


if __name__ == "__main__":
    from pprint import pprint

    pdf = Path(input("Enter PDF: "))
    with pdfium.PdfDocument(pdf) as document:
        res = start_parser(document, pdf, hashlib.sha256(pdf.read_bytes()).hexdigest())
    pprint(res)
