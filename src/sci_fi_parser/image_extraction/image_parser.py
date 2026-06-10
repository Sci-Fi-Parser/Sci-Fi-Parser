from uuid import uuid4, UUID
from dataclasses import dataclass, field

import pymupdf

from PIL import Image


MAX_IMAGE_SIZE = 1000


@dataclass
class DocumentMetadata:
    uid: UUID
    file_name: str
    page_count: int
    metadata: dict = field(default_factory=dict)


@dataclass
class ExtractedImage:
    page_number: int
    image: Image.Image


@dataclass
class ParserResult:
    pdf_data: DocumentMetadata
    image_data: dict[UUID, ExtractedImage]


def start_parser(doc: pymupdf.Document) -> ParserResult:
    """Parse a PyMuPDF Document to collect PDF-level metadata and extract images.

    Creates a mapping containing PDF metadata and extracted images (both embedded raster images
    and clustered vector drawings). Image extraction is performed via extract_images and
    extract_drawings, which populate the image_data mapping with UUID keys.

    Args:
        doc (pymupdf.Document): An open PyMuPDF Document to parse.

    Returns:
        ParserResult: A mapping with two keys:
        - "pdf_data" (DocumentMetadata): PDF-level metadata entries keyed by UUID.
        Each value is a mapping such as DocumentMetadata.page_count.
        - "image_data" (dict[UUID, ExtractedImage]): Extracted images keyed by UUID.
        Each value is an ExtractedImage with attributes .image and .page_number (1-based). For example image_data[uid].image
    """
    image_data: dict[UUID, ExtractedImage] = {}
    extract_images(doc, image_data)
    extract_drawings(doc, image_data)
 
    return ParserResult(
        pdf_data=extract_document_metadata(doc),
        image_data=image_data,
    )


def extract_document_metadata(doc: pymupdf.Document) -> DocumentMetadata:
    """Extract basic document-level metadata from a PyMuPDF Document.

    Generates a UUID for this document and collects the document's file name,
    page count, and the PDF Info/XMP metadata (if present).

    Note that this function does not modify `doc`.

    Args:
        doc (pymupdf.Document): An open PyMuPDF Document to extract metadata from.

    Returns:
        DocumentMetadata: Dataclass with fields:
        - uid (UUID): generated identifier for this document.
        - file_name (str | None): value of doc.name (may be None or a path).
        - page_count (int): total number of pages.
        - metadata (dict): PDF Info/XMP metadata mapping (empty if unavailable).
    """
    return DocumentMetadata(
        uid=uuid4(),
        file_name=doc.name,
        page_count=doc.page_count,
        metadata=doc.metadata or {},
    )


def extract_images(doc: pymupdf.Document, image_data: dict) -> None:
    """Extract embedded images from a document and store them in a provided mapping.

    Args:
        doc (pymupdf.Document): An open PyMuPDF `Document` object to extract images from.
        image_data (dict[UUID, Image]): Mutable mapping populated in-place. Keys are UUIDs for extracted images.
        Values are `ExtractedImage` types where page_number is 1-based and image is a Pillow Image object.
    """
    xref_seen = set()
    for page in doc:
        for img in page.get_images():
            xref = img[0]
            if xref in xref_seen:
                continue
            xref_seen.add(xref)
            for rect in page.get_image_rects(xref):
                pix = page.get_pixmap(dpi=300, clip=rect)
                img = _downsize(pix)

                img_uid = uuid4()
                true_page_number = page.number + 1

                image_data[img_uid] = ExtractedImage(page_number=true_page_number, image=img)


def extract_drawings(doc: pymupdf.Document, image_data: dict) -> None:
    """Cluster and extract vector graphics from a document and store them in a provided mapping.

    Args:
        doc (pymupdf.Document): An open PyMuPDF `Document` object to extract drawings from.
        image_data (dict[UUID, Image]): Mutable mapping populated in-place. Keys are UUIDs for extracted images.
        Values are `ExtractedImage` types where page_number is 1-based and image is a Pillow Image object.
    """
    for page in doc:
        for drawing in page.cluster_drawings(x_tolerance=75, y_tolerance=75):
            pix = page.get_pixmap(dpi=300, clip=drawing,)
            img = _downsize(pix)

            img_uid = uuid4()
            true_page_number = page.number + 1

            image_data[img_uid] = ExtractedImage(page_number=true_page_number, image=img)

def _downsize(pix: pymupdf.Pixmap) -> Image:
    """Takes a pixmap object and downsizes it so that the longer side is equal to the
    length of MAX_SIZE. Preserves the aspect ratio of the image.

    Args:
        pix (pymupdf.Pixmap): Pixmap object of an image/drawing from a document.

    Returns:
        Image: Pillow Image object.
    """
    img = pix.pil_image()
    img.thumbnail((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE))
    return img


if __name__ == "__main__":
    from pprint import pprint
    pdf = input("Enter PDF: ")
    document = pymupdf.open(pdf)
    res = start_parser(document)
    pprint(res)
