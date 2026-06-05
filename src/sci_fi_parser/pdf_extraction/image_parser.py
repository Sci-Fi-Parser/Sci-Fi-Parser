import pymupdf

from pymupdf import Document, Pixmap
from PIL import Image


MAX_SIZE = 1000

def extract_images(doc: Document):
    """Saves images in PDF files as PNG images.

    Args:
        doc (Document): Document object
    """
    xref_seen = set()
    for page in doc:
        for img in page.get_images():
            xref = img[0]
            if xref in xref_seen:
                pass
            xref_seen.add(xref)
            for index, rect in enumerate(page.get_image_rects(xref)):
                pix = page.get_pixmap(dpi=300, clip=rect)
                img = resize(pix)
                img.save(f"pymupdf_outputs/{xref}-{index}.png")

def extract_drawings(doc: Document):
    """Saves vector graphics in PDF files as PNG images.

    Args:
        doc (Document): Document object
    """
    i = 0 # index naming is temporary
    for page in doc:
        for drawing in page.cluster_drawings(x_tolerance=75, y_tolerance=75):
            pix = page.get_pixmap(dpi=300, clip=drawing,)
            img = resize(pix)
            img.save(f"pymupdf_outputs/vector-{i}.png")
            i += 1

def resize(pix: Pixmap) -> Image:
    img = pix.pil_image()
    img.thumbnail((MAX_SIZE, MAX_SIZE))
    return img

if __name__ == "__main__":
    pdf = input("Enter PDF: ")
    document = pymupdf.open(pdf)
    extract_drawings(document)
    extract_images(document)
