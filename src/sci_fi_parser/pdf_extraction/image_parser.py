import pymupdf

from pymupdf import Document
from PIL import Image

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
            pix = pymupdf.Pixmap(doc, xref)
            pix.save(f"pymupdf_outputs/{xref}.png")

def extract_drawings(doc: Document):
    """Saves vector graphics in PDF files as PNG images.

    Args:
        doc (Document): Document object
    """
    MAX_SIZE = 1000
    i = 0 # index naming is temporary
    for page in doc:
        for drawing in page.cluster_drawings(x_tolerance=75, y_tolerance=75):
            pix = page.get_pixmap(dpi=300, clip=drawing,)

            mode = "RGBA" if pix.alpha else "RGB"
            img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
            img.thumbnail((MAX_SIZE, MAX_SIZE))
            img.save(f"pymupdf_outputs/vector-{i}.png")
            i += 1

if __name__ == "__main__":
    pdf = input("Enter PDF: ")
    document = pymupdf.open(pdf)
    extract_drawings(document)
    extract_images(document)
