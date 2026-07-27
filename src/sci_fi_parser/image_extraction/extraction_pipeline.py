"""Image extraction pipeline stage.

This module bridges PDF parsing with the rest of the application pipeline. It accepts
one PDF or a directory of PDFs, calls ``start_parser`` for each document, saves the
returned Pillow images into the explicitly provided extraction folder, and records only
string metadata in the supplied image and PDF metadata sets.

The extraction folder is temporary working storage for later pipeline stages. Image paths
are intentionally not written into metadata because this folder is not permanent
storage.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pymupdf
from diskcache import Cache
from tqdm import tqdm

from sci_fi_parser.image_extraction.image_parser import start_parser
from sci_fi_parser.schema import ImageSet, PdfSet

DEFAULT_CACHE_DIR = "temp"


def start_extraction(
    input_path: Path,
    image_set: ImageSet,
    pdf_set: PdfSet,
    extracted_image_folder: Path | None,
) -> None:
    """Extract images and metadata from one PDF file or a flat PDF directory.

    Args:
        input_path (Path): input of a pdf or a folder of pdfs
        image_set (dict): set where image data is stored
        pdf_set (dict): set where pdf data is stored
        extracted_image_folder (Path | None): Path where images are written,
        if None then not writing the images into folders.
    """
    pdf_paths = _find_pdfs(input_path)
    if extracted_image_folder:
        extracted_image_folder.mkdir(parents=True, exist_ok=True)

    for pdf_path in tqdm(pdf_paths):
        with Cache(DEFAULT_CACHE_DIR) as pdf_cache:
            if is_duplicate(pdf_path, pdf_cache):
                continue

        with pymupdf.open(pdf_path) as doc:
            pdf_data, image_data = start_parser(doc)

        for pdf_id, pdf_metadata in pdf_data.items():
            pdf_set.add(pdf_id, {"metadata": pdf_metadata})

        for image_id, image_payload in image_data.items():
            image, img_metadata = image_payload
            if image:
                image_path = None
                if extracted_image_folder:
                    image_path = extracted_image_folder / f"{image_id}.png"
                    image.save(image_path)
                image_set.add_extracted_image(
                    image_id,
                    image_path=image_path or Path(f"{image_id}.png"),
                    extraction_metadata=img_metadata,
                )


def is_duplicate(pdf_path: Path, cache: Cache) -> bool:
    """Check the on-disk cache for the hash of a PDF.

    Args:
        pdf_path (Path): Path to a PDF.
        cache (Cache): Initialized cache.

    Returns:
        True if PDF is in cache. False otherwise.
    """
    pdf_hash = _hash_pdf(pdf_path)
    with cache as conn:
        if pdf_hash in conn:
            return True
        conn.add(pdf_hash, pdf_path.name)
    return False


def _hash_pdf(pdf: Path) -> str:
    """Create a hash of a PDF. Hash is constructed from the contents of the PDF.

    Args:
        pdf (Path): Path to a PDF.

    Returns:
        String of the PDF contents in hexadecimal
    """
    with open(pdf, "rb") as f:
        file_contents = f.read()
    hash = hashlib.sha256(file_contents).hexdigest()
    return hash


def _find_pdfs(input_path: Path) -> list[Path]:
    """Return PDF files to process from a file or one directory level.

    A PDF file input returns a one-item list. A directory input returns sorted direct
    child files with a case-insensitive ``.pdf`` suffix. Nested directories are not
    scanned in the current pipeline.
    """
    if input_path.is_file() and input_path.suffix.lower() == ".pdf":
        return [input_path]

    if input_path.is_dir():
        return sorted(
            path for path in input_path.iterdir() if path.is_file() and path.suffix.lower() == ".pdf"
        )

    raise ValueError(f"Input path must be a PDF or directory of PDFs: {input_path}")
