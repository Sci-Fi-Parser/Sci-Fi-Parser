import hashlib
from pathlib import Path

from diskcache import Cache

DEFAULT_CACHE_LOCATION = Path.cwd() / "temp"


def init_cache(cache_location=DEFAULT_CACHE_LOCATION) -> Cache:
    return Cache(cache_location)

def in_cache(pdf, cache: Cache) -> bool:
    return _hash_pdf(pdf) in cache


def add_to_cache(pdf_path: Path, cache: Cache) -> None:
    pdf_hash = _hash_pdf(pdf_path)
    cache.add(pdf_hash, pdf_path.name)


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
