from pathlib import Path

from PIL import Image

from sci_fi_parser.image_extraction import extraction_pipeline
from sci_fi_parser.schema import ImageSet, PdfSet


class DummyDocument:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_start_extraction_writes_images_and_metadata(tmp_path, monkeypatch):
    pdf_path = tmp_path / "source.pdf"
    pdf_path.write_bytes(b"%PDF-1.7")
    output_path = tmp_path / "images"

    def fake_open(path: Path):
        assert path == pdf_path
        return DummyDocument()

    def fake_start_parser(doc: DummyDocument, hash: str):
        image = Image.new("RGB", (1, 1), "white")
        pdf_data = {"pdf-1": {"title": "source"}}
        image_data = {"image-1": (image, {"pdf_id": "pdf-1"})}
        return pdf_data, image_data

    image_set = ImageSet()
    pdf_set = PdfSet()
    monkeypatch.setattr(extraction_pipeline.pymupdf, "open", fake_open)
    monkeypatch.setattr(extraction_pipeline, "start_parser", fake_start_parser)

    extraction_pipeline.start_extraction(pdf_path, image_set, pdf_set, output_path)

    saved_image = output_path / "image-1.png"
    assert saved_image.exists()
    image_record = image_set.get("image-1")
    assert image_record["metadata"]["extraction"] == {"pdf_id": "pdf-1"}
    assert image_record["output"]["path"] == saved_image
    assert pdf_set.get("pdf-1") == {"metadata": {"title": "source"}}


def test_find_pdfs_accepts_directory(tmp_path):
    first = tmp_path / "a.pdf"
    second = tmp_path / "b.PDF"
    ignored = tmp_path / "notes.txt"
    first.touch()
    second.touch()
    ignored.touch()

    assert extraction_pipeline._find_pdfs(tmp_path) == [first, second]
