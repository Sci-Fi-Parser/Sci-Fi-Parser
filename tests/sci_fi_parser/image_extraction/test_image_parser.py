import logging
from unittest.mock import MagicMock

import pytest
from PIL import Image
from pymupdf import IRect, Pixmap, csRGB

from sci_fi_parser.image_extraction import image_parser


def test_start_parser_returns_pdf_metadata() -> None:
    mock_doc = MagicMock()
    mock_doc.name = "/tmp/test/report.pdf"
    mock_doc.page_count = 5
    mock_doc.__iter__ = lambda s: iter([])

    pdf_data, image_data = image_parser.start_parser(mock_doc)

    assert len(pdf_data) == 1
    meta = next(iter(pdf_data.values()))
    assert meta["file_name"] == "report.pdf"
    assert meta["page_count"] == "5"
    assert not image_data


def test_start_parser_raises_on_empty_doc_name():
    mock_doc = MagicMock()
    mock_doc.name = ""
    with pytest.raises(ValueError, match="no name"):
        image_parser.start_parser(mock_doc)


def test_extract_images_adds_entry() -> None:
    mock_page = MagicMock()
    mock_page.number = 0
    mock_page.get_images.return_value = [(42, ...)]
    mock_page.get_image_rects.return_value = [MagicMock()]

    mock_pix = MagicMock(width=100, height=100)
    mock_pix.pil_image.return_value = Image.new("RGB", (100, 100))
    mock_page.get_pixmap.return_value = mock_pix

    mock_doc = MagicMock()
    mock_doc.pages.return_value = [mock_page]

    image_data = {}
    image_parser.extract_images(mock_doc, image_data, "pdf-123")
    assert len(image_data) == 1
    _, meta = next(iter(image_data.values()))
    assert meta["source_type"] == "embedded_image"
    assert meta["pdf_id"] == "pdf-123"


def test_extract_images_duplicate_xref() -> None:
    mock_page = MagicMock()
    mock_page.number = 0
    mock_page.get_images.return_value = [(42,), (42,)]  # duplicate xref
    mock_page.get_image_rects.return_value = [MagicMock()]

    mock_pix = MagicMock(width=100, height=100)
    mock_pix.pil_image.return_value = Image.new("RGB", (100, 100))
    mock_page.get_pixmap.return_value = mock_pix

    mock_doc = MagicMock()
    mock_doc.pages.return_value = [mock_page]

    image_data = {}
    image_parser.extract_images(mock_doc, image_data, "pdf-123")
    assert len(image_data) == 1


def test_extract_images_skips_empty_pixmap() -> None:
    mock_page = MagicMock()
    mock_page.number = 0
    mock_page.get_images.return_value = [(42,)]
    mock_page.get_image_rects.return_value = [MagicMock()]

    mock_pix = MagicMock(width=0, height=0)  # _downsize returns None
    mock_pix.pil_image.return_value = Image.new("RGB", (0, 0))
    mock_page.get_pixmap.return_value = mock_pix

    mock_doc = MagicMock()
    mock_doc.pages.return_value = [mock_page]

    image_data = {}
    image_parser.extract_images(mock_doc, image_data, "pdf-123")
    assert len(image_data) == 0


def test_extract_images_skips_failed_pixmap(caplog) -> None:
    mock_page = MagicMock()
    mock_page.number = 0
    mock_page.get_images.return_value = [(42,), (67,)]
    mock_page.get_image_rects.return_value = [MagicMock()]

    good_pix = MagicMock(width=100, height=100)
    good_pix.pil_image.return_value = Image.new("RGB", (100, 100))
    mock_page.get_pixmap.side_effect = [RuntimeError("corrupt"), good_pix]

    mock_doc = MagicMock()
    mock_doc.pages.return_value = [mock_page]

    image_data = {}

    with caplog.at_level(logging.WARNING):
        image_parser.extract_images(mock_doc, image_data, "pdf-123")

    assert len(image_data) == 1
    assert "Failed" in caplog.text


def test_extract_drawings_adds_entry() -> None:
    mock_page = MagicMock()
    mock_page.number = 0
    mock_page.cluster_drawings.return_value = [(42, ...)]

    mock_pix = MagicMock(width=100, height=100)
    mock_pix.pil_image.return_value = Image.new("RGB", (100, 100))
    mock_page.get_pixmap.return_value = mock_pix

    mock_doc = MagicMock()
    mock_doc.pages.return_value = [mock_page]

    image_data = {}
    image_parser.extract_drawings(mock_doc, image_data, "pdf-123")
    assert len(image_data) == 1
    _, meta = next(iter(image_data.values()))
    assert meta["source_type"] == "vector_drawing"
    assert meta["pdf_id"] == "pdf-123"


def test_extract_drawings_skips_empty_pixmap() -> None:
    mock_page = MagicMock()
    mock_page.number = 0
    mock_page.cluster_drawings.return_value = [(42, ...)]

    mock_pix = MagicMock(width=0, height=0)  # _downsize returns None
    mock_pix.pil_image.return_value = Image.new("RGB", (100, 100))
    mock_page.get_pixmap.return_value = mock_pix

    mock_doc = MagicMock()
    mock_doc.pages.return_value = [mock_page]

    image_data = {}
    image_parser.extract_drawings(mock_doc, image_data, "pdf-123")
    assert len(image_data) == 0


def test_extract_drawings_skips_failed_pixmap(caplog) -> None:
    mock_page = MagicMock()
    mock_page.number = 0
    mock_page.cluster_drawings.return_value = [(42,), (67,)]

    good_pix = MagicMock(width=100, height=100)
    good_pix.pil_image.return_value = Image.new("RGB", (100, 100))
    mock_page.get_pixmap.side_effect = [RuntimeError("corrupt"), good_pix]

    mock_doc = MagicMock()
    mock_doc.pages.return_value = [mock_page]

    image_data = {}

    with caplog.at_level(logging.WARNING):
        image_parser.extract_drawings(mock_doc, image_data, "pdf-123")

    assert len(image_data) == 1
    assert "Failed" in caplog.text


def test_downsize_does_not_upsize_small_image() -> None:
    pixmap = Pixmap(csRGB, IRect(p0=(0, 0), p1=(100, 50)))

    image = image_parser._downsize(pixmap)

    assert isinstance(image, Image.Image)
    assert image.size == (100, 50)


def test_downsize_returns_none_for_empty_pixmap() -> None:
    pixmap = Pixmap(csRGB, IRect(p0=(0, 0), p1=(0, 0)))

    assert image_parser._downsize(pixmap) is None


def test_downsize_caps_longest_side() -> None:
    pixmap = Pixmap(csRGB, IRect(p0=(0, 0), p1=(2000, 1000)))

    image = image_parser._downsize(pixmap)

    assert isinstance(image, Image.Image)
    assert image.size == (1000, 500)
