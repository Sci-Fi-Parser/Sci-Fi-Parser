import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pypdfium2 as pdfium
import pytest
from PIL import Image

from sci_fi_parser.image_extraction import image_parser

PAGE_WIDTH = 612.0
PAGE_HEIGHT = 792.0


def _fake_object(bounds, data=b"stream", container=None):
    obj = MagicMock()
    obj.get_bounds.return_value = bounds
    obj.get_data.return_value = data
    obj.container = container
    return obj


def _fake_doc(page):
    doc = MagicMock()
    doc.__len__.return_value = 1
    doc.__getitem__.return_value = page
    return doc


def _fake_page(objects):
    page = MagicMock()
    page.get_size.return_value = (PAGE_WIDTH, PAGE_HEIGHT)
    page.get_rotation.return_value = 0
    page.get_objects.return_value = objects
    return page


def _patch_render(monkeypatch, result):
    if isinstance(result, list):
        calls = iter(result)

        def render(*args, **kwargs):
            value = next(calls)
            if isinstance(value, Exception):
                raise value
            return value
    else:

        def render(*args, **kwargs):
            return result

    monkeypatch.setattr(image_parser, "_render_clip", render)


def test_start_parser_returns_pdf_metadata() -> None:
    mock_doc = MagicMock()
    mock_doc.__len__.return_value = 5
    mock_doc.__getitem__.return_value = _fake_page([])

    pdf_data, image_data = image_parser.start_parser(mock_doc, Path("/tmp/test/report.pdf"), "pdf-hash")

    assert len(pdf_data) == 1
    assert next(iter(pdf_data)) == "pdf-hash"
    meta = next(iter(pdf_data.values()))
    assert meta["file_name"] == "report.pdf"
    assert meta["page_count"] == "5"
    assert not image_data


def test_start_parser_raises_on_pathless_source():
    mock_doc = MagicMock()
    with pytest.raises(ValueError, match="no name"):
        image_parser.start_parser(mock_doc, Path(""), "pdf-hash")


def test_extract_images_adds_entry(monkeypatch) -> None:
    page = _fake_page([_fake_object((100.0, 100.0, 300.0, 300.0))])
    _patch_render(monkeypatch, Image.new("RGB", (100, 100)))

    image_data = {}
    image_parser.extract_images(_fake_doc(page), image_data, "pdf-123")

    assert len(image_data) == 1
    _, meta = next(iter(image_data.values()))
    assert meta["source_type"] == "embedded_image"
    assert meta["pdf_id"] == "pdf-123"
    assert meta["page_number"] == "1"


def test_extract_images_duplicate_stream(monkeypatch) -> None:
    duplicate = [
        _fake_object((100.0, 100.0, 300.0, 300.0), data=b"same"),
        _fake_object((400.0, 400.0, 500.0, 500.0), data=b"same"),
    ]
    page = _fake_page(duplicate)
    _patch_render(monkeypatch, Image.new("RGB", (100, 100)))

    image_data = {}
    image_parser.extract_images(_fake_doc(page), image_data, "pdf-123")

    assert len(image_data) == 1


def test_extract_images_skips_empty_render(monkeypatch) -> None:
    page = _fake_page([_fake_object((100.0, 100.0, 300.0, 300.0))])
    _patch_render(monkeypatch, None)

    image_data = {}
    image_parser.extract_images(_fake_doc(page), image_data, "pdf-123")

    assert len(image_data) == 0


@pytest.mark.parametrize(
    "error",
    [pdfium.PdfiumError("corrupt"), ValueError("Crop exceeds page dimensions")],
)
def test_extract_images_skips_failed_render(monkeypatch, caplog, error) -> None:
    page = _fake_page(
        [
            _fake_object((100.0, 100.0, 300.0, 300.0), data=b"a"),
            _fake_object((100.0, 100.0, 300.0, 300.0), data=b"b"),
        ]
    )
    _patch_render(monkeypatch, [error, Image.new("RGB", (100, 100))])

    image_data = {}
    with caplog.at_level(logging.WARNING):
        image_parser.extract_images(_fake_doc(page), image_data, "pdf-123")

    assert len(image_data) == 1
    assert "Failed" in caplog.text


def test_extract_drawings_adds_entry(monkeypatch) -> None:
    page = _fake_page([_fake_object((100.0, 100.0, 400.0, 400.0))])
    _patch_render(monkeypatch, Image.new("RGB", (100, 100)))

    image_data = {}
    image_parser.extract_drawings(_fake_doc(page), image_data, "pdf-123")

    assert len(image_data) == 1
    _, meta = next(iter(image_data.values()))
    assert meta["source_type"] == "vector_drawing"
    assert meta["pdf_id"] == "pdf-123"
    assert meta["page_number"] == "1"


def test_extract_drawings_skips_empty_render(monkeypatch) -> None:
    page = _fake_page([_fake_object((100.0, 100.0, 400.0, 400.0))])
    _patch_render(monkeypatch, None)

    image_data = {}
    image_parser.extract_drawings(_fake_doc(page), image_data, "pdf-123")

    assert len(image_data) == 0


def test_extract_drawings_skips_failed_render(monkeypatch, caplog) -> None:
    page = _fake_page(
        [
            _fake_object((100.0, 100.0, 400.0, 400.0)),
            _fake_object((500.0, 500.0, 600.0, 700.0)),
        ]
    )
    _patch_render(monkeypatch, [pdfium.PdfiumError("corrupt"), Image.new("RGB", (100, 100))])

    image_data = {}
    with caplog.at_level(logging.WARNING):
        image_parser.extract_drawings(_fake_doc(page), image_data, "pdf-123")

    assert len(image_data) == 1
    assert "Failed" in caplog.text


def test_extract_drawings_drops_hairline_clusters(monkeypatch) -> None:
    page = _fake_page([_fake_object((87.7, 698.9, 524.1, 699.6))])
    _patch_render(monkeypatch, Image.new("RGB", (100, 100)))

    image_data = {}
    image_parser.extract_drawings(_fake_doc(page), image_data, "pdf-123")

    assert len(image_data) == 0


def test_extract_drawings_drops_paths_outside_page(monkeypatch) -> None:
    page = _fake_page([_fake_object((100.0, 100.0, PAGE_WIDTH + 10.0, 400.0))])
    _patch_render(monkeypatch, Image.new("RGB", (100, 100)))

    image_data = {}
    image_parser.extract_drawings(_fake_doc(page), image_data, "pdf-123")

    assert len(image_data) == 0


def test_page_space_bounds_without_container() -> None:
    obj = _fake_object((10.0, 20.0, 30.0, 40.0))

    assert image_parser._page_space_bounds(obj) == (10.0, 20.0, 30.0, 40.0)


def test_page_space_bounds_applies_container_matrix() -> None:
    form = SimpleNamespace(
        get_matrix=lambda: SimpleNamespace(a=6.2947, b=0.0, c=0.0, d=2.3646, e=72.0, f=144.94),
        container=None,
    )
    obj = _fake_object((0.0, -4.2, 81.3, 72.0), container=form)

    left, bottom, right, top = image_parser._page_space_bounds(obj)

    assert left == pytest.approx(72.0, abs=0.01)
    assert bottom == pytest.approx(135.01, abs=0.01)
    assert right == pytest.approx(583.76, abs=0.01)
    assert top == pytest.approx(315.19, abs=0.01)


@pytest.mark.parametrize(
    ("rotation", "rendered_size", "expected"),
    [
        (0, (612.0, 792.0), (100.0, 200.0, 312.0, 292.0)),
        (90, (792.0, 612.0), (200.0, 100.0, 292.0, 312.0)),
        (180, (612.0, 792.0), (312.0, 292.0, 100.0, 200.0)),
        (270, (792.0, 612.0), (292.0, 312.0, 200.0, 100.0)),
    ],
)
def test_render_crop_rotations(rotation, rendered_size, expected) -> None:
    crop = image_parser._render_crop((100.0, 200.0, 300.0, 500.0), *rendered_size, rotation)

    assert crop == pytest.approx(expected)


def test_render_crop_clamps_negative() -> None:
    crop = image_parser._render_crop(
        (-50.0, -50.0, PAGE_WIDTH + 50.0, PAGE_HEIGHT + 50.0), PAGE_WIDTH, PAGE_HEIGHT, 0
    )

    assert crop == (0.0, 0.0, 0.0, 0.0)


def test_downsize_does_not_upsize_small_image() -> None:
    image = image_parser._downsize(Image.new("RGB", (100, 50)))

    assert isinstance(image, Image.Image)
    assert image.size == (100, 50)


def test_downsize_returns_none_for_empty_image() -> None:
    assert image_parser._downsize(Image.new("RGB", (0, 0))) is None


def test_downsize_caps_longest_side() -> None:
    image = image_parser._downsize(Image.new("RGB", (2000, 1000)))

    assert isinstance(image, Image.Image)
    assert image.size == (1000, 500)
