from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from sci_fi_parser.api import ParseResult, parse_folder
from sci_fi_parser.schema import ImageSet, PdfSet


def test_summary():
    image_set = ImageSet()
    pdf_set = PdfSet()

    image_set.add("img1", {})
    image_set.add("img2", {})

    pdf_set.add("pdf1", {})

    result = ParseResult(image_set, pdf_set)

    assert result.summary() == {
        "pdf_count": 1,
        "image_count": 2,
    }


@patch("sci_fi_parser.api.start_vlm")
@patch("sci_fi_parser.api.start_ocr")
@patch("sci_fi_parser.api.start_classification")
@patch("sci_fi_parser.api.start_extraction")
@patch("sci_fi_parser.api.Path.iterdir")
@patch("sci_fi_parser.api.Ocr")
@patch("sci_fi_parser.api.in_cache")
@patch("sci_fi_parser.api.add_to_cache")
def test_parse_folder_returns_parse_result(
    mock_add_to_cache,
    mock_in_cache,
    mock_ocr_inst,
    mock_iterdir,
    mock_extraction,
    mock_classification,
    mock_ocr,
    mock_vlm,
):
    mock_iterdir.return_value = [Path("file1.pdf")]
    mock_in_cache.return_value = False

    result = parse_folder("dummy_folder")

    assert isinstance(result, ParseResult)
    mock_extraction.assert_called()
    assert isinstance(result, ParseResult)

    mock_extraction.assert_called_once()
    mock_classification.assert_called_once()
    mock_ocr.assert_called_once()
    mock_vlm.assert_called_once()


@patch("sci_fi_parser.api.start_vlm")
@patch("sci_fi_parser.api.start_ocr")
@patch("sci_fi_parser.api.start_classification")
@patch("sci_fi_parser.api.start_extraction")
@patch("sci_fi_parser.api.Path.iterdir")
@patch("sci_fi_parser.api.Ocr")
@patch("sci_fi_parser.api.in_cache")
@patch("sci_fi_parser.api.add_to_cache")
def test_parse_folder_respects_pipeline_flags(
    mock_add_to_cache,
    mock_in_cache,
    mock_ocr_inst,
    mock_iterdir,
    mock_extraction,
    mock_classification,
    mock_ocr,
    mock_vlm,
):
    mock_iterdir.return_value = [Path("file1.pdf")]
    mock_in_cache.return_value = False

    parse_folder(
        "dummy_folder",
        classify=False,
        ocr=False,
        vlm=False,
    )

    mock_extraction.assert_called_once()

    mock_classification.assert_not_called()
    mock_ocr.assert_not_called()
    mock_vlm.assert_not_called()


@patch("sci_fi_parser.api.save_image_set")
def test_save_updates_output_dir(
    mock_save,
    tmp_path,
):
    result = ParseResult(
        image_set=ImageSet(),
        pdf_set=PdfSet(),
    )

    result.save(tmp_path)

    mock_save.assert_called_once()

    assert result.output_dir == tmp_path


def test_points_dataframe(tmp_path):
    tables_dir = tmp_path / "tables"
    tables_dir.mkdir()

    df = pd.DataFrame(
        {
            "x": [1, 2],
            "y": [3, 4],
        }
    )

    df.to_parquet(tables_dir / "points.parquet")

    result = ParseResult(
        image_set=ImageSet(),
        pdf_set=PdfSet(),
        output_dir=tmp_path,
    )

    loaded = result.points_dataframe()

    pd.testing.assert_frame_equal(loaded, df)


def test_series_dataframe(tmp_path):
    tables_dir = tmp_path / "tables"
    tables_dir.mkdir()

    df = pd.DataFrame(
        {
            "series_id": [1],
            "name": ["test"],
        }
    )

    df.to_parquet(tables_dir / "series.parquet")

    result = ParseResult(
        image_set=ImageSet(),
        pdf_set=PdfSet(),
        output_dir=tmp_path,
    )

    loaded = result.series_dataframe()

    pd.testing.assert_frame_equal(loaded, df)


def test_charts_dataframe(tmp_path):
    tables_dir = tmp_path / "tables"
    tables_dir.mkdir()

    df = pd.DataFrame(
        {
            "chart_id": ["chart_1"],
            "chart_type": ["line"],
        }
    )

    df.to_parquet(tables_dir / "charts.parquet")

    result = ParseResult(
        image_set=ImageSet(),
        pdf_set=PdfSet(),
        output_dir=tmp_path,
    )

    loaded = result.charts_dataframe()

    pd.testing.assert_frame_equal(loaded, df)


def test_dataframe_methods_raise_without_output_dir():
    result = ParseResult(
        image_set=ImageSet(),
        pdf_set=PdfSet(),
    )

    with pytest.raises(ValueError):
        result.charts_dataframe()

    with pytest.raises(ValueError):
        result.series_dataframe()

    with pytest.raises(ValueError):
        result.points_dataframe()
