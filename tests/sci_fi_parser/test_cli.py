from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sci_fi_parser.cli import cli_clear_cache, cli_parse_folder


@patch("sci_fi_parser.cli.parse_folder")
@patch("sys.argv", ["scifi-parser", "train_data/small_pdfs"])
def test_cli_defaults(mock_parse_folder):
    mock_parse_folder.return_value.summary.return_value = {
        "pdf_count": 2,
        "image_count": 10,
    }

    cli_parse_folder()

    mock_parse_folder.assert_called_once_with(
        Path("train_data/small_pdfs"),
        output_dir="output",
        classify=True,
        ocr=True,
        vlm=True,
    )


@patch("sci_fi_parser.cli.parse_folder")
@patch(
    "sys.argv",
    [
        "scifi-parser",
        "train_data/small_pdfs",
        "--output",
        "results",
    ],
)
def test_cli_custom_output(mock_parse_folder):
    mock_parse_folder.return_value.summary.return_value = {}

    cli_parse_folder()

    mock_parse_folder.assert_called_once_with(
        Path("train_data/small_pdfs"),
        output_dir="results",
        classify=True,
        ocr=True,
        vlm=True,
    )


@patch("sci_fi_parser.cli.parse_folder")
@patch(
    "sys.argv",
    [
        "scifi-parser",
        "train_data/small_pdfs",
        "--no-classify",
        "--no-ocr",
        "--no-vlm",
    ],
)
def test_cli_pipeline_flags(mock_parse_folder):
    mock_parse_folder.return_value.summary.return_value = {}

    cli_parse_folder()

    mock_parse_folder.assert_called_once_with(
        Path("train_data/small_pdfs"),
        output_dir="output",
        classify=False,
        ocr=False,
        vlm=False,
    )


@patch("sys.argv", ["scifi-parser", "--help"])
def test_cli_help():
    with pytest.raises(SystemExit):
        cli_parse_folder()


@patch("sci_fi_parser.cli.Cache")
@patch("sys.argv", [])
def test_cli_clear_cache_clears_cache(mock_cache):
    conn = MagicMock()
    mock_cache.return_value.__enter__.return_value = conn

    cli_clear_cache()

    mock_cache.assert_called_once_with("temp")
    conn.clear.assert_called_once()
