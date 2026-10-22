import json
from pathlib import Path

import pandas as pd
import pytest

from sci_fi_parser.storage.writer import (
    _infer_x_type,
    _json_default,
    _safe_float,
    _safe_int,
    save_image_set,
    save_pdf_set,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    return tmp_path / "output"


@pytest.fixture
def make_image_record():
    """Factory for a minimal, valid image record."""

    def _make(
        pdf_id: str = "pdf-1",
        page_number: int = 1,
        chart_type: str = "line",
        series: list[dict] | None = None,
        image_path: str | None = "img.png",
    ) -> dict:
        return {
            "metadata": {
                "extraction": {"pdf_id": pdf_id, "page_number": page_number, "source_type": "raster"},
                "classification": {},
                "ocrcv": {},
                "vlm": {},
            },
            "output": {"path": image_path},
            "classification": {"result": chart_type, "raw": {}},
            "ocrcv": {"result": "", "raw": {}},
            "vlm": {
                "result": {"chart_type": chart_type, "series": series or []},
                "raw": {"model": "test-model", "created_at": "2026-01-01"},
            },
        }

    return _make


@pytest.fixture
def make_pdf_record():
    """Factory for a minimal, valid pdf record."""

    def _make(file_name: str = "Letter.pdf", page_count: str = "11") -> dict:
        return {"metadata": {"file_name": file_name, "page_count": page_count}}

    return _make


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# _json_default
# ---------------------------------------------------------------------------


class _WithModelDump:
    def model_dump(self):
        return {"a": 1}


class _WithTolist:
    def tolist(self):
        return [1, 2, 3]


@pytest.mark.parametrize(
    "obj, expected",
    [
        (_WithModelDump(), {"a": 1}),
        (_WithTolist(), [1, 2, 3]),
        (Path("some/path.png"), "some/path.png"),
        (object(), None),  # handled separately below, see test_json_default_fallback
    ],
)
def test_json_default(obj, expected):
    if expected is None:
        assert isinstance(_json_default(obj), str)
    else:
        assert _json_default(obj) == expected


# ---------------------------------------------------------------------------
# helper coercion functions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [("3.5", 3.5), (3, 3.0), (None, None), ("not-a-number", None)],
)
def test_safe_float(value, expected):
    assert _safe_float(value) == expected


@pytest.mark.parametrize(
    "value, expected",
    [("3", 3), (3.9, 3), (None, None), ("not-a-number", None)],
)
def test_safe_int(value, expected):
    assert _safe_int(value) == expected


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, "unknown"),
        ("2024", "year"),
        ("3.14", "numeric"),
        ("Q1", "category"),
        (2023, "year"),
    ],
)
def test_infer_x_type(value, expected):
    assert _infer_x_type(value) == expected


# ---------------------------------------------------------------------------
# save_image_set
# ---------------------------------------------------------------------------


def test_save_image_set_writes_jsonl_with_image_id(output_dir, make_image_record):
    image_set = {"img-1": make_image_record()}

    save_image_set(image_set, output_dir)

    records = _read_jsonl(output_dir / "raw" / "image_set.jsonl")
    assert len(records) == 1
    assert records[0]["image_id"] == "img-1"


def test_save_image_set_appends_across_calls(output_dir, make_image_record):
    save_image_set({"img-1": make_image_record()}, output_dir)
    save_image_set({"img-2": make_image_record()}, output_dir)

    records = _read_jsonl(output_dir / "raw" / "image_set.jsonl")
    assert {r["image_id"] for r in records} == {"img-1", "img-2"}


def test_save_image_set_charts_table_reflects_full_history_not_just_latest_batch(
    output_dir, make_image_record
):
    """Regression test: charts.parquet must accumulate across calls, not get
    overwritten with only the most recently saved batch."""
    save_image_set({"img-1": make_image_record(pdf_id="pdf-1")}, output_dir)
    save_image_set({"img-2": make_image_record(pdf_id="pdf-2")}, output_dir)

    charts = pd.read_parquet(output_dir / "tables" / "charts.parquet")

    assert len(charts) == 2
    assert set(charts["chart_id"]) == {"img-1", "img-2"}
    assert set(charts["pdf_id"]) == {"pdf-1", "pdf-2"}


def test_save_image_set_charts_table_fields(output_dir, make_image_record):
    save_image_set({"img-1": make_image_record(pdf_id="pdf-1", page_number=5, chart_type="bar")}, output_dir)

    charts = pd.read_parquet(output_dir / "tables" / "charts.parquet")
    row = charts.iloc[0]

    assert row["chart_id"] == "img-1"
    assert row["pdf_id"] == "pdf-1"
    assert row["page_number"] == 5
    assert row["chart_type"] == "bar"
    assert row["model"] == "test-model"


def test_save_image_set_builds_series_and_points_tables(output_dir, make_image_record):
    series = [
        {
            "name": "Revenue",
            "points": [
                {"x": "2020", "y": 1.5, "y_unit": "USD"},
                {"x": "2021", "y": 2.5, "y_unit": "USD"},
            ],
        }
    ]
    save_image_set({"img-1": make_image_record(series=series)}, output_dir)

    series_df = pd.read_parquet(output_dir / "tables" / "series.parquet")
    points_df = pd.read_parquet(output_dir / "tables" / "points.parquet")

    assert len(series_df) == 1
    assert series_df.iloc[0]["series_name"] == "Revenue"
    assert series_df.iloc[0]["chart_id"] == "img-1"

    assert len(points_df) == 2
    assert list(points_df["x_raw"]) == ["2020", "2021"]
    assert list(points_df["x_type"]) == ["year", "year"]
    assert list(points_df["y"]) == [1.5, 2.5]


def test_save_image_set_handles_missing_image_path(output_dir, make_image_record):
    save_image_set({"img-1": make_image_record(image_path=None)}, output_dir)

    charts = pd.read_parquet(output_dir / "tables" / "charts.parquet")
    assert charts.iloc[0]["image_path"] is None


def test_save_image_set_creates_expected_directory_structure(output_dir, make_image_record):
    save_image_set({"img-1": make_image_record()}, output_dir)

    assert (output_dir / "raw" / "image_set.jsonl").exists()
    assert (output_dir / "tables" / "charts.parquet").exists()
    assert (output_dir / "tables" / "series.parquet").exists()
    assert (output_dir / "tables" / "points.parquet").exists()


# ---------------------------------------------------------------------------
# save_pdf_set
# ---------------------------------------------------------------------------


def test_save_pdf_set_writes_jsonl_with_pdf_id(output_dir, make_pdf_record):
    pdf_set = {"hash-1": make_pdf_record(file_name="Letter.pdf")}

    save_pdf_set(pdf_set, output_dir)

    records = _read_jsonl(output_dir / "raw" / "pdf_set.jsonl")
    assert len(records) == 1
    assert records[0]["pdf_id"] == "hash-1"
    assert records[0]["metadata"]["file_name"] == "Letter.pdf"


def test_save_pdf_set_appends_and_rebuilds_full_table(output_dir, make_pdf_record):
    """Regression test: pdfs.parquet must accumulate across calls."""
    save_pdf_set({"hash-1": make_pdf_record(file_name="a.pdf")}, output_dir)
    save_pdf_set({"hash-2": make_pdf_record(file_name="b.pdf")}, output_dir)

    records = _read_jsonl(output_dir / "raw" / "pdf_set.jsonl")
    assert {r["pdf_id"] for r in records} == {"hash-1", "hash-2"}

    pdfs = pd.read_parquet(output_dir / "tables" / "pdfs.parquet")
    assert len(pdfs) == 2
    assert set(pdfs["pdf_id"]) == {"hash-1", "hash-2"}


def test_save_pdf_set_creates_expected_directory_structure(output_dir, make_pdf_record):
    save_pdf_set({"hash-1": make_pdf_record()}, output_dir)

    assert (output_dir / "raw" / "pdf_set.jsonl").exists()
    assert (output_dir / "tables" / "pdfs.parquet").exists()
