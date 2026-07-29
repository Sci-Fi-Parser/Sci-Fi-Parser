from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def save_image_set(image_set, output_dir: Path) -> None:
    output_dir = Path(output_dir)
    raw_dir = output_dir / "raw"
    table_dir = output_dir / "tables"

    raw_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    raw_path = raw_dir / "image_set.jsonl"

    charts = []
    series_rows = []
    points = []

    with raw_path.open("a", encoding="utf-8", newline="\n") as f:
        for image_id, image_record in image_set.items():
            record = _record_to_dict(image_record)

            # Always preserve full raw record
            f.write(json.dumps(record, default=_json_default, ensure_ascii=False) + "\n")

            metadata = record.get("metadata", {})
            extraction = metadata.get("extraction", {})
            output = record.get("output", {})
            vlm = record.get("vlm", {})
            vlm_result = vlm.get("result", {}) or {}
            vlm_raw = vlm.get("raw", {}) or {}

            chart_id = str(image_id)

            charts.append(
                {
                    "chart_id": chart_id,
                    "pdf_id": extraction.get("pdf_id"),
                    "page_number": _safe_int(extraction.get("page_number")),
                    "source_type": extraction.get("source_type"),
                    "image_path": str(output.get("path")) if output.get("path") is not None else None,
                    "chart_type": vlm_result.get("chart_type"),
                    "model": vlm_raw.get("model"),
                    "created_at": vlm_raw.get("created_at"),
                    "total_duration": vlm_raw.get("total_duration"),
                    "load_duration": vlm_raw.get("load_duration"),
                    "eval_count": vlm_raw.get("eval_count"),
                    "eval_duration": vlm_raw.get("eval_duration"),
                }
            )

            for series_index, s in enumerate(vlm_result.get("series", []) or []):
                series_id = str(uuid4())

                series_rows.append(
                    {
                        "series_id": series_id,
                        "chart_id": chart_id,
                        "series_index": series_index,
                        "series_name": s.get("name"),
                    }
                )

                for point_index, p in enumerate(s.get("points", []) or []):
                    x_raw = p.get("x")

                    points.append(
                        {
                            "point_id": str(uuid4()),
                            "chart_id": chart_id,
                            "series_id": series_id,
                            "point_index": point_index,
                            "x_raw": str(x_raw) if x_raw is not None else None,
                            "x_numeric": _safe_float(x_raw),
                            "x_type": _infer_x_type(x_raw),
                            "y": _safe_float(p.get("y")),
                            "y_unit": p.get("y_unit"),
                        }
                    )

    pd.DataFrame(charts).to_parquet(table_dir / "charts.parquet", index=False)
    pd.DataFrame(series_rows).to_parquet(table_dir / "series.parquet", index=False)
    pd.DataFrame(points).to_parquet(table_dir / "points.parquet", index=False)


def _record_to_dict(record: Any) -> dict[str, Any]:
    if isinstance(record, dict):
        return record
    if hasattr(record, "model_dump"):
        return record.model_dump()
    if hasattr(record, "__dict__"):
        return record.__dict__
    raise TypeError(f"Cannot serialize record of type {type(record)}")


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _infer_x_type(value: Any) -> str:
    if value is None:
        return "unknown"

    value_str = str(value)

    if value_str.isdigit() and len(value_str) == 4:
        return "year"

    try:
        float(value_str)
        return "numeric"
    except ValueError:
        return "category"
