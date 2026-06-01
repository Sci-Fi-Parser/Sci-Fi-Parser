import json
import shutil
from pathlib import Path
from datetime import datetime, timezone
from typing import Any
import cv2
import numpy as np


class DatasetWriter:
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)

        self.evidence_dir = self.output_dir / "evidence"
        self.chart_crops_dir = self.evidence_dir / "chart_crops"
        self.vlm_inputs_dir = self.evidence_dir / "vlm_inputs"
        self.overlays_dir = self.evidence_dir / "overlays"
        self.raw_responses_dir = self.evidence_dir / "raw_responses"
        self.raw_ocr_dir = self.evidence_dir / "raw_ocr"

    def setup(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.chart_crops_dir.mkdir(parents=True, exist_ok=True)
        self.vlm_inputs_dir.mkdir(parents=True, exist_ok=True)
        self.overlays_dir.mkdir(parents=True, exist_ok=True)
        self.raw_responses_dir.mkdir(parents=True, exist_ok=True)

    def write_jsonl(self, filename: str, record: dict[str, Any]) -> None:
        path = self.output_dir / filename

        record = {
            **record,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }

        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def write_run(self, record: dict) -> None:
        path = self.output_dir / "runs.jsonl"

        with path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(record, ensure_ascii=False)
                + "\n"
            )

    def write_ocr_result(self, record: dict[str, Any]) -> None:
        self.write_jsonl("ocr_results.jsonl", record)

    def write_vlm_result(self, record: dict[str, Any]) -> None:
        self.write_jsonl("vlm_results.jsonl", record)

    def write_error(self, record: dict[str, Any]) -> None:
        self.write_jsonl("errors.jsonl", record)

    
    def save_chart_crop(
        self,
        image: np.ndarray,
        chart_id: str,
    ) -> str:

        destination = self.chart_crops_dir / f"{chart_id}.png"

        cv2.imwrite(str(destination), image)

        return str(destination.relative_to(self.output_dir))
    
    def save_chart_crop(
        self,
        image: np.ndarray,
        chart_id: str,
    ) -> str:

        destination = self.chart_crops_dir / f"{chart_id}.png"

        cv2.imwrite(str(destination), image)

        return str(destination.relative_to(self.output_dir))

    def save_overlay(
        self,
        image: np.ndarray,
        chart_id: str,
    ) -> str:

        destination = self.overlays_dir / f"{chart_id}.png"

        cv2.imwrite(str(destination), image)

        return str(destination.relative_to(self.output_dir))
    
    def save_raw_ocr(
        self,
        chart_id: str,
        data: dict,
    ) -> str:

        destination = self.raw_ocr_dir / f"{chart_id}.json"

        with destination.open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                data,
                f,
                indent=2,
                ensure_ascii=False,
            )

        return str(
            destination.relative_to(
                self.output_dir
            )
        )
    
