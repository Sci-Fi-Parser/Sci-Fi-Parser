"""Run the OCR -> VLM -> offload pipeline on a folder of chart images.

Edit ``TARGET_FOLDER`` / ``OUTPUT_FOLDER`` below; no CLI flags. The work
lives in :mod:`sci_fi_parser.data_pipeline` (data containers + offloader),
:mod:`sci_fi_parser.cv.pipeline` (OCR stage), and
:mod:`sci_fi_parser.vlm.pipeline` (VLM stage).
"""

from pathlib import Path

from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4

from sci_fi_parser.data_pipeline import OCRSet, VLMSet
from sci_fi_parser.vlm.pipeline import start_vlm
from sci_fi_parser.cv.pipeline import start_ocr
from sci_fi_parser.storage.writer import DatasetWriter


BASE_DIR = Path(__file__).resolve().parent

TARGET_FOLDER = BASE_DIR / "input_folder"
OUTPUT_FOLDER = BASE_DIR / "output_folder"

def main() -> None:

    writer = DatasetWriter(OUTPUT_FOLDER)
    writer.setup()

    run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"

    run_record = {
        "run_id": run_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "input_folder": str(TARGET_FOLDER),
        "output_folder": str(OUTPUT_FOLDER),
    }
    
    print("Current working directory:", Path.cwd())
    print("Input folder exists:", TARGET_FOLDER.exists())
    print("Input folder absolute path:", TARGET_FOLDER.resolve())

    if TARGET_FOLDER.exists():
        print("Files in input:")
        for f in TARGET_FOLDER.iterdir():
            print("  ", f)

    try:
        ocr_set = OCRSet()

        start_ocr(
            TARGET_FOLDER,
            ocr_set,
            writer,
            run_record["run_id"],
        )

        vlm_set = VLMSet()

        start_vlm(
            TARGET_FOLDER,
            ocr_set,
            vlm_set,
            writer,
            run_id
        )

        run_record["finished_at"] = (
            datetime.now(timezone.utc).isoformat()
        )

        run_record["status"] = "completed"

        run_record["charts_processed"] = len(ocr_set._data)

    except Exception as exc:

        run_record["finished_at"] = (
            datetime.now(timezone.utc).isoformat()
        )

        run_record["status"] = "failed"

        run_record["error_type"] = type(exc).__name__
        run_record["error_message"] = str(exc)

        raise

    finally:
        writer.write_run(run_record)
   
if __name__ == "__main__":

    main()
