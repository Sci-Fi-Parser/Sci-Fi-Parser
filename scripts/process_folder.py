"""Run the VLM over every chart image in a folder and print results.

Usage:
    python scripts/process_folder.py path/to/charts/
    python scripts/process_folder.py path/to/charts/ --backend api --model qwen2.5vl:7b
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from sci_fi_parser.vlm import APIConfig, VLMConfig, build_vlm

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path, help="Folder containing chart images")
    p.add_argument("--backend", default="api", choices=["api", "hf_transformers"])
    p.add_argument("--model", default="qwen2.5vl:7b")
    p.add_argument("--base-url", default="http://localhost:8080/v1",
                   help="OpenAI-compatible endpoint (api backend only)")
    p.add_argument("--schema-mode", default="schema", choices=["schema", "object"],
                   help="'schema' uses strict response_format=json_schema "
                        "(llama.cpp, OpenAI). 'object' uses loose json_object "
                        "(ollama's /v1/).")
    args = p.parse_args()

    if not args.folder.is_dir():
        print(f"error: {args.folder} is not a directory", file=sys.stderr)
        return 2

    images = sorted(
        f for f in args.folder.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        print(f"No images found in {args.folder}")
        return 0

    vlm = build_vlm(VLMConfig(
        backend=args.backend,
        model=args.model,
        api=APIConfig(base_url=args.base_url, schema_mode=args.schema_mode),
    ))
    print(f"Backend: {type(vlm).__name__} ({vlm.name})")
    print(f"Found {len(images)} image(s) in {args.folder}\n")

    for img in images:
        t0 = time.monotonic()
        try:
            chart = vlm.extract(img)
        except Exception as e:  # noqa: BLE001
            print(f"{img.name}: ERROR — {type(e).__name__}: {e}")
            continue
        elapsed = time.monotonic() - t0
        n_points = sum(len(s.points) for s in chart.series)
        print(f"{img.name}  [{elapsed:.1f}s]  {chart.chart_type}  "
              f"{len(chart.series)} series, {n_points} points")
        for s in chart.series:
            preview = ", ".join(f"({p.x}, {p.y})" for p in s.points[:5])
            tail = " ..." if len(s.points) > 5 else ""
            print(f"    {s.name}: {preview}{tail}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
