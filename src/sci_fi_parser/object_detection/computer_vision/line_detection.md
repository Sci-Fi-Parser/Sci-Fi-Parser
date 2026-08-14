## Components

- `lines.py` detects horizontal and vertical lines with Hough transforms and
  directional morphology.
- `axis_candidates.py` fuses vertical observations from both detectors.
- `axis.py` optionally ranks fused lines as left Y-axis candidates. It accepts
  numeric labels and bars already classified by upstream components; it does
  not run OCR, parse label text, detect ticks, or map pixels to values.
- `axis_ocr.py` is a small replaceable adapter from PaddleOCR output to numeric
  labels. It is used by development tools until the label classifier replaces
  it.
- `scripts/debug_lines.py` writes raw and merged line overlays.
- `scripts/debug_y_axis.py` writes source, fusion, evidence, and selection
  overlays plus JSON reports and an HTML review page.
- `line_benchmark.py` scores synthetic structural truth, or measures runtime and
  candidate counts on unlabeled folders.
- `axis_benchmark.py` scores Y-axis selection with oracle classified labels.

Bounding boxes use `(left, top, right, bottom)` image coordinates. Image Y
coordinates increase downwards, so a conventional numeric Y scale must have a
negative correlation between label value and label center Y.

## Debug a folder

```bash
uv run python scripts/debug_lines.py path/to/charts --output output/line_debug
```

Use `--morphology` to write morphology results instead of merged Hough lines,
and `--limit N` for a quick sample.

## Debug Y-axis evidence

```bash
uv run python scripts/debug_y_axis.py path/to/charts \
  --output output/y_axis_review \
  --ocr-cache temp/y_axis_ocr_cache
```

The debugger runs PaddleOCR and bar detection by default. OCR results are
cached. Use `--refresh-ocr` after changing images or OCR settings, and
`--no-ocr` for structural-only inspection.

## Benchmark a folder

```bash
uv run python -m sci_fi_parser.benchmark.line_benchmark path/to/charts \
  --output reports/line_benchmark.json \
  --runs 3
```

The report contains per-image dimensions, runtime, Hough and morphology line
counts, fused vertical-candidate counts, and aggregate runtime statistics.

Generate and score structural truth:

```bash
uv run python -m scripts.synthetic.cli \
  --axis-challenge \
  --out train_data/axis_challenge \
  --refresh

uv run python -m sci_fi_parser.benchmark.line_benchmark \
  --data train_data/axis_challenge \
  --output reports/line_benchmark.json

uv run python -m sci_fi_parser.benchmark.axis_benchmark \
  --data train_data/axis_challenge \
  --output reports/axis_benchmark.json \
  --mode both
```

Use `--mode oracle` to isolate line generation and ranking, or `--mode
detected` to include PaddleOCR, numeric parsing, and bar detection.
